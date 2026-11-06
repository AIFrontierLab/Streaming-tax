"""`streamreplace download`: checkpoints and benchmark data at the pinned revisions.

    streamreplace download models   [ID ...|all] [--processors-only]   model checkpoints (configs/models.yaml)
    streamreplace download judges   [ID ...|all]                       judge checkpoints (configs/judges.yaml)
    streamreplace download datasets [NAME ...|SUITE|GROUP|all]         benchmark data (manifests/datasets.yaml)
    streamreplace download list                                        what the manifests pin, with licences

Checkpoints go to <checkpoints>/<org>/<name> (or where a paths file maps them) at exactly the revision the
configuration pins, which must agree with manifests/checkpoints.yaml; datasets go to <data>/<name>/raw/, the folder
`streamreplace prepare` reads. Every file is checked against its pinned size and checksum.

--dry-run resolves every pin (Hugging Face API at the revision, GitHub files and folder digests at the commit,
fixed-URL sizes, Google Drive ids) and prints the plan without downloading anything; add --offline to print the plan
without any network request. --verify-only checks what is already on disk (no network).

Unpinned sources are refused (OpenUAV-QA's ModelScope frames need --allow-unpinned). Licences that need an explicit
agreement (LingoQA) need --accept-terms NAME; gated Hugging Face data (Video-SafetyBench) print the access step.
Real downloads need huggingface_hub, and gdown for Google Drive files (TOMATO videos, LingoQA). See docs/data.md."""
from __future__ import annotations

import argparse
import json
import sys

from . import context


def register(subparsers) -> None:
    parser = subparsers.add_parser("download", help="download checkpoints and benchmark data at pinned revisions",
                                   description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=("models", "judges", "datasets", "list"), help="what to download")
    parser.add_argument("names", nargs="*", metavar="NAME", help="model / judge ids or repositories, benchmark names, "
                        "suites (paper-core, paper-supp, finetune), groups, or all (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="resolve every pin and print the plan; download nothing")
    parser.add_argument("--offline", action="store_true", help="with --dry-run: print the plan without network requests")
    parser.add_argument("--verify-only", action="store_true", help="check the files on disk against the pins (offline)")
    parser.add_argument("--size-only-above", type=int, metavar="BYTES",
                        help="with --verify-only: check files larger than this by size only (quick check)")
    parser.add_argument("--processors-only", action="store_true",
                        help="models / judges: skip the weight files (tokenizers, processors and configs only)")
    parser.add_argument("--accept-terms", action="append", default=[], metavar="NAME",
                        help="confirm that you accept the licence of dataset NAME (repeatable; needed for lingoqa)")
    parser.add_argument("--allow-unpinned", action="store_true",
                        help="allow sources without an immutable revision (OpenUAV-QA frames on ModelScope)")
    parser.add_argument("--frames-for", metavar="OUTPUTS",
                        help="openuav_qa: download the frames of the trajectories selected in OUTPUTS "
                             "(subset.json / train_order.json of `streamreplace finetune prepare openuav_qa`)")
    parser.add_argument("--workers", type=int, default=8, help="parallel file downloads (default 8)")
    parser.add_argument("--force", action="store_true", help="download again even when the files verify")
    parser.add_argument("--json", action="store_true", help="dry run / verify: print the result as JSON")
    parser.set_defaults(func=run)


def _print_list(checkpoints, datasets) -> int:
    print("checkpoints (manifests/checkpoints.yaml):")
    for c in checkpoints.values():
        users = ", ".join(f"{k}" if v == "checkpoint" else f"{k} ({v})" for k, v in c.used_by.items())
        print(f"  {c.role:6s} {c.repo}@{c.revision[:10]}  licence: {c.license}  used by: {users}")
    print("\ndatasets (manifests/datasets.yaml):")
    for d in datasets.values():
        flags = (" [needs --accept-terms]" if d.requires_acceptance else "") + (" [unpinned source]" if d.unpinned else "")
        kinds = [s.type for s in d.sources]
        kinds = " + ".join(f"{kinds.count(k)}x {k}" if kinds.count(k) > 1 else k for k in dict.fromkeys(kinds))
        print(f"  {d.group:11s} {d.name:18s} {kinds:18s} licence: {d.license}{flags}")
    return 0


def _done(args) -> str:
    if args.dry_run:
        return "planned (pins not resolved: --offline)" if args.offline else "resolved"
    return "verified" if args.verify_only else "downloaded and verified"


def _checkpoints(args, ctx, download) -> int:
    manifest = download.load_checkpoints()
    plans = download.plan_checkpoints(ctx.config, args.what, args.names, ctx.paths, manifest,
                                      processors_only=args.processors_only)
    results, failed = [], 0
    for plan in plans:
        entry = {"repo": plan.repo, "revision": plan.revision, "target": str(plan.target), "used_by": plan.users}
        if args.dry_run:
            if args.offline:
                entry.update(ok=True, detail="not resolved (--offline)")
            else:
                try:
                    resolution = download.resolve_checkpoint(plan)
                    entry.update(ok=resolution.ok, detail=resolution.detail)
                except Exception as exc:           # noqa: BLE001 - reported per checkpoint
                    entry.update(ok=False, detail=f"{type(exc).__name__}: {exc}")
        elif args.verify_only:
            problems = download.verify_checkpoint(plan)
            entry.update(ok=not problems, detail="; ".join(problems) or "revision recorded")
        else:
            try:
                download.download_checkpoint(plan, workers=args.workers, force=args.force)
                problems = download.verify_checkpoint(plan)
                entry.update(ok=not problems, detail="; ".join(problems) or "downloaded")
            except download.DownloadError as exc:
                entry.update(ok=False, detail=str(exc))
        failed += not entry["ok"]
        results.append(entry)
        if not args.json:
            print(f"{'ok  ' if entry['ok'] else 'FAIL'} {plan.label} -> {plan.target}  [{', '.join(plan.users)}]"
                  f"{'  licence: ' + plan.license if plan.license and args.dry_run else ''}\n       {entry['detail']}")
    if args.json:
        print(json.dumps({"what": args.what, "results": results}, indent=1))
    else:
        print(f"{len(plans) - failed}/{len(plans)} checkpoints {_done(args)}")
    return 1 if failed else 0


def _datasets(args, ctx, download) -> int:
    manifest = download.load_datasets()
    suites = ctx.config.suites if hasattr(ctx.config, "suites") else {}
    selected = download.select_datasets(manifest, args.names, suites)
    if not args.verify_only and not (args.dry_run and args.offline):
        if args.dry_run:                 # a dry run reports refusals instead of stopping at the first one
            refused = []
            for ds in selected:
                try:
                    download.check_allowed([ds], args.accept_terms, args.allow_unpinned)
                except download.Refused as exc:
                    refused.append(str(exc))
        else:
            download.check_allowed(selected, args.accept_terms, args.allow_unpinned)
            refused = []
    else:
        refused = []
    results, failed = [], 0
    for ds in selected:
        raw = ctx.paths.dataset_dir(ds.name) / "raw"
        entry = {"dataset": ds.name, "target": str(raw), "license": ds.license, "sources": []}
        if args.dry_run:
            for source in ds.sources:
                if args.offline:
                    resolution = download.Resolution(f"{ds.name} {source.label}", True, "not resolved (--offline)")
                else:
                    try:
                        resolution = download.resolve_source(ds, source)
                    except Exception as exc:       # noqa: BLE001 - one unresolvable source must not stop the plan
                        resolution = download.Resolution(f"{ds.name} {source.label}", False,
                                                         f"{type(exc).__name__}: {exc}")
                entry["sources"].append({"source": source.label, "ok": resolution.ok, "detail": resolution.detail})
            entry["ok"] = all(s["ok"] for s in entry["sources"])
        elif args.verify_only:
            problems = download.verify_dataset(ds, raw, args.size_only_above)
            entry.update(ok=not problems, problems=problems)
        else:
            try:
                problems = download.download_dataset(ds, raw, workers=args.workers, force=args.force,
                                                     frames_for=args.frames_for)
                entry.update(ok=not problems, problems=problems)
            except download.DownloadError as exc:
                entry.update(ok=False, problems=[str(exc)])
        failed += not entry["ok"]
        results.append(entry)
        if not args.json:
            print(f"{'ok  ' if entry['ok'] else 'FAIL'} {ds.name} -> {raw}  (licence: {ds.license})")
            for s in entry["sources"]:
                print(f"       {'ok  ' if s['ok'] else 'FAIL'} {s['source']}: {s['detail']}")
            for problem in entry.get("problems", [])[:20]:
                print(f"       {problem}")
            if args.dry_run and ds.terms:
                print(f"       note: {ds.terms}")
    if args.json:
        print(json.dumps({"what": "datasets", "results": results, "refused": refused}, indent=1))
    else:
        for message in refused:
            print(f"note: a real download would be refused: {message}")
        print(f"{len(selected) - failed}/{len(selected)} datasets {_done(args)}")
    return 1 if failed else 0


def run(args) -> int:
    from ..provisioning import download
    ctx = context(args)
    if args.offline and not args.dry_run:
        print("streamreplace download: --offline only goes with --dry-run", file=sys.stderr)
        return 2
    try:
        if args.what == "list":
            return _print_list(download.load_checkpoints(), download.load_datasets())
        if args.what in ("models", "judges"):
            return _checkpoints(args, ctx, download)
        return _datasets(args, ctx, download)
    except (download.Refused, download.ManifestError) as exc:
        print(f"streamreplace download: {exc}", file=sys.stderr)
        return 1
