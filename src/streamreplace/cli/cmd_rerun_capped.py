"""`streamreplace rerun-capped`: rerun the offline rows that stopped at the generation cap, at a larger cap.

  streamreplace rerun-capped select                 capped ids of every pair member -> <root>/<pair>/subset.json and
                                                    <root>/capped.json
  streamreplace rerun-capped run --gpus 0,1         generate the selected ids again at 8,192 tokens, every member of
                                                    each pair, in <root>/<pair>/ (resumable; --dry-run, --launcher print)
  streamreplace --outputs <root>/<pair> score       score each rerun folder (rules + the judge fallback)
  streamreplace rerun-capped report                 <root>/TRUNC.md and trunc.json: formal vs merged per pair
  streamreplace stats                               the contrasts read the rerun rows on the capped ids (default)

<root> defaults to <outputs>/rerun_capped; pairs, benchmarks, the rerun cap and the margin default to configs/
analysis.yaml (overlays.truncation_rerun: p2-p5 on MathVista, MMMU and MMStar at 8,192 tokens, as in the paper).
A row is capped when its finish is "length"; rows without a recorded stop reason are counted with the model's own
tokenizer (in its environment, on CPU) and count as capped at >= cap - margin tokens. The paper's selection ships as
paper/frozen/rerun_capped/capped.json. See run/rerun_capped.py and docs/statistics.md."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import context, csv_list, gpu_list


def register(subparsers) -> None:
    parser = subparsers.add_parser("rerun-capped", help="select, rerun and report the rows capped by max_new_tokens",
                                   description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    actions = parser.add_subparsers(dest="action", metavar="<action>", required=True)

    def common(p):
        p.add_argument("--root", metavar="DIR", help="rerun folder (default <outputs>/rerun_capped)")
        p.add_argument("--pairs", type=csv_list, help="pair ids (default: the analysis profile's)")
        p.add_argument("--benchmarks", type=csv_list, help="benchmarks (default: the analysis profile's)")

    s = actions.add_parser("select", help="write <root>/<pair>/subset.json and <root>/capped.json")
    common(s)
    s.add_argument("--margin", type=int, help="token margin for rows without a recorded finish (default 8)")
    s.add_argument("--cap", type=int, help="rerun cap recorded in capped.json (default 8192)")
    s.add_argument("--token-counts", metavar="FILE",
                   help="JSON {model: {id: tokens}} for rows without a recorded finish (skips the tokenizer run)")
    r = actions.add_parser("run", help="generate the selected ids again at the rerun cap")
    common(r)
    r.add_argument("--cap", type=int, help="max_new_tokens of the rerun (default 8192)")
    r.add_argument("--gpus", type=gpu_list, help="GPU slots (default: run.gpus in configs)")
    r.add_argument("--shards", type=int, default=1)
    r.add_argument("--threads", type=int)
    r.add_argument("--fresh", action="store_true", help="archive rerun jobs whose identity changed and redo them")
    r.add_argument("--dry-run", action="store_true", help="print the planned jobs; run nothing")
    r.add_argument("--launcher", choices=("local", "print"), default="local")
    t = actions.add_parser("report", help="write <root>/TRUNC.md and trunc.json")
    t.add_argument("--root", metavar="DIR", help="rerun folder (default <outputs>/rerun_capped)")
    t.add_argument("--capped", metavar="FILE", help="the selection (default <root>/capped.json, else the paper's "
                                                    "frozen list of configs/analysis.yaml)")
    parser.set_defaults(func=handler)


def _spec(ctx, args) -> dict:
    from ..analysis.specs import profile_from_config
    profile = profile_from_config(ctx.config)
    spec = dict((profile.overlays or {}).get("truncation_rerun") or {})
    root = Path(args.root) if getattr(args, "root", None) else Path(ctx.paths.outputs) / spec.get("rows", "rerun_capped")
    return {"root": root, "pairs": getattr(args, "pairs", None) or list(spec.get("pairs", [])),
            "datasets": (ctx.config.expand_benchmarks(args.benchmarks) if getattr(args, "benchmarks", None)
                         else list(spec.get("datasets", []))),
            "cap": getattr(args, "cap", None) or int(spec.get("cap", 8192)),
            "margin": getattr(args, "margin", None) or int(spec.get("margin", 8)),
            "shared": dict(spec.get("shared") or {}),
            "frozen": profile.path(spec["ids"]) if spec.get("ids") else None}


def handler(args) -> int:
    from ..run import rerun_capped
    ctx = context(args)
    spec = _spec(ctx, args)
    if args.action != "report" and (not spec["pairs"] or not spec["datasets"]):
        raise ValueError("no pairs / benchmarks: pass --pairs and --benchmarks (or set overlays.truncation_rerun in "
                         "configs/analysis.yaml)")
    if args.action == "select":
        counts = json.loads(Path(args.token_counts).read_text(encoding="utf-8")) if args.token_counts else None
        summary = rerun_capped.select(ctx.config, ctx.paths, ctx.paths.outputs, spec["root"], spec["pairs"],
                                      spec["datasets"], spec["cap"], spec["margin"], counts)
        for key, value in summary.items():
            print(f"{key}: " + ", ".join(f"{k} {v}" for k, v in value.items()))
        print(f"wrote {spec['root'] / rerun_capped.CAPPED_FILE} and <pair>/subset.json")
        return 0
    if args.action == "run":
        from ..run import dispatch
        from .cmd_run import print_states
        jobs = rerun_capped.plan(ctx.config, ctx.paths, spec["root"], spec["pairs"], spec["datasets"], spec["cap"])
        if not jobs:
            print("nothing to rerun: no capped ids selected")
            return 0
        inputs = dispatch.Inputs(ctx.paths)
        if args.dry_run:
            print_states(jobs, inputs, identities=True)
            print(f"{len(jobs)} rerun jobs planned in {spec['root']}")
            return 0
        gpus = args.gpus or list(ctx.config.run.get("gpus") or [0])
        status = dispatch.run(jobs, gpus=gpus, shards=args.shards, threads=args.threads, fresh=args.fresh,
                              launcher=args.launcher, paths=ctx.paths, inputs=inputs)
        if args.launcher == "print":
            return 0
        incomplete = {name: s for name, s in status.items() if s != "complete"}
        for name, s in incomplete.items():
            print(f"  {name}: {s}")
        folders = sorted({str(job.outputs) for job in jobs})
        print("next: " + "; ".join(f"streamreplace --outputs {f} score" for f in folders))
        return 2 if incomplete else 0
    capped = args.capped or (None if (spec["root"] / rerun_capped.CAPPED_FILE).is_file() else spec["frozen"])
    target = rerun_capped.report(ctx.config, ctx.paths.outputs, spec["root"], capped_file=capped,
                                 shared=spec["shared"])
    print(f"wrote {target} and {target.with_name('trunc.json')}")
    return 0
