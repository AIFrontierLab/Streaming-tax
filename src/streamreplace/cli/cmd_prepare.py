"""`streamreplace prepare`: raw benchmark data -> the prepared samples every run reads, checked against the data the
paper's runs used.

    streamreplace prepare paper-core                 # prepare every missing benchmark of the suite, then check it
    streamreplace prepare videomme --verify          # check only (writes nothing)
    streamreplace prepare videomme --video-meta      # also decode every video and compare with manifests/video_meta

For each benchmark, `<data>/<benchmark>/raw` is read and `<data>/<benchmark>/prepared/samples.jsonl` (+ manifest.json,
and train.jsonl for fine-tuning sets) is written; existing files are kept unless --force. Then the checks run: sample
count (the benchmark's expected size and manifests/prepared.yaml), sha256 of samples.jsonl and the fingerprint of all
rendered prompts (manifests/prepared.yaml), and the list of videos (manifests/video_meta/<benchmark>.jsonl). With
--video-meta the videos are also measured with the local libraries (decodable frames with PyAV, fps with decord) and
compared with the manifest; this fills the frame-count cache the runs read. One JSON line per benchmark; differences
are printed as warnings and give exit code 1. `verified` says whether manifests/prepared.yaml has a row to compare with
(every benchmark of the paper has one; a benchmark you added is reported as not verified, with a note)."""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import context


def register(subparsers) -> None:
    parser = subparsers.add_parser("prepare", help="raw benchmark data -> prepared samples (+ checks)",
                                   description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benchmarks", nargs="+", help="benchmark or suite names, or 'all'")
    parser.add_argument("--force", action="store_true",
                        help="rebuild samples.jsonl (and train.jsonl) even when present (extracted media are reused)")
    parser.add_argument("--verify", action="store_true", help="only check already prepared benchmarks; write nothing")
    parser.add_argument("--video-meta", action="store_true",
                        help="measure the videos with the local libraries and compare them with manifests/video_meta")
    parser.add_argument("--video-sample", type=float, default=1.0,
                        help="fraction of the videos --video-meta measures (seeded random choice; default: all)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="processes measuring videos")
    parser.set_defaults(func=handler)


def handler(args) -> int:
    from ..data.prepare import check_prepared, check_videos, prepare_dataset
    ctx = context(args)
    names = ctx.config.expand_benchmarks([n for item in args.benchmarks for n in item.split(",") if n])
    failed = False
    for name in names:
        record: dict = {"dataset": name}
        problems: list[str] = []
        try:
            if not args.verify:
                result = prepare_dataset(name, args.force, ctx.paths)
                record.update(status=result.get("status"), samples=result.get("samples"), kinds=result.get("kinds"))
            checks = check_prepared(name, ctx.paths)
            videos = check_videos(name, ctx.paths, fraction=args.video_sample if args.video_meta else 0.0,
                                  workers=args.workers)
        except Exception as exc:                  # noqa: BLE001 - reported per benchmark; the others still run
            record["error"] = f"{type(exc).__name__}: {exc}"
            print(json.dumps(record, ensure_ascii=False), flush=True)
            failed = True
            continue
        problems = checks["problems"] + videos["problems"]
        record.update(count=checks["count"], samples_sha256=checks["samples_sha256"],
                      prompts_sha256=checks.get("prompts_sha256"),
                      expected=sorted(checks["expected"]) or None, verified=bool(checks["expected"]),
                      videos={k: v for k, v in videos.items() if k != "problems"} or None,
                      ok=not problems)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        for problem in problems:
            print(f"WARNING {name}: {problem}", file=sys.stderr, flush=True)
        for note in checks.get("notes", []):
            print(f"NOTE {name}: {note}", file=sys.stderr, flush=True)
        failed |= bool(problems)
    return 1 if failed else 0
