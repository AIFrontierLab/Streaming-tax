"""`streamreplace score`: write `<job>.scores.jsonl` for the job files of the outputs folder (scoring/score.py).

  streamreplace score                         rules + cached judge answers (rows waiting for a judge stay unscored)
  streamreplace score --judge --judge-gpus 0,1   also send requests without a cached answer to their judges
  streamreplace score --cache-only            reproduce scores from the judge caches alone; exit 1, writing nothing,
                                              if any judge request has no cached answer
  streamreplace score --selftest --judge-gpus 0,1   only run the judges' self-tests (RJScore, HarmBench classifiers)

Judges and their checkpoints are configured in configs/judges.yaml; see docs/judges.md. Before new RJScore requests
are sent, the RJScore judge checks itself against frozen reference scores on the given GPUs and refuses to judge on a
disagreement (--skip-selftest disables this). Scoring refuses when the offline jobs of a pair (configs/pairs.yaml)
sampled different frames for a row (--no-frame-check disables this)."""
from __future__ import annotations

import argparse
import sys

from . import context, csv_list, gpu_list


def register(subparsers) -> None:
    parser = subparsers.add_parser("score", help="score generation rows (rules, then cached or new judge answers)",
                                   description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmarks", type=csv_list, help="benchmarks or suites (default: every benchmark folder)")
    parser.add_argument("--models", type=csv_list, help="model ids (a plugin arm also as <model>@<variant>)")
    parser.add_argument("--modes", type=csv_list, help="offline and/or stream")
    parser.add_argument("--jobs", type=csv_list, help="exact job names, e.g. aura_8b.stream")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--judge", action="store_true", help="send requests without a cached answer to their judges")
    group.add_argument("--cache-only", action="store_true",
                       help="use cached judge answers only; fail (exit 1, nothing written) on any missing answer")
    group.add_argument("--selftest", action="store_true", help="only run the judges' self-tests on --judge-gpus")
    parser.add_argument("--judge-gpus", type=gpu_list,
                        help="GPUs for the judges (default: each judge's own: 0; the RJScore judge 0,1)")
    parser.add_argument("--judges", type=csv_list,
                        help="with --judge/--selftest: only these judges (extract, harmbench_cls, rjscore, lingo)")
    parser.add_argument("--skip-selftest", action="store_true",
                        help="with --judge: do not self-test the RJScore judge before sending requests")
    parser.add_argument("--no-frame-check", action="store_true",
                        help="do not require the offline jobs of a pair to have sampled identical frames")
    parser.set_defaults(func=handler)


def _selftests(ctx, args) -> int:
    from ..judges.registry import build, load_specs
    from ..judges.selftest import SelfTestFailed
    section = ctx.config.section("judges")
    names = args.judges or [n for n in ("rjscore", "harmbench_cls") if n in load_specs(section)]
    failed = 0
    for name in names:
        judge = build(name, paths=ctx.paths, section=section)
        try:
            judge.self_test(args.judge_gpus or list(judge.DEFAULT_GPUS))     # RJScore: 0,1; the others: 0
        except SelfTestFailed as exc:
            print(exc, file=sys.stderr)
            failed += 1
    return 1 if failed else 0


def handler(args) -> int:
    from ..judges.selftest import SelfTestFailed
    from ..scoring.score import CacheMiss, score
    ctx = context(args)
    if args.selftest:
        return _selftests(ctx, args)
    benchmarks = ctx.config.expand_benchmarks(args.benchmarks) if args.benchmarks else None
    pair_of = None if args.no_frame_check else {m: spec.pair for m, spec in ctx.config.models.items() if spec.pair}
    try:
        report = score(ctx.paths.outputs, benchmarks=benchmarks, models=args.models, modes=args.modes,
                       jobs=args.jobs, run_judges=args.judge, cache_only=args.cache_only,
                       judge_gpus=args.judge_gpus, only_judges=args.judges, selftest=not args.skip_selftest,
                       judge_section=ctx.config.section("judges"), pair_of=pair_of, paths=ctx.paths)
    except CacheMiss as exc:
        print(f"streamreplace score --cache-only: {exc}; nothing was written", file=sys.stderr)
        for name, files in exc.report.missed.items():
            for job, n in sorted(files.items()):
                print(f"  {name:14s} {job}: {n} rows", file=sys.stderr)
        return 1
    except SelfTestFailed as exc:                     # the RJScore judge refused this setup: no score file is written
        print(exc, file=sys.stderr)
        return 1
    for name, n in report.requests.items():
        print(f"judge {name}: {n} requests, {report.misses.get(name, 0)} without a cached answer "
              f"({report.caches[name]})")
    if not report.jobs and not report.skipped:
        print(f"no job files found under {ctx.paths.outputs}")
    return 0
