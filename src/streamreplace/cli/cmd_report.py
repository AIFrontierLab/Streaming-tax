"""`streamreplace report`: metrics and completion counts of every scored job in the outputs folder.

  streamreplace report                          <outputs>/metrics.json, RESULTS.md, pairs.csv
  streamreplace --outputs runs/quick report     the same for another folder (a subset.json there is respected)

metrics.json holds, per job (benchmark x model[@plugin arm] x mode), the benchmark's own metrics (headline first,
percentages rounded to 2 decimals as the benchmark computes them), the rule-only metrics of judged jobs and the counts
(expected, done, errors, no_answer, forced, parse_fail, truncated, unscored). The paper's absolute-score tables are
rendered from it (`streamreplace tables`). RESULTS.md is the development report: one table per pair with each job's
headline and its difference to the parent, an exact McNemar test per benchmark stratum, the VMDT benchmarks' own
paired bootstrap (2,000 draws), and per-benchmark details; pairs.csv lists every (parent, candidate) question pair.
Run `streamreplace score` first; rows waiting for a judge are not counted (see `unscored`)."""
from __future__ import annotations

import argparse

from . import context, csv_list


def register(subparsers) -> None:
    parser = subparsers.add_parser("report", help="metrics.json, RESULTS.md and pairs.csv of the scored jobs",
                                   description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmarks", type=csv_list, help="only these benchmarks or suites (default: all found)")
    parser.add_argument("--profile", metavar="FILE",
                        help="analysis profile YAML instead of configs/analysis.yaml (pairs, report order, names)")
    parser.set_defaults(func=handler)


def handler(args) -> int:
    from ..analysis.pipeline import report
    from ..analysis.records import Catalog
    from ..analysis.specs import profile_from_config
    ctx = context(args)
    profile = profile_from_config(ctx.config, args.profile)
    datasets = ctx.config.expand_benchmarks(args.benchmarks) if args.benchmarks else None
    files = report(ctx.paths.outputs, profile, Catalog(), datasets)
    for name, path in files.items():
        print(f"wrote {path}")
    return 0
