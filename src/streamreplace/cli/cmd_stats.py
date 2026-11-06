"""`streamreplace stats`: every C-P, S-C and S-P contrast of the analysis profile (configs/analysis.yaml).

  streamreplace stats                 <outputs>/analysis/contrasts.json, contrasts.csv, GAPS.md
  streamreplace stats --no-overlay    the same on the formal rows only, in <outputs>/analysis/no-overlay/

Each contrast: both arms' values, delta = b - a in points (+ = the child, or its stream run, is better; ASR and
location leakage as parent - child), a paired cluster-bootstrap 95% / 90% interval (10,000 multinomial draws, a fresh
numpy default_rng(0) per contrast), exact McNemar or cluster sign-flip p (VMDT: the official composite bootstrap, 20,000
draws), Holm within the declared and the main-text families, TOST at +-3 / +-2 points, non-inferiority at -2 / -3 / -5,
response coverage, the replacement verdict and the cell's flags (forced, invalid, non_answer, chance, artifact).
docs/statistics.md documents every choice.

Generation-cap reruns (overlay `truncation_rerun`, on by default): on the ids the parent's or the child's formal
offline row stopped at the generation cap, both offline arms read the rows rerun at 8,192 tokens from
<outputs>/rerun_capped/<pair>/ (see `streamreplace rerun-capped`); ids from <that folder>/capped.json, else
paper/frozen/rerun_capped/capped.json (the paper's), unless --rerun-ids.
The run fails when that folder is missing; use --no-overlay for the formal rows only.

Arm values are checked against the benchmarks' own metrics(); disagreements are printed and recorded under
checks.official_mismatches (exit code 1 with --strict). A contrast whose two jobs exist but share no counted row (rows
not scored yet, e.g. safety rows waiting for `score --judge`) is left out like a missing job, printed and listed under
checks.uncounted. numpy's random streams are version-dependent in principle;
the paper's numbers reproduce under numpy 2.2-2.5 (docs/statistics.md)."""
from __future__ import annotations

import argparse
import sys

from . import context


def register(subparsers) -> None:
    parser = subparsers.add_parser("stats", help="paired P/C/S contrasts with Holm, TOST / NI, verdicts and flags",
                                   description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", metavar="FILE", help="analysis profile YAML instead of configs/analysis.yaml")
    parser.add_argument("--no-overlay", action="store_true",
                        help="formal rows only (no generation-cap reruns); default output folder analysis/no-overlay")
    parser.add_argument("--rerun-ids", metavar="FILE",
                        help="capped ids of the rerun overlay (default: <rerun rows>/capped.json written by `rerun-capped "
                             "select`, else the profile's paper/frozen/rerun_capped/capped.json)")
    parser.add_argument("--rerun-rows", metavar="DIR",
                        help="folder holding <pair>/<benchmark>/<model>.offline.jsonl of the reruns (default "
                             "<outputs>/rerun_capped)")
    parser.add_argument("--flags", metavar="DIR",
                        help="folder of external flag tables (default: the profile's, paper/flags)")
    parser.add_argument("--out", metavar="DIR", help="output folder (default <outputs>/analysis[/no-overlay])")
    parser.add_argument("--strict", action="store_true", help="exit 1 when an arm value differs from the benchmark's "
                                                              "own metric")
    parser.set_defaults(func=handler)


def handler(args) -> int:
    from ..analysis.pipeline import stats
    from ..analysis.records import Catalog
    from ..analysis.specs import profile_from_config
    ctx = context(args)
    profile = profile_from_config(ctx.config, args.profile)
    doc, files = stats(ctx.paths.outputs, profile, Catalog(), out=args.out, overlay=not args.no_overlay,
                       rerun_ids=args.rerun_ids, rerun_rows=args.rerun_rows, flags=args.flags)
    from ..analysis.render.gaps_md import summary_lines
    for line in summary_lines(doc["rows"]):
        print(line)
    bad = doc["checks"]["official_mismatches"]
    for line in bad:
        print(f"warning: arm value differs from the benchmark's metric: {line}", file=sys.stderr)
    for line in doc["checks"].get("uncounted", []):
        print(f"warning: contrast left out: {line}", file=sys.stderr)
    for path in files.values():
        print(f"wrote {path}")
    print(f"{len(doc['rows'])} contrasts" + (f" with the {doc['overlay']['name']} overlay" if doc["overlay"] else ""))
    return 1 if bad and args.strict else 0
