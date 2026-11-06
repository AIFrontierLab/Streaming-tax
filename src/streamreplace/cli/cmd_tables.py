"""`streamreplace tables`: the paper's tables and figure data from the files of `report` and `stats`.

  streamreplace report && streamreplace stats && streamreplace tables --paper     -> <outputs>/tables/

Written (each .tex with a .csv of unrounded values next to it):
  gaps.tex, gaps.json, gaps.csv     Table 2 (offline C-P on the image headlines, their mean, HarmBench, prompt
                                    robustness) and its cell data, with the generation-cap reruns merged
  gaps_figure.json / .csv           the gap figure: C-P and S-P on the video, safety and VMDT columns (108 cells)
  master_cp, master_sp (.tex/.md/.csv)  every C-P and every S-P contrast
  eval_image, eval_video, eval_safety, eval_vmdt   absolute scores (formal rows, from metrics.json)
  eval_robustness, eval_instruction  prompt robustness (from the robustness report's results.json)
  eval_finetuning, eval_finetuning_gains  fine-tuning (from the fine-tuning report's ft_tables/results.json)
  eval_common_response              the VSB response-form audit (paper/flags/common_response.csv)

Inputs (defaults under the outputs folder): analysis/contrasts.json (`stats`), metrics.json (`report`),
robustness/report/contrasts.csv and results.json (`robustness report`; the Table 2 robustness columns and two
supplement tables), finetune/ft_tables/cells.csv and results.json (`finetune report`; the fine-tuning column of
gaps.json and two supplement tables). Tables whose inputs are missing are skipped and listed. --paper: the paper's
display conventions (the master tables omit the coverage mark).

--check [FILE] then compares every written table and the gap figure's printed cells with an expected tables file,
cell by cell (default paper/expected/tables.json, the paper's printed cells), lists each differing cell and gives exit
code 1 when one differs. On the paper's rows (`tables --paper --check`) exactly four master-table cells differ:
ThinkStream VMDT cells that the paper rendered with an unreleased scoring variant (docs/reproducing-the-paper.md,
"Known differences")."""
from __future__ import annotations

import argparse

from . import context


def register(subparsers) -> None:
    parser = subparsers.add_parser("tables", help="the paper's LaTeX tables (+ csv) and figure data",
                                   description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper", action="store_true", help="the paper's display conventions")
    parser.add_argument("--out", metavar="DIR", help="output folder (default <outputs>/tables)")
    parser.add_argument("--profile", metavar="FILE", help="analysis profile YAML instead of configs/analysis.yaml")
    parser.add_argument("--contrasts", metavar="FILE", help="contrasts.json of `stats` (default "
                                                            "<outputs>/analysis/contrasts.json)")
    parser.add_argument("--metrics", metavar="FILE", help="metrics.json of `report` (default <outputs>/metrics.json)")
    parser.add_argument("--robustness", metavar="FILE",
                        help="prompt-robustness contrasts.csv (default <outputs>/robustness/report/contrasts.csv)")
    parser.add_argument("--robustness-results", metavar="FILE",
                        help="prompt-robustness results.json (default: next to --robustness)")
    parser.add_argument("--finetune", metavar="FILE",
                        help="fine-tuning cells.csv (default <outputs>/finetune/ft_tables/cells.csv)")
    parser.add_argument("--finetune-results", metavar="FILE",
                        help="fine-tuning results.json (default: next to --finetune)")
    parser.add_argument("--common-response", metavar="FILE",
                        help="response-form audit table (default <profile flags folder>/common_response.csv)")
    parser.add_argument("--check", nargs="?", const="paper", metavar="FILE",
                        help="compare the written tables cell by cell with an expected tables file (default "
                             "paper/expected/tables.json)")
    parser.set_defaults(func=handler)


def handler(args) -> int:
    from ..analysis.pipeline import tables
    from ..analysis.specs import profile_from_config
    ctx = context(args)
    profile = profile_from_config(ctx.config, args.profile)
    result = tables(ctx.paths.outputs, profile, out=args.out, paper=args.paper, contrasts=args.contrasts,
                    metrics=args.metrics, robustness=args.robustness, robustness_results=args.robustness_results,
                    finetune=args.finetune, finetune_results=args.finetune_results,
                    common_response=args.common_response)
    for path in result["files"].values():
        print(f"wrote {path}")
    for what, why in result["skipped"].items():
        print(f"skipped {what}: {why}")
    if not result["files"]:
        return 1
    if args.check:
        from pathlib import Path
        from ..analysis.pipeline import check_tables
        expected = profile.path("paper/expected/tables.json") if args.check == "paper" else Path(args.check)
        out = Path(args.out) if args.out else ctx.paths.outputs / "tables"
        found = check_tables(out, expected)
        for line in found["differences"]:
            print(f"differs: {line}")
        if found["missing"]:
            print(f"not compared (not written): {', '.join(found['missing'])}")
        print(f"check against {expected}: {found['cells']} cells compared, {len(found['differences'])} differ")
        return 1 if found["differences"] else 0
    return 0
