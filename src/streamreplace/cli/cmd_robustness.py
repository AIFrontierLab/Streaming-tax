"""`streamreplace robustness ...`: the prompt-robustness experiment (docs/generalization/robustness.md).

Steps, in order (the experiment root is <outputs root>/robustness unless --root; one folder per condition):
  subset      write the same subset.json into every condition folder (instruction-level folders: a nested subset)
              and subset.info.json; once (an existing file must be identical)
  paraphrase  paraphrase the stems of the `para` condition through an OpenAI-compatible endpoint (cached; only
              missing units are asked). Not exactly regenerable: see the documentation
  edits       write each condition folder's prompt_overrides.jsonl (once)
  check       verify every folder's subset and edits against the rendered prompts; --frozen also compares the item
              lists and the override files with the paper's (paper/frozen/robustness)
  audit       audit/<condition>.md: clean vs perturbed text of a few units, for manual review
  runs        print the run and score commands of every condition folder (with this invocation's global options,
              --spec and the experiment root)
  report      ROBUSTNESS.md, results.json and the tables (report/) from the scored condition folders
  freeze      write items.json and edits.json (ids, skip reasons and sha256 only) for these condition folders

Each condition folder is then run and scored like any experiment folder, e.g.
  streamreplace --outputs <root>/typo run --pairs p1,p2,p3,p4,p5 --protocols P,C --benchmarks videomme,mmbench
  streamreplace --outputs <root>/typo score --judge"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import command_line, context, csv_list, global_options


def register(subparsers) -> None:
    parser = subparsers.add_parser("robustness", help="prompt robustness: perturbed prompts, runs and the analysis",
                                   description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, help="specification file (default: configs/generalization/robustness.yaml)")
    parser.add_argument("--root", type=Path, help="experiment root (default: <outputs root>/<spec outputs>)")
    actions = parser.add_subparsers(dest="robustness_action", required=True)
    actions.add_parser("subset", help="write every condition folder's subset.json (once)")
    p = actions.add_parser("paraphrase", help="paraphrase the stems through an OpenAI-compatible endpoint (cached)")
    p.add_argument("--endpoint", help="base URL, e.g. http://127.0.0.1:8000/v1 (default: paraphrase.endpoint)")
    p.add_argument("--model", help="served model name (default: paraphrase.model)")
    p.add_argument("--threads", type=int, help="requests in flight (default: paraphrase.threads)")
    e = actions.add_parser("edits", help="write each condition folder's prompt_overrides.jsonl (once)")
    e.add_argument("--conditions", type=csv_list, help="only these conditions")
    c = actions.add_parser("check", help="verify subsets and edits against the rendered prompts")
    c.add_argument("--frozen", nargs="?", const="paper", metavar="DIR",
                   help="also compare with the paper's frozen ids and edit digests (default paper/frozen/robustness)")
    a = actions.add_parser("audit", help="audit/<condition>.md: clean vs perturbed text for manual review")
    a.add_argument("--n", type=int, default=30, help="units per dataset (default 30)")
    a.add_argument("--conditions", type=csv_list, help="only these conditions")
    actions.add_parser("runs", help="print the run and score commands of every condition folder")
    r = actions.add_parser("report", help="the analysis: ROBUSTNESS.md, results.json, csv tables")
    r.add_argument("--out", type=Path, help="report folder (default: <root>/report)")
    f = actions.add_parser("freeze", help="write items.json and edits.json (ids and digests) for these folders")
    f.add_argument("--out", type=Path, required=True, help="folder to write the two files to")
    parser.set_defaults(func=handler)


def frozen_dir(value: str) -> Path:
    from ..paths import project_root
    return project_root() / "paper" / "frozen" / "robustness" if value == "paper" else Path(value)


def experiment(args):
    from ..generalization.robustness.make import bind, load_spec
    ctx = context(args)
    spec = load_spec(ctx.config_dir, args.spec, ctx.config.section("robustness"))
    return bind(spec, ctx.config, ctx.paths, args.root)


def handler(args) -> int:
    from ..generalization.robustness import edits, make
    exp = experiment(args)
    action = args.robustness_action
    if action == "subset":
        result = make.make_subset(exp)
    elif action == "paraphrase":
        from ..generalization.robustness.paraphrase import generate
        result = generate(exp, args.endpoint, args.model, args.threads)
    elif action == "edits":
        result = edits.make_edits(exp, args.conditions)
    elif action == "check":
        result = {"conditions": edits.check(exp)}
        if args.frozen:
            folder = frozen_dir(args.frozen)
            items = json.loads((folder / "items.json").read_text(encoding="utf-8"))
            digests = json.loads((folder / "edits.json").read_text(encoding="utf-8"))["conditions"]
            result["frozen"] = {"items": make.compare_items(exp, items) or "identical",
                                "edits": edits.compare_digests(exp, digests)}
    elif action == "audit":
        result = edits.audit(exp, args.n, args.conditions)
    elif action == "runs":
        pairs, benchmarks = ",".join(exp.pairs), ",".join(exp.datasets())
        prefix = global_options(args)
        for condition in exp.conditions:
            folder = os.path.abspath(exp.folder(condition))
            sets = ",".join(exp.datasets(condition))
            print(command_line([*prefix, "--outputs", folder, "run", "--pairs", pairs, "--protocols", "P,C",
                                "--benchmarks", sets]))
            print(command_line([*prefix, "--outputs", folder, "score", "--judge"]))
        spec = ["--spec", os.path.abspath(args.spec)] if args.spec else []
        print(command_line([*prefix, "robustness", *spec, "--root", os.path.abspath(exp.root), "report"])
              + f"   # after every folder is scored ({benchmarks})")
        return 0
    elif action == "report":
        from ..generalization.robustness.report import write_report
        result = write_report(exp, args.out)
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        items = make.item_lists(exp)
        (args.out / "items.json").write_text(json.dumps(items, indent=1) + "\n", encoding="utf-8")
        frozen = {"seed": exp.seed, "file": edits.OVERRIDES_FILE, "conditions": edits.digests(exp)}
        (args.out / "edits.json").write_text(json.dumps(frozen, indent=1, ensure_ascii=False) + "\n",
                                             encoding="utf-8")
        result = [str(args.out / "items.json"), str(args.out / "edits.json")]
    print(json.dumps(result, indent=1, ensure_ascii=False))
    return 0
