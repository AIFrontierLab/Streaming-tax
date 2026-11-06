"""`streamreplace finetune ...`: the LoRA fine-tuning experiment (docs/generalization/finetune.md).

The experiment root is <outputs root>/finetune unless --root (configs/generalization/finetune.yaml `outputs`; a
`finetune:` block in configs/local.yaml overrides the specification). Steps, in order:
  prepare DATASET...  fix each dataset's test subset and training order: subset.json, train_order.json and
                      selection.json in the root, written once. --frozen copies the paper's lists instead
                      (paper/frozen/finetune), e.g. when the near-duplicate check's frames are not on disk
  check               compare the root's lists (and the control folders' questions) with the paper's
  models              the evaluation model ids as a models file for the core commands (`--config FILE`)
  units               every training unit and its state
  train               train one unit in its model's environment on one GPU (resumes from its checkpoint)
  eval                evaluate the zero-shot and the trained ids through the run pipeline: the formal folder,
                      a control's folder (--control NAME) or the seed-replication folder (--seeds)
  commands            print every step of the whole experiment as shell commands (for a scheduler); every line
                      repeats this invocation's global options, --spec and the experiment root
  report              FT_RESULTS.md, READOUT.md and ft_tables/ from the scored folders
  seeds               SEED_RESULTS.md and seed_tables/ (training-seed replication)

Scoring is the ordinary scorer, once per folder (LingoQA needs the Lingo-Judge: --judge):
  streamreplace --outputs <root> score --judge --judge-gpus 0"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import command_line, context, csv_list, global_options, gpu_list


def int_list(value: str) -> list[int]:
    return [int(v) for v in csv_list(value)]


def register(subparsers) -> None:
    parser = subparsers.add_parser("finetune", help="LoRA fine-tuning: selection, training units, evaluations, report",
                                   description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, help="specification file (default: configs/generalization/finetune.yaml)")
    parser.add_argument("--root", type=Path, help="experiment root (default: <outputs root>/<spec outputs>)")
    actions = parser.add_subparsers(dest="finetune_action", required=True)

    p = actions.add_parser("prepare", help="fix the test subset and training order of datasets (written once)")
    p.add_argument("datasets", nargs="+", help="datasets of the experiment, e.g. lingoqa eyepcr")
    p.add_argument("--frozen", nargs="?", const="paper", metavar="DIR",
                   help="copy the paper's lists (default paper/frozen/finetune) instead of selecting")
    p.add_argument("--redo", action="store_true", help="archive the dataset's selection to <root>/_old/ and select "
                   "again (only before anything was trained or evaluated on it)")
    p.add_argument("--workers", type=int, default=16, help="processes for the near-duplicate thumbnails (default 16)")

    c = actions.add_parser("check", help="compare the root's lists with the paper's")
    c.add_argument("--frozen", default="paper", metavar="DIR", help="default paper/frozen/finetune")

    m = actions.add_parser("models", help="write the evaluation ids as a models file (for --config)")
    m.add_argument("--out", type=Path, help="file to write (default: print to stdout)")

    u = actions.add_parser("units", help="list the training units and their state")
    u.add_argument("--datasets", type=csv_list, default=[], help="only these datasets (default: all)")
    u.add_argument("--models", type=csv_list, default=[], help="only these pair members (default: all)")
    u.add_argument("--seeds", action="store_true", help="the training-seed replication's units")

    t = actions.add_parser("train", help="train one unit in its model's environment (resumes from its checkpoint)")
    t.add_argument("--model", required=True, help="pair member, e.g. aura_8b (`finetune units` lists them)")
    t.add_argument("--dataset", required=True, help="dataset of the experiment, e.g. lingoqa")
    t.add_argument("--budget", type=int, required=True,
                   help="training examples of the unit, one of the specification's budgets (e.g. 1000)")
    t.add_argument("--tag", help="an extra unit's tag (e.g. e3) or a seed unit's (s1, s2)")
    t.add_argument("--gpus", "--gpu", dest="gpu", default="0", help="the one GPU slot of the unit (default 0)")
    t.add_argument("--threads", type=int, help="CPU threads (default: CPU cores / 8)")
    t.add_argument("--fresh", action="store_true", help="archive the unit's folder to <adapters>/_old/ first")
    t.add_argument("--max-steps", type=int, help="pilot: stop after N steps (checkpoint, no adapter)")
    t.add_argument("--dir", type=Path, help="pilot: write here instead of the unit's folder")
    t.add_argument("--launcher", choices=("local", "print"), default="local",
                   help="local: run it here (default); print: write the payload and print the command")

    e = actions.add_parser("eval", help="evaluate zero-shot and trained ids through the run pipeline")
    e.add_argument("--datasets", type=csv_list, default=[])
    e.add_argument("--models", type=csv_list, default=[], help="pair members (default: all, or the control's)")
    e.add_argument("--budgets", type=int_list, help="0 = zero-shot (default: 0 and every budget, or the control's)")
    e.add_argument("--tags", type=csv_list, default=[], help="extra units' tags, e.g. e3")
    group = e.add_mutually_exclusive_group()
    group.add_argument("--control", help="a control of the specification (its own folder), e.g. blank")
    group.add_argument("--seeds", action="store_true", help="the training-seed replication's units")
    e.add_argument("--gpus", type=gpu_list, help="GPU slots (default: run.gpus in configs)")
    e.add_argument("--shards", type=int, default=1, help="workers per job, each taking every N-th sample (default 1)")
    e.add_argument("--threads", type=int, help="CPU threads per worker (default: CPU cores / GPU slots)")
    e.add_argument("--fresh", action="store_true", help="archive outputs whose identity changed and redo them")
    e.add_argument("--dry-run", action="store_true",
                   help="print the planned jobs; run nothing (a control or seed folder still gets its subset.json)")
    e.add_argument("--launcher", choices=("local", "print"), default="local",
                   help="local: run the workers here (default); print: write payloads, print the commands")

    k = actions.add_parser("commands", help="print every step of the experiment as shell commands")
    k.add_argument("--gpus", type=gpu_list, default=[0],
                   help="GPU slots the printed steps use; training units are spread over them (default 0)")

    actions.add_parser("report", help="FT_RESULTS.md, READOUT.md and ft_tables/ from the scored folders")
    actions.add_parser("seeds", help="SEED_RESULTS.md and seed_tables/ (training-seed replication)")
    parser.set_defaults(func=handler)


def experiment(args):
    from ..generalization.finetune import load_experiment
    ctx = context(args)
    return load_experiment(ctx.config_dir, args.spec, args.root, ctx.paths.outputs, ctx.config.section("finetune"))


def frozen_dir(value: str) -> Path:
    from ..generalization.finetune.data import frozen_dir as paper
    return paper() if value == "paper" else Path(value)


def handler(args) -> int:
    from ..generalization.finetune import data, launch
    ctx = context(args)
    exp = experiment(args)
    action = args.finetune_action
    if action == "prepare":
        unknown = sorted(set(args.datasets) - set(exp.datasets))
        if unknown:
            raise ValueError(f"not datasets of the experiment: {unknown}; configured: {', '.join(exp.datasets)}")
        if args.frozen:
            result = data.use_frozen(exp.root, args.datasets, frozen_dir(args.frozen))
        else:
            result = [data.prepare(exp, d, workers=args.workers, redo=args.redo) for d in args.datasets]
    elif action == "check":
        folder = frozen_dir(args.frozen)
        problems = data.compare_frozen(exp.root, exp.datasets, folder) + data.compare_controls(exp, folder)
        result = {"root": str(exp.root), "problems": problems or "identical to the paper's lists"}
        print(json.dumps(result, indent=1))
        return 1 if problems else 0
    elif action == "models":
        if args.out is None:
            import yaml
            print(yaml.safe_dump({"models": launch.model_entries(exp, ctx.config)}, sort_keys=False), end="")
            return 0
        result = str(launch.write_models_file(exp, ctx.config, args.out))
    elif action == "units":
        for line in launch.describe_units(exp, launch.select_units(exp, args.datasets, args.models, args.seeds)):
            print(line)
        return 0
    elif action == "train":
        return launch.run_training(exp, ctx.config, ctx.paths, args.model, args.dataset, args.budget, args.tag,
                                   gpu=args.gpu, threads=args.threads, fresh=args.fresh, directory=args.dir,
                                   max_steps=args.max_steps, launcher=args.launcher)
    elif action == "eval":
        return evaluate(args, exp)
    elif action == "commands":
        prefix = global_options(args)
        spec = ["--spec", os.path.abspath(args.spec)] if args.spec else []
        finetune = command_line([*prefix, "finetune", *spec, "--root", os.path.abspath(exp.root)])
        for line in launch.commands(exp, prefix=command_line(prefix), gpus=args.gpus, finetune=finetune):
            print(line)
        return 0
    elif action == "report":
        from ..generalization.finetune.report import write_readout, write_report
        result = [str(write_report(exp)), str(write_readout(exp))]
    else:
        from ..generalization.finetune.seeds import write_report
        result = str(write_report(exp))
    print(json.dumps(result, indent=1, ensure_ascii=False, default=str))
    return 0


def evaluate(args, exp) -> int:
    from ..generalization.finetune.launch import plan_evaluations
    from ..run import dispatch
    from .cmd_run import print_states
    ctx = context(args)
    jobs, waiting, folder = plan_evaluations(exp, ctx.config, ctx.paths, args.datasets, args.models, args.budgets,
                                             args.control, args.seeds, args.tags)
    if waiting:
        print(f"not trained yet (skipped): {', '.join(waiting)}")
    if not jobs:
        print(f"nothing to evaluate in {folder}")
        return 0
    inputs = dispatch.Inputs(ctx.paths)
    if args.dry_run:
        states = print_states(jobs, inputs, identities=True)
        print(f"{len(jobs)} jobs planned ({sum(s['state'] == 'complete' for s in states)} complete); outputs: {folder}")
        return 0
    gpus = args.gpus or list(ctx.config.run.get("gpus") or [0])
    status = dispatch.run(jobs, gpus=gpus, shards=args.shards, threads=args.threads, fresh=args.fresh,
                          launcher=args.launcher, paths=ctx.paths, inputs=inputs)
    if args.launcher == "print":
        return 0
    incomplete = {name: s for name, s in status.items() if s != "complete"}
    for name, s in incomplete.items():
        print(f"  {name}: {s}")
    print("all jobs complete" if not incomplete else f"{len(incomplete)} of {len(status)} jobs incomplete")
    return 2 if incomplete else 0
