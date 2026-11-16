"""`streamreplace experiment list|run|label|report`: the diagnostic and supplementary experiments of the paper
(Sec. 5, Supp. C and D), declared in configs/experiments.yaml (docs/experiments.md maps each to its table or figure).

list     the experiments, their paper location and their arms
run      generate an experiment's own arms into <outputs>/experiments/<name>/ (resumable, as `run`); the reference arms
         P, C and S are the formal jobs of the outputs root and are not rerun. --dry-run prints the planned jobs
label    experiments of kind `forms` and `probe`: send the arms' replies to the response-form judge (--judge-gpus)
report   <outputs>/experiments/<name>/report.json and REPORT.md

Order: run, then `streamreplace --outputs <outputs>/experiments/<name>[/<folder>] score [--judge]` for benchmark-scored
arms (each experiment folder is an ordinary outputs folder; `run` prints the folders), then label / report.

Examples:
    streamreplace experiment list
    streamreplace experiment run window --dry-run
    streamreplace experiment run question_timing --children aura_8b --gpus 0
    streamreplace experiment report window"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from . import command_line, context, csv_list, global_options, gpu_list


def register(subparsers) -> None:
    parser = subparsers.add_parser("experiment", help="diagnostic and supplementary experiments (Sec. 5, Supp. C-D)",
                                   description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    actions = parser.add_subparsers(dest="experiment_action", required=True)
    actions.add_parser("list", help="the configured experiments").set_defaults(func=_list)

    run = actions.add_parser("run", help="generate an experiment's arms (resumable)")
    run.add_argument("name")
    run.add_argument("--children", type=csv_list, default=[], help="children (default: the experiment's)")
    run.add_argument("--arms", type=csv_list, default=[], help="arm labels (default: every arm of its own)")
    run.add_argument("--benchmarks", type=csv_list, default=[], help="benchmarks (default: the experiment's)")
    run.add_argument("--gpus", type=gpu_list, help="GPU slots, one worker each (default: run.gpus in configs)")
    run.add_argument("--shards", type=int, default=1, help="split every arm into N workers (default 1)")
    run.add_argument("--threads", type=int, help="CPU threads per worker")
    run.add_argument("--fresh", action="store_true", help="archive outputs whose identity changed and redo them")
    run.add_argument("--dry-run", action="store_true", help="print the planned jobs and their state; run nothing")
    run.add_argument("--launcher", choices=("local", "print"), default="local")
    run.set_defaults(func=_run)

    label = actions.add_parser("label", help="label replies with the response-form judge (kinds forms, probe)")
    label.add_argument("name")
    label.add_argument("--judge-gpus", type=gpu_list, help="GPUs of the judge (default: GPU 0)")
    label.add_argument("--formal", type=Path, help="outputs folder of the formal P/C/S jobs (default: the outputs root)")
    label.set_defaults(func=_label)

    report = actions.add_parser("report", help="write report.json and REPORT.md")
    report.add_argument("name")
    report.add_argument("--children", type=csv_list, default=[], help="children (default: the experiment's)")
    report.add_argument("--formal", type=Path, help="outputs folder of the formal P/C/S jobs (default: the outputs root)")
    report.add_argument("--profile", metavar="FILE", help="analysis profile YAML instead of configs/analysis.yaml")
    report.set_defaults(func=_report)


def _list(args) -> int:
    from .. import experiments
    ctx = context(args)
    for name in experiments.names(ctx.config):
        exp = experiments.load(ctx.config, ctx.paths, name)
        own = [label for label, arm in exp.arms.items() if not arm.formal]
        print(f"{name:<20} [{exp.kind}] {exp.spec.get('title', '')}")
        print(f"{'':<20} paper: {exp.spec.get('paper', '-')}; children: {', '.join(exp.children) or '-'}; "
              f"arms: {', '.join(own) or '-'}")
    return 0


def _run(args) -> int:
    from .. import experiments
    from ..run import dispatch
    from .cmd_run import print_states
    ctx = context(args)
    exp = experiments.load(ctx.config, ctx.paths, args.name)
    if exp.kind == "probe":
        from ..experiments import probe
        return probe.run(exp, ctx, args)
    jobs = experiments.plan(exp, ctx.config, ctx.paths, args.children, args.arms, args.benchmarks)
    if not jobs:
        print("no jobs: the experiment has no arm of its own for this selection")
        return 0
    inputs = dispatch.Inputs(ctx.paths)
    folders = list(dict.fromkeys(str(j.outputs) for j in jobs))
    if args.dry_run:
        states = print_states(jobs, inputs, identities=True)
        print(f"{len(jobs)} jobs planned ({sum(s['state'] == 'complete' for s in states)} complete)")
    else:
        gpus = args.gpus or list(ctx.config.run.get("gpus") or [0])
        status = dispatch.run(jobs, gpus=gpus, shards=args.shards, threads=args.threads, fresh=args.fresh,
                              launcher=args.launcher, paths=ctx.paths, inputs=inputs)
        if args.launcher != "print":
            incomplete = {name: s for name, s in status.items() if s != "complete"}
            for name, s in incomplete.items():
                print(f"  {name}: {s}")
            print("all jobs complete" if not incomplete else f"{len(incomplete)} of {len(status)} jobs incomplete")
            if incomplete:
                return 2
    for folder in folders:                               # every experiment folder is scored like an outputs folder
        print("score: " + command_line([*global_options(args), "--outputs", folder, "score", "--judge"]))
    return 0


def _label(args) -> int:
    from .. import experiments
    from ..experiments import forms
    ctx = context(args)
    exp = experiments.load(ctx.config, ctx.paths, args.name)
    if exp.kind not in ("forms", "probe"):
        raise ValueError(f"experiment {exp.name} is of kind {exp.kind}: its arms are scored by `score`, not labelled")
    pending = forms.label(exp, ctx.config, ctx.paths, args.formal or ctx.paths.outputs, args.judge_gpus or [0])
    print(f"{exp.name}: {pending} replies sent to the response-form judge")
    return 0


def _report(args) -> int:
    from .. import experiments
    from ..analysis.records import Catalog
    from ..analysis.specs import profile_from_config
    from ..core.io import atomic_text, write_json
    ctx = context(args)
    exp = experiments.load(ctx.config, ctx.paths, args.name)
    profile = profile_from_config(ctx.config, args.profile)
    formal = args.formal or ctx.paths.outputs
    names = {c: profile.name_of(c) for c in exp.children}
    if exp.kind == "forms":
        from ..experiments import forms
        result = forms.report(exp, ctx.config, ctx.paths, formal, args.children or None)
        text = forms.markdown(result, names)
    elif exp.kind == "probe":
        from ..experiments import probe
        result = probe.report(exp, ctx.config, ctx.paths, args.children or None)
        text = probe.markdown(result, names)
    else:
        from ..experiments.report import Report, markdown
        result = Report(exp, ctx.config, profile, Catalog(), formal).compute(args.children or None)
        text = markdown(result, names)
    write_json(exp.root / "report.json", result)
    atomic_text(exp.root / "REPORT.md", text)
    for problem in result.get("problems") or []:
        print("incomplete:", problem, file=sys.stderr)
    print(f"wrote {exp.root}/report.json, REPORT.md")
    return 2 if result.get("problems") else 0
