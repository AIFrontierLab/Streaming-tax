"""`streamreplace run`: generate the outputs of the P/C/S jobs (and plugin arms), resumably.

Examples:
    streamreplace run --pairs p2 --protocols P,C,S --benchmarks paper-core --gpus 0,0,1,1
    streamreplace run --pairs all --benchmarks paper-core,paper-supp --dry-run        # the paper's 235 jobs
    streamreplace --outputs runs/quick run --pairs p1 --benchmarks videomme --limit 3 --gpus 0
    streamreplace --config finetuned.yaml run --models my_model --protocols C --benchmarks lingoqa

Jobs: P = the parent offline, C = the child in the same offline pipeline, S = the child's native stream (video
benchmarks only); --variants adds plugin arms of the children's streams. Each job writes
<outputs>/<benchmark>/<model>[@<variant>].<offline|stream>.jsonl and its .meta.json (run identity).

Resume: rerunning the same command continues where it stopped and retries failed samples. A job whose settings,
checkpoint, adapter, plugin, --limit, subset or prompt overrides changed, or whose stored prompts differ from the
rendered ones, is refused; --fresh archives just those jobs to <outputs>/_archive/ and starts them over.

Workers: one process per arm (and per --shards slice), each on one GPU slot of --gpus; repeating an id packs several
workers on one card (--gpus 0,0,1,1). Logs: <outputs>/logs/. --launcher print writes the worker payloads and prints
the commands instead of running them (e.g. for a batch scheduler). Exit code 2 when a job is incomplete."""
from __future__ import annotations

import argparse

from . import context, csv_list, gpu_list


def add_selection(parser: argparse.ArgumentParser) -> None:
    """The job-selection options shared by `run` and `status`."""
    parser.add_argument("--pairs", type=csv_list, default=[], help="pair ids (configs/pairs.yaml), or 'all'")
    parser.add_argument("--models", type=csv_list, default=[],
                        help="individual model ids, e.g. models added with --config (run under the conditions of "
                             "their role), or <child>@<plugin arm> for that one plugin arm (e.g. aura_8b@sfix)")
    parser.add_argument("--protocols", type=csv_list, default=["P", "C", "S"],
                        help="conditions: P (parent, offline), C (child, same offline pipeline), S (child, native "
                             "stream; video benchmarks only). Default: P,C,S")
    parser.add_argument("--variants", type=csv_list, default=[],
                        help="plugin arms of the children's streams (configs/plugin.yaml arms: sfix, sfix_prs, record, "
                             "inert, hf)")
    parser.add_argument("--benchmarks", type=csv_list, required=True,
                        help="benchmark or suite names (paper-core, paper-supp, ...), or 'all'")
    parser.add_argument("--limit", type=int,
                        help="only the first N samples of each benchmark, in its smoke order (whole groups first). "
                             "Part of the run identity, so give such a run its own --outputs folder")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a generation setting, e.g. generation.max_new_tokens=8192 (keys under "
                             "generation, image, video, stream, system; changes the run identity)")


def plan_jobs(args):
    from ..config.settings import parse_sets
    from ..run.plan import plan
    ctx = context(args)
    return plan(ctx.config, ctx.paths, benchmarks=args.benchmarks, pairs=args.pairs, models=args.models,
                protocols=args.protocols, variants=args.variants, limit=args.limit, sets=parse_sets(args.set))


def print_states(jobs, inputs, identities: bool = False) -> list[dict]:
    """One line per job: name, [adapter, env, identity], done/expected samples, errors, state."""
    from ..run.dispatch import job_state
    states = []
    for job in jobs:
        state = job_state(job, inputs)
        states.append(state)
        counts = "-" if state["expected"] is None else f"{state['done']}/{state['expected']}"
        errors = f"{state['errors']} errors" if state["errors"] else ""
        detail = state["detail"] if len(state["detail"]) <= 100 else state["detail"][:97] + "..."
        extra = f"{job.adapter:<20} {job.env:<10} {job.identity or '?':<16} " if identities else ""
        print(f"{job.name:<44} {extra}{counts:>11} {errors:<10} {state['state']}{': ' + detail if detail else ''}")
    return states


def register(subparsers) -> None:
    parser = subparsers.add_parser("run", help="generate outputs of P/C/S jobs and plugin arms (resumable)",
                                   description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_selection(parser)
    parser.add_argument("--gpus", type=gpu_list,
                        help="GPU slots, one worker each; repeat an id to pack workers (default: run.gpus in configs)")
    parser.add_argument("--shards", type=int, default=1,
                        help="split every arm into N workers, worker k taking samples k, k+N, ... (default 1)")
    parser.add_argument("--threads", type=int, help="CPU threads per worker (default: CPU cores / GPU slots)")
    parser.add_argument("--fresh", action="store_true",
                        help="archive outputs whose identity or prompts changed to <outputs>/_archive/ and redo them")
    parser.add_argument("--dry-run", action="store_true", help="print the planned jobs and their state; run nothing")
    parser.add_argument("--launcher", choices=("local", "print"), default="local",
                        help="local: run the workers here (default); print: write payloads, print the commands")
    parser.set_defaults(func=handler)


def handler(args) -> int:
    from ..run import dispatch
    ctx = context(args)
    jobs = plan_jobs(args)
    if not jobs:
        print("no jobs: no selected model has an arm for the selected conditions and benchmarks")
        return 0
    inputs = dispatch.Inputs(ctx.paths)
    if args.dry_run:
        states = print_states(jobs, inputs, identities=True)
        print(f"{len(jobs)} jobs planned ({sum(s['state'] == 'complete' for s in states)} complete); outputs: "
              f"{jobs[0].outputs}")
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
