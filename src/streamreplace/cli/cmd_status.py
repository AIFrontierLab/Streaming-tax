"""`streamreplace status`: progress of the planned jobs, without running or writing anything.

Takes the job-selection options of `streamreplace run` and prints, per job, the samples done out of those expected,
the samples whose last row is an error, and its state: new, partial, complete, changed (its run identity differs
from the stored one: `run` would refuse it without --fresh), drift (stored prompts differ from the rendered ones),
unprepared (benchmark data or module missing) or unavailable (adapter module missing). Exit code 2 unless every job
is complete.

    streamreplace status --pairs all --benchmarks paper-core,paper-supp"""
from __future__ import annotations

import argparse

from . import context
from .cmd_run import add_selection, plan_jobs, print_states


def register(subparsers) -> None:
    parser = subparsers.add_parser("status", help="progress of P/C/S jobs and plugin arms (read-only)",
                                   description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_selection(parser)
    parser.set_defaults(func=handler)


def handler(args) -> int:
    from ..run.dispatch import Inputs
    jobs = plan_jobs(args)
    states = print_states(jobs, Inputs(context(args).paths))
    complete = sum(state["state"] == "complete" for state in states)
    print(f"{complete} of {len(states)} jobs complete")
    return 0 if complete == len(states) else 2
