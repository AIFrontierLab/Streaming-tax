"""`streamreplace env`: the Python environments of the models, the judges and the orchestrator.

    streamreplace env list                                   what is defined, built, and which models use it
    streamreplace env build NAME... | all [--dry-run]        uv venv + uv pip sync <lock> + checked patches
    streamreplace env check NAME... | all [--no-probe]       interpreter, packages vs lock, patches, imports
    streamreplace env patch NAME...                          apply the checked patches to an existing environment

Environments are defined in envs/<name>/ (requirements.lock, requirements.in, python-version) and built under the
envs root (STREAMREPLACE_ENVS, default $STREAMREPLACE_HOME/envs). `ffmpeg` builds the FFmpeg 6 shared libraries
that torchcodec needs (with micromamba, mamba or conda). See docs/environments.md."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

from . import context


def register(subparsers) -> None:
    parser = subparsers.add_parser("env", help="build and check the Python environments", description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    actions = parser.add_subparsers(dest="env_action", metavar="<action>")

    listing = actions.add_parser("list", help="defined environments, whether they are built, who uses them")
    listing.add_argument("--json", action="store_true", help="print JSON")
    listing.set_defaults(env_func=_list)

    build = actions.add_parser("build", help="build environments from their locks")
    build.add_argument("names", nargs="+", metavar="NAME", help="environment names, or all")
    build.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    build.add_argument("--recreate", action="store_true", help="recreate the virtual environment from scratch")
    build.add_argument("--uv", help="uv executable (default: $UV, else uv on PATH)")
    build.add_argument("--conda", help="micromamba / mamba / conda executable for the ffmpeg libraries")
    build.set_defaults(env_func=_build)

    check = actions.add_parser("check", help="compare built environments with their locks")
    check.add_argument("names", nargs="+", metavar="NAME", help="environment names, or all")
    check.add_argument("--no-probe", action="store_true", help="skip the import probes")
    check.add_argument("--json", action="store_true", help="print JSON")
    check.set_defaults(env_func=_check)

    patch = actions.add_parser("patch", help="apply the checked patches to existing environments")
    patch.add_argument("names", nargs="+", metavar="NAME", help="environment names")
    patch.set_defaults(env_func=_patch)

    def dispatch(args) -> int:
        if getattr(args, "env_func", None) is None:
            parser.print_help()
            return 2
        return args.env_func(args)

    parser.set_defaults(func=dispatch)


def _users(ctx) -> dict[str, list[str]]:
    """env -> the model jobs and judges that run in it (from the configuration)."""
    users: dict[str, list[str]] = {}
    try:
        config = ctx.config
    except (ValueError, OSError):
        return users
    for model in config.models.values():
        for mode, block in (("offline", model.offline), ("stream", model.stream)):
            if block is not None:
                users.setdefault(block.env, []).append(f"{model.id}.{mode}")
    judges = config.section("judges")
    for judge, entry in (judges.get("judges", judges) or {}).items():     # judge name -> {env: ..., ...}
        if isinstance(entry, dict) and isinstance(entry.get("env"), str):
            users.setdefault(entry["env"], []).append(f"judge:{judge}")
    return users


def _list(args) -> int:
    from ..provisioning import envs
    ctx = context(args)
    root, users = ctx.paths.envs, _users(ctx)
    rows = []
    for name in envs.names():
        if name == envs.FFMPEG:
            rows.append({"name": name, "python": "-", "packages": envs.FFMPEG_SPEC,
                         "built": (root / name / "lib").is_dir(), "used_by": ["torchcodec (ffmpeg: true models)"]})
            continue
        definition = envs.definitions()[name]
        rows.append({"name": name, "python": definition.python, "packages": len(definition.packages()),
                     "built": (root / name / "bin" / "python").exists(),
                     "used_by": users.get(name, ["orchestrator"] if name == "core" else [])})
    if args.json:
        print(json.dumps({"root": str(root), "envs": rows}, indent=2))
        return 0
    print(f"envs root: {root}")
    for row in rows:
        print(f"  {row['name']:<9} python {row['python']:<8} {str(row['packages']):<28} "
              f"{'built' if row['built'] else 'not built':<9} {', '.join(row['used_by'])}")
    return 0


def _build(args) -> int:
    from ..provisioning import envs
    ctx = context(args)
    selected = envs.expand(args.names)
    uv = args.uv or os.environ.get("UV") or shutil.which("uv")
    if uv is None and any(name != envs.FFMPEG for name in selected):
        if not args.dry_run:
            print("streamreplace env build: uv not found (https://docs.astral.sh/uv/; or pass --uv)", file=sys.stderr)
            return 1
        uv = "uv"
    for name in selected:
        print(f"== {name} -> {ctx.paths.envs / name}")
        try:
            record = envs.build(name, ctx.paths.envs, uv=uv, conda=args.conda, recreate=args.recreate,
                                dry_run=args.dry_run)
        except RuntimeError as exc:
            print(f"streamreplace env build: {exc}", file=sys.stderr)
            return 1
        for patch in record.get("patches", []):
            print(f"   patch {patch['name']}: {patch['status']}")
    return 0


def _check(args) -> int:
    from ..provisioning import envs
    ctx = context(args)
    ffmpeg_lib = ctx.paths.envs / envs.FFMPEG / "lib"
    reports = [envs.check(name, ctx.paths.envs, probes=not args.no_probe, ffmpeg_lib=ffmpeg_lib)
               for name in envs.expand(args.names)]
    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        for report in reports:
            print(f"{report['name']:<9} {'ok' if report['ok'] else 'PROBLEMS'}  {report['path']}")
            for problem in report["problems"]:
                print(f"    problem: {problem}")
            for warning in report["warnings"]:
                print(f"    warning: {warning}")
    return 0 if all(r["ok"] for r in reports) else 1


def _patch(args) -> int:
    from ..provisioning import envs
    ctx = context(args)
    for name in envs.expand(args.names):
        try:
            results = envs.patch_env(name, ctx.paths.envs)
        except RuntimeError as exc:
            print(f"streamreplace env patch: {name}: {exc}", file=sys.stderr)
            return 1
        for result in results:
            print(f"{name}: {result['name']}: {result['status']}")
    return 0
