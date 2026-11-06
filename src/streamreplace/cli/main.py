"""`streamreplace [global options] <command> ...`: the root of the command-line interface.

Global options (before the command): --home, --paths, --outputs (see paths.py), --config-dir, --config (extra
config files merged over configs/), -v. Commands are discovered from the cli/cmd_*.py modules: each one registers
itself with register(subparsers). A command module that fails to import is still listed under the command name its
source registers, and reports its import error when run, so one broken environment never disables the whole tool.
A command that stops on a missing package (ImportError) says which interpreter lacked it."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
from pathlib import Path
import pkgutil
import re
import sys
import traceback

from .. import __version__
from . import context

DESCRIPTION = ("Evaluate streaming video-LLMs against the offline parents they were initialised from: P (parent, "
               "offline), C (child, same offline pipeline) and S (child, native stream, question at the end), with "
               "S - P = (C - P) + (S - C); the streaming plugin; generalization experiments.")


def command_modules() -> list[str]:
    from .. import cli
    return sorted(m.name for m in pkgutil.iter_modules(cli.__path__) if m.name.startswith("cmd_"))


def command_name(module: str) -> str:
    """The command a cmd_*.py module registers, read from its source (the first subparsers.add_parser("...") call;
    the module itself could not be imported), else the module name without cmd_."""
    try:
        spec = importlib.util.find_spec(f"{__package__}.{module}")
        source = Path(spec.origin).read_text(encoding="utf-8")
        found = re.search(r"subparsers\.add_parser\(\s*[\"']([\w-]+)[\"']", source)
        if found:
            return found.group(1)
    except Exception:                                    # noqa: BLE001 - no readable source: the module name
        pass
    return module.removeprefix("cmd_").replace("_", "-")


def _unavailable(subparsers, module: str, error: BaseException) -> None:
    name = command_name(module)
    message = f"{type(error).__name__}: {error}"
    try:
        parser = subparsers.add_parser(name, help=f"(unavailable: {message})")
    except (argparse.ArgumentError, ValueError):       # the module registered the name before failing
        return

    def handler(args) -> int:
        print(f"streamreplace {name}: this command could not be loaded: {message}", file=sys.stderr)
        return 1

    parser.set_defaults(func=handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="streamreplace", description=DESCRIPTION)
    parser.add_argument("--version", action="version", version=f"streamreplace {__version__}")
    parser.add_argument("--home", help="workspace root (default: $STREAMREPLACE_HOME, else workspace/ in the checkout)")
    parser.add_argument("--paths", metavar="FILE", help="YAML file mapping roots (data, checkpoints, envs, outputs, "
                        "cache, upstream) to folders (default: $STREAMREPLACE_PATHS)")
    parser.add_argument("--outputs", help="outputs folder, e.g. for a side experiment (default: <home>/outputs)")
    parser.add_argument("--config-dir", help="configuration folder (default: configs/ in the checkout)")
    parser.add_argument("--config", action="append", default=[], metavar="FILE",
                        help="extra config file merged over the configuration folder (repeatable)")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="more logging (-vv: debug)")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for module_name in command_modules():
        try:
            module = importlib.import_module(f"{__package__}.{module_name}")
            register = getattr(module, "register")
        except Exception as exc:                        # noqa: BLE001 - reported by the placeholder command
            _unavailable(subparsers, module_name, exc)
            continue
        try:
            register(subparsers)
        except Exception as exc:                        # noqa: BLE001
            _unavailable(subparsers, module_name, exc)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * min(args.verbose, 2),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 2
    try:
        from .. import paths
        paths.set_current(context(args).paths)          # library code (paths.current()) sees the flags too
        return int(func(args) or 0)
    except (ValueError, OSError, KeyError) as exc:
        if args.verbose:
            traceback.print_exc()
        print(f"streamreplace: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        if args.verbose:
            traceback.print_exc()
        print(f"streamreplace: {type(exc).__name__}: {exc}\n  {sys.executable} lacks a package this step needs. "
              "Steps that load a model run in that model's environment (docs/environments.md); `prepare` needs "
              "pyarrow for parquet releases (pip install -e \".[prepare]\").", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted; finished samples are kept and resume automatically.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
