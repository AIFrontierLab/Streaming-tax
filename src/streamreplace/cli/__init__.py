"""Command-line interface. cli/main.py builds the `streamreplace` command; every subcommand lives in its own
`cli/cmd_<name>.py` module exposing `register(subparsers)`, which adds a parser and sets `func=handler`
(handler(args) -> exit code). main.py imports every cmd_*.py module it finds, so adding a command never touches
main.py.

Handlers get the global options on `args` and the loaded configuration and resolved paths from context(args):

    from . import context
    def handler(args) -> int:
        ctx = context(args)
        ctx.config.settings("videomme", "aura_8b", "stream")
        ctx.paths.outputs

A command that starts or prints another streamreplace command line repeats this invocation's global options with
global_options(args), so the inner command sees the same configuration files, workspace and paths."""
from __future__ import annotations

import argparse
import os
import shlex
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config.schema import Config
    from ..paths import Paths


class Context:
    """The configuration and paths of one command-line invocation, loaded on first use."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._config: Config | None = None
        self._paths: Paths | None = None

    @property
    def config_dir(self):
        from ..paths import default_config_dir
        return getattr(self.args, "config_dir", None) or default_config_dir()

    @property
    def config(self) -> Config:
        if self._config is None:
            from ..config.load import load_config
            self._config = load_config(self.config_dir, getattr(self.args, "config", None) or ())
        return self._config

    @property
    def paths(self) -> Paths:
        if self._paths is None:
            from .. import paths
            try:
                block = self.config.paths_block
            except ValueError:                     # an invalid configuration still lets `doctor` show the paths
                block = paths.local_config_paths(self.config_dir)
            self._paths = paths.resolve(home=getattr(self.args, "home", None),
                                        paths_file=getattr(self.args, "paths", None),
                                        outputs=getattr(self.args, "outputs", None), config_paths=block)
        return self._paths


def context(args: argparse.Namespace) -> Context:
    ctx = getattr(args, "_context", None)
    if ctx is None:
        ctx = Context(args)
        args._context = ctx
    return ctx


def global_options(args: argparse.Namespace, outputs: bool = False) -> list[str]:
    """This invocation's global options as command-line arguments for another streamreplace command: --home, --paths,
    --config-dir, every --config (in order) and -v, and with outputs=True --outputs as resolved. Paths are made
    absolute, so the line means the same from any folder."""
    def absolute(value) -> str:
        return os.path.abspath(os.path.expanduser(str(value)))

    argv: list[str] = []
    for flag, name in (("--home", "home"), ("--paths", "paths"), ("--config-dir", "config_dir")):
        value = getattr(args, name, None)
        if value:
            argv += [flag, absolute(value)]
    for path in getattr(args, "config", None) or ():
        argv += ["--config", absolute(path)]
    argv += ["-v"] * min(int(getattr(args, "verbose", 0) or 0), 2)
    if outputs:
        argv += ["--outputs", str(context(args).paths.outputs)]
    return argv


def command_line(argv: list[str], program: str = "streamreplace") -> str:
    """A shell-quoted command line: `program` followed by `argv`."""
    return " ".join([program, *(shlex.quote(str(a)) for a in argv)])


def csv_list(value: str) -> list[str]:
    """"a,b,,c" -> ["a", "b", "c"] (argparse type for comma-separated lists)."""
    return [v for v in value.split(",") if v]


def gpu_list(value: str) -> list[int]:
    """"0,0,1" -> [0, 0, 1]: GPU slots; repeating an id packs several workers on one card."""
    return [int(v) for v in csv_list(value)]
