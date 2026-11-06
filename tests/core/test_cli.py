"""The command-line root (cli/main.py): every cli/cmd_*.py module registers its command, a module that fails to
import is still listed (and reports its error when run), and `doctor` reports the installation."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

from streamreplace.cli import main as cli_main

SRC = Path(__file__).resolve().parents[2] / "src"


def subcommands(parser) -> tuple[dict, dict]:
    """({command: parser}, {command: help text}) of a parser's subcommands."""
    action = next(a for a in parser._actions if a.dest == "command")
    return dict(action.choices), {a.dest: a.help for a in action._choices_actions}


def registered_by(module: str) -> list[str]:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    importlib.import_module(f"streamreplace.cli.{module}").register(subparsers)
    return list(subparsers.choices)


def test_every_command_module_registers_commands_that_help_lists():
    modules = cli_main.command_modules()
    assert "cmd_doctor" in modules and all(m.startswith("cmd_") for m in modules)
    commands, helps = subcommands(cli_main.build_parser())
    for module in modules:
        names = registered_by(module)
        assert names and set(names) <= set(commands), module
    assert not [name for name, text in helps.items() if "unavailable" in (text or "")]
    env = {k: v for k, v in os.environ.items() if not k.startswith("STREAMREPLACE_")}
    proc = subprocess.run([sys.executable, "-m", "streamreplace", "--help"], capture_output=True, text=True,
                          env={**env, "PYTHONPATH": str(SRC)}, timeout=300)
    assert proc.returncode == 0, proc.stderr
    listed = {line.split()[0] for line in proc.stdout.splitlines() if line.startswith("    ") and line.split()}
    assert set(helps) <= listed


def test_a_broken_command_module_is_listed_as_unavailable(monkeypatch, capsys):
    broken = types.ModuleType("streamreplace.cli.cmd_broken")

    def register(subparsers):
        raise RuntimeError("needs a library this environment lacks")
    broken.register = register
    monkeypatch.setitem(sys.modules, "streamreplace.cli.cmd_broken", broken)
    monkeypatch.setattr(cli_main, "command_modules", lambda: ["cmd_doctor", "cmd_broken"])
    commands, helps = subcommands(cli_main.build_parser())
    assert set(commands) == {"doctor", "broken"}
    assert helps["broken"] == "(unavailable: RuntimeError: needs a library this environment lacks)"
    assert cli_main.main(["broken"]) == 1
    assert "could not be loaded" in capsys.readouterr().err


def test_no_command_prints_help(capsys):
    assert cli_main.main([]) == 2
    assert "doctor" in capsys.readouterr().out


def test_doctor_reports_paths_configuration_and_missing_pieces(tmp_path, capsys, monkeypatch, config_dir):
    for name in list(os.environ):
        if name.startswith("STREAMREPLACE_"):
            monkeypatch.delenv(name)
    home = tmp_path / "home"
    (home / "upstream" / "videomme").mkdir(parents=True)
    assert cli_main.main(["--home", str(home), "doctor", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["paths"]["home"] == {"path": str(home), "source": "command line", "exists": True}
    assert report["config"]["models"] == 11 and report["config"]["pairs"] == 5
    assert report["upstream"]["present"] == ["videomme"] and "videomme" not in report["upstream"]["missing"]
    expected = set(report["upstream"]["expected"])      # the fetchable entries of manifests/upstream.yaml
    assert {"videomme", "harmbench"} <= expected and not {"version", "upstream", "aura", "moss"} & expected
    assert report["upstream"]["states"]["videomme"] == "incomplete" and "harmbench" in report["upstream"]["missing"]
    assert report["envs"]["modern"]["exists"] is False and report["data"]["mme"]["prepared"] is False
    assert all(report["adapters"].values())
    assert cli_main.main(["--home", str(home), "doctor", "--strict"]) == 1         # things are missing
    text = capsys.readouterr().out
    assert "paths:" in text and "missing" in text
    broken = config_dir(pairs={"p": {"parent": "ghost", "children": ["child"]}})
    assert cli_main.main(["--home", str(home), "--config-dir", str(broken), "doctor"]) == 1
    assert "unknown model 'ghost'" in capsys.readouterr().out


def test_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli_main.main(["--version"])
    assert exit_info.value.code == 0 and capsys.readouterr().out.startswith("streamreplace 0.1.0")
