"""Commands that start or print other streamreplace commands keep this invocation's global options (cli.global_options):
`plugin run` (its inner `run`), the GPU stages of `plugin calibrate` (run in the model environment), the command
names of modules that fail to import, and the message for a missing package."""
from __future__ import annotations

import json
import os
import sys
import types

import pytest

from streamreplace.cli import cmd_plugin, context, global_options
from streamreplace.cli import main as cli_main


@pytest.fixture(autouse=True)
def no_roots_from_the_caller(monkeypatch):
    for name in list(os.environ):
        if name.startswith("STREAMREPLACE_"):
            monkeypatch.delenv(name)


def parse(argv: list[str]):
    return cli_main.build_parser().parse_args(argv)


def test_global_options_repeat_the_invocation_with_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("a.yaml", "b.yaml"):
        (tmp_path / name).write_text("{}\n")
    args = parse(["--home", "ws", "--config", "a.yaml", "--config", str(tmp_path / "b.yaml"), "-vv", "--outputs", "out",
                  "status", "--benchmarks", "videomme"])
    assert global_options(args) == ["--home", str(tmp_path / "ws"), "--config", str(tmp_path / "a.yaml"),
                                    "--config", str(tmp_path / "b.yaml"), "-v", "-v"]
    assert global_options(args, outputs=True)[-2:] == ["--outputs", str(tmp_path / "out")]
    assert global_options(parse(["status", "--benchmarks", "videomme"])) == []


def test_plugin_run_hands_the_global_options_to_its_run(tmp_path, monkeypatch, capsys):
    live = tmp_path / "live.yaml"
    live.write_text("plugin: {arms: {sfix_prs: {gate: live}}}\n")
    items = tmp_path / "items.json"
    items.write_text(json.dumps({"videomme": ["001-1", "001-2"]}))
    out, home = tmp_path / "out", tmp_path / "home"
    real = cli_main.main
    inner = []
    monkeypatch.setattr(cli_main, "main", lambda argv: inner.append(argv) or 0)
    argv = ["--home", str(home), "--config", str(live), "--outputs", str(out), "plugin", "run", "--children", "aura_8b",
            "--arms", "sfix_prs", "--items", str(items), "--", "--dry-run"]
    assert real(argv) == 0
    expected = ["--home", str(home), "--config", str(live), "--outputs", str(out), "run", "--models", "aura_8b@sfix_prs",
                "--protocols", "S", "--benchmarks", "videomme", "--dry-run"]
    assert inner == [expected]                                   # no --gpus: the inner run takes run.gpus
    assert capsys.readouterr().out.splitlines()[0] == "streamreplace " + " ".join(expected)
    assert json.loads((out / "subset.json").read_text()) == {"videomme": ["001-1", "001-2"]}
    assert real([*argv[:-2], "--gpus", "0,0,1", "--print"]) == 0 and len(inner) == 1     # --print runs nothing
    assert capsys.readouterr().out.strip().endswith("--benchmarks videomme --gpus 0,0,1")


def test_calibration_stages_run_in_the_model_environment(tmp_path, monkeypatch):
    home = tmp_path / "home"
    args = parse(["--home", str(home), "plugin", "calibrate", "aura_8b", "--stage", "harmless", "--clips", "clips",
                  "--gpus", "3"])
    ctx = context(args)
    block = cmd_plugin.stage_block(ctx.config, "harmless", "aura_8b", None)
    assert (block.env, block.ffmpeg) == ("aura", True)            # the child's stream environment
    command, variables = cmd_plugin.stage_command(args, ctx, block, "aura_8b", None, tmp_path / "work")
    assert command[:3] == [str(home / "envs" / "aura" / "bin" / "python"), "-m", "streamreplace"]
    assert command[3:5] == ["--home", str(home)] and command[command.index("--clips") + 1] == os.path.abspath("clips")
    assert command[command.index("--stage") + 1] == "harmless" and "--parent" not in command
    assert variables["CUDA_VISIBLE_DEVICES"] == "3" and variables[cmd_plugin.WORKER] == "1"
    assert variables["PYTHONPATH"].endswith("src") and variables["HF_HUB_OFFLINE"] == "1"
    directions = cmd_plugin.stage_block(ctx.config, "directions", "aura_8b", "qwen3vl_8b")
    assert directions.env == "modern"                             # child and parent share the offline environment
    with pytest.raises(ValueError, match="offline environments differ"):
        cmd_plugin.stage_block(ctx.config, "directions", "aura_8b", "llava_ov_7b")
    # the environment is not built here: a clear error instead of an import failure in this interpreter
    code = cli_main.main(["--home", str(home), "plugin", "calibrate", "aura_8b", "--stage", "harmless", "--clips", "c"])
    assert code == 1
    monkeypatch.setenv(cmd_plugin.WORKER, "1")                    # inside the stage's process: no second launch
    assert cmd_plugin._in_model_env(args, ctx, "aura_8b", None, tmp_path / "work") is None


def test_plugin_report_names_a_child_without_a_pair(tmp_path, capsys):
    extra = tmp_path / "ghost.yaml"
    extra.write_text("plugin: {children: [mmduet_7b, ghost]}\n")
    code = cli_main.main(["--home", str(tmp_path / "home"), "--config", str(extra), "--outputs", str(tmp_path / "o"),
                          "plugin", "report", "--children", "ghost"])
    assert code == 1 and "no parent for ['ghost']" in capsys.readouterr().err


def test_unimportable_modules_keep_their_command_names():
    assert cli_main.command_name("cmd_media") == "video-meta"
    assert cli_main.command_name("cmd_fetch") == "fetch-upstream"
    assert cli_main.command_name("cmd_rerun_capped") == "rerun-capped"
    assert cli_main.command_name("cmd_does_not_exist") == "does-not-exist"


def test_a_missing_package_names_the_interpreter(monkeypatch, capsys):
    module = types.ModuleType("streamreplace.cli.cmd_needs_torch")

    def register(subparsers):
        def handler(args):
            raise ModuleNotFoundError("No module named 'torch'")
        subparsers.add_parser("needs-torch").set_defaults(func=handler)
    module.register = register
    monkeypatch.setitem(sys.modules, "streamreplace.cli.cmd_needs_torch", module)
    monkeypatch.setattr(cli_main, "command_modules", lambda: ["cmd_needs_torch"])
    assert cli_main.main(["needs-torch"]) == 1
    err = capsys.readouterr().err
    assert "No module named 'torch'" in err and sys.executable in err
