"""`streamreplace robustness ...` end to end on the synthetic benchmark: subset, paraphrase (a stub endpoint),
edits, check, freeze + check --frozen, audit, runs, report."""
from __future__ import annotations

import json

import yaml
from robustness_helpers import DEMO, write_job

from streamreplace import paths as paths_module
from streamreplace.cli.main import main
from streamreplace.generalization.robustness import make, paraphrase

CONFIG = {
    "defaults.yaml": {"defaults": {"generation": {"max_new_tokens": 16}, "image": {"max_pixels": 1003520},
                                   "video": {"nframes": 8, "max_pixels": 200704}, "stream": {"fps": 1}}},
    "models.yaml": {"models": {
        "parent": {"role": "parent", "checkpoint": "test/parent@0", "offline": {"adapter": "mock", "env": "local"}},
        "child": {"role": "child", "checkpoint": "test/child@0", "offline": {"adapter": "mock", "env": "local"}}}},
    "pairs.yaml": {"pairs": {"p": {"parent": "parent", "children": ["child"]}}},
    "benchmarks.yaml": {"benchmarks": {DEMO: {"media": "image"}}, "suites": {}},
}


def test_command_line_end_to_end(demo, tmp_path, spec, monkeypatch, capsys):
    configs = tmp_path / "configs"
    for name, data in CONFIG.items():
        (configs / name).parent.mkdir(parents=True, exist_ok=True)
        (configs / name).write_text(yaml.safe_dump(data))
    spec_file = tmp_path / "robustness.yaml"
    spec_file.write_text(yaml.safe_dump({**spec, "paraphrase": {"endpoint": "http://127.0.0.1:9/v1"}}))
    root = tmp_path / "exp"

    def cli(*argv) -> int:
        try:
            return main(["--config-dir", str(configs), "--home", str(demo.home), "robustness", "--spec",
                         str(spec_file), "--root", str(root), *argv])
        finally:
            paths_module.set_current(demo)

    def answer(self, prompt):                          # the "endpoint": prefix-reword, always equivalent
        if "Do the two versions" in prompt:
            return "yes"
        return "Please tell me, " + prompt.split("\n\n", 1)[1].removeprefix("Please tell me, ")
    monkeypatch.setattr(paraphrase.Endpoint, "__call__", answer)
    assert cli("subset") == 0 and (root / "clean" / "subset.json").exists()
    capsys.readouterr()
    assert cli("paraphrase") == 0
    assert json.loads(capsys.readouterr().out.split("\n", 1)[1])["ok"] == 8
    assert cli("edits") == 0 and cli("check") == 0
    assert cli("freeze", "--out", str(tmp_path / "frozen")) == 0
    frozen = json.loads((tmp_path / "frozen" / "edits.json").read_text())
    assert frozen["file"] == "prompt_overrides.jsonl" and frozen["conditions"]["typo"]["rows"] == 16
    capsys.readouterr()
    assert cli("check", "--frozen", str(tmp_path / "frozen")) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["frozen"]["items"] == "identical" and set(result["frozen"]["edits"].values()) == {"identical"}
    assert cli("audit", "--n", "2") == 0 and (root / "audit" / "para.md").exists()
    capsys.readouterr()
    assert cli("runs") == 0
    commands = capsys.readouterr().out
    prefix = f"streamreplace --home {demo.home} --config-dir {configs}"          # this invocation's global options
    assert f"{prefix} --outputs {root / 'typo'} run --pairs p --protocols P,C --benchmarks {DEMO}" in commands
    assert f"{prefix} robustness --spec {spec_file} --root {root} report" in commands
    exp = make.Experiment(make.load_spec(path=spec_file), root, {"p": ("parent", "child")}, demo)
    for condition in exp.conditions:
        for model in ("parent", "child"):
            write_job(exp, condition, model)
    assert cli("report") == 0
    assert (root / "report" / "ROBUSTNESS.md").exists() and (root / "report" / "instruction.csv").exists()
