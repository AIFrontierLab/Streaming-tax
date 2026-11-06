"""`streamreplace finetune ...` end to end on the command line: prepare and check (also from the paper's frozen lists),
the models file, units, commands, an evaluation dry run, the reports."""
import json

import pytest
import yaml

from finetune_helpers import DEMO, spec, write_scored

from streamreplace.cli.main import main
from streamreplace.config.load import load_config
from streamreplace.generalization.finetune import data


def run(capsys, *argv) -> tuple[int, str]:
    code = main(list(argv))
    return code, capsys.readouterr().out


@pytest.fixture
def demo_spec(tmp_path):
    path = tmp_path / "finetune.yaml"
    path.write_text(yaml.safe_dump(spec()), encoding="utf-8")
    return path


def test_the_command_lists_its_steps(capsys):
    with pytest.raises(SystemExit) as stop:
        main(["finetune", "--help"])
    assert stop.value.code == 0
    out = capsys.readouterr().out
    for action in ("prepare", "check", "models", "units", "train", "eval", "commands", "report", "seeds"):
        assert action in out


def test_prepare_check_and_report_on_the_command_line(demo, demo_spec, tmp_path, capsys):
    root = tmp_path / "root"
    base = ["--home", str(tmp_path / "home"), "finetune", "--spec", str(demo_spec), "--root", str(root)]
    code, out = run(capsys, *base, "prepare", DEMO, "--workers", "1")
    assert code == 0 and json.loads(out)[0]["test_subset"] == 12
    for name in data.FILES:
        assert (root / f"{name}.json").is_file()
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    for name in data.FILES:
        (frozen / f"{name}.json").write_bytes((root / f"{name}.json").read_bytes())
    code, out = run(capsys, *base, "check", "--frozen", str(frozen))
    assert code == 0 and "identical to the paper's lists" in out
    (frozen / "subset.json").write_text(json.dumps({DEMO: ["q00"]}))
    code, out = run(capsys, *base, "check", "--frozen", str(frozen))
    assert code == 1 and "differs from the paper's" in out
    code, out = run(capsys, *base, "prepare", "lingoqa")
    assert code == 1                                      # not a dataset of this experiment
    ids = data.subset_ids(root, DEMO)
    for model, budget, right in (("par", 0, 6), ("par", 6, 9), ("par", 12, 9), ("kid", 0, 3), ("kid", 6, 9),
                                 ("kid", 12, 12)):
        write_scored(root, f"{model}@{DEMO}-{budget}", ids, right)
    code, out = run(capsys, *base, "report")
    assert code == 0 and [p.rsplit("/", 1)[-1] for p in json.loads(out)] == ["FT_RESULTS.md", "READOUT.md"]


def test_the_papers_lists_models_units_commands_and_an_evaluation_plan(home, tmp_path, capsys):
    root = tmp_path / "ft"
    base = ["--home", str(tmp_path / "home"), "finetune", "--root", str(root)]
    code, out = run(capsys, *base, "prepare", "lingoqa", "eyepcr", "openuav_qa", "--frozen")
    assert code == 0 and len(json.loads(out)) == 3
    assert run(capsys, *base, "check")[0] == 0
    code, out = run(capsys, *base, "models", "--out", str(tmp_path / "models.yaml"))
    assert code == 0 and "qwen25vl_3b@lingoqa-5k" in load_config(extra=[tmp_path / "models.yaml"]).models
    code, out = run(capsys, *base, "models")
    assert code == 0 and "mmduet_7b_merged@eyepcr-0:" in out
    code, out = run(capsys, *base, "units", "--datasets", "lingoqa", "--models", "aura_8b")
    assert code == 0 and len(out.splitlines()) == 3 and out.splitlines()[0].split()[:3] == ["aura_8b", "lingoqa", "1k"]
    code, out = run(capsys, *base, "commands", "--gpus", "0,1")
    prefix = f"streamreplace --home {tmp_path / 'home'}"            # the global options and the root on every line
    assert code == 0 and f"{prefix} finetune --root {root} eval --control blank --gpus 0,1" in out.splitlines()
    assert f"{prefix} --outputs {root} score --judge --judge-gpus 0" in out.splitlines()
    code, out = run(capsys, *base, "eval", "--datasets", "lingoqa", "--models", "qwen25vl_3b,thinkstream_3b",
                    "--dry-run")
    lines = out.splitlines()
    assert code == 0 and lines[0].startswith("not trained yet (skipped): qwen25vl_3b@lingoqa-1k")
    assert [l.split()[0] for l in lines[1:3]] == ["lingoqa/qwen25vl_3b@lingoqa-0.offline",
                                                  "lingoqa/thinkstream_3b@lingoqa-0.offline"]
    assert lines[-1].startswith("2 jobs planned")
    code, out = run(capsys, *base, "eval", "--control", "images", "--dry-run")
    assert code == 0 and "openuav_qa/qwen25vl_3b@openuav_qa-0.offline" in out
    assert json.loads((root / "images" / "subset.json").read_text())["openuav_qa"] == data.subset_ids(root, "openuav_qa")
