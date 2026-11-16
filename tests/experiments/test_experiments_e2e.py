"""End to end on a mock project: the arms of an experiment are planned and run through `streamreplace experiment run`
(the real dispatcher and worker, mock models on the CPU), scored through the command line and reported."""
from __future__ import annotations

import json

import yaml

from e2e_project import make_project

from streamreplace import experiments
from streamreplace.core.io import read_jsonl
from streamreplace.paths import project_root

MODELS = {
    "parent": {"role": "parent", "checkpoint": "test/parent@0", "offline": {"adapter": "mock", "env": "local"},
               "options": {"accuracy": 1.0}},
    "cand": {"role": "child", "checkpoint": "test/cand@0", "offline": {"adapter": "mock", "env": "local"},
             "stream": {"adapter": "mock", "env": "local"}, "options": {"accuracy": 0.0}},
}
ANALYSIS = {"engine": {"reps": 200},
            "metrics": [{"dataset": "demo", "name": "acc", "kind": "acc", "cluster": "video", "family": "video"}]}
IDS = ["mcq-1", "mcq-2", "yn-1", "yn-2"]


def project_with(tmp_path, spec, models=None):
    items = tmp_path / "items.json"
    items.write_text(json.dumps({"demo": IDS}))
    judges = yaml.safe_load((project_root() / "configs" / "judges.yaml").read_text(encoding="utf-8"))
    sections = {"analysis": ANALYSIS, "judges": judges,
                "experiments": {"models": models or {},
                                "experiments": {"control": {"title": "A control", "paper": "none",
                                                            "items": str(items), "children": ["cand"], **spec}}}}
    return make_project(tmp_path, models=MODELS, sections=sections)


def test_a_control_runs_scores_and_reports(demo, tmp_path, capsys):
    project = project_with(
        tmp_path,
        {"arms": {"C_right": "cand_right.offline",                                     # a derived model id
                  "S_fps2": {"job": "{child}.stream", "folder": "fps2", "set": {"stream": {"fps": 2}}}},
         "contrasts": ["C_right - C", "S_fps2 - S", "C - P"]},
        models={"cand_right": {"from": "cand", "options": {"accuracy": 1.0}}})
    assert project.cli("run", "--pairs", "p", "--benchmarks", "demo", "--gpus", "0") == 0      # the formal P, C, S
    assert project.cli("score") == 0
    assert project.cli("experiment", "list") == 0
    assert "control" in capsys.readouterr().out
    assert project.cli("experiment", "run", "control", "--dry-run") == 0
    assert "2 jobs planned (0 complete)" in capsys.readouterr().out
    assert project.cli("experiment", "run", "control", "--gpus", "0") == 0
    out = capsys.readouterr().out
    root = project.outputs / "experiments" / "control"
    assert f"--outputs {root} score" in out and f"--outputs {root / 'fps2'} score" in out
    for folder in (root, root / "fps2"):
        assert project.cli("--outputs", str(folder), "score") == 0
    right = read_jsonl(root / "demo" / "cand_right.offline.jsonl")
    assert sorted(r["id"] for r in right) == sorted(IDS)                               # only the experiment's items
    assert json.loads((root / "fps2" / "demo" / "cand.stream.meta.json").read_text())["identity"]
    assert project.cli("experiment", "report", "control") == 0
    report = json.loads((root / "report.json").read_text())
    (cell,) = report["cells"]
    assert (cell["child"], cell["dataset"], cell["metric"]) == ("cand", "demo", "acc")
    assert list(cell["arms"]) == ["P", "C", "S", "C_right", "S_fps2"]                  # references, then as declared
    assert {arm: v["value"] for arm, v in cell["arms"].items()} == {"P": 100, "C": 0, "S": 0, "C_right": 100,
                                                                    "S_fps2": 0}
    assert all(v["n"] == len(IDS) for v in cell["arms"].values())
    assert cell["contrasts"]["C_right - C"]["delta"] == 100 and cell["contrasts"]["S_fps2 - S"]["delta"] == 0
    assert cell["contrasts"]["C - P"]["delta"] == -100
    text = (root / "REPORT.md").read_text()
    assert "A control" in text and "C_right - C" in text
    assert project.cli("experiment", "label", "control") == 1                          # scored by `score`, not labelled


def test_harmless_requests_are_asked_and_labelled(demo, tmp_path):
    from streamreplace.experiments import forms
    from streamreplace.plugin.calibrate.harmless import HOWTO
    from streamreplace.plugin.recipe import VIDEO_PREFIX
    project = project_with(tmp_path, {"kind": "forms", "options": {"requests": {"howto": "howto"}},
                                      "arms": {"A": {"job": "cand_right.offline", "folder": "howto"},
                                               "B": {"job": "{child}.offline", "folder": "howto"}},
                                      "contrasts": ["B - A"]},
                           models={"cand_right": {"from": "cand", "options": {"accuracy": 1.0}}})
    assert project.cli("experiment", "run", "control", "--gpus", "0") == 0
    exp = experiments.load(project.config(), project.paths, "control")
    rows = {job: {r["id"]: r for r in read_jsonl(exp.root / "howto" / "demo" / f"{job}.offline.jsonl")}
            for job in ("cand_right", "cand")}
    assert [rows["cand"][i]["prompt"] for i in IDS] == [VIDEO_PREFIX + HOWTO[k] for k in range(len(IDS))]
    assert all(rows["cand"][i]["raw"] != rows["cand_right"][i]["raw"] for i in IDS)
    assert project.cli("experiment", "report", "control") == 2                         # no label yet
    answers = []                                # the judge's answers: arm B answers one request, arm A all of them
    for job, label_of in (("cand_right", lambda i: "A"), ("cand", lambda i: "A" if i == "mcq-1" else "V")):
        for i, row in rows[job].items():
            request = forms.harmless_request(row["prompt"], row["raw"])
            answers.append({"key": request["key"], "reply": json.dumps({"label": label_of(i)})})
    forms.judge(project.paths).cache(exp.root).add(answers)
    assert project.cli("experiment", "report", "control") == 0
    (cell,) = json.loads((exp.root / "report.json").read_text())["cells"]
    assert cell["split"] == "harmless" and list(cell["arms"]) == ["A", "B"]
    assert cell["arms"]["A"]["answered"] == 100 and cell["arms"]["B"]["answered"] == 25
    assert cell["arms"]["B"]["off-task"] == 75
    assert cell["contrasts"]["B - A"]["answered"]["delta"] == -75
