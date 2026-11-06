"""The prompt-robustness analysis on fake scored outputs whose answers are known (report.py, instruction.py)."""
from __future__ import annotations

import csv
import hashlib
import json
import math

import pytest
from robustness_helpers import DEMO, prepared, write_job

from streamreplace.generalization.robustness import instruction, report

GOLD, WEAK = 2.0, 1.0           # first-token logit of the gold letter (the other letter: 0)


def p_gold(logit: float) -> float:
    return math.exp(logit) / (math.exp(logit) + 1)


def scenario(exp, child_wrong: dict | None = None, jobs: dict | None = None) -> dict:
    """Subset, paraphrases (one unit rejected), edits, then outputs: the parent is right everywhere; the child is
    right except on rotation 1 of the units listed per condition. The child's gold logit is lower under stress.
    jobs: (condition, model) -> extra write_job arguments (errored or unscored ids)."""
    prepared(exp)
    units = sorted(exp.units(DEMO, "clean"))
    wrong = child_wrong if child_wrong is not None else {"typo": units[:2], "stress": units[2:3]}
    for condition in exp.conditions:
        for model in ("parent", "child"):
            bad = set(wrong.get(condition, [])) if model == "child" else set()
            logit = WEAK if (model, condition) == ("child", "stress") else GOLD
            write_job(exp, condition, model,
                      correct=lambda s, bad=bad: not (str(s.metadata["question"]) in bad and s.metadata["rotation"] == 1),
                      logits=lambda s, logit=logit: {s.answer: logit, next(k for k in s.choices if k != s.answer): 0.0},
                      **(jobs or {}).get((condition, model), {}))
    return {"units": units, "wrong": wrong}


def test_unit_outcomes_need_every_rotation_and_count_errors_wrong(experiment):
    exp = experiment
    prepared(exp)
    units = exp.units(DEMO, "clean")
    first, second = sorted(units)[:2]
    errored = units[first][1].id                          # rotation 1 of the first unit: an errored generation
    unscored = units[second][0].id
    scenario(exp, {}, {("typo", "child"): {"error_ids": [errored]}, ("irrelevant", "child"): {"unscored_ids": [unscored]}})
    tables = report.Tables(exp)
    typo = tables.units("typo", DEMO, "child")
    assert len(typo) == 8 and set(typo) == set(units)
    assert typo[first]["correct"] is False and typo[first]["preds"][1] is None and typo[first]["answered"] is False
    assert all(v["correct"] for u, v in typo.items() if u != first)
    assert typo[first]["cluster"] == f"{DEMO}:{first}" and typo[second]["bare"] == [True, True]
    assert tables.units("irrelevant", DEMO, "child") is None       # one row is not scored yet: incomplete
    assert tables.probs("irrelevant", DEMO, "child") == {}
    assert tables.probs("clean", DEMO, "parent")[first]["p"] == pytest.approx(p_gold(GOLD))
    assert len(tables.skips("para", DEMO)) == 1 and tables.skips("typo", DEMO) == set()


def test_primary_contrasts_by_hand(experiment):
    exp = experiment
    scenario(exp)
    tables = report.Tables(exp)
    results = report.primary(exp, tables)
    typo, para, distraction = (results[("child", f)] for f in ("typo", "para", "distraction"))
    # typo: the child loses 2 of 8 units the parent keeps -> -25 points
    assert typo["d_rob"]["value"] == -25.0 and typo["d_rob"]["n"] == 8
    assert typo["d_rob"]["low"] <= -25.0 <= typo["d_rob"]["high"] <= 0.0
    assert 0.4 < typo["d_rob"]["p"] < 0.6                # two clusters of -1: same sign with probability 1/2
    assert para["d_rob"]["value"] == 0.0 and para["d_rob"]["n"] == 7       # the rejected paraphrase leaves the item
    assert distraction["d_rob"]["value"] == pytest.approx(-100 * 0.5 / 8)  # one unit wrong under stress, mean of two
    assert typo["acc"]["value"] == typo["d_rob"]["value"] and typo["acc"]["n"] == 8
    assert {"lower_0.05", "lower_0.00277778", "tost_p", "equivalent", "low90", "high90"} <= set(typo["acc"])
    # gold-option probability: the child's gold share drops under stress only
    expected = 100 * -(p_gold(GOLD) - p_gold(WEAK)) / 2
    assert distraction["prob"]["value"] == pytest.approx(expected) and typo["prob"]["value"] == pytest.approx(0)
    holm = sorted(v["d_rob"]["p_holm"] for v in results.values())
    assert all(v["d_rob"]["p_holm"] >= v["d_rob"]["p"] for v in results.values()) and holm[-1] <= 1.0
    # PDR: the child's relative loss 2/8 under typo vs none for the parent
    assert typo["pdr"]["value"] == pytest.approx(-25.0)


def test_report_files(experiment, tmp_path):
    exp = experiment
    scenario(exp)
    files = report.write_report(exp, tmp_path / "out")
    assert set(files) == {"report", "results", "items", "primary", "contrasts", "instruction"}
    with open(files["items"], newline="", encoding="utf-8") as f:
        items = list(csv.DictReader(f))
    assert len(items) == 5 * 2 * 8 and list(items[0]) == ["condition", "dataset", "model", "unit", "cluster",
                                                          "correct", "preds"]
    assert open(files["items"], "rb").read().count(b"\r\n") == len(items) + 1          # csv default dialect
    with open(files["primary"], newline="", encoding="utf-8") as f:
        primary = {r["family"]: r for r in csv.DictReader(f)}
    assert float(primary["typo"]["d_rob_pp"]) == -25.0 and primary["para"]["items"] == "7"
    with open(files["contrasts"], newline="", encoding="utf-8") as f:
        contrasts = list(csv.DictReader(f))
    assert len(contrasts) == 6 and {r["readout"] for r in contrasts} == {"acc", "prob"}
    assert {"child", "family", "readout", "n", "value", "low", "high", "p_holm", "equivalent"} <= set(contrasts[0])
    results = json.loads(open(files["results"], encoding="utf-8").read())
    entry = results["robustness_by_benchmark"][0]
    assert entry["n_clean"] == 8 and entry["clean_parent"] == 100.0 and entry["typo"]["delta_rob_pp"] == -25.0
    assert entry["typo"]["rob_tax_pp"] == 25.0 and entry["para"]["n"] == 7
    accuracy = results["per_benchmark"][DEMO]["accuracy"]
    child = next(r for r in accuracy if r["model"] == "child")
    assert child["conditions"]["typo"]["drop"] == 25.0 and child["conditions"]["typo"]["pdr"] == 0.25
    text = open(files["report"], encoding="utf-8").read()
    assert "## Primary: ΔRob per child × family (pooled over rdemo; Holm over 3 tests" in text
    assert "| p | parent | child | -25.0 [" in text and "| child | 100.0 | 75.0 (+25.0; PDR +0.25)" in text


def test_incomplete_tables_leave_their_contrasts_out(experiment, tmp_path):
    exp = experiment
    scenario(exp, jobs={("stress", "child"): {"unscored_ids": [s.id for s in exp.prepared(DEMO)]}})
    results = report.primary(exp, report.Tables(exp))
    assert results[("child", "distraction")]["d_rob"] is None
    assert results[("child", "typo")]["d_rob"]["p_holm"] == pytest.approx(min(1.0, 2 * results[("child", "typo")]["d_rob"]["p"]))
    text = open(report.write_report(exp, tmp_path / "out")["report"], encoding="utf-8").read()
    assert "| stress | 1/2 models |" in text


def test_instruction_level_analysis(experiment, tmp_path):
    exp = experiment
    prepared(exp)
    nested = sorted(exp.units(DEMO, "ins_typo"))
    s = scenario(exp, {"ins_typo": nested[:1]})
    level = instruction.analyse(exp, report.Tables(exp))
    assert [g["conditions"] for g in level["groups"]] == [["ins_typo", "ins_para", "ins_irrelevant"], ["ins8_typo"]]
    first = level["groups"][0]
    result = first["results"][("child", "ins_typo")]
    assert result["instruction"]["value"] == -25.0 and result["instruction"]["n"] == 4
    assert result["stem"]["value"] == 0.0 and result["did"]["value"] == -25.0
    row = first["rows"][0]
    assert row["n_selected"] == 4 and row["typo"]["condition"] == "ins_typo" and row["typo"]["delta_rob_pp"] == -25.0
    assert row["typo"]["instruction_parent"] == 100.0 and row["typo"]["instruction_child"] == 75.0
    skipped = next(iter(report.Tables(exp).skips("para", DEMO)))
    assert row["para"]["n"] == 4 - (skipped in nested)
    rep = level["groups"][1]["replication"]
    assert rep["earlier"] == {"ins8_typo": "ins_typo"}
    assert rep["identical"][("parent", "ins8_typo")] == (8, 8) and rep["identical"][("child", "ins8_typo")] == (7, 8)
    assert rep["split"][("child", "ins8_typo", "earlier")]["n"] == 4
    assert len(level["contrasts"]) == 4 and all("p_holm" in v["prob"] for v in level["contrasts"].values())
    lines = instruction.markdown(exp, level)
    assert any(line.startswith("# Instruction-level perturbations (vs the question-stem twin, same items): ins_typo")
               for line in lines)
    assert "| child | 7/8 |" in lines and s["units"]


def test_instruction_span_requires_a_single_instruction(demo):
    assert instruction.instruction_span(DEMO) == "Answer with the option's letter from the given choices directly."
    assert instruction.instruction_span("mmbench") == "Answer with the option's letter from the given choices directly."
    with pytest.raises(ValueError, match="single instruction sentence"):
        instruction.instruction_span("videomme")
    texts = ["one", "two", "three"]
    index = int(hashlib.sha256(b"0|demo|9|ins_para").hexdigest(), 16) % 3
    assert instruction.choice(texts, "0|demo|9|ins_para") == texts[index]
