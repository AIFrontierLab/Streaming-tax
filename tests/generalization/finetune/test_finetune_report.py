"""The fine-tuning reports on synthetic scored rows: FT_RESULTS.md and ft_tables (accuracy per budget, dAcc / dGain,
the primary endpoint with Holm, the blank-frame 2x2), READOUT.md with cells.csv / pooled.csv, and the seed
replication's SEED_RESULTS.md with its decision rules."""
import csv
import json

import pytest

from finetune_helpers import DEMO, write_scored

from streamreplace.core.io import write_json
from streamreplace.generalization.finetune import data, report, seeds


def rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def scored(exp):
    """The demo experiment prepared (12 test questions) with every evaluation scored: parent 50 -> 75 -> 75%,
    child 25 -> 75 -> 100% (budgets 6 and 12), so dAcc@0 = -25, dGain = +25 / +50 and the primary endpoint +37.5."""
    data.prepare(exp, DEMO, workers=1)
    ids = data.subset_ids(exp.root, DEMO)
    for model, budget, right in (("par", 0, 6), ("par", 6, 9), ("par", 12, 9), ("kid", 0, 3), ("kid", 6, 9),
                                 ("kid", 12, 12)):
        write_scored(exp.root, f"{model}@{DEMO}-{budget}", ids, right)
    return exp, ids


def test_ft_results_tables(scored):
    exp, ids = scored
    text = report.write_report(exp).read_text(encoding="utf-8")
    assert f"## {DEMO} (12 test questions)" in text
    assert "| model | acc@0 | acc@6 | acc@12 | gain@6 | gain@12 | norm gain@12 | letter acc@0 | letter acc@12 |" in text
    assert "| par | 50.0 | 75.0 | 75.0 | 25.0 | 25.0 | 0.50 | – | – |" in text
    assert "| kid | 25.0 | 75.0 | 100.0 | 50.0 | 75.0 | 1.0 | – | – |" in text
    line = next(l for l in text.splitlines() if l.startswith("| p | kid |"))
    cells = line.split(" | ")
    assert cells[2].startswith("-25.0 [") and cells[3].startswith("+0.0 [") and cells[4].startswith("+25.0 [")
    assert cells[5].startswith("+25.0 [") and cells[6].startswith("+50.0 [") and cells[7].startswith("+37.5 [")
    assert " p=" not in line                      # questions share cases: no McNemar
    primary = next(l for l in text.splitlines() if l.startswith(f"| {DEMO} | p | kid | +37.50 | ["))
    p, holm = primary.strip("| ").split(" | ")[-2:]
    assert p == holm                              # a one-cell Holm family
    accuracy = rows(exp.root / "ft_tables" / "accuracy.csv")
    assert [(r["model"], r["acc_0"], r["acc_12"], r["norm_gain"]) for r in accuracy] == [
        ("par", "50.0", "75.0", "0.5"), ("kid", "25.0", "100.0", "1.0")]
    pairs = rows(exp.root / "ft_tables" / "pairs.csv")
    assert len(pairs) == 1 and float(pairs[0]["primary"]) == pytest.approx(37.5) and pairs[0]["in_primary"] == "True"
    assert pairs[0]["d_acc_0_ci"].count(",") == 1 and "6 for 3 epochs" in text and "| par | 75.0 | – |" in text


def test_results_json_holds_the_paper_table_data(scored):
    exp, _ = scored
    report.write_report(exp)
    data_ = json.loads((exp.root / "ft_tables" / "results.json").read_text(encoding="utf-8"))
    assert data_["finetuning_accuracy"][DEMO]["par"] == {"acc@0": 50.0, "acc@6": 75.0, "acc@12": 75.0, "gain@6": 25.0,
                                                          "gain@12": 25.0, "norm gain@12": 0.5, "letter acc@0": None,
                                                          "letter acc@12": None}
    [primary] = data_["finetuning_primary"]
    assert primary["dataset"] == DEMO and primary["child"] == "kid" and primary["DeltaGain"] == "+37.50"
    assert primary["CI"].startswith("[") and primary["p"] == primary["Holm"]
    [common] = data_["finetuning_max_common_budget"]
    assert common["train_budget"] == 12 and common["parent_final"] == 75.0 and common["child_final"] == 100.0
    assert common["delta_gain_pp"] == 50.0 and common["budget_average_delta_gain_pp"] == pytest.approx(37.5)
    assert set(common["curve"]) == {"6", "12"} and common["n_test"] == 12 and common["in_original_primary"] is True


def test_an_excluded_dataset_keeps_its_tables_but_leaves_the_primary_endpoint(scored):
    exp, _ = scored
    exp.spec["primary_excludes"] = {DEMO: "answerable without the video"}
    text = report.write_report(exp).read_text(encoding="utf-8")
    assert "**Not in the primary endpoint:** answerable without the video" in text
    assert "## Primary endpoint" not in text and rows(exp.root / "ft_tables" / "pairs.csv")[0]["in_primary"] == "False"


def test_incomplete_evaluations_are_shown_not_guessed(scored):
    exp, ids = scored
    write_scored(exp.root, f"kid@{DEMO}-12", ids[:-1], 11)
    text = report.write_report(exp).read_text(encoding="utf-8")
    assert "| p | kid | incomplete |" in text and "| kid | 25.0 | 75.0 | – |" in text


def test_blank_frame_control_2x2(scored):
    exp, ids = scored
    blank_ids = data.control_subset(exp, DEMO, exp.controls["blank"]["size"])
    data.write_subset(exp.folder("blank"), {DEMO: blank_ids})
    for model, budget, right in (("par", 0, 1), ("par", 12, 2), ("kid", 0, 0), ("kid", 12, 4)):
        write_scored(exp.folder("blank"), f"{model}@{DEMO}-{budget}", blank_ids, right)
    text = report.write_report(exp).read_text(encoding="utf-8")
    assert f"Blank frames, zero-shot vs 12 ({DEMO}, 4 questions;" in text
    table = rows(exp.root / "ft_tables" / f"blank_2x2_{DEMO}.csv")
    assert [r["model"] for r in table] == ["par", "kid"]
    par = table[0]
    real = {k: 100 * sum(i in set(ids[:n]) for i in blank_ids) / 4 for k, n in (("r0", 6), ("rt", 9))}
    assert float(par["acc_r0"]) == real["r0"] and float(par["acc_b0"]) == 25.0 and float(par["acc_bt"]) == 50.0
    assert float(par["gain"]) == pytest.approx(float(par["gain_blank"]) + float(par["dvd"]))     # the identity
    assert "ΔVis = ΔVD_child − ΔVD_parent" in text


def test_readout_cells_and_pooled_estimate(scored):
    exp, _ = scored
    report.write_report(exp)
    text = report.write_readout(exp).read_text(encoding="utf-8")
    assert "(max |Δ| over value and p = 0)" in text and f"| {DEMO} | p | kid | 12 | -25.0 [" in text
    assert "Non-inferiority of post-fine-tuning accuracy (ftdemo; 1 cells)" in text
    cells = {r["contrast"]: r for r in rows(exp.root / "ft_tables" / "cells.csv")}
    assert set(cells) == set(report.READOUT_KEYS)
    assert float(cells["d_gain_avg"]["value"]) == pytest.approx(37.5) and float(cells["d_acc_0"]["value"]) == -25.0
    assert float(cells["d_acc_avg"]["value"]) == pytest.approx(12.5)              # (0 + 25) / 2
    assert cells["d_gain_avg"]["holm12"] == cells["d_gain_avg"]["p"] == cells["d_gain_avg"]["holm18"]
    pooled = rows(exp.root / "ft_tables" / "pooled.csv")
    assert {(r["datasets"], r["contrast"]) for r in pooled} == {(n, k) for n in ("primary", "all")
                                                                for k in ("d_gain_avg", "d_acc_avg", "d_acc_0")}
    assert float(next(r for r in pooled if r["contrast"] == "d_gain_avg")["value"]) == pytest.approx(37.5)


def test_reading_rule():
    entry = {"dataset": "x", "d_gain_avg": {"holm12": 0.01}, "d_acc_avg": {"low": -1.0, "high": 2.0}}
    assert report.reading(entry, ["x"]) == "zero-shot difference (post-FT parity)"
    entry["d_acc_avg"] = {"low": 0.5, "high": 2.0}
    assert report.reading(entry, ["x"]) == "ΔGain and post-FT difference"
    entry["d_gain_avg"] = {"holm12": 0.5}
    assert report.reading(entry, ["x"]) == "post-FT difference only"
    assert report.reading({"dataset": "y", "d_gain_avg": {"holm12": 0.01, "holm18": 0.5},
                           "d_acc_avg": {"low": -1.0, "high": 1.0}}, ["x"]) == "no difference"


def test_seed_report_spreads_and_decision_rules(exp):
    """The parent is always right; the child's accuracy at budget 6 differs by seed.
    Mean / SD / range of dAcc over seeds 0-2, dGain = dAcc - dAcc(0), the decision rules, and an unscored evaluation
    shown as missing."""
    ids = ["q00", "q01", "q02", "q03", "q04", "q05"]
    write_json(exp.root / "subset.json", {DEMO: ids})
    for budget in (0, 6):
        write_scored(exp.root, f"par@{DEMO}-{budget}", ids, 6)
    write_scored(exp.root, f"kid@{DEMO}-0", ids, 2)
    write_scored(exp.root, f"kid@{DEMO}-6", ids, 3)
    folder = exp.seeds_folder()
    write_json(folder / "subset.json", {DEMO: ids})
    for seed, right in ((1, 4), (2, 1)):
        write_scored(folder, f"par@{DEMO}-6-s{seed}", ids, 6)
        write_scored(folder, f"kid@{DEMO}-6-s{seed}", ids, right)
    text = seeds.write_report(exp).read_text(encoding="utf-8")
    assert "| p | kid | 6 | -50.0 [" in text and " | -33.3 [" in text and " | -83.3 [" in text
    assert "| -55.6 | 25.5 | 50.0 | [" in text and "| +11.1 [" in text               # dGain = dAcc - (-66.7)
    assert f"| {DEMO} | p | kid | 6 | -55.6 (25.5, 50.0, [" in text and "unresolved within seed noise | run (25.5 pp) |" in text
    assert "| kid | 33.3 | 50.0 | 66.7 | 16.7 | 44.4 ± 25.5 (k=3) |" in text and "| avg |" not in text
    summary = rows(folder / "seed_tables" / "summary.csv")
    assert len(summary) == 1 and summary[0]["k"] == "3" and float(summary[0]["d_acc_range"]) == pytest.approx(50.0)
    assert [r["seed"] for r in rows(folder / "seed_tables" / "seeds.csv")] == ["0", "1", "2"]
    write_scored(folder, f"kid@{DEMO}-6-s2", ids, 1, scored=False)
    text = seeds.write_report(exp).read_text(encoding="utf-8")       # unscored: shown missing, no decision yet
    assert "| kid | 33.3 | 50.0 | 66.7 | – | 58.3 ± 11.8 (k=2) |" in text and f"| {DEMO} | p | kid |" not in text


def test_seed_pairs_keep_the_replicated_members_only(exp):
    exp.spec["seeds"]["models"] = ["par"]
    assert seeds.pairs(exp) == {}
    exp.spec["seeds"]["models"] = ["par", "kid"]
    assert seeds.pairs(exp) == {"p": ["par", "kid"]}
