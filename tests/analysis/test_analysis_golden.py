"""The paper's golden numbers (paper/expected/): the shipped files themselves (contrasts, the plugin recovery cells and
their consistency with the shipped plugin artefacts), and - when the paper's scored rows are in the outputs folder - a
full recomputation of the contrasts and metrics against them."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import pytest

from streamreplace import paths as paths_module
from streamreplace.analysis.specs import load_profile, repo_root

EXPECTED = repo_root() / "paper" / "expected"
GOLDEN = {   # (dataset, metric, child, gap) -> delta_pp, ci95_pp, p (the paper's supplement, Supp. B)
    ("videomme", "acc", "mmduet_7b", "C-P"): (-3.7778, (-5.6667, -1.8889), 0.00029997),
    ("mathvista", "acc", "mmduet_7b", "C-P"): (-2.2, (-3.5, -0.9), 0.00126),
    ("mme", "perception", "mmduet2_3b", "C-P"): (-6.0643, (-8.8993, -3.1535), None),
    ("dynamath", "avg_acc", "mmduet_7b", "C-P"): (-0.1397, (-1.7964, 1.5569), 0.8881),
}


def rows(name: str) -> dict:
    data = json.loads((EXPECTED / name).read_text(encoding="utf-8"))
    return {(r["dataset"], r["metric"], r["child"], r["gap"]): r for r in data["rows"]}


@pytest.mark.parametrize("name", ["contrasts.json", "contrasts_no_overlay.json"])
def test_shipped_contrasts_hold_the_documented_values(name):
    table = rows(name)
    assert len(table) == 334          # 354 metric x child x gap cells, 20 of them not reported (see below)
    for key, (delta, ci, p) in GOLDEN.items():
        r = table[key]
        assert round(r["delta_pp"], 4) == pytest.approx(delta, abs=1e-4)
        assert [round(x, 4) for x in r["ci95_pp"]] == pytest.approx(list(ci), abs=1e-4)
        if p is not None:
            assert r["p"] == pytest.approx(p, rel=1e-3)
    mathvista = table[("mathvista", "acc", "mmduet_7b", "C-P")]
    assert mathvista["test"] == "mcnemar" and (mathvista["a_only"], mathvista["b_only"]) == (33, 11)


def test_rerun_overlay_changes_only_the_rerun_benchmarks():
    merged, formal = rows("contrasts.json"), rows("contrasts_no_overlay.json")
    changed = {k for k in merged if merged[k]["delta"] != formal[k]["delta"]}
    assert changed and {k[0] for k in changed} <= {"mathvista", "mmmu", "mmstar"}
    assert {k[3] for k in changed} == {"C-P"}
    aura = ("mathvista", "acc", "aura_8b", "C-P")
    assert round(formal[aura]["delta_pp"], 1) == -7.8 and round(merged[aura]["delta_pp"], 1) == -8.5


def test_unreported_contrasts_are_the_silent_stream_cells():
    """A stream arm that answers none of a set's requests has no ASR or privacy score there, and MMDuet's one rated
    fairness reply leaves three fairness scores undefined: those S-C and S-P contrasts are not reported, which gives
    the Holm families of the paper (safety 51 / 21, VMDT 89 / 41)."""
    data = json.loads((EXPECTED / "contrasts.json").read_text(encoding="utf-8"))
    cells = Counter((u["dataset"], u["metric"], u["child"]) for u in data["unreported"])
    assert set(cells.values()) == {2} and {u["gap"] for u in data["unreported"]} == {"S-C", "S-P"}
    assert sorted(cells) == sorted(
        [("mcv_safetybench", "asr_explicit", "mmduet2_3b"), ("mcv_safetybench", "asr_implicit", "mmduet2_3b"),
         ("mcv_safetybench", "asr_implicit", "mmduet_7b")]
        + [("vmdt_fairness", m, "mmduet_7b") for m in ("fairness", "stereotype", "decision")]
        + [("vmdt_privacy", m, "mmduet_7b") for m in ("privacy", "state", "city", "zipcode")])
    families = Counter(r["holm_broad_family"] for r in data["rows"])
    assert [families[k] for k in ("safety/parent", "safety/S-C", "vmdt/parent", "vmdt/S-C")] == [51, 21, 89, 41]
    native = rows("contrasts.json")[("videomme", "acc", "mmduet_7b", "S-C")]          # MMDuet's native stream
    assert round(native["b"], 1) == 0.8 and round(native["delta_pp"], 1) == -64.8


def test_plugin_recovery_golden_file():
    """paper/expected/plugin_recovery.json: Stream++'s recovery cells on the paper's 10,140 evaluation rows per
    child (6 children x 9 metrics), the figure cells as plugin.report.figure_data reads them, the per-metric means of
    Fig. 6 and the thresholds, which are the shipped artefacts' (plugin_artefacts/<child>/manifest.json)."""
    from streamreplace.plugin.report import figure_data, means
    data = json.loads((EXPECTED / "plugin_recovery.json").read_text(encoding="utf-8"))
    assert len({(c["child"], c["dataset"], c["metric"]) for c in data["cells"]}) == len(data["cells"]) == 54
    assert figure_data({"cells": data["cells"], "means": data["means"]})["cells"] == data["figure"]
    assert means(data["cells"]) == data["means"]
    change = {(m["dataset"], m["metric"]): (len(m["children"]), round(m["change"], 1)) for m in data["means"]}
    assert change[("videomme", "acc")] == (6, 12.5)                    # the paper's headline numbers (Sec. 6.2)
    assert change[("video_safetybench", "asr_harmful")] == (6, 7.7)
    assert change[("vmdt_fairness", "fairness")] == (5, 4.0) and change[("vmdt_privacy", "privacy")] == (5, 5.6)
    for child, tau in data["thresholds"].items():
        manifest = json.loads((repo_root() / "plugin_artefacts" / child / "manifest.json").read_text(encoding="utf-8"))
        assert tau == {k: manifest["thresholds"][k] for k in ("closed", "free")}, child


def test_shipped_profile_matches_the_golden_rows():
    profile = load_profile()
    table = rows("contrasts.json")
    keys = [(m.dataset, m.name) for m in profile.metrics if m.family != "streaming"]     # Supp. D.4: its own report
    assert list(dict.fromkeys((k[0], k[1]) for k in table)) == keys
    assert {k[2] for k in table} == set(profile.children)


def paper_outputs() -> Path | None:
    """The outputs folder when it holds the paper's scored jobs (all 235 score files)."""
    try:
        outputs = Path(paths_module.current().outputs)
    except (ValueError, OSError):
        return None
    files = list(outputs.glob("*/*.scores.jsonl"))
    return outputs if len(files) >= 235 and (outputs / "vmdt_privacy" / "moss_vl_rt.stream.scores.jsonl").is_file() \
        else None


def same(x, y) -> bool:
    if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
        return True
    if isinstance(x, dict) and isinstance(y, dict):
        return x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return len(x) == len(y) and all(same(a, b) for a, b in zip(x, y))
    return x == y


@pytest.mark.requires_data
@pytest.mark.parametrize("overlay,name", [(False, "contrasts_no_overlay.json"), (True, "contrasts.json")])
def test_recomputed_contrasts_equal_the_paper(overlay, name):
    """Minutes of CPU: every contrast on the paper's scored rows (and the reruns)."""
    outputs = paper_outputs()
    if outputs is None:
        pytest.skip("the outputs folder does not hold the paper's 235 scored jobs")
    from streamreplace.analysis.pipeline import compute
    from streamreplace.analysis.records import Catalog
    doc = compute(outputs, load_profile(), Catalog(), overlay=overlay)
    expected = json.loads((EXPECTED / name).read_text(encoding="utf-8"))
    assert len(doc["rows"]) == len(expected["rows"])
    for mine, theirs in zip(doc["rows"], expected["rows"]):
        assert same({k: mine.get(k) for k in theirs}, theirs), (theirs["dataset"], theirs["metric"], theirs["child"],
                                                                theirs["gap"])
    assert same({k: doc["summary"][k] for k in expected["summary"]}, expected["summary"])


@pytest.mark.requires_data
def test_recomputed_metrics_equal_the_paper():
    outputs = paper_outputs()
    if outputs is None:
        pytest.skip("the outputs folder does not hold the paper's 235 scored jobs")
    from streamreplace.analysis.metrics import collect, to_json
    from streamreplace.analysis.records import Catalog
    mine = {(r["dataset"], r["model"], r["mode"]): r for r in to_json(collect(outputs, Catalog()))}
    for r in json.loads((EXPECTED / "metrics.json").read_text(encoding="utf-8"))["jobs"]:
        m = mine[(r["dataset"], r["model"], r["mode"])]
        assert same({k: m[k] for k in r}, r), (r["dataset"], r["model"], r["mode"])
