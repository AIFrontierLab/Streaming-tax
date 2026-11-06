"""Item selection, condition folders and frozen prompt edits on the synthetic benchmark (make.py, edits.py,
paraphrase.py), and the runtime's prompt-override hook applying them (data/prompts.py)."""
from __future__ import annotations

import hashlib
import json

import pytest
from robustness_demo import POST_PROMPT
from robustness_helpers import DEMO, REWRITES, prepared

from streamreplace.core.io import read_jsonl, write_jsonl
from streamreplace.data.prompts import OVERRIDES_FILE, read_overrides, render_prompts
from streamreplace.generalization.robustness import edits, make, perturb

INSTRUCTION = POST_PROMPT.strip()


def test_stratified_largest_remainder_and_seeded_order():
    class S:
        def __init__(self, i, g):
            self.id, self.metadata = i, {"g": g}
    samples = [S(f"a{i}", "a") for i in range(5)] + [S(f"b{i}", "b") for i in range(3)] + [S("c0", "c")]
    chosen = make.stratified(samples, 4, ("g",), 0)
    # quotas 2.22 / 1.33 / 0.44: floors 2 / 1 / 0, the remaining unit goes to the largest remainder (c)
    assert len(chosen) == 4 and sum(i.startswith("a") for i in chosen) == 2 and "c0" in chosen
    order = sorted((s for s in samples if s.metadata["g"] == "a"), key=lambda s: make.key(f"0:{s.id}"))
    assert [i for i in chosen if i.startswith("a")] == sorted(s.id for s in order[:2])
    assert make.stratified(samples, 20, ("g",), 0) == sorted(s.id for s in samples)


def test_subsets_are_shared_nested_and_written_once(experiment):
    exp = experiment
    status = make.make_subset(exp)
    assert status[DEMO] == 16 and set(status["folders"].values()) == {"written"}
    shared = exp.subset("clean")
    assert all(exp.subset(c) == shared for c in ("typo", "para", "stress", "irrelevant"))
    assert len(shared[DEMO]) == 16 and shared[DEMO] == sorted(shared[DEMO])
    units = exp.units(DEMO, "clean")
    assert len(units) == 8 and all(len(v) == 2 for v in units.values())          # every rotation of a question
    categories = [v[0].metadata["l2"] for v in units.values()]
    assert {c: categories.count(c) for c in set(categories)} == {"counting": 3, "objects": 3, "shapes": 2}
    nested = exp.subset("ins_typo")[DEMO]
    assert len(nested) == 8 and set(nested) < set(shared[DEMO])
    assert exp.subset("ins_para") == exp.subset("ins_typo") and exp.subset("ins8_typo") == shared
    info = json.loads((exp.root / make.INFO_FILE).read_text())
    assert info["seed"] == 0 and info["datasets"][DEMO]["units"] == 8
    assert info["datasets"][DEMO]["ids_sha256"] == hashlib.sha256("\n".join(shared[DEMO]).encode()).hexdigest()
    assert set(make.make_subset(exp)["folders"].values()) == {"unchanged"}
    (exp.folder("typo") / make.SUBSET_FILE).write_text(json.dumps({DEMO: shared[DEMO][:2]}))
    with pytest.raises(ValueError, match="different content"):
        make.make_subset(exp)


def test_one_question_per_video_keeps_the_lowest_hash(experiment, monkeypatch):
    """Video-MME style: with one_per, a single unit per value of that key enters the selection."""
    exp = experiment
    spec = make.DATASETS[DEMO]
    monkeypatch.setitem(make.DATASETS, DEMO, make.UnitSpec(spec.stem, unit=lambda s: s.id, cluster=spec.cluster,
                                                           strata=(), one_per="l2"))
    ids = make.select(exp, DEMO, 10)
    assert len(ids) == 3                                   # one sample per category
    for category in ("counting", "objects", "shapes"):
        members = [s.id for s in exp.samples(DEMO) if s.metadata["l2"] == category]
        assert min(members, key=lambda i: make.key(f"0:{i}")) in ids


def test_edits_rows_skips_and_seed_as(experiment):
    exp = experiment
    rejected = prepared(exp)
    rows = {c: read_jsonl(exp.folder(c) / OVERRIDES_FILE) for c in exp.conditions if c != "clean"}
    assert not (exp.folder("clean") / OVERRIDES_FILE).exists()
    assert {c: len(r) for c, r in rows.items()} == {"typo": 16, "para": 16, "stress": 16, "irrelevant": 16,
                                                    "ins_typo": 8, "ins_para": 8, "ins_irrelevant": 8, "ins8_typo": 16}
    assert list(rows["typo"][0]) == ["dataset", "id", "unit", "condition", "edits", "skip"]
    by_unit: dict = {}
    for r in rows["typo"]:
        by_unit.setdefault(r["unit"], []).append(r["edits"])
    assert all(len(e) == 2 and e[0] == e[1] for e in by_unit.values())         # rotations share one edit
    skipped = [r for r in rows["para"] if r["skip"]]
    assert len(skipped) == 2 and skipped[0]["skip"] == "judged not equivalent" and skipped[0]["edits"] == []
    assert all(r["edits"][0][1] == "Tell me please, " + r["edits"][0][0] for r in rows["para"] if not r["skip"])
    assert all(r["edits"][0][0] != rejected for r in rows["para"] if r["edits"])
    for r in rows["stress"]:
        assert r["edits"][0][1] == perturb.append(r["edits"][0][0], perturb.STRESS_TEST)
    for r in rows["irrelevant"]:
        old, new = r["edits"][0]
        assert new.startswith(old + " ") and new[len(old) + 1:] in perturb.irrelevant_pool()
    for c in ("ins_typo", "ins_para", "ins_irrelevant"):
        assert all(r["edits"][0][0] == INSTRUCTION for r in rows[c])
    assert all(r["edits"][0][1] in REWRITES for r in rows["ins_para"])
    earlier = {r["id"]: r["edits"] for r in rows["ins_typo"]}
    later = {r["id"]: r["edits"] for r in rows["ins8_typo"]}
    assert all(later[i] == e for i, e in earlier.items()) and len(later) == 16       # seed_as: same edits
    assert set(edits.make_edits(exp)[c]["file"] for c in rows) == {"unchanged"}
    rows["typo"][0]["edits"][0][1] += "!"
    write_jsonl(exp.folder("typo") / OVERRIDES_FILE, rows["typo"])
    with pytest.raises(ValueError, match="different content"):
        edits.make_edits(exp)


def test_overrides_reach_the_rendered_prompts_and_check_passes(experiment):
    exp = experiment
    prepared(exp)
    report = edits.check(exp)
    assert report["clean"][DEMO]["changed"] == 0 and report["typo"][DEMO]["changed"] == 16
    assert report["para"][DEMO]["changed"] == 14 and report["ins_irrelevant"][DEMO]["changed"] == 8
    samples = exp.samples(DEMO, "stress")
    clean = render_prompts(DEMO, samples)
    stressed = render_prompts(DEMO, samples, read_overrides(exp.folder("stress"))[DEMO])
    for s in samples:
        stem = s.metadata["raw_question"]
        assert stressed[s.id] == clean[s.id].replace(stem, perturb.append(stem, perturb.STRESS_TEST))
    rows = read_jsonl(exp.folder("typo") / OVERRIDES_FILE)
    rows[0]["edits"] = [[rows[0]["edits"][0][0] + " There are several options:", "Something else entirely"]]
    write_jsonl(exp.folder("typo") / OVERRIDES_FILE, rows)
    with pytest.raises(ValueError, match="reaches outside the stem"):
        edits.check(exp)


def test_check_refuses_folders_that_are_not_nested(experiment):
    exp = experiment
    prepared(exp, reject=False)
    other = [s.id for s in exp.samples(DEMO) if s.id not in exp.subset("clean")[DEMO]]
    (exp.folder("ins_para") / make.SUBSET_FILE).write_text(json.dumps({DEMO: other[:2]}))
    with pytest.raises(ValueError, match="not nested"):
        edits.check(exp)


def test_missing_paraphrase_is_an_error(experiment):
    make.make_subset(experiment)
    with pytest.raises(ValueError, match="no paraphrase"):
        edits.make_edits(experiment)
    assert edits.make_edits(experiment, only=["typo"])["typo"]["rows"] == 16


def test_audit_digests_and_frozen_comparisons(experiment, tmp_path):
    exp = experiment
    prepared(exp)
    written = edits.audit(exp, n=3)
    assert len(written) == 8 and "| unit | clean stem | perturbed stem |" in (exp.root / "audit/typo.md").read_text()
    assert "method: back_translation" in (exp.root / "audit/para.md").read_text()
    frozen = edits.digests(exp)
    assert frozen["para"]["rows"] == 16 and sum(len(u) for u in frozen["para"]["skipped"].values()) == 1
    assert set(edits.compare_digests(exp, frozen).values()) == {"identical"}
    assert edits.compare_digests(exp, {"typo": {**frozen["typo"], "sha256": "0" * 64}}) == {"typo": "differs"}
    items = make.item_lists(exp)
    assert make.compare_items(exp, items) == []
    assert make.compare_items(exp, {**items, "seed": 1}) and make.compare_items(exp, {**items, "subset": {DEMO: []}})


def test_specification_validation(demo, spec):
    make.validate_spec(spec)
    for change, message in [({"typo": {"edit": "typo"}}, "needs a `family`"),
                            ({"typo": {"edit": "swap", "family": "typo"}}, "edit must be one of"),
                            ({"ins_typo": {"edit": "typo", "target": "instruction", "twin": "nope"}}, "stem `twin`"),
                            ({"ins_para": {"edit": "choice", "target": "instruction", "twin": "para"}}, "needs `texts`"),
                            ({"typo": {"edit": "typo", "family": "typo", "colour": 1}}, "unknown keys")]:
        bad = {**spec, "conditions": {**spec["conditions"], **change}}
        with pytest.raises(ValueError, match=message):
            make.validate_spec(bad)
    with pytest.raises(ValueError, match="no unit definition"):
        make.validate_spec({**spec, "datasets": {"tomato": 10}})
