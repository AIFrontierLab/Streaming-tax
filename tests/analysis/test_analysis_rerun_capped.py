"""Generation-cap reruns: selection rule, folders written, rerun plan and the sensitivity report
(run/rerun_capped.py)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from analysis_fakes import N, catalog, write_job
from streamreplace.run import rerun_capped as R


def test_capped_rule():
    rows = {"a": {"finish": "length"}, "b": {"finish": "eos"}, "c": {"finish": "length", "error": "x"},
            "d": {"finish": None, "raw": "..."}, "e": {"finish": "unknown"}}
    assert R.capped(rows, 64, {"d": 56, "e": 55}) == ["a", "d"]           # >= cap - 8 tokens without a finish
    with pytest.raises(ValueError):
        R.capped(rows, 64, {"d": 56})                                     # e has no finish and no count
    assert R.unfinished({"finish": "unknown"}) and not R.unfinished({"finish": "length"})
    assert not R.unfinished({"finish": None, "error": "boom"})


class FakeConfig:
    """The two configuration lookups selection needs: pair members and resolved settings."""

    def pair(self, pair):
        return SimpleNamespace(members=("par", "kid"))

    def settings(self, dataset, model, mode):
        return {"generation": {"max_new_tokens": 64}}

    def model(self, model):
        raise AssertionError("no tokenizer run expected")


def test_select_writes_subset_and_capped(tmp_path):
    formal, root = tmp_path / "formal", tmp_path / "rerun"
    write_job(formal, "img", "par.offline", lambda k: True, length={4})
    write_job(formal, "img", "kid.offline", lambda k: True, length={1, 4, 9})
    summary = R.select(FakeConfig(), None, formal, root, ["p"], ["img"], rerun_cap=8192)
    assert summary == {"p/img": {"rerun ids": 3, "par": 1, "kid": 3}}
    assert json.loads((root / "p" / "subset.json").read_text()) == {"img": ["img-001", "img-004", "img-009"]}
    capped = json.loads((root / "capped.json").read_text())
    assert capped["rerun_cap"] == 8192 and capped["pairs"]["p"]["img"]["kid"] == {
        "cap": 64, "rows_without_finish": 0, "ids": ["img-001", "img-004", "img-009"]}
    # a second selection must give the same ids (a folder's ids never change under a running job)
    write_job(formal, "img", "kid.offline", lambda k: True, length={2})
    with pytest.raises(ValueError):
        R.select(FakeConfig(), None, formal, root, ["p"], ["img"], rerun_cap=8192)


def test_select_counts_rows_without_a_finish_from_given_token_counts(tmp_path):
    formal = tmp_path / "formal"
    write_job(formal, "img", "par.offline", lambda k: True)
    write_job(formal, "img", "kid.offline", lambda k: True)
    path = formal / "img" / "kid.offline.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for r in rows:
        r.pop("finish")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    counts = {"kid": {r["id"]: (60 if r["id"] == "img-007" else 10) for r in rows}}
    R.select(FakeConfig(), None, formal, tmp_path / "rerun", ["p"], ["img"], 8192, token_counts=counts)
    capped = json.loads((tmp_path / "rerun" / "capped.json").read_text())["pairs"]["p"]["img"]["kid"]
    assert capped == {"cap": 64, "rows_without_finish": N, "ids": ["img-007"]}


def test_report_formal_vs_merged(tmp_path):
    formal, root = tmp_path / "formal", tmp_path / "rerun"
    write_job(formal, "img", "par.offline", lambda k: k % 2 == 0)
    write_job(formal, "img", "kid.offline", lambda k: k % 2 == 0 and k != 4, length={4})
    for job in ("par.offline", "kid.offline"):
        write_job(root / "p", "img", job, lambda k: k % 2 == 0)
    (root / "capped.json").write_text(json.dumps({"cap_margin": 8, "rerun_cap": 8192, "pairs": {"p": {"img": {
        "par": {"cap": 64, "ids": []}, "kid": {"cap": 64, "ids": ["img-004"]}}}}}))
    target = R.report(FakeConfig(), formal, root, catalog())
    data = json.loads((root / "trunc.json").read_text())["p/img"]
    pair = data["pairs"]["kid"]
    assert pair["rerun_ids"] == 1 and pair["formal"]["delta"] == pytest.approx(-100 / N)
    assert pair["merged"]["delta"] == 0.0 and pair["merged"]["mcnemar_p"] is None
    kid = next(m for m in data["members"] if m["model"] == "kid")
    assert (kid["capped"], kid["to_right"], kid["to_wrong"], kid["still_capped"]) == (1, 1, 0, 0)
    assert "| kid | 1 |" in target.read_text()


def test_report_lists_pending_reruns(tmp_path):
    formal, root = tmp_path / "formal", tmp_path / "rerun"
    write_job(formal, "img", "par.offline", lambda k: True)
    write_job(formal, "img", "kid.offline", lambda k: True)
    (root / "p").mkdir(parents=True)
    (root / "capped.json").write_text(json.dumps({"pairs": {"p": {"img": {"par": {"cap": 64, "ids": ["img-001"]},
                                                                           "kid": {"cap": 64, "ids": []}}}}}))
    text = R.report(FakeConfig(), formal, root, catalog()).read_text()
    assert "pending: par 0/1, kid 0/1" in text


def test_rerun_ids_of_a_contrast_and_of_a_shared_group():
    from streamreplace.analysis.arms import TruncationRerun, rerun_ids
    per_model = {"par": {"ids": ["1", "2"]}, "kid": {"ids": ["2", "3"]}, "kid2": {"ids": ["9"]}}
    assert rerun_ids(per_model, "par", "kid") == {"1", "2", "3"}             # the ids either of the two capped
    assert rerun_ids(per_model, "par", "kid2") == {"1", "2", "9"}
    assert rerun_ids(per_model, "par", "kid", shared=True) == {"1", "2", "3", "9"}      # one merge for the group
    assert rerun_ids(per_model, "par", "kid2", shared=True) == rerun_ids(per_model, "par", "kid", shared=True)
    rerun = TruncationRerun({"p": {"img": per_model, "txt": per_model}}, root=None, pairs=["p"], datasets=["img", "txt"],
                            shared={"p": ["img"]})
    assert rerun.ids("p", "img", "par", "kid") == {"1", "2", "3", "9"}
    assert rerun.ids("p", "txt", "par", "kid") == {"1", "2", "3"}            # not shared on this dataset
    assert rerun.ids("q", "img", "par", "kid") == frozenset() == rerun.ids("p", "other", "par", "kid")
