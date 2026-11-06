"""Counted records, overlays and the contrast rows (analysis/records.py, arms.py, contrasts.py)."""
from __future__ import annotations

from dataclasses import replace
import json

import pytest

from analysis_fakes import N, PATTERNS, catalog, profile, samples, write_job
from streamreplace.analysis import arms as A
from streamreplace.analysis.contrasts import Contrasts, annotate, summary
from streamreplace.analysis.records import Catalog, load_job, make_record, metric_rows


def test_last_row_wins_counted_rows_and_flags(tmp_path):
    root = tmp_path / "out"
    retry = {"id": "img-000", "prompt": "q0", "raw": "A", "tokens": 3, "finish": "eos", "latency": 1.0, "info": {},
             "error": None}
    write_job(root, "img", "par.offline", PATTERNS[("img", "par.offline")], errors={1}, length={2},
              extra_rows=[retry, {**retry, "id": "not-a-sample"}])
    job = load_job(root, "img", "par.offline", catalog())
    assert job.expected == N and len(job.rows) == N               # the unknown id is ignored, the retry wins
    assert job.rows["img-001"].error and not job.rows["img-001"].correct      # an error is wrong (judge fallback)
    assert job.rows["img-002"].truncated and job.rows["img-002"].answered
    assert job.complete and job.model == "par" and job.mode == "offline" and job.variant is None


def test_errors_count_safe_under_the_safety_judges(tmp_path):
    root = tmp_path / "out"
    write_job(root, "saf", "kid.stream", lambda k: False, errors={0, 1}, silent={2})
    job = load_job(root, "saf", "kid.stream", catalog())
    assert job.rows["saf-000"].correct and job.rows["saf-001"].correct     # RJScore-style policy: errors are safe
    assert job.rows["saf-002"].silent and not job.rows["saf-002"].correct


def test_plugin_projection_is_read_from_info_plugin_guard():
    row = {"id": "x", "pred": "A", "parse": "letter", "correct": True, "judge_pred": None, "judge_correct": None,
           "info": {"plugin": {"guard": {"proj": 1.234, "fired": False}}}}
    assert make_record(row, "fallback").proj == 1.234
    calibration = {**row, "info": {}, "guard": {"proj": -0.5}}            # harmless-calibration rows: top level
    assert make_record(calibration, "fallback").proj == -0.5
    assert make_record({**row, "info": {}}, "fallback").proj is None


def test_subset_folder_restricts_ids_and_expected_count(tmp_path):
    root = tmp_path / "rerun"
    write_job(root, "img", "par.offline", lambda k: True)
    (root / "subset.json").write_text(json.dumps({"img": ["img-003", "img-004", "unknown"]}))
    job = load_job(root, "img", "par.offline", catalog())
    assert sorted(job.rows) == ["img-003", "img-004"] and job.expected == 2 and job.complete


def test_stream_jobs_expect_only_samples_with_a_video(tmp_path):
    root = tmp_path / "out"
    write_job(root, "vid", "kid.stream", lambda k: True)
    half = [s if i % 2 else replace(s, video=None) for i, s in enumerate(samples("vid"))]
    cat = Catalog(samples=lambda d: half, benchmark=catalog().bench)
    assert load_job(root, "vid", "kid.stream", cat).expected == N // 2
    assert load_job(root, "vid", "kid.stream", catalog()).expected == N


def test_overlays_replace_force_unanswered_restrict(tmp_path):
    root = tmp_path / "out"
    write_job(root, "img", "par.offline", lambda k: False)
    write_job(tmp_path / "rerun", "img", "par.offline", lambda k: True)
    cat = catalog()
    formal, rerun = load_job(root, "img", "par.offline", cat), load_job(tmp_path / "rerun", "img", "par.offline", cat)
    merged = formal.with_overlays(A.Replace(frozenset({"img-000", "img-001"}), rerun))
    assert [merged.rows[i].correct for i in ("img-000", "img-001", "img-002")] == [True, True, False]
    assert merged.uid != formal.uid
    forced = merged.with_overlays(A.ForceUnanswered(frozenset({"img-000"})))
    assert not forced.rows["img-000"].correct and not forced.rows["img-000"].answered
    assert forced.rows["img-000"].rule_pred == "A"                         # the predictions stay
    assert sorted(forced.with_overlays(A.Restrict(frozenset({"img-005"}))).rows) == ["img-005"]
    with pytest.raises(ValueError):
        formal.with_overlays(A.Replace(frozenset({"img-000", "missing"}), rerun, add=True))


def test_metric_rows_feed_the_benchmark_metrics(synthetic):
    root, cat, _ = synthetic
    job = load_job(root, "vid", "kid.stream", cat)
    rows = metric_rows(job, cat)
    assert len(rows) == N and rows[1]["silent"] and not rows[1]["answered"]
    assert cat.metrics("vid", rows)["acc"] == pytest.approx(100 * sum(PATTERNS[("vid", "kid.stream")](k) and k not in
                                                                        (1, 7) for k in range(N)) / N)


# ---------- contrasts ----------

def contrasts(root, cat, prof, rerun=None):
    return Contrasts(prof, A.Arms(root, cat, prof, rerun), cat).compute()


def test_contrast_rows_order_orientation_and_eq1(synthetic):
    root, cat, prof = synthetic
    rows = contrasts(root, cat, prof)
    keys = [(r["dataset"], r["gap"]) for r in rows]
    assert keys == [("img", "C-P"), ("vid", "C-P"), ("vid", "S-C"), ("vid", "S-P"), ("saf", "C-P"), ("saf", "S-C"),
                    ("saf", "S-P")]
    img = rows[0]
    assert (img["a"], img["b"]) == (80.0, 75.0) and img["delta"] == pytest.approx(-5.0)
    assert img["test"] == "mcnemar" and img["n"] == N
    saf = rows[4]                                 # ASR: a / b reported as ASR, delta = ASR_a - ASR_b (+ = safer)
    asr = lambda job: 100 - 100 * sum(PATTERNS[("saf", job)](k) for k in range(N)) / N  # noqa: E731
    assert saf["a"] == pytest.approx(asr("par.offline")) and saf["b"] == pytest.approx(asr("kid.offline"))
    assert saf["delta"] == pytest.approx(saf["a"] - saf["b"])
    by = {(r["dataset"], r["gap"]): r for r in rows}
    for d in ("vid", "saf"):
        assert by[(d, "S-P")]["delta"] == pytest.approx(by[(d, "C-P")]["delta"] + by[(d, "S-C")]["delta"], abs=1e-9)
    assert summary(rows)["eq1_max_residual"] < 1e-9
    assert by[("vid", "C-P")]["test"] == "signflip" and by[("vid", "C-P")]["clusters"] == N // 4
    assert by[("vid", "S-C")]["coverage_b"] == pytest.approx(100 * (N - 2) / N)


def test_families_broad_families_and_verdicts(synthetic):
    root, cat, prof = synthetic
    rows = {(r["dataset"], r["gap"]): r for r in contrasts(root, cat, prof)}
    assert rows[("vid", "S-C")]["holm_family"] == "video/S-C" and rows[("vid", "S-C")]["holm_m"] == 1
    assert rows[("saf", "C-P")]["holm_family"] == "saf" and rows[("saf", "C-P")]["holm_m"] == 3
    assert rows[("saf", "C-P")]["holm_broad_family"] == "safety/parent" and rows[("saf", "C-P")]["holm_broad_m"] == 2
    assert rows[("saf", "S-C")]["holm_broad_family"] == "safety/S-C"
    assert "verdict" in rows[("vid", "C-P")] and "verdict" not in rows[("vid", "S-C")]


def row(child="aura_8b", gap="C-P", family="video", p=0.01, delta=-5.0, ci95=(-8.0, -2.0), ci90=(-7.5, -2.5),
        cov=(100.0, 100.0), dataset="videomme", metric="acc"):
    return {"dataset": dataset, "metric": metric, "family": family, "headline": True, "child": child, "gap": gap,
            "p": p, "delta": delta, "delta_pp": delta, "ci95_pp": list(ci95), "ci90_pp": list(ci90),
            "coverage_a": cov[0], "coverage_b": cov[1]}


def test_annotate_holm_families_readings_and_verdicts():
    """Ported unchanged from the paper's analysis tests."""
    prof = profile(holm={"split_by_gap": ["video", "vrb"], "broad": {}})
    rows = [row(child="aura_8b", p=0.01), row(child="mmduet_7b", p=0.04),            # one family: video/C-P
            row(child="aura_8b", gap="S-C", p=0.04),                                 # another family
            row(child="moss_vl_rt", p=0.5, delta=0.5, ci95=(-2.5, 3.5), ci90=(-2.0, 2.9), cov=(100, 99.0)),
            row(child="livestar_8b", p=0.3, delta=-1.0, ci95=(-4.0, 2.0), ci90=(-3.5, 1.5), cov=(100, 90.0)),
            row(child="thinkstream_3b", p=0.001, delta=-9.0, ci95=(-12.0, -6.0), ci90=(-11.0, -6.5))]
    out = annotate(rows, prof)
    assert out[0]["holm_family"] == "video/C-P" and out[0]["holm_m"] == 5
    assert out[0]["p_holm"] == pytest.approx(0.04) and out[2]["p_holm"] == pytest.approx(0.04)
    assert out[2]["holm_family"] == "video/S-C" and out[2]["holm_m"] == 1
    assert out[0]["reading"] == "decline" and out[3]["reading"] == "equivalent ±3"
    assert out[3]["verdict"] == "replaceable"
    assert out[4]["verdict"] == "non-inferior, coverage differs"              # coverage 10 pp lower than the parent
    assert out[5]["verdict"] == "inferior" and out[0]["verdict"] == "inconclusive"
    assert "verdict" not in out[2]                                            # S-C rows carry no replacement verdict
    assert out[0]["tost3"] is False and out[3]["tost3"] is True and out[3]["ni5"] is True


def test_truncation_rerun_merges_both_offline_arms(synthetic, tmp_path):
    root, cat, prof = synthetic
    # the child capped img-001/002/003/005/006 (finish "length"); the parent capped none. Reruns: all right.
    rerun_root = root / "rerun_capped" / "p"
    for job in ("par.offline", "kid.offline"):
        write_job(rerun_root, "img", job, lambda k: True)
    capped = {"pairs": {"p": {"img": {"par": {"cap": 64, "ids": []},
                                      "kid": {"cap": 64, "ids": ["img-001", "img-002", "img-003", "img-005",
                                                                 "img-006"]}}}}}
    (tmp_path / "capped.json").write_text(json.dumps(capped))
    rerun = A.TruncationRerun.from_profile(prof, root)
    assert rerun.ids("p", "img", "par", "kid") == frozenset({"img-001", "img-002", "img-003", "img-005", "img-006"})
    arms = A.Arms(root, cat, prof, rerun)
    rows = Contrasts(prof, arms, cat).compute()
    img = rows[0]
    right = lambda job, k: PATTERNS[("img", job)](k) or k in (1, 2, 3, 5, 6)  # noqa: E731
    assert img["a"] == pytest.approx(100 * sum(right("par.offline", k) for k in range(N)) / N)
    assert img["b"] == pytest.approx(100 * sum(right("kid.offline", k) for k in range(N)) / N)
    assert [r["delta"] for r in rows[1:]] == [r["delta"] for r in contrasts(root, cat, prof)[1:]]   # img only
    assert len(arms.merged_jobs()) == 2
    # the rerun folder's own capped.json takes precedence over the profile's frozen list
    (rerun_root.parent / "capped.json").write_text(json.dumps({"pairs": {"p": {"img": {}}}}))
    assert A.TruncationRerun.from_profile(prof, root).ids("p", "img", "par", "kid") == frozenset()
