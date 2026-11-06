"""The two bootstrap engines and their tests (analysis/stats.py, analysis/stats_index.py)."""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import pytest

from streamreplace.analysis import stats, stats_index


def test_mcnemar_exact_matches_binomial():
    for b, c in [(0, 0), (1, 0), (3, 7), (21, 1), (40, 40), (96, 3)]:
        n = b + c
        if n == 0:
            assert stats.mcnemar_exact(b, c) == 1.0
            assert stats_index.mcnemar_p(b, c) is None and stats_index.mcnemar(b, c) == 1.0
            continue
        tail = sum(Fraction(math.comb(n, i), 2 ** n) for i in range(min(b, c) + 1))
        expected = float(min(Fraction(1), 2 * tail))
        assert stats.mcnemar_exact(b, c) == pytest.approx(expected, rel=1e-12)
        assert stats_index.mcnemar_p(b, c) == pytest.approx(expected, rel=1e-12)


def test_holm_skips_none_and_keeps_ties_stable():
    adjusted = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
    assert adjusted == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    assert "d" not in adjusted
    assert stats_index.holm({"x": 0.2, "y": 0.01}) == pytest.approx({"y": 0.02, "x": 0.2})
    assert stats.holm({"x": 0.5, "y": 0.5}) == {"x": 1.0, "y": 1.0}


def test_paired_contrast_binary_uses_mcnemar():
    # 100 items: arm a right on 60, arm b right on 50; 15 a-only, 5 b-only
    a = [1] * 45 + [1] * 15 + [0] * 5 + [0] * 35
    b = [1] * 45 + [0] * 15 + [1] * 5 + [0] * 35
    u = stats.units({("all", i): (a[i], b[i], 1) for i in range(100)})
    r = stats.paired_contrast(u, reps=2000)
    assert (r["a"], r["b"], r["delta"]) == pytest.approx((60.0, 50.0, -10.0))
    assert r["test"] == "mcnemar" and (r["a_only"], r["b_only"]) == (15, 5)
    assert r["p"] == pytest.approx(stats.mcnemar_exact(15, 5))
    assert r["ci95"][0] < -10 < r["ci95"][1] and r["ci90"][0] > r["ci95"][0]


def test_paired_contrast_ratio_terms_match_mme_definition():
    # two categories; per image cluster: (right questions, both right) for each arm; MME = sum(acc + acc+)
    clusters = {}
    for cat, images in [("x", 10), ("y", 20)]:
        for k in range(images):
            right_a, right_b = (2 if k % 2 else 1), (2 if k % 3 else 1)
            num_a = [right_a if cat == "x" else 0, int(right_a == 2) if cat == "x" else 0,
                     right_a if cat == "y" else 0, int(right_a == 2) if cat == "y" else 0]
            num_b = [right_b if cat == "x" else 0, int(right_b == 2) if cat == "x" else 0,
                     right_b if cat == "y" else 0, int(right_b == 2) if cat == "y" else 0]
            den = [2 if cat == "x" else 0, 1 if cat == "x" else 0, 2 if cat == "y" else 0, 1 if cat == "y" else 0]
            clusters[(cat, f"{cat}{k}")] = (num_a, num_b, den)
    u = stats.units(clusters, weight=[1, 1, 1, 1])
    r = stats.paired_contrast(u, reps=1000, scale=4)

    def mme(right):
        total = 0.0
        for cat, images in [("x", 10), ("y", 20)]:
            rights = [right(k) for k in range(images)]
            total += 100 * sum(rights) / (2 * images) + 100 * sum(x == 2 for x in rights) / images
        return total
    assert r["a"] == pytest.approx(mme(lambda k: 2 if k % 2 else 1))
    assert r["b"] == pytest.approx(mme(lambda k: 2 if k % 3 else 1))
    assert r["test"] == "signflip" and 0 < r["p"] <= 1
    assert r["delta_pp"] == pytest.approx(r["delta"] / 4)


def test_paired_contrast_drops_empty_clusters_and_handles_no_units():
    u = stats.units({("all", "x"): (1, 0, 1), ("all", "y"): (0, 0, 0), ("all", "z"): (1, 1, 1)})
    r = stats.paired_contrast(u, reps=200)
    assert r["clusters"] == 2 and r["items"] == 2.0
    empty = stats.paired_contrast(stats.units({("all", "y"): (0, 0, 0)}), reps=200)
    assert empty["ci95"] is None and empty["p"] is None


def test_paired_contrast_is_reproducible_and_seeded_per_call():
    rng = np.random.default_rng(7)
    a, b = rng.integers(0, 2, 300), rng.integers(0, 2, 300)
    table = {("all", f"c{i // 3}"): (0, 0, 0) for i in range(300)}
    for i in range(300):
        key = ("all", f"c{i // 3}")
        na, nb, n = table[key]
        table[key] = (na + int(a[i]), nb + int(b[i]), n + 1)
    first = stats.paired_contrast(stats.units(table), reps=3000)
    second = stats.paired_contrast(stats.units(table), reps=3000)
    assert first == second                         # a fresh default_rng(seed) for every contrast
    assert stats.paired_contrast(stats.units(table), reps=3000, seed=1)["ci95"] != first["ci95"]


def test_signflip_detects_a_shift_and_not_noise():
    assert stats.cluster_signflip(np.ones(30), reps=2000) < 0.001
    assert stats.cluster_signflip(np.array([1.0, -1.0] * 20), reps=2000) == 1.0
    assert stats.cluster_signflip(np.zeros(5)) == 1.0


def test_tost_and_noninferiority():
    assert stats.equivalent([-2.9, 2.9], 3) and not stats.equivalent([-3.1, 0.0], 3)
    assert stats.noninferior([-4.9, 1.0], 5) and not stats.noninferior([-5.1, 1.0], 5)
    assert stats.equivalent(None, 3) is None and stats.noninferior(None, 3) is None


def test_binom_tail_exact():
    n, p = 40, 0.0625
    exact = sum(Fraction(math.comb(n, i)) * Fraction(1, 16) ** i * Fraction(15, 16) ** (n - i) for i in range(5))
    assert stats.binom_tail(4, n, p, upper=False) == pytest.approx(float(exact), rel=1e-9)
    assert stats.binom_tail(5, n, p) == pytest.approx(float(1 - exact), rel=1e-9)


# ---------- the integer-index engine ----------

def test_bootstrap_ci_tail_is_exactly_2_5_and_5():
    """round(100 * (1 - level) / 2, 9): the 95% interval uses exactly the 2.5 / 97.5 percentiles."""
    rng = np.random.default_rng(3)
    values = rng.integers(-1, 2, 400).astype(float)
    clusters = [f"c{i // 3}" for i in range(400)]
    strata = ["a" if i < 250 else "b" for i in range(400)]
    boot = stats_index.bootstrap_means(values, clusters, strata, reps=500)
    assert stats_index.bootstrap_ci(values, clusters, strata, reps=500) == tuple(
        float(x) for x in np.percentile(boot, [2.5, 97.5]))
    assert stats_index.bootstrap_ci(values, clusters, strata, reps=500, level=0.90) == tuple(
        float(x) for x in np.percentile(boot, [5, 95]))


def test_draws_equal_bootstrap_means_and_summary_bounds():
    rng = np.random.default_rng(5)
    values = rng.integers(0, 2, 120).astype(float) - rng.integers(0, 2, 120)
    clusters = [i // 4 for i in range(120)]
    strata = [0] * 120
    assert np.array_equal(stats_index.means(values, clusters, strata, reps=300),
                          stats_index.bootstrap_means(values, clusters, strata, reps=300))
    s = stats_index.summary(values, clusters, strata, alphas=(0.05,), margin=3, reps=300)
    assert s["n"] == 120 and s["low"] <= s["high"]
    assert s["lower_0.05"] >= s["low"] and s["low90"] == pytest.approx(s["lower_0.05"])
    assert 0 < s["p"] <= 1 and 0 < s["tost_p"] <= 1


def test_index_engine_signflip_keeps_zero_clusters():
    """The integer-index engine's sign-flip keeps clusters that contribute 0 (one block of draws), unlike the main
    engine."""
    values = [1.0] * 10 + [0.0] * 10
    clusters = list(range(20))
    p = stats_index.signflip_p(values, clusters, reps=1000)
    assert 0 < p < 0.01


def test_mann_whitney_and_kappa():
    r = stats_index.mann_whitney_less([1, 2, 3], [4, 5, 6])
    assert r["u"] == 0 and r["auc"] == 0 and r["p_less"] < 0.05
    assert stats_index.kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    assert stats_index.kappa([], []) is None
