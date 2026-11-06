"""The integer-index cluster bootstrap and its companion tests (numpy + stdlib only).

This is the second statistics engine of the paper. It resamples whole clusters by index
(`rng.integers(0, K, (B, K))` per stratum) and takes the ratio of summed cluster sums to summed counts. It is NOT
the engine behind the P/C/S gap contrasts (see `stats.py`); it is kept, draw for draw, because these reported
numbers were produced with it:

- prompt robustness (Delta_Rob, Holm over 18 tests) and fine-tuning (Delta_Gain, Holm over 12 cells);
- the generation-cap (truncation rerun) report;
- the exact McNemar p of RESULTS.md's pair tables.

Functions:
- mcnemar_p: exact two-sided McNemar; returns None when there is no discordant pair (the `stats.py` version
  returns 1.0 there; both behaviours are kept where each was used).
- bootstrap_ci / bootstrap_means: stratified cluster bootstrap of a mean. The percentile tail is
  round(100 * (1 - level) / 2, 9): without the rounding 100 * (1 - 0.95) / 2 = 2.500000000000002, and the
  reported 95% intervals were computed with exactly 2.5 (and 5 / 95 for the 90% interval).
- signflip_p: cluster sign-flip test (one block of `reps` flips, zero clusters kept, tolerance 1e-12).
- holm: Holm step-down (no None handling).
- draws / means / summary / tost / mcnemar: the same bootstrap returning the draws, so several statistics,
  one-sided bounds or a TOST can be read off the same resamples.
- mann_whitney_less, kappa: small descriptive helpers used by the robustness analysis."""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

BOOT, PERM, SEED = 10_000, 10_000, 0


def mcnemar_p(b: int, c: int) -> float | None:
    """Exact two-sided McNemar test on the discordant counts (b = regressions, c = improvements)."""
    n = b + c
    if n == 0:
        return None
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def bootstrap_ci(values, clusters, strata, reps: int = BOOT, seed: int = SEED, level: float = 0.95) -> tuple[float, float]:
    """`level` (default 95%) interval of the mean of per-question `values`, resampling whole clusters within each
    stratum (level 0.90 gives the two one-sided 5% tests of a TOST equivalence check)."""
    tail = round(100 * (1 - level) / 2, 9)          # exactly 2.5 for 95%: the reported intervals stay bit-identical
    low, high = np.percentile(bootstrap_means(values, clusters, strata, reps, seed), [tail, 100 - tail])
    return float(low), float(high)


def bootstrap_means(values, clusters, strata, reps: int = BOOT, seed: int = SEED):
    """The `reps` resampled means behind bootstrap_ci (numpy array); their SD is the bootstrap standard error.
    Strata and clusters are visited in first-seen order."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    groups: dict = {}
    for i, (c, s) in enumerate(zip(clusters, strata)):
        groups.setdefault(s, {}).setdefault(c, []).append(i)
    sums, counts = [], []
    for stratum in groups.values():
        members = list(stratum.values())
        cluster_sum = np.array([values[m].sum() for m in members])
        cluster_n = np.array([len(m) for m in members])
        pick = rng.integers(0, len(members), size=(reps, len(members)))
        sums.append(cluster_sum[pick].sum(1))
        counts.append(cluster_n[pick].sum(1))
    return np.sum(sums, 0) / np.sum(counts, 0)


def signflip_p(values, clusters, reps: int = PERM, seed: int = SEED) -> float:
    """Two-sided p of mean(values) = 0 under exchangeability of parent and child within each cluster."""
    rng = np.random.default_rng(seed)
    totals: dict = {}
    for v, c in zip(values, clusters):
        totals[c] = totals.get(c, 0.0) + float(v)
    t = np.array(list(totals.values()))
    observed = abs(t.sum())
    flips = rng.choice([-1.0, 1.0], size=(reps, len(t)))
    return float((1 + (np.abs(flips @ t) >= observed - 1e-12).sum()) / (reps + 1))


def holm(pvalues: dict) -> dict:
    """Holm step-down adjusted p values (monotone, capped at 1). Every value must be a number."""
    order = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


# ---------- the same draws, returned for several statistics ----------

def draws(columns, clusters, strata, reps: int = BOOT, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """(reps, k) bootstrap column sums and (reps,) item counts: whole clusters resampled with replacement within each
    stratum (exactly the draws of bootstrap_means). columns: n values or an (n, k) array (several statistics on the
    same draws)."""
    values = np.asarray(columns, dtype=float)
    values = values[:, None] if values.ndim == 1 else values
    rng = np.random.default_rng(seed)
    groups: dict = {}
    for i, (c, s) in enumerate(zip(clusters, strata)):
        groups.setdefault(s, {}).setdefault(c, []).append(i)
    sums, counts = np.zeros((reps, values.shape[1])), np.zeros(reps)
    for stratum in groups.values():
        members = list(stratum.values())
        cluster_sum = np.array([values[m].sum(0) for m in members])
        cluster_n = np.array([len(m) for m in members])
        pick = rng.integers(0, len(members), size=(reps, len(members)))
        for j in range(values.shape[1]):
            sums[:, j] += cluster_sum[:, j][pick].sum(1)
        counts += cluster_n[pick].sum(1)
    return sums, counts


def means(values, clusters, strata, reps: int = BOOT, seed: int = SEED) -> np.ndarray:
    sums, counts = draws(values, clusters, strata, reps, seed)
    return sums[:, 0] / counts


def summary(values, clusters, strata, scale: float = 100.0, alphas=(0.05,), margin: float | None = None,
            reps: int = BOOT, seed: int = SEED) -> dict:
    """Mean, 95 % percentile interval, one-sided lower bounds (the alpha-quantile for each alpha: 'the true value is
    at least ...'), cluster sign-flip p, and with `margin` the TOST equivalence decision at that margin (90 %
    interval inside +-margin) and its bootstrap p. Values in `scale` units."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"n": 0}
    boot = scale * means(values, clusters, strata, reps, seed)
    out = {"n": int(len(values)), "value": float(scale * values.mean()),
           "low": float(np.percentile(boot, 2.5)), "high": float(np.percentile(boot, 97.5)),
           "p": signflip_p(list(map(float, values)), list(clusters))}
    for a in alphas:
        out[f"lower_{a:g}"] = float(np.percentile(boot, 100 * a))
    if margin is not None:
        out.update(tost(boot, margin))
    return out


def tost(boot: np.ndarray, margin: float) -> dict:
    """Two one-sided tests against +-margin on bootstrap draws (percentile inversion): p = the larger of the two
    one-sided bootstrap p values; equivalent at 5 % iff the 90 % interval lies inside (-margin, margin)."""
    below = (int((boot <= -margin).sum()) + 1) / (len(boot) + 1)
    above = (int((boot >= margin).sum()) + 1) / (len(boot) + 1)
    low, high = np.percentile(boot, [5, 95])
    return {"margin": margin, "tost_p": max(below, above), "equivalent": bool(-margin < low and high < margin),
            "low90": float(low), "high90": float(high)}


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar on the discordant counts (1.0 when there are none)."""
    value = mcnemar_p(b, c)
    return 1.0 if value is None else value


def mann_whitney_less(x, y) -> dict:
    """One-sided Mann-Whitney U test that x tends to be smaller than y (normal approximation with tie correction).
    Descriptive when the values are not independent (e.g. contrasts that share a model)."""
    x, y = list(map(float, x)), list(map(float, y))
    pooled = sorted((v, g) for g, vs in ((0, x), (1, y)) for v in vs)
    i = 0
    rank_x, ties = 0.0, 0.0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        t = j - i + 1
        ties += t ** 3 - t
        rank_x += r * sum(1 for k in range(i, j + 1) if pooled[k][1] == 0)
        i = j + 1
    n1, n2 = len(x), len(y)
    u = rank_x - n1 * (n1 + 1) / 2
    n = n1 + n2
    sd = math.sqrt(n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1)))) if n > 1 else 0.0
    z = (u - n1 * n2 / 2) / sd if sd else 0.0
    return {"u": u, "auc": u / (n1 * n2), "z": z, "p_less": NormalDist().cdf(z)}


def kappa(a: list, b: list) -> float | None:
    """Cohen's kappa of two raters' labels on the same items (None labels are a category)."""
    if not a:
        return None
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    cats = set(a) | set(b)
    expected = sum((sum(x == c for x in a) / n) * (sum(y == c for y in b) / n) for c in cats)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)
