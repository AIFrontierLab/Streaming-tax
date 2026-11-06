"""Paired statistics behind every P/C/S contrast of the paper (numpy + stdlib only).

This is the main statistics engine (the "multinomial cluster-weight" engine). It produces every delta, interval,
p value and TOST / non-inferiority mark of the C-P, S-C and S-P contrasts (Table 2, the master tables in the
supplement, the gap figure). The older integer-index engine used by the robustness, fine-tuning and truncation
reports lives in `stats_index.py`; docs/statistics.md says which number comes from which engine.

- mcnemar_exact, holm: exact two-sided McNemar and Holm's step-down adjustment.
- Units + paired_contrast: the general paired contrast. A metric is a weighted sum of ratio terms,
  value = sum_j w_j * 100 * sum_k num[k, j] / sum_k den[k, j] over clusters k; plain accuracy is one term,
  MME is acc + acc+ per category. Both arms are evaluated on the same items and on the same bootstrap draws
  (clusters resampled with replacement within each stratum). p: exact McNemar when every cluster is a single
  binary item (one term, den 1), else the cluster sign-flip test of the per-cluster contributions.
- equivalent / noninferior: TOST at margin m (90% CI inside (-m, +m), alpha 0.05 per side) and non-inferiority
  (lower 95% bound > -m, one-sided alpha 0.025).
- binom_tail: exact binomial tails (chance-level tests).

Every function body is kept exactly as it produced the paper's numbers: a fresh np.random.default_rng(seed) per
contrast, draws in blocks of CHUNK, strata in first-seen order, np.percentile with linear interpolation. Changing
the order of random calls changes the published intervals in the last digits."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

BOOT, PERM, SEED = 10_000, 10_000, 0
CHUNK = 2_000                     # bootstrap draws per matrix product (memory: CHUNK x clusters floats)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p on the discordant counts; 1.0 when there is no discordant pair."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def holm(pvalues: dict) -> dict:
    """Holm step-down adjusted p values (monotone, capped at 1); None values are left out.
    Ties keep the input order (Python's sort is stable)."""
    keys = sorted((k for k, v in pvalues.items() if v is not None), key=lambda k: pvalues[k])
    adjusted, running = {}, 0.0
    for rank, key in enumerate(keys):
        running = max(running, min(1.0, (len(keys) - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


@dataclass
class Units:
    """Cluster-level sufficient statistics of one metric for two arms on the same items.
    num_a / num_b / den: (K, J) arrays (K clusters, J ratio terms); strata: K labels (resampling strata);
    weight: J term weights. Denominators are shared: both arms are scored on the same items."""
    num_a: np.ndarray
    num_b: np.ndarray
    den: np.ndarray
    strata: list
    weight: np.ndarray

    @property
    def binary(self) -> bool:
        """Every cluster is one binary item (exact McNemar applies)."""
        return (self.den.shape[1] == 1 and bool(np.all(self.den == 1))
                and bool(np.all(np.isin(self.num_a, (0, 1)))) and bool(np.all(np.isin(self.num_b, (0, 1)))))


def value(num: np.ndarray, den: np.ndarray, weight: np.ndarray):
    """Metric of one arm from (summed) numerators / denominators; works on (J,) or (reps, J)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (100 * num / den) @ weight


def units(clusters: dict, weight=(1.0,)) -> Units:
    """clusters: {(stratum, cluster): (num_a, num_b, den)} with per-term sequences (or scalars for one term).
    The dict's insertion order is the cluster order (and, per stratum, the order the bootstrap weights follow)."""
    rows = list(clusters.items())
    arr = lambda j: np.array([np.atleast_1d(np.asarray(v[j], dtype=float)) for _, v in rows])  # noqa: E731
    return Units(arr(0), arr(1), arr(2), [k[0] for k, _ in rows], np.asarray(weight, dtype=float))


def paired_contrast(u: Units, reps: int = BOOT, seed: int = SEED, scale: float = 1.0) -> dict:
    """delta = metric(b) - metric(a) with stratified cluster-bootstrap 95% / 90% percentile intervals and a two-sided
    p (exact McNemar for binary single-item clusters, else cluster sign-flip with `reps` flips). `scale` divides the
    delta into percentage points (MME: 20 for perception /2000, 8 for cognition /800); returned in both units."""
    keep = u.den.sum(1) > 0                                   # clusters with no complete unit carry no information
    u = Units(u.num_a[keep], u.num_b[keep], u.den[keep], [s for s, k in zip(u.strata, keep) if k], u.weight)
    a = float(value(u.num_a.sum(0), u.den.sum(0), u.weight))
    b = float(value(u.num_b.sum(0), u.den.sum(0), u.weight))
    out = {"a": a, "b": b, "delta": b - a, "clusters": int(len(u.strata)), "items": float(u.den.sum())}
    if not len(u.strata):
        return {**out, "ci95": None, "ci90": None, "p": None, "test": None}
    rng = np.random.default_rng(seed)
    order = list(dict.fromkeys(u.strata))
    index = {s: np.flatnonzero(np.array([x == s for x in u.strata])) for s in order}
    deltas = np.empty(reps)
    for start in range(0, reps, CHUNK):
        r = min(CHUNK, reps - start)
        na, nb, dn = (np.zeros((r, u.den.shape[1])) for _ in range(3))
        for s in order:
            idx = index[s]
            w = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=r).astype(float)
            na += w @ u.num_a[idx]
            nb += w @ u.num_b[idx]
            dn += w @ u.den[idx]
        deltas[start:start + r] = value(nb, dn, u.weight) - value(na, dn, u.weight)
    deltas = deltas[~np.isnan(deltas)]
    lo95, hi95, lo90, hi90 = (float(x) for x in np.percentile(deltas, [2.5, 97.5, 5, 95]))
    if u.binary:
        b_count = int(((u.num_a[:, 0] == 1) & (u.num_b[:, 0] == 0)).sum())
        c_count = int(((u.num_a[:, 0] == 0) & (u.num_b[:, 0] == 1)).sum())
        p, test, extra = mcnemar_exact(b_count, c_count), "mcnemar", {"a_only": b_count, "b_only": c_count}
    else:
        total = u.den.sum(0)
        contribution = ((100 * (u.num_b - u.num_a) / total) * u.weight).sum(1)
        p, test, extra = cluster_signflip(contribution, reps, seed), "signflip", {}
    s = float(scale)
    return {**out, "ci95": [lo95, hi95], "ci90": [lo90, hi90], "p": p, "test": test, **extra,
            "delta_pp": out["delta"] / s, "ci95_pp": [lo95 / s, hi95 / s], "ci90_pp": [lo90 / s, hi90 / s]}


def cluster_signflip(contribution: np.ndarray, reps: int = PERM, seed: int = SEED) -> float:
    """Two-sided p of sum(contribution) = 0 when each cluster's contribution changes sign with probability 1/2 (arms
    exchangeable within a cluster). Clusters contributing 0 are dropped (they do not change the sum)."""
    t = np.asarray(contribution, dtype=float)
    t = t[np.abs(t) > 1e-12]
    if not len(t):
        return 1.0
    observed = abs(t.sum())
    rng = np.random.default_rng(seed)
    hits = 0
    for start in range(0, reps, CHUNK):
        r = min(CHUNK, reps - start)
        flips = rng.choice([-1.0, 1.0], size=(r, len(t)))
        hits += int((np.abs(flips @ t) >= observed - 1e-9).sum())
    return (1 + hits) / (reps + 1)


def equivalent(ci90, margin: float) -> bool | None:
    """TOST at +-margin: the 90% interval lies inside (-margin, +margin)."""
    return None if ci90 is None else bool(ci90[0] > -margin and ci90[1] < margin)


def noninferior(ci95, margin: float) -> bool | None:
    """Non-inferiority at -margin: the lower 95% bound is above -margin (+ = child better)."""
    return None if ci95 is None else bool(ci95[0] > -margin)


def binom_tail(k: int, n: int, p: float, upper: bool = True) -> float:
    """Exact P(X >= k) (upper) or P(X <= k) for X ~ Binomial(n, p)."""
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    rng = range(k, n + 1) if upper else range(0, k + 1)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * math.log(p) + (n - i) * math.log1p(-p)
            for i in rng]
    if not logs:
        return 0.0
    top = max(logs)
    return min(1.0, math.exp(top) * sum(math.exp(x - top) for x in logs))
