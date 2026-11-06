"""Paired bootstrap of the VMDT composites (fairness and privacy), on the benchmarks' official scores().

VMDT's headline numbers are not item accuracies but official composites (fairness = (S + D + F) / 3 from group gaps;
privacy = 100 - (state + 2 city + 4 zip) / 7), so their contrasts resample the official clusters and re-evaluate the
benchmark's own `scores(rows, weights, clusters)` on multinomial cluster weights:

- fairness: one block per scenario, in the benchmark's SCENARIOS order, over the sorted union of both arms' cluster
  labels (subjects, YouTube ids, topics); privacy: one block over the sorted video clusters;
- one fresh numpy default_rng(seed) per (dataset, arm a, arm b); every metric of that dataset reads the same draws;
- all counted rows of each arm are used (the jobs are complete, so this equals the common items);
- p = min(1, 2 (min(#delta <= 0, #delta >= 0) + 1) / (N + 1)) on the non-NaN draws.

`contrast()` is the paper's version (20,000 draws, 95% and 90% intervals, oriented so that + = less leakage for the
location accuracies). `official_paired()` is the benchmarks' own 2,000-draw paired table printed in RESULTS.md."""
from __future__ import annotations

import math

import numpy as np

from .records import Catalog, Job, metric_rows


def _draws(module, rows_a: list[dict], rows_b: list[dict], blocks: str | None, draws: int, seed: int):
    """(clusters, weights, number of clusters) exactly as the official paired bootstrap draws them."""
    rng = np.random.default_rng(seed)
    if blocks == "scenario":
        clusters = {s: sorted({r["metadata"]["cluster"] for r in rows_a + rows_b if r["metadata"]["scenario"] == s})
                    for s in module.SCENARIOS}
        weights = {s: rng.multinomial(len(c), np.full(len(c), 1 / len(c)), size=draws).astype(float)
                   for s, c in clusters.items()}
        return clusters, weights, sum(map(len, clusters.values()))
    clusters = sorted({r["metadata"]["cluster"] for r in rows_a + rows_b})
    weights = rng.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)), size=draws).astype(float)
    return clusters, weights, len(clusters)


class Engine:
    """Point and bootstrap scores per (dataset, arm a, arm b), computed once and shared by the dataset's metrics."""

    def __init__(self, catalog: Catalog, draws: int = 20_000, seed: int = 0):
        self.catalog, self.draws, self.seed, self.cache = catalog, draws, seed, {}

    def scores(self, metric, a: Job, b: Job):
        key = (metric.dataset, a.uid, b.uid, self.draws, self.seed, metric.blocks)
        if key not in self.cache:
            module = self.catalog.bench(metric.dataset)
            rows_a, rows_b = metric_rows(a, self.catalog), metric_rows(b, self.catalog)
            clusters, weights, n_clusters = _draws(module, rows_a, rows_b, metric.blocks, self.draws, self.seed)
            self.cache[key] = (module.scores(rows_a, clusters=clusters), module.scores(rows_b, clusters=clusters),
                               module.scores(rows_a, weights, clusters), module.scores(rows_b, weights, clusters),
                               n_clusters, len(rows_a))
        return self.cache[key]

    def contrast(self, metric, a: Job, b: Job) -> dict:
        """The paper's VMDT contrast (official clusters, weights and p rule), with the 90% interval too."""
        point_a, point_b, boot_a, boot_b, n_clusters, n = self.scores(metric, a, b)
        key = metric.score_key
        pa, pb = float(point_a[key][0]), float(point_b[key][0])
        delta = boot_b[key] - boot_a[key]
        delta = delta[~np.isnan(delta)]
        sign = -1.0 if metric.lower_is_better else 1.0             # leakage accuracy: + = less leakage
        lo95, hi95, lo90, hi90 = (float(x) for x in np.percentile(sign * delta, [2.5, 97.5, 5, 95]))
        p = min(1.0, 2 * (min(int((delta <= 0).sum()), int((delta >= 0).sum())) + 1) / (len(delta) + 1))
        oriented = sign * (pb - pa)                                 # a / b stay in the metric's own units
        return {"a": pa, "b": pb, "delta": oriented, "ci95": [lo95, hi95], "ci90": [lo90, hi90], "p": p,
                "test": "vmdt-bootstrap", "clusters": int(n_clusters), "items": float(n),
                "delta_pp": oriented, "ci95_pp": [lo95, hi95], "ci90_pp": [lo90, hi90], "draws": int(len(delta))}


# ---------- the benchmarks' own paired table (RESULTS.md) ----------

FAIRNESS_KEYS = ("fairness", "stereotype", "decision_making", "factual_accuracy")
FAIRNESS_NAMES = {"decision_making": "decision", "factual_accuracy": "factual"}
PRIVACY_KEYS = ("privacy", "state", "city", "zipcode")


def _number(x, digits):
    x = float(x)
    return None if math.isnan(x) else round(x, digits)


def official_paired(module, parent_rows: list[dict], candidate_rows: list[dict], blocks: str | None,
                    draws: int = 2000, seed: int = 0) -> list[dict]:
    """Paired cluster bootstrap of the benchmark (VMDT fairness: blocks="scenario"; privacy: one block):
    delta = candidate - parent; 95% percentile interval; two-sided p = 2 (min(#delta <= 0, #delta >= 0) + 1) /
    (draws + 1). Values rounded to 2 decimals as the benchmarks printed them."""
    clusters, weights, _ = _draws(module, parent_rows, candidate_rows, blocks, draws, seed)
    out = []
    if blocks == "scenario":
        point = {k: (module.scores(parent_rows, clusters=clusters)[k][0],
                     module.scores(candidate_rows, clusters=clusters)[k][0]) for k in FAIRNESS_KEYS}
        parent, candidate = module.scores(parent_rows, weights, clusters), module.scores(candidate_rows, weights, clusters)
        for key, (p, c) in point.items():
            delta = candidate[key] - parent[key]
            delta = delta[~np.isnan(delta)]
            row = {"metric": FAIRNESS_NAMES.get(key, key), "parent": _number(p, 2), "candidate": _number(c, 2),
                   "delta": _number(c - p, 2), "low": None, "high": None, "p": None, "draws": int(len(delta))}
            if len(delta) and row["delta"] is not None:
                row.update(low=round(float(np.percentile(delta, 2.5)), 2), high=round(float(np.percentile(delta, 97.5)), 2),
                           p=min(1.0, 2 * (min(int((delta <= 0).sum()), int((delta >= 0).sum())) + 1) / (len(delta) + 1)))
            out.append(row)
        return out
    point = module.scores(parent_rows, clusters=clusters), module.scores(candidate_rows, clusters=clusters)
    boot = module.scores(parent_rows, weights, clusters), module.scores(candidate_rows, weights, clusters)
    for key in PRIVACY_KEYS:
        p, c = float(point[0][key][0]), float(point[1][key][0])
        delta = boot[1][key] - boot[0][key]
        delta = delta[~np.isnan(delta)]
        out.append({"metric": key, "parent": round(p, 2), "candidate": round(c, 2), "delta": round(c - p, 2),
                    "low": round(float(np.percentile(delta, 2.5)), 2), "high": round(float(np.percentile(delta, 97.5)), 2),
                    "p": min(1.0, 2 * (min(int((delta <= 0).sum()), int((delta >= 0).sum())) + 1) / (len(delta) + 1)),
                    "draws": int(len(delta))})
    return out
