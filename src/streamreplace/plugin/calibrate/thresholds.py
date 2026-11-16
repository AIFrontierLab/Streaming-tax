"""Gate thresholds of a child (the paper's rule; the shipped manifests hold the results).

The gate refuses an item whose logged score is strictly above the threshold of its answer format. Each threshold is
a 97.5th percentile of the score on harmless requests of that format, from three sets of harmless requests, none of
which is an evaluation item:

  dev_closed_q975      round(np.quantile(p, 0.975), 4) over the record arm's logged projections p (3 decimals, as
                       info.plugin.guard.proj holds them) on the 301 harmless closed-choice dev-slice items of
                       paper/frozen/plugin/threshold_items.json (Video-MME 45, TOMATO 75, TimeBlind 80, VMDT fairness
                       101). Every item must have a projection.
  free_harmless_q975   np.quantile(q, 0.975) over the S_fix arm's logged projections q on the 100 harmless free-form
                       "fit" requests (how-to / describe requests on benign clips cut from the steering calibration
                       videos; paper/frozen/plugin/harmless_requests.json role "fit").
  closed_pool_q975     np.quantile over the 301 dev-slice projections pooled with the complete plugin's (sfix_prs)
                       logged projections on the 576 streaming-benchmark pilot rows of paper/frozen/plugin/
                       stream_pilot_items.json (closed-choice questions asked during the video).

  tau_free   = max(dev_closed_q975, free_harmless_q975)     free-form items
  tau_closed = max(tau_free, closed_pool_q975)               multiple-choice and yes/no items

np.quantile uses numpy's default linear interpolation. Both floors were added after pilots (the paper says so): with
the harmless free-form quantile alone the gate refused most open-ended fairness items of three children, and the
dev-slice quantile alone refused closed-choice questions asked during a stream for one child. `rule()` is the whole
rule; a child calibrated without the streaming pilot gets tau_closed = max(tau_free, dev_closed_q975) = tau_free."""
from __future__ import annotations

from pathlib import Path

import numpy as np

QUANTILE = 0.975
CLOSED_DECIMALS = 4


def guard_proj(row: dict) -> float | None:
    """The logged gate projection of a generation row: info.plugin.guard.proj, or guard.proj at the top level of a
    harmless-request row (None when absent or the row errored)."""
    from ...analysis.records import plugin_guard
    if row is None or row.get("error"):
        return None
    return plugin_guard(row).get("proj")


def tau_closed(values, quantile: float = QUANTILE, decimals: int = CLOSED_DECIMALS) -> float:
    if not len(values):
        raise ValueError("no projections")
    return round(float(np.quantile(values, quantile)), decimals)


def harmless_quantile(values, quantile: float = QUANTILE, minimum: int = 100) -> float:
    if len(values) < minimum:
        raise ValueError(f"{len(values)} of {minimum} harmless fit projections")
    return float(np.quantile(values, quantile))


def pool_quantile(dev_values, pilot_values, quantile: float = QUANTILE) -> float:
    """closed_pool_q975: the quantile of the dev-slice projections pooled with the streaming-pilot projections."""
    pool = [*dev_values, *pilot_values]
    if not pool:
        raise ValueError("no projections")
    return float(np.quantile(pool, quantile))


def rule(dev_closed: float, free_harmless: float, closed_pool: float | None = None) -> dict:
    """{"free", "closed"} from the three quantiles (module docstring); closed_pool None = no streaming pilot."""
    free = max(dev_closed, free_harmless)
    return {"free": free, "closed": free if closed_pool is None else max(free, closed_pool)}


def record_projections(rows_by_dataset: dict[str, dict], items: dict[str, list[str]],
                       complete: bool = True) -> list[float]:
    """An arm's logged projection on every item of an item set (the record arm on the threshold items; the complete
    plugin on the streaming pilot). complete: refuse a missing, errored or projection-less item (the threshold
    items); otherwise such rows are left out (a stream that could not run a clip)."""
    values, problems = [], []
    for d, ids in items.items():
        rows = rows_by_dataset.get(d, {})
        for i in ids:
            p = guard_proj(rows.get(i))
            if p is None:
                problems.append(f"{d} {i}")
            else:
                values.append(p)
    if problems and complete:
        raise ValueError(f"no projection on {len(problems)} threshold item(s), e.g. {problems[:3]}")
    return values


def latest_ok(rows) -> dict:
    """id -> row, a later row replacing an earlier one unless the later one errored (the harmless-request runner's rule)."""
    out: dict = {}
    for r in rows:
        if not r.get("error") or r["id"] not in out:
            out[r["id"]] = r
    return out


def fit_projections(rows, fit_ids) -> list[float]:
    """The S_fix arm's projections on the harmless fit requests (rows: the harmless-request run's rows, any order)."""
    fit = set(fit_ids)
    return [p for i, r in latest_ok(rows).items() if i in fit and (p := guard_proj(r)) is not None]


def fit_ids(path: Path) -> list[str]:
    import json
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [it["id"] for it in data["items"] if it["role"] == "fit"]
