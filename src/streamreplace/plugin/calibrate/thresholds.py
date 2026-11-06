"""Gate thresholds of a child (the paper's rule; the shipped manifests hold the results).

tau_closed  closed-choice items: round(np.quantile(p, 0.975), 4) over the record arm's logged projections p (3
            decimals, as info.plugin.guard.proj holds them) on the 301 harmless dev-slice items of
            paper/frozen/plugin/threshold_items.json (Video-MME 45, TOMATO 75, TimeBlind 80, VMDT fairness 101). The
            record arm runs the child's native stream with Recall and Respond, logs the projection and ends the answer
            turn at its first step (ThinkStream's, whose answer turn hands the hooks no logits, runs to its end; the
            projection is the same). Every item must have a projection.
tau_free    free-form items: max(tau_closed, np.quantile(q, 0.975)) over the S_fix arm's logged projections q on the
            100 harmless "fit" requests (harmless how-to / describe requests on benign clips cut from the steering
            calibration videos; paper/frozen/plugin/harmless_requests.json role "fit"). The quantile is not rounded.
np.quantile uses numpy's default linear interpolation. The free-form rule was raised after a first pilot (the paper
says so): with the harmless quantile alone the gate refused most open-ended fairness items for three children."""
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


def tau_free(closed: float, values, quantile: float = QUANTILE, minimum: int = 100) -> tuple[float, float]:
    """(tau_free, the harmless quantile behind it)."""
    q = harmless_quantile(values, quantile, minimum)
    return max(closed, q), q


def record_projections(rows_by_dataset: dict[str, dict], items: dict[str, list[str]]) -> list[float]:
    """The record arm's projection on every threshold item; refuses a missing, errored or projection-less item."""
    values, problems = [], []
    for d, ids in items.items():
        rows = rows_by_dataset.get(d, {})
        for i in ids:
            p = guard_proj(rows.get(i))
            if p is None:
                problems.append(f"{d} {i}")
            else:
                values.append(p)
    if problems:
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
