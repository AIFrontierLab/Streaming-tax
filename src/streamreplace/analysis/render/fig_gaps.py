"""Data of the paper's gap figure: C-P and S-P of every child on 9 video / safety / VMDT columns (108 cells).

Both child conditions are compared with the offline parent: C-P (offline panel) and S-P (stream panel), so the
difference between the two bars is S-C. Each cell is the full contrast row plus panel, column and flags; the
figure prints the delta with '+.1f' (|x| < 0.05 as +0.0), a dagger for a forced stream (S-dagger) and a double
dagger for more than 10% capped rows in an arm. A Mean row averages each column over the six children.
`printed()` returns exactly the strings the figure shows, for a cell-by-cell comparison."""
from __future__ import annotations

from .latex import zeroed

COLUMNS = [("videomme", "videomme", "acc"), ("tomato", "tomato", "acc"), ("timeblind", "timeblind", "I_Acc"),
           ("vsb_harmful", "video_safetybench", "asr_harmful"), ("vsb_benign", "video_safetybench", "asr_benign"),
           ("mcv_explicit", "mcv_safetybench", "asr_explicit"), ("mcv_implicit", "mcv_safetybench", "asr_implicit"),
           ("vmdt_fairness", "vmdt_fairness", "fairness"), ("vmdt_privacy", "vmdt_privacy", "privacy")]
PANELS = (("offline", "C-P"), ("stream", "S-P"))


def build(rows: list[dict], children: list[str]) -> dict:
    """Cells (contrast rows with panel, column, flags) and per-column means over the children."""
    index = {(r["dataset"], r["metric"], r["child"], r["gap"]): r for r in rows}
    cells, means = [], []
    for panel, gap in PANELS:
        for column, dataset, metric in COLUMNS:
            deltas = []
            for child in children:
                r = index.get((dataset, metric, child, gap))
                if r is None:
                    continue
                cell = {k: v for k, v in r.items() if k != "flag_notes"}
                cell.update(panel=panel, column=column, flags=list(r.get("flags", [])))
                cells.append(cell)
                deltas.append(r["delta"])
            if len(deltas) == len(children):
                means.append({"panel": panel, "column": column, "delta": sum(deltas) / len(deltas)})
    return {"sign": "+ = higher understanding / fairness / privacy score, or lower ASR; both conditions are "
                    "compared with the offline parent",
            "children": children, "cells": cells, "means": means}


def mark(flags) -> str:
    flags = set(flags)
    return ("\\dagger" if "forced" in flags else "") + ("\\ddagger" if flags & {"invalid", "truncation_pending"} else "")


def value(d: float, mark_: str = "") -> str:
    return "$" + f"{d:+.1f}" + (f"^{{{mark_}}}" if mark_ else "") + "$"


def printed(data: dict) -> dict:
    """(panel, child or 'mean', column) -> the string the figure prints."""
    out = {}
    for c in data["cells"]:
        out[(c["panel"], c["child"], c["column"])] = value(zeroed(c["delta"]), mark(c["flags"]))
    for m in data["means"]:
        out[(m["panel"], "mean", m["column"])] = value(zeroed(m["delta"]))
    return out


CSV_FIELDS = ["panel", "column", "child", "gap", "delta", "ci_low", "ci_high", "p_holm_broad", "flags", "printed"]


def flat(data: dict, names: dict | None = None) -> list[dict]:
    """One row per figure cell (and per column mean) with the unrounded delta and the printed string."""
    names = names or {}
    shown = printed(data)
    out = []
    for c in data["cells"]:
        ci = c.get("ci95_pp") or [None, None]
        out.append({"panel": c["panel"], "column": c["column"], "child": names.get(c["child"], c["child"]),
                    "gap": c["gap"], "delta": c["delta"], "ci_low": ci[0], "ci_high": ci[1],
                    "p_holm_broad": c.get("p_holm_broad"), "flags": "|".join(c["flags"]),
                    "printed": shown[(c["panel"], c["child"], c["column"])]})
    for m in data["means"]:
        out.append({"panel": m["panel"], "column": m["column"], "child": "mean", "gap": dict(PANELS)[m["panel"]],
                    "delta": m["delta"], "printed": shown[(m["panel"], "mean", m["column"])]})
    return out
