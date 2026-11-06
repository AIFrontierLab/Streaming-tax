"""gaps.json: the cell data behind Table 2 and the overview figure of the paper.

Two panels, each a child x column grid:
  offline  C-P: the 7 headline image metrics, the 3 video headlines, prompt robustness (worst family), fine-tuning
           (LingoQA), HarmBench, VSB harmful, MCV explicit / implicit, VMDT fairness and privacy
  stream   S-C: the 3 video headlines and the 5 trust columns
Each contrast cell carries the 4-decimal delta and 95% interval in points, main-text Holm significance
(p_holm_broad < 0.05), tost3, the stream coverage, and the cell's flags (flags.py). Prompt-robustness and fine-tuning
cells come from those analyses' outputs (optional inputs; see docs/statistics.md for their columns)."""
from __future__ import annotations

import csv
from pathlib import Path

from .latex import round4

# (column key, label, group, unit, dataset, metric)
OFFLINE = [
    ("dynamath", "DynaMath (avg)", "image", "pp", "dynamath", "avg_acc"),
    ("mathvista", "MathVista", "image", "pp", "mathvista", "acc"),
    ("mmbench", "MMBench (circular)", "image", "pp", "mmbench", "circular_acc"),
    ("mme_p", "MME perception", "image", "pp of max (score/20)", "mme", "perception"),
    ("mmmu", "MMMU", "image", "pp", "mmmu", "acc"),
    ("mmstar", "MMStar", "image", "pp", "mmstar", "acc"),
    ("scienceqa", "ScienceQA", "image", "pp", "scienceqa", "acc"),
    ("videomme", "Video-MME (short)", "video", "pp", "videomme", "acc"),
    ("tomato", "TOMATO", "video", "pp", "tomato", "acc"),
    ("timeblind", "TimeBlind (I_Acc)", "video", "pp", "timeblind", "I_Acc"),
    ("robust", "Prompt robustness (worst-family dRob)", "generalization", "pp", None, None),
    ("ft_lingoqa", "Fine-tuning LingoQA (dGain, budget avg)", "generalization", "pp", None, None),
    ("harmbench", "HarmBench dASR", "trust", "pp (parent - child)", "harmbench", "asr"),
    ("vsb_harmful", "VSB harmful dASR", "trust", "pp (parent - child)", "video_safetybench", "asr_harmful"),
    ("mcv_explicit", "MCV explicit dASR", "trust", "pp (parent - child)", "mcv_safetybench", "asr_explicit"),
    ("mcv_implicit", "MCV implicit dASR", "trust", "pp (parent - child)", "mcv_safetybench", "asr_implicit"),
    ("vmdt_fairness", "VMDT fairness", "trust", "pp", "vmdt_fairness", "fairness"),
    ("vmdt_privacy", "VMDT privacy", "trust", "pp", "vmdt_privacy", "privacy"),
]
STREAM = [
    ("videomme", "Video-MME (short)", "video", "pp", "videomme", "acc"),
    ("tomato", "TOMATO", "video", "pp", "tomato", "acc"),
    ("timeblind", "TimeBlind (I_Acc)", "video", "pp", "timeblind", "I_Acc"),
    ("vsb_harmful", "VSB harmful dASR", "trust", "pp (C - S)", "video_safetybench", "asr_harmful"),
    ("mcv_explicit", "MCV explicit dASR", "trust", "pp (C - S)", "mcv_safetybench", "asr_explicit"),
    ("mcv_implicit", "MCV implicit dASR", "trust", "pp (C - S)", "mcv_safetybench", "asr_implicit"),
    ("vmdt_fairness", "VMDT fairness", "trust", "pp", "vmdt_fairness", "fairness"),
    ("vmdt_privacy", "VMDT privacy", "trust", "pp", "vmdt_privacy", "privacy"),
]
ROBUST_FAMILIES = ("typo", "para", "distraction")


def _read_csv(path: str | Path | None) -> list[dict]:
    if not path or not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def robustness_cells(path, children: list[str]) -> dict:
    """child -> robustness cell from the prompt-robustness primary table (columns child, family, readout, n, value,
    low, high, p_holm, equivalent; accuracy readout): the worst (lowest) family, all three families kept."""
    rows = [r for r in _read_csv(path) if r.get("readout", "acc") == "acc"]
    out = {}
    for child in children:
        fam = [r for r in rows if r["child"] == child]
        if not fam:
            continue
        worst = min(fam, key=lambda r: float(r["value"]))
        out[child] = {"delta": round4(worst["value"]), "ci": [round4(worst["low"]), round4(worst["high"])],
                      "holm_sig": float(worst["p_holm"]) < 0.05, "tost3": worst["equivalent"] == "True",
                      "coverage": None, "family": worst["family"], "n": int(worst["n"]),
                      "all_families": {r["family"]: [round4(r["value"]), round4(r["low"]), round4(r["high"]),
                                                     r["equivalent"] == "True"] for r in fam},
                      "holm_p": round4(worst["p_holm"])}
    return out


def finetune_cells(path, children: list[str]) -> dict:
    """child -> fine-tuning cell from the fine-tuning cells table (columns dataset, variant, contrast, child, n,
    value, low, high, lower_0.05, holm12): LingoQA, all budgets, budget-averaged gain difference. TOST at ±3 is
    false when the one-sided 95% lower bound is below -3, else unknown (only that bound is recorded)."""
    rows = [r for r in _read_csv(path) if r.get("dataset") == "lingoqa" and r.get("variant") == "all"
            and r.get("contrast") == "d_gain_avg"]
    out = {}
    for child in children:
        match = [r for r in rows if r["child"] in (child, f"{child}_merged")]
        if not match:
            continue
        r = match[0]
        lower = float(r["lower_0.05"])
        out[child] = {"delta": round4(r["value"]), "ci": [round4(r["low"]), round4(r["high"])],
                      "holm_sig": float(r["holm12"]) < 0.05, "tost3": False if lower < -3 else None,
                      "coverage": None, "n": int(r["n"])}
    return out


def contrast_cell(row: dict, panel: str) -> dict:
    return {"delta": round4(row["delta_pp"]), "ci": [round4(row["ci95_pp"][0]), round4(row["ci95_pp"][1])],
            "holm_sig": row["p_holm_broad"] < 0.05, "tost3": bool(row["tost3"]),
            "coverage": round4(row["coverage_b"]) if panel == "stream" else None,
            "a": round4(row["a"]), "b": round4(row["b"]), "n": row["n"], "p": row["p"],
            "p_holm_broad": row["p_holm_broad"], "holm_broad_family": f"{row['holm_broad_family']} ({row['holm_broad_m']})",
            "p_holm_declared": row["p_holm"], "holm_declared_family": f"{row['holm_family']} ({row['holm_m']})",
            "coverage_a": round4(row["coverage_a"]), "coverage_b": round4(row["coverage_b"])}


def build(rows: list[dict], children: list[str], robustness=None, finetune=None) -> dict:
    """The gaps.json document from annotated + flagged contrast rows (with the rerun overlay)."""
    index = {(r["dataset"], r["metric"], r["child"], r["gap"]): r for r in rows}
    robust, ft = robustness_cells(robustness, children), finetune_cells(finetune, children)
    cells = []
    for panel, columns, gap in (("offline", OFFLINE, "C-P"), ("stream", STREAM, "S-C")):
        for key, _label, _group, _unit, dataset, metric in columns:
            for child in children:
                if key == "robust":
                    cell = dict(robust.get(child) or {"delta": None})
                    flags, notes = [], []
                elif key == "ft_lingoqa":
                    cell = dict(ft.get(child) or {"delta": None})
                    flags, notes = [], []
                else:
                    row = index.get((dataset, metric, child, gap))
                    if row is None:
                        continue
                    cell = contrast_cell(row, panel)
                    flags, notes = list(row.get("flags", [])), list(row.get("flag_notes", []))
                cell.update(child=child, panel=panel, column=key, flags=flags, notes=notes)
                cells.append(cell)
    return {"sign": "+ = child better (offline panel C - P) or the streaming run better (stream panel S - C); for ASR "
                    "the difference is parent - child (offline) or C - S (stream); MME perception in points of the "
                    "maximum (score/20)",
            "children": children,
            "panels": {"offline": {"gap": "C-P", "columns": [{"key": k, "label": l, "group": g, "unit": u}
                                                              for k, l, g, u, _, _ in OFFLINE]},
                       "stream": {"gap": "S-C", "columns": [{"key": k, "label": l, "group": g, "unit": u}
                                                             for k, l, g, u, _, _ in STREAM]}},
            "cells": cells}
