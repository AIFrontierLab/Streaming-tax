"""Table 2 of the paper (tables/gaps.tex) and its companion for MCV-SafetyBench (tables/mcv_gaps.tex): the
adaptation tax C - P and the execution tax S - C on the trust benchmarks, and the generalization contrasts.

Table 2 columns, all in points with + = the child (or its streaming run) better:
  HarmBench C-P; VSB harmful and benign-query C-P and S-C (ASR reductions: a negative value is a higher ASR);
  VMDT fairness and privacy C-P and S-C; prompt robustness per perturbation type, (C - P)_pert - (C - P)_clean pooled
  over Video-MME and MMBench (= the parent's accuracy drop minus the child's); fine-tuning dGain and dAcc, the mean
  over LingoQA and EyePCR of the budget-averaged contrasts.
Grey = not significant after Holm correction (the main-text families of the trust columns; the robustness
family of the prompt-robustness report); fine-tuning columns are descriptive means. "--" = the stream arm has no
score (analysis/contrasts.py, no-reply rule). The Mean row weights the children equally and is starred where a child
has no score. |x| < 0.05 prints as +0.0."""
from __future__ import annotations

import csv
from pathlib import Path

from .latex import zeroed

GREY = "8A939B"
# (key, dataset, metric, gap)
TRUST = (("harmbench C-P", "harmbench", "asr", "C-P"),
         ("vsb_harmful C-P", "video_safetybench", "asr_harmful", "C-P"),
         ("vsb_harmful S-C", "video_safetybench", "asr_harmful", "S-C"),
         ("vsb_benign C-P", "video_safetybench", "asr_benign", "C-P"),
         ("vsb_benign S-C", "video_safetybench", "asr_benign", "S-C"),
         ("fairness C-P", "vmdt_fairness", "fairness", "C-P"), ("fairness S-C", "vmdt_fairness", "fairness", "S-C"),
         ("privacy C-P", "vmdt_privacy", "privacy", "C-P"), ("privacy S-C", "vmdt_privacy", "privacy", "S-C"))
MCV = (("explicit C-P", "mcv_safetybench", "asr_explicit", "C-P"), ("explicit S-C", "mcv_safetybench", "asr_explicit", "S-C"),
       ("implicit C-P", "mcv_safetybench", "asr_implicit", "C-P"), ("implicit S-C", "mcv_safetybench", "asr_implicit", "S-C"))
ROBUST = (("typo", "Typo"), ("para", "Paraphrase"), ("distraction", "Distraction"))
FINETUNE = (("d_gain_avg", r"$\Delta$Gain"), ("d_acc_avg", r"$\Delta$Acc"))
FINETUNE_TASKS = ("lingoqa", "eyepcr")
CAPTION = (r"Retention of trustworthiness and generalization ability in streaming models. All differences are reported "
           r"in points. Negative values indicate worse performance on the reported metric. Safety columns report ASR "
           r"reductions, so a negative value means a higher ASR. Grey marks nonsignificant differences after Holm "
           r"correction. --: no score under streaming because the child stays silent. $^{\ast}$Mean of the children "
           r"with a score.")
MCV_CAPTION = (r"MCV-SafetyBench ASR reduction in percentage points. Positive values indicate fewer successful "
               r"attacks. Grey values are non-significant under the same Holm correction as Table~\ref{tab:gaps}. "
               r"Mean weights the children equally. --: the child answers none of the requests under streaming, and "
               r"no ASR is reported. $^{\ast}$Mean of the children with a score.")


def trust_cells(rows: list[dict], columns=TRUST) -> dict:
    """(child, key) -> {"delta", "ci", "sig"} from the contrast rows (absent: not reported)."""
    index = {(r["dataset"], r["metric"], r["child"], r["gap"]): r for r in rows}
    out = {}
    for key, dataset, metric, gap in columns:
        for (d, m, child, g), r in index.items():
            if (d, m, g) == (dataset, metric, gap):
                out[(child, key)] = {"delta": r["delta_pp"], "ci": r.get("ci95_pp"),
                                     "sig": r.get("p_holm_broad") is not None and r["p_holm_broad"] < 0.05}
    return out


def robustness_cells(path: Path | None, children: list[str]) -> dict:
    """(child, family) -> {"delta", "ci"}: the pooled accuracy contrasts of `robustness report` (its contrasts.csv:
    columns child, family, readout, value, low, high)."""
    out = {}
    if path is None or not Path(path).is_file():
        return out
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for r in csv.DictReader(handle):
            if r.get("readout", "acc") == "acc" and r.get("child") in children and r.get("family") in dict(ROBUST):
                holm = r.get("p_holm")
                out[(r["child"], r["family"])] = {"delta": float(r["value"]), "ci": [float(r["low"]), float(r["high"])],
                                                  "sig": bool(holm) and float(holm) < 0.05}
    return out


def finetune_cells(path: Path | None, ids: dict) -> dict:
    """(child, contrast) -> {"delta"}: the mean over LingoQA and EyePCR of the budget-averaged contrast of
    `finetune report` (cells.csv). ids: child -> its id in the fine-tuning report (MMDuet's merged checkpoint)."""
    out = {}
    if path is None or not Path(path).is_file():
        return out
    with Path(path).open(encoding="utf-8", newline="") as handle:
        table = [r for r in csv.DictReader(handle) if r["variant"] == "all" and r["dataset"] in FINETUNE_TASKS]
    for child, name in ids.items():
        for contrast, _ in FINETUNE:
            values = [float(r["value"]) for r in table if r["child"] == name and r["contrast"] == contrast]
            if len(values) == len(FINETUNE_TASKS):
                # rounded first, so that a mean of one-decimal cells prints as its decimal value does (3.85 -> +3.9)
                out[(child, contrast)] = {"delta": round(sum(values) / len(values), 6), "sig": True}
    return out


def number(cell: dict | None) -> str:
    if cell is None or cell.get("delta") is None:
        return "--"
    text = "$" + f"{zeroed(cell['delta']):+.1f}" + "$"
    return text if cell.get("sig", True) else r"\textcolor[HTML]{" + GREY + "}{" + text + "}"


def mean(cells: dict, children: list[str], key: str) -> str:
    values = [cells[(c, key)]["delta"] for c in children if (c, key) in cells]
    if not values:
        return "--"
    star = r"^{\ast}" if len(values) < len(children) else ""
    return "$" + f"{zeroed(sum(values) / len(values)):+.1f}" + star + "$"


def build(rows: list[dict], children: list[str], robustness: Path | None = None, finetune: Path | None = None,
          finetune_ids: dict | None = None) -> dict:
    """Every cell of Table 2 and of the MCV table: {"children", "cells": {"<child>|<column>": {...}}}."""
    ids = {c: (finetune_ids or {}).get(c, c) for c in children}
    cells = {**trust_cells(rows), **trust_cells(rows, MCV), **robustness_cells(robustness, children),
             **finetune_cells(finetune, ids)}
    return {"children": list(children), "cells": {f"{c}|{k}": v for (c, k), v in cells.items()}}


def _cells(data: dict) -> dict:
    return {tuple(k.split("|", 1)): v for k, v in data["cells"].items()}


def latex(data: dict, names: dict) -> str:
    cells, children = _cells(data), data["children"]
    keys = [k for k, *_ in TRUST] + [k for k, _ in ROBUST] + [k for k, _ in FINETUNE]
    lines = ["% Generated by `streamreplace tables`; do not edit by hand.", r"\begin{table*}[t]", r"\centering",
             r"\caption{" + CAPTION + "}", r"\label{tab:gaps}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\renewcommand{\arraystretch}{1.12}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{@{}l*{14}{c}@{}}", r"\toprule",
             r" & \multicolumn{5}{c}{Safety} & \multicolumn{4}{c}{VMDT} & \multicolumn{3}{c}{Prompt robustness} & "
             r"\multicolumn{2}{c}{Fine-tuning} \\",
             r"\cmidrule(lr){2-6}\cmidrule(lr){7-10}\cmidrule(lr){11-13}\cmidrule(lr){14-15}",
             r" & HarmBench & \multicolumn{2}{c}{VSB harmful} & \multicolumn{2}{c}{VSB benign} & "
             r"\multicolumn{2}{c}{Fairness} & \multicolumn{2}{c}{Privacy} & "
             r"\multicolumn{3}{c}{$(C-P)_{\rm pert}-(C-P)_{\rm clean}$} & \multicolumn{2}{c}{Task mean} \\",
             r"\cmidrule(lr){2-2}\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}"
             r"\cmidrule(lr){11-13}\cmidrule(lr){14-15}",
             "Model & " + " & ".join(["$C-P$"] + ["$C-P$", "$S-C$"] * 4 + [label for _, label in ROBUST]
                                     + [label for _, label in FINETUNE]) + r" \\", r"\midrule"]
    for child in children:
        lines.append(" & ".join([names.get(child, child)] + [number(cells.get((child, k))) for k in keys]) + r" \\")
    lines += [r"\midrule", " & ".join([r"\textit{Mean}"] + [mean(cells, children, k) for k in keys]) + r" \\",
              r"\bottomrule", r"\end{tabular}%", "}", r"\end{table*}", ""]
    return "\n".join(lines)


def mcv_latex(data: dict, names: dict) -> str:
    cells, children = _cells(data), data["children"]
    keys = [k for k, *_ in MCV]
    lines = ["% Generated by `streamreplace tables`; do not edit by hand.", r"\begin{table}[!htbp]", r"\centering",
             r"\caption{" + MCV_CAPTION + "}", r"\label{tab:mcv-gaps}", r"\footnotesize",
             r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Explicit} & \multicolumn{2}{c}{Implicit} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"Model & $C-P$ & $S-C$ & $C-P$ & $S-C$ \\", r"\midrule"]
    for child in children:
        lines.append(" & ".join([names.get(child, child)] + [number(cells.get((child, k))) for k in keys]) + r" \\")
    lines += [r"\midrule", " & ".join([r"\textit{Mean}"] + [mean(cells, children, k) for k in keys]) + r" \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


CSV_FIELDS = ["child", "column", "delta", "ci_low", "ci_high", "significant", "printed"]


def flat(data: dict) -> list[dict]:
    """One row per cell of the two tables with its unrounded value and the printed LaTeX (gaps.csv)."""
    cells = _cells(data)
    out = []
    for child in data["children"]:
        for key in [k for k, *_ in TRUST + MCV] + [k for k, _ in ROBUST + FINETUNE]:
            r = cells.get((child, key))
            ci = (r or {}).get("ci") or [None, None]
            out.append({"child": child, "column": key, "delta": (r or {}).get("delta"), "ci_low": ci[0],
                        "ci_high": ci[1], "significant": (r or {}).get("sig"), "printed": number(r)})
    return out
