"""Master tables (supplement): every C-P contrast in one table and every S-P contrast in another.

Each table is a benchmark x child grid of deltas in points (+ = the child or its stream run is better; ASR and
location leakage as parent minus child), written as LaTeX (the paper's master_cp.tex / master_sp.tex), Markdown
and CSV (with intervals and p values). Marks: bold = Holm p < 0.05 in the main-text ("broad") family;
^{\\approx} = equivalent within ±3 points (TOST) and not significant; optionally ^{c} = response coverage differs
from the parent's by more than 2 points (the paper omits this mark). The S-P CSV also carries the matching C-P and
S-C so that S-P = (C-P) + (S-C) can be read off each row.

The paper's C-P table is rendered from the contrasts with the generation-cap reruns merged (`stats` default)."""
from __future__ import annotations

from .latex import interval, master_delta, md_table, signed, to_csv

FIELDS = ["dataset", "metric", "child", "gap", "headline", "n", "a", "b", "delta_pp", "ci95_low", "ci95_high",
          "p", "p_holm", "p_holm_broad", "tost3", "ni5", "coverage_a", "coverage_b", "coverage_gap", "reading",
          "verdict"]

LABELS = {
    ("dynamath", "avg_acc"): "DynaMath", ("dynamath", "worst_acc"): r"\quad worst case", ("mathvista", "acc"): "MathVista",
    ("mmbench", "circular_acc"): "MMBench", ("mme", "perception"): "MME perception", ("mme", "cognition"): "MME cognition",
    ("mmmu", "acc"): "MMMU", ("mmstar", "acc"): "MMStar", ("scienceqa", "acc"): "ScienceQA", ("visulogic", "acc"): "VisuLogic",
    ("videomme", "acc"): "Video-MME short", ("tomato", "acc"): "TOMATO", ("timeblind", "I_Acc"): r"TimeBlind $I_{\mathrm{Acc}}$",
    ("videoreasonbench", "acc"): "VideoReasonBench", ("harmbench", "asr"): "HarmBench ASR",
    ("video_safetybench", "asr_harmful"): "VSB harmful ASR", ("video_safetybench", "asr_benign"): "VSB benign ASR",
    ("mcv_safetybench", "asr_explicit"): "MCV explicit ASR", ("mcv_safetybench", "asr_implicit"): "MCV implicit ASR",
    ("vmdt_fairness", "fairness"): "VMDT fairness", ("vmdt_fairness", "stereotype"): r"\quad stereotype",
    ("vmdt_fairness", "decision"): r"\quad decision", ("vmdt_fairness", "factual"): r"\quad factual",
    ("vmdt_privacy", "privacy"): "VMDT privacy", ("vmdt_privacy", "state"): r"\quad state",
    ("vmdt_privacy", "city"): r"\quad city", ("vmdt_privacy", "zipcode"): r"\quad zip code",
}
GROUPS = {"Image understanding": ["dynamath", "mathvista", "mmbench", "mme", "mmmu", "mmstar", "scienceqa", "visulogic"],
          "Video understanding": ["videomme", "tomato", "timeblind", "videoreasonbench"],
          "Safety": ["harmbench", "video_safetybench", "mcv_safetybench"],
          "Fairness and privacy": ["vmdt_fairness", "vmdt_privacy"]}
CAPTION = {
    "C-P": "the child offline ($C-P$)",
    "S-P": "the child in native streaming ($S-P$)",
}


def significant(r: dict, alpha: float = 0.05) -> bool:
    p = r.get("p_holm_broad", r.get("p_holm"))
    return p is not None and p < alpha


def metrics(rows: list[dict]) -> list[tuple[str, str]]:
    """(dataset, metric) in the order the contrasts were computed, which is the paper's order."""
    seen: list[tuple[str, str]] = []
    for r in rows:
        if (r["dataset"], r["metric"]) not in seen:
            seen.append((r["dataset"], r["metric"]))
    return seen


def md_cell(r: dict | None) -> str:
    if r is None:
        return "–"
    text = signed(r["delta_pp"])
    if significant(r):
        text = f"**{text}**"
    elif r.get("tost3"):
        text += " ≈"
    if r.get("coverage_gap") is not None and abs(r["coverage_gap"]) > 2:
        text += " †"
    return text


def tex_cell(r: dict | None, coverage_marks: bool = False) -> str:
    if r is None:
        return "--"
    number = master_delta(r["delta_pp"])
    marks = [m for m, on in ((r"\approx", r.get("tost3") and not significant(r)),
                             ("c", coverage_marks and r.get("coverage_gap") is not None and abs(r["coverage_gap"]) > 2))
             if on]
    return "$" + (rf"\mathbf{{{number}}}" if significant(r) else number) + \
        ("^{" + " ".join(marks) + "}" if marks else "") + "$"


def flat(r: dict) -> dict:
    ci = r.get("ci95_pp") or [None, None]
    return {**{k: r.get(k) for k in FIELDS}, "ci95_low": ci[0], "ci95_high": ci[1]}


def markdown(rows: list[dict], gap: str, children: list[str], names: dict, source: str = "contrasts.json") -> tuple[str, str]:
    """(markdown, csv) for one gap ('C-P' or 'S-P')."""
    index = {(r["dataset"], r["metric"], r["child"], r["gap"]): r for r in rows}
    order = metrics([r for r in rows if r["gap"] == gap])
    lead = "C" if gap == "C-P" else "S"
    lines = [f"# {gap}: every {'child offline' if gap == 'C-P' else 'child streaming'} vs parent offline contrast", "",
             f"Source: `{source}`. Δ in pp, + = the child is better (ASR and leakage: parent − child). "
             "**bold** = Holm p < 0.05 (main-text family), ≈ = equivalent within ±3 pp, † = coverage differs > 2 pp.",
             "", "## Grid", ""]
    grid = []
    for dataset, metric in order:
        any_row = index[next((dataset, metric, c, gap) for c in children if (dataset, metric, c, gap) in index)]
        name = f"{dataset} · {metric}" + ("" if any_row["headline"] else " (secondary)")
        grid.append([name] + [md_cell(index.get((dataset, metric, c, gap))) for c in children])
    lines += md_table(["benchmark · metric"] + [names.get(c, c) for c in children], grid)
    header = ["benchmark · metric", "child", "n", "P", lead, "Δ", "95% CI", "Holm p", "reading", "verdict"]
    if gap == "S-P":
        header[-2:-2] = ["C−P", "S−C"]
    body, records = [], []
    for dataset, metric in order:
        for child in children:
            r = index.get((dataset, metric, child, gap))
            if r is None:
                continue
            ci = r.get("ci95_pp") or [None, None]
            p = r.get("p_holm_broad", r.get("p_holm"))
            row = [f"{dataset} · {metric}", names.get(child, child), r["n"], f"{r['a']:.1f}" if r["a"] is not None else None,
                   f"{r['b']:.1f}" if r["b"] is not None else None, md_cell(r), interval(*ci),
                   None if p is None else f"{p:.3g}", r.get("reading"), r.get("verdict")]
            record = flat(r)
            if gap == "S-P":
                cp, sc = index.get((dataset, metric, child, "C-P")), index.get((dataset, metric, child, "S-C"))
                row[-2:-2] = [md_cell(cp), md_cell(sc)]
                record.update(cp_delta_pp=cp and cp["delta_pp"], sc_delta_pp=sc and sc["delta_pp"])
            body.append(row)
            records.append(record)
    lines += ["", "## Detail", ""] + md_table(header, body) + [""]
    extra = ["cp_delta_pp", "sc_delta_pp"] if gap == "S-P" else []
    return "\n".join(lines), to_csv(records, FIELDS + extra)


def latex(rows: list[dict], gap: str, children: list[str], names: dict, coverage_marks: bool = False) -> str:
    """One table* per gap: benchmark x child grid of deltas in points, grouped as the paper's evaluation dimensions."""
    index = {(r["dataset"], r["metric"], r["child"], r["gap"]): r for r in rows}
    order = metrics([r for r in rows if r["gap"] == gap])
    body = []
    for group, datasets in GROUPS.items():
        keys = [k for k in order if k[0] in datasets]
        if not keys:
            continue
        body += [r"\midrule", rf"\multicolumn{{{len(children) + 1}}}{{@{{}}l}}{{\textit{{{group}}}}} \\"]
        for key in keys:
            body.append(" & ".join([LABELS.get(key, f"{key[0]} {key[1]}")] +
                                   [tex_cell(index.get((*key, c, gap)), coverage_marks) for c in children]) + r" \\")
    stem = gap.replace("-", "").lower()
    coverage = (r" $^{c}$: response coverage differs from the parent's by more than 2 points." if coverage_marks else "")
    return "\n".join([
        "% Generated by `streamreplace tables` from contrasts.json; do not edit by hand.",
        r"\begin{table*}[t]", r"\centering",
        rf"\caption{{All differences between {CAPTION[gap]} and its offline parent, in points; $+$ means the child is better "
        r"(ASR and location leakage are reported as parent minus child). \textbf{Bold}: Holm-significant in the main-text "
        r"comparison family; $^{\approx}$: equivalent within $\pm3$ points (TOST)." + coverage +
        r" Indented rows are secondary metrics. Intervals and $p$-values are in the accompanying CSV.}",
        rf"\label{{tab:master-{stem}}}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l" + "r" * len(children) + "@{}}", r"\toprule",
        " & ".join(["Benchmark"] + [names.get(c, c) for c in children]) + r" \\",
        *body, r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
