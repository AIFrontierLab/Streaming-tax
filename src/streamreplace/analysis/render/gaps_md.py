"""GAPS.md: every P/C/S contrast of the paper in readable tables, written by `streamreplace stats` next to
contrasts.json.

Sections: the Eq. 1 gap table (P, C, S and the three gaps per child on the video and trust headline metrics and
VideoReasonBench, with the stream's response coverage), the image C-P contrasts with TOST at +-3 / +-2, every video
and trust contrast with its test, Holm family and main-text ("broad") Holm p, the replacement verdicts, the Video-MME
S-C split by video duration, and the summary counts quoted in the text. Cells read 'delta* [95% CI]' in points, with
* = Holm p < 0.05 within the row's declared family (the 'reading' column uses the same family)."""
from __future__ import annotations

from ..contrasts import sig, summary
from ..specs import GAPS
from .latex import interval, md_table

IMAGE_FAMILIES = ("image-headline", "image-secondary")
VIDEO_FAMILIES = ("video", "vrb")
# How the summary names a video metric and a broad Holm family (others: "<dataset> <metric>", the family's name).
VIDEO_LABELS = {("videomme", "acc"): "Video-MME", ("tomato", "acc"): "TOMATO", ("timeblind", "I_Acc"): "TimeBlind I_Acc"}
BROAD_LABELS = {"vmdt": "VMDT"}


def _cell(r: dict | None, digits: int = 1) -> str:
    if r is None:
        return "–"
    mark = "*" if sig(r) else ""
    return f"{r['delta_pp']:+.{digits}f}{mark} {interval(*r['ci95_pp'], digits)}"


def _value(v, digits=1):
    return "–" if v is None else f"{v:.{digits}f}"


def _video_family(rows: list[dict]) -> str:
    """'Video-MME, TOMATO, TimeBlind I_Acc x 6' for the paper's roster: the family's metrics x its children."""
    if not rows:
        return "none"
    metrics = dict.fromkeys((r["dataset"], r["metric"]) for r in rows)
    return (", ".join(VIDEO_LABELS.get(m, f"{m[0]} {m[1]}") for m in metrics)
            + f" x {len({r['child'] for r in rows})}")


def _broad_families(rows: list[dict]) -> str:
    """'safety parent 54, safety S-C 24, VMDT parent 96, VMDT S-C 48' for the paper's roster: each main-text Holm
    family of the trust rows with its number of tests, in order of appearance."""
    sizes = {}
    for r in rows:
        sizes.setdefault(r.get("holm_broad_family"), r.get("holm_broad_m"))
    parts = []
    for family, m in sizes.items():
        name, _, kind = str(family).partition("/")
        parts.append(f"{BROAD_LABELS.get(name, name)}{' ' + kind if kind else ''} {m}")
    return ", ".join(parts) or "none"


def summary_lines(rows: list[dict]) -> list[str]:
    s = summary(rows)
    residual = s["eq1_max_residual"]                      # None without a metric that has all three gaps
    lines = [f"- Eq. 1 additivity: max |(S − P) − (C − P) − (S − C)| = {residual:.2g} over all three-gap rows."
             if residual is not None else "- Eq. 1 additivity: no metric with all three gaps.",
             f"- Image headline C − P: {s['image_tost3']}/{s['image_headline']} equivalent at ±3 pp, "
             f"{s['image_tost2']}/{s['image_headline']} at ±2 pp; Holm-significant declines {s['image_holm_decline']}, "
             f"increases {s['image_holm_increase']}."]
    for gap in GAPS:
        v = s[f"video_{gap}"]
        family = _video_family([r for r in rows if r["family"] == "video" and r["gap"] == gap])
        lines.append(f"- Video {gap} ({family}): {v['holm_decline']} Holm-significant "
                     f"declines, {v['holm_increase']} increases, {v['tost3']} equivalent at ±3 pp, of {v['tests']}.")
    for mode, counts in s["replacement"].items():
        lines.append(f"- Replacement ({mode} vs P, headline metrics): " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    t = s["trust"]
    lines.append(f"- Trust: {t['holm_significant']}/{t['tests']} contrasts Holm-significant ({t['raw_significant']} at raw "
                 f"p < 0.05); lost under Holm: " + ("; ".join(s["trust_lost_by_holm"]) or "none"))
    trust = [r for r in rows if r["family"] not in IMAGE_FAMILIES + VIDEO_FAMILIES]
    lines.append(f"- Trust, broad families ({_broad_families(trust)}): "
                 f"{s['trust_broad']['holm_significant']}/{t['tests']} Holm-significant; additionally lost: "
                 + ("; ".join(s["trust_lost_by_broad_holm_only"]) or "none"))
    return lines


def render(rows: list[dict], profile, durations: list[dict] | None = None, title: str | None = None) -> str:
    """The GAPS.md text of annotated contrast rows (contrasts.Contrasts.compute())."""
    short = {c: profile.name_of(c) for c in profile.children}
    image = [m for m in profile.metrics if m.family in IMAGE_FAMILIES]
    video = [m for m in profile.metrics if m.family in VIDEO_FAMILIES]
    trust = [m for m in profile.metrics if m not in image and m not in video]
    idx = {(r["dataset"], r["metric"], r["child"], r["gap"]): r for r in rows}
    lines = [title or "# Paired contrasts, Holm, TOST / non-inferiority, Eq. 1 gaps and replacement verdicts", "",
             "Δ in percentage points, **+ = the child (or its stream run) is better**: accuracy / fairness / privacy "
             "Δ = b − a; ASR and VMDT location accuracies Δ = a − b. Intervals: paired cluster bootstrap 95%; * = "
             "Holm-adjusted p < 0.05 within the row's family. MME in pp of the maximum (perception /20, cognition /8). "
             "Items: full formal sets; silence and errors wrong (understanding), errors safe / silence judged (safety).", ""]
    # Eq. 1
    lines += ["## Eq. 1 gap table: S − P = (C − P) + (S − C)", ""]
    header = ["child", "benchmark", "metric", "P", "C", "S", "C − P", "S − C", "S − P", "S cover"]
    body = []
    for metric in video + trust:
        if not metric.headline and metric.family != "vrb":
            continue
        for child in profile.children:
            cp, sc, sp = (idx.get((metric.dataset, metric.name, child, g)) for g in GAPS)
            if cp is None:
                continue
            body.append([short[child], metric.dataset, metric.name, _value(cp["a"]), _value(cp["b"]),
                         _value(sp["b"]) if sp else "–", _cell(cp), _cell(sc), _cell(sp),
                         _value(sp["coverage_b"]) if sp else "–"])
    lines += md_table(header, body)
    # image
    headline = sum(r["family"] == "image-headline" for r in rows)
    lines += ["", "## Image understanding, C − P (offline)", "",
              f"TOST: 90% interval inside ±3 (±2) pp. Holm over the {headline} headline contrasts; DynaMath worst, MME "
              "cognition and VisuLogic form a secondary family.", ""]
    body = []
    for metric in image:
        for child in profile.children:
            r = idx.get((metric.dataset, metric.name, child, "C-P"))
            if r:
                body.append([short[child], f"{metric.dataset} {metric.name}", _value(r["a"]), _value(r["b"]), _cell(r),
                             f"{r['p']:.3g}", f"{r['p_holm']:.3g}", "y" if r["tost3"] else "–", "y" if r["tost2"] else "–",
                             r["reading"]])
    lines += md_table(["child", "metric", "P", "C", "Δ [95% CI]", "p", "Holm p", "TOST ±3", "TOST ±2", "reading"], body)
    # video and trust detail
    lines += ["", "## Video and trust contrasts (all gaps)", ""]
    body = []
    for metric in video + trust:
        for child in profile.children:
            for gap in GAPS:
                r = idx.get((metric.dataset, metric.name, child, gap))
                if r:
                    body.append([short[child], f"{metric.dataset} {metric.name}", gap, r["n"], _value(r["a"]),
                                 _value(r["b"]), _cell(r), r["test"], f"{r['p']:.3g}", f"{r['p_holm']:.3g}",
                                 f"{r['holm_family']} ({r['holm_m']})", f"{r['p_holm_broad']:.3g}",
                                 "y" if r["tost3"] else "–", r["reading"]])
    lines += md_table(["child", "metric", "gap", "n", "a", "b", "Δ [95% CI]", "test", "p", "Holm p", "family (m)",
                       "Holm p broad", "TOST ±3", "reading"], body)
    # replacement
    lines += ["", "## Replacement verdicts (child vs parent, per mode; margin −5 pp)", "",
              "replaceable = lower 95% bound > −5 pp and response coverage within 2 pp of the parent's; inferior = "
              "upper bound < −5 pp; NI@−2 / −3 = the same rule's bound at those margins (coverage not required).", ""]
    body = []
    for metric in profile.metrics:
        if not metric.headline:
            continue
        for child in profile.children:
            for gap in ("C-P", "S-P"):
                r = idx.get((metric.dataset, metric.name, child, gap))
                if r:
                    body.append([short[child], f"{metric.dataset} {metric.name}", "C" if gap == "C-P" else "S", _cell(r),
                                 _value(r["coverage_a"]), _value(r["coverage_b"]), "y" if r["ni2"] else "–",
                                 "y" if r["ni3"] else "–", "y" if r["ni5"] else "–", r["verdict"]])
    lines += md_table(["child", "metric", "mode", "Δ [95% CI]", "P cover", "child cover", "NI@−2", "NI@−3", "NI@−5",
                       "verdict"], body)
    if durations:
        lines += ["", "## Video-MME S − C by video duration (descriptive; video-cluster bootstrap)", ""]
        lines += md_table(["child", "duration", "n", "C", "S", "S − C [95% CI]"],
                          [[short[d["child"]], d["bin"], d["n"], _value(d["a"]), _value(d["b"]), _cell(d)] for d in durations])
    lines += ["", "## Summary", "", *summary_lines(rows)]
    return "\n".join(lines) + "\n"
