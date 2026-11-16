"""The three analysis steps behind `streamreplace report`, `streamreplace stats` and `streamreplace tables`.

  report(outputs)  per-job metrics and counts of every scored job in an outputs folder:
                   <outputs>/metrics.json, RESULTS.md, pairs.csv
  stats(outputs)   every P/C/S contrast of the analysis profile (configs/analysis.yaml), annotated with Holm families,
                   TOST / non-inferiority marks, verdicts and cell flags; by default with the generation-cap rerun
                   overlay: <out>/contrasts.json, contrasts.csv, GAPS.md (out = <outputs>/analysis)
  tables(outputs)  the paper's tables and figure data from those files (no statistics are computed here):
                   <out>/gaps.tex (Table 2), mcv_gaps.tex, gaps.json and gaps_figure.json (the gap figures' data),
                   eval_*.tex (+ .csv for the absolute scores), and the generalization tables from the
                   prompt-robustness and fine-tuning reports when they exist (out = <outputs>/tables)
  check_tables(out, expected)  the written tables against an expected tables file (paper/expected/tables.json: the
                   paper's printed cell strings), cell by cell

Every file is rewritten atomically; nothing under <outputs>/<dataset>/ is modified."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import platform

from ..core.io import atomic_text, write_json
from .records import Catalog
from .specs import Profile

CONTRAST_FIELDS = ["dataset", "metric", "family", "headline", "child", "gap", "a_job", "b_job", "n", "a", "b",
                   "delta_pp", "ci95_low", "ci95_high", "ci90_low", "ci90_high", "test", "p", "p_holm", "holm_family",
                   "p_holm_broad", "holm_broad_family", "tost3", "tost2", "ni2", "ni3", "ni5", "coverage_a",
                   "coverage_b", "coverage_gap", "reading", "verdict", "flags"]


def versions() -> dict:
    import numpy
    from .. import __version__
    return {"streamreplace": __version__, "numpy": numpy.__version__, "python": platform.python_version()}


# ---------- report ----------

def limited(outputs: Path, results: dict) -> bool:
    """True when any job of the folder was run with --limit (a smoke run)."""
    for r in results.values():
        meta = Path(outputs) / r["dataset"] / f"{r['job']}.meta.json"
        try:
            payload = json.loads(meta.read_text(encoding="utf-8")).get("payload") or {}
        except (OSError, ValueError):
            continue
        if payload.get("limit") is not None:
            return True
    return False


def report(outputs: str | Path, profile: Profile, catalog: Catalog, datasets: list[str] | None = None,
           generated: datetime | None = None) -> dict:
    """Writes metrics.json, RESULTS.md and pairs.csv into `outputs`; returns {name: path}."""
    from . import metrics
    from .render import results_md
    outputs = Path(outputs)
    results = metrics.collect(outputs, catalog, datasets)
    text, pairs_csv = results_md.render(results, profile, catalog, generated=generated,
                                        smoke=limited(outputs, results))
    files = {"metrics": outputs / "metrics.json", "results": outputs / "RESULTS.md", "pairs": outputs / "pairs.csv"}
    write_json(files["metrics"], {"versions": versions(), "jobs": metrics.to_json(results)})
    atomic_text(files["results"], text)
    atomic_text(files["pairs"], pairs_csv)
    return {k: str(v) for k, v in files.items()}


# ---------- stats ----------

def flat_row(r: dict) -> dict:
    ci95, ci90 = r.get("ci95_pp") or [None, None], r.get("ci90_pp") or [None, None]
    return {**{k: r.get(k) for k in CONTRAST_FIELDS}, "ci95_low": ci95[0], "ci95_high": ci95[1],
            "ci90_low": ci90[0], "ci90_high": ci90[1], "flags": "|".join(r.get("flags") or [])}


def _shown(path: Path | None, *bases: Path | None) -> str | None:
    """`path` relative to the first base that contains it (no machine-specific prefix in the outputs)."""
    if path is None:
        return None
    path = Path(path).resolve()
    for base in bases:
        if base is not None and path.is_relative_to(Path(base).resolve()):
            return str(path.relative_to(Path(base).resolve()))
    return path.name


def compute(outputs: str | Path, profile: Profile, catalog: Catalog, overlay: bool = True,
            rerun_ids: str | Path | None = None, rerun_rows: str | Path | None = None,
            flags: str | Path | None = None) -> dict:
    """The contrasts document (rows, summary, checks) of one outputs folder; see stats()."""
    from .arms import Arms, TruncationRerun
    from .contrasts import Contrasts, check_official, duration_split, summary
    from .flags import External, Flagger
    outputs = Path(outputs)
    rerun = TruncationRerun.from_profile(profile, outputs, rerun_ids, rerun_rows) if overlay else None
    arms = Arms(outputs, catalog, profile, rerun)
    contrasts = Contrasts(profile, arms, catalog)
    rows = contrasts.compute()
    if not rows:
        raise ValueError(f"no contrasts: {outputs} holds no scored parent/child jobs of the profile (run `score` "
                         "first, with --judge for judge-scored benchmarks)")
    external = flags if flags is not None else (profile.flags or {}).get("external")
    flagger = Flagger(profile, arms, catalog, External(profile.path(external) if external else None))
    flagger.annotate(rows)
    return {
        "profile": profile.name, "versions": versions(),
        "overlay": None if rerun is None else {"name": "truncation_rerun", "pairs": rerun.pairs,
                                               "datasets": rerun.datasets,
                                               "ids": _shown(rerun.ids_file, profile.root, outputs),
                                               "rows": _shown(rerun.root, outputs)},
        "engine": {"reps": profile.reps, "seed": profile.seed, "vmdt_draws": profile.vmdt_draws},
        "summary": summary(rows),
        "checks": {"official_mismatches": check_official(rows, arms, catalog, profile),
                   **({"uncounted": contrasts.uncounted} if contrasts.uncounted else {})},
        "unreported": contrasts.unreported,
        "rows": rows,
        "videomme_duration": duration_split(profile, arms, catalog),
        "merged_truncation": flagger.merged_truncation(),
    }


def stats(outputs: str | Path, profile: Profile, catalog: Catalog, out: str | Path | None = None,
          overlay: bool = True, rerun_ids: str | Path | None = None, rerun_rows: str | Path | None = None,
          flags: str | Path | None = None) -> tuple[dict, dict]:
    """Writes contrasts.json, contrasts.csv and GAPS.md; returns (the document, {name: path})."""
    from .render import gaps_md
    from .render.latex import to_csv
    doc = compute(outputs, profile, catalog, overlay, rerun_ids, rerun_rows, flags)
    out = Path(out) if out else Path(outputs) / "analysis" / ("" if overlay else "no-overlay")
    files = {"contrasts": out / "contrasts.json", "csv": out / "contrasts.csv", "gaps": out / "GAPS.md"}
    write_json(files["contrasts"], doc)
    atomic_text(files["csv"], to_csv([flat_row(r) for r in doc["rows"]], CONTRAST_FIELDS))
    title = None if overlay else ("# Paired contrasts, Holm, TOST / non-inferiority, Eq. 1 gaps and replacement "
                                  "verdicts (formal rows, no generation-cap reruns)")
    atomic_text(files["gaps"], gaps_md.render(doc["rows"], profile, doc["videomme_duration"], title))
    return doc, {k: str(v) for k, v in files.items()}


# ---------- tables ----------

def tables(outputs: str | Path, profile: Profile, out: str | Path | None = None,
           contrasts: str | Path | None = None, metrics: str | Path | None = None,
           robustness: str | Path | None = None, robustness_results: str | Path | None = None,
           finetune: str | Path | None = None, finetune_results: str | Path | None = None) -> dict:
    """Renders every table and figure data file whose inputs exist; returns {name: path} and the skipped ones.

    contrasts: contrasts.json of `stats` (default <outputs>/analysis/contrasts.json); metrics: metrics.json of `report`
    (default <outputs>/metrics.json). Optional inputs of the generalization experiments: robustness = the
    prompt-robustness contrasts.csv (default <outputs>/robustness/report/contrasts.csv), robustness_results = its
    results.json (same folder), finetune = the fine-tuning cells.csv (default <outputs>/finetune/ft_tables/cells.csv),
    finetune_results = its results.json (same folder)."""
    from .render import eval_tables, fig_gaps, gaps_json, table_gaps
    from .render.latex import to_csv
    outputs = Path(outputs)
    out = Path(out) if out else outputs / "tables"
    contrasts = Path(contrasts) if contrasts else outputs / "analysis" / "contrasts.json"
    metrics = Path(metrics) if metrics else outputs / "metrics.json"
    robustness = Path(robustness) if robustness else outputs / "robustness" / "report" / "contrasts.csv"
    robustness_results = (Path(robustness_results) if robustness_results
                          else robustness.parent / "results.json")
    finetune = Path(finetune) if finetune else outputs / "finetune" / "ft_tables" / "cells.csv"
    finetune_results = Path(finetune_results) if finetune_results else finetune.parent / "results.json"
    files, skipped = {}, {}

    def write(name: str, text: str) -> None:
        atomic_text(out / name, text)
        files[name] = str(out / name)

    children, names = profile.children, {c: profile.name_of(c) for c in profile.models}
    unreported: list[dict] = []
    if contrasts.is_file():
        document = json.loads(contrasts.read_text(encoding="utf-8"))
        rows, unreported = document["rows"], document.get("unreported") or []
        table = table_gaps.build(rows, children, robustness, finetune, profile.display.get("finetune_ids"))
        gap_names = {c: profile.name_of(c, "gap_table_names") for c in children}
        write("gaps.tex", table_gaps.latex(table, gap_names))
        write("mcv_gaps.tex", table_gaps.mcv_latex(table, gap_names))
        write("gaps.csv", to_csv(table_gaps.flat(table), table_gaps.CSV_FIELDS))
        gaps = gaps_json.build(rows, children, robustness=robustness if robustness.is_file() else None,
                               finetune=finetune if finetune.is_file() else None)
        write("gaps.json", json.dumps(gaps, indent=1, ensure_ascii=False) + "\n")
        figure = fig_gaps.build(rows, children)
        write("gaps_figure.json", json.dumps(figure, indent=1, ensure_ascii=False) + "\n")
        write("gaps_figure.csv", to_csv(fig_gaps.flat(figure, names), fig_gaps.CSV_FIELDS))
        if not robustness.is_file():
            skipped["gaps.tex robustness columns"] = f"{robustness} not found (streamreplace robustness report)"
        if not finetune.is_file():
            skipped["gaps.tex fine-tuning columns"] = f"{finetune} not found (streamreplace finetune report)"
    else:
        skipped["gaps, figure data"] = f"{contrasts} not found (streamreplace stats)"
    if metrics.is_file():
        rows = json.loads(metrics.read_text(encoding="utf-8"))["jobs"]
        for name, (text, table_rows, fields) in eval_tables.build(rows, profile, unreported).items():
            write(f"{name}.tex", text)
            write(f"{name}.csv", to_csv(table_rows, fields))
    else:
        skipped["eval_image, eval_video, eval_safety, eval_vmdt"] = f"{metrics} not found (streamreplace report)"
    if robustness_results.is_file():
        document = json.loads(robustness_results.read_text(encoding="utf-8"))
        for name, text in eval_tables.robustness(document, profile).items():
            write(f"{name}.tex", text)
    else:
        skipped["eval_robustness, eval_instruction"] = f"{robustness_results} not found (streamreplace robustness report)"
    if finetune_results.is_file():
        document = json.loads(finetune_results.read_text(encoding="utf-8"))
        for name, text in eval_tables.finetuning(document, profile).items():
            write(f"{name}.tex", text)
    else:
        skipped["eval_finetuning, eval_finetuning_gains"] = (f"{finetune_results} not found (streamreplace finetune "
                                                             "report)")
    return {"files": files, "skipped": skipped}


def _columns(text: str) -> list[str]:
    """The last header line of a generated LaTeX table, split into column names."""
    head = text.split(r"\toprule", 1)[-1].split(r"\midrule", 1)[0].strip().splitlines()
    return [c.strip() for c in head[-1].rstrip("\\ ").split(" & ")] if head else []


def check_tables(out: str | Path, expected: str | Path) -> dict:
    """Compares the tables written to `out` (<name>.tex, by tables()) with an expected tables file:
    {"tables": {name: [[cell, ...], ...]}}, the format of paper/expected/tables.json (the paper's printed cell
    strings). Returns {"cells": number compared, "differences": ["<table> / <row> / <column>: got vs expected", ...],
    "missing": [tables not written]}."""
    from .render.latex import tex_cells
    out = Path(out)
    data = json.loads(Path(expected).read_text(encoding="utf-8"))
    cells, differences, missing = 0, [], []
    for name, rows in (data.get("tables") or {}).items():
        path = out / f"{name}.tex"
        if not path.is_file():
            missing.append(name)
            continue
        text = path.read_text(encoding="utf-8")
        got, columns = tex_cells(text), _columns(text)
        if len(got) != len(rows):
            differences.append(f"{name}: {len(got)} rows, expected {len(rows)}")
        for mine, theirs in zip(got, rows):
            if len(mine) != len(theirs):
                differences.append(f"{name} / {theirs[0]}: {len(mine)} cells, expected {len(theirs)}")
                continue
            for j, (a, b) in enumerate(zip(mine, theirs)):
                cells += 1
                if a != b:
                    column = columns[j] if j < len(columns) else str(j)
                    differences.append(f"{name} / {theirs[0]} / {column}: {a} vs {b}")
    return {"cells": cells, "differences": differences, "missing": missing}
