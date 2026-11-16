"""The report of a `qa` experiment: every arm's score on the experiment's items and the paired contrasts between
arms, computed by the analysis engine of the main results (analysis/contrasts.py), so a control is read with the
same metric definitions, analysis units and bootstrap as the P/C/S gaps.

Per child and metric of the experiment's benchmarks:
  arms       label -> {job, n, value (the benchmark's own metric), coverage (%), silent (rows without a reply)}
  contrasts  "B - A" -> {n, a, b, delta, ci95, p} on the items both arms counted (delta in points, + = B better; ASR
             and leakage are oriented so that + = safer)
An arm whose job is missing or incomplete on the items is listed under `problems` and left out. Optional sections
(experiment `report`): `duration` (Video-MME contrasts by video duration: under 60 s, 60-90 s, 90-121 s, and the
difference between the longest and the shortest group) and `timing` (question timing: when the first reply came and
how first and last replies score).

Writes <outputs>/experiments/<name>/report.json and REPORT.md."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..analysis.arms import Restrict
from ..analysis.contrasts import Contrasts, coverage, no_reply, official_metrics, scope
from ..analysis.records import Catalog, Job, load_job
from . import Experiment, parent_of

DURATION_BINS = ((0, 60), (60, 90), (90, 121))          # seconds, as the duration split of the main analysis


def restrict(job: Job, ids) -> Job:
    return job.with_overlays(Restrict(frozenset(ids), label=f"items-{len(ids)}"))


class Report:
    def __init__(self, exp: Experiment, config, profile, catalog: Catalog, formal: Path):
        self.exp, self.config, self.profile, self.catalog, self.formal = exp, config, profile, catalog, Path(formal)
        self.engine = Contrasts(profile, None, catalog)
        self.problems: list[str] = []
        self._jobs: dict = {}

    # ---------- arms ----------

    def metrics(self, dataset: str) -> list:
        wanted = self.exp.spec.get("metrics")
        return [m for m in self.profile.metrics if m.dataset == dataset
                and (m.name in wanted if wanted else m.headline)]

    def job(self, label: str, dataset: str, child: str) -> Job | None:
        """The arm's job on `dataset` for `child`, restricted to the experiment's items; None (and a problem noted)
        when it is missing or lacks rows."""
        arm = self.exp.arms[label]
        name = arm.job_name(child, parent_of(self.config, child))
        if name is None:
            return None
        root = self.formal if arm.formal else self.exp.folder(arm)
        key = (str(root), dataset, name)
        if key not in self._jobs:
            job = load_job(root, dataset, name, self.catalog)
            items = self.exp.items()
            if job is not None and items is not None:
                want = [i for i in items[dataset] if i in self.catalog.samples(dataset)]
                if job.mode == "stream":
                    want = [i for i in want if self.catalog.samples(dataset)[i].video]
                missing = len(set(want) - set(job.rows))
                if missing:
                    self.problems.append(f"{dataset}/{name} ({root}): {missing} of {len(want)} items missing or "
                                         "unscored")
                job = restrict(job, want)
            elif job is None:
                self.problems.append(f"{dataset}/{name}: no rows in {root}")
            self._jobs[key] = job
        return self._jobs[key]

    def value(self, metric, job: Job) -> dict:
        ids = scope(metric, job, self.catalog)
        try:
            value = official_metrics(job, self.catalog).get(metric.official_key)
        except (IndexError, ZeroDivisionError, ValueError):       # a VMDT component without a rated item
            value = None
        if no_reply(metric, job, ids):
            value = None
        return {"job": job.job, "n": len(ids), "value": value, "coverage": coverage(metric, job, ids),
                "silent": sum(job.rows[i].silent for i in ids)}

    def contrast(self, metric, a: Job, b: Job, ids=None) -> dict:
        if ids is not None:
            a, b = restrict(a, ids), restrict(b, ids)
        common = sorted(set(scope(metric, a, self.catalog)) & set(scope(metric, b, self.catalog)))
        if not common or no_reply(metric, a, common) or no_reply(metric, b, common):
            return {"n": len(common), "a": None, "b": None, "delta": None, "ci95": None, "p": None}
        try:
            r = self.engine.pair(metric, a, b)
        except (IndexError, ZeroDivisionError, ValueError):
            if metric.kind != "vmdt":
                raise
            return {"n": len(common), "a": None, "b": None, "delta": None, "ci95": None, "p": None}
        return {"n": r["n"], "a": r["a"], "b": r["b"], "delta": r.get("delta_pp", r["delta"]),
                "ci95": r.get("ci95_pp"), "p": r.get("p")}

    # ---------- tables ----------

    def cells(self, children=None) -> list[dict]:
        out = []
        extras = set(self.exp.spec.get("report") or [])
        used = {label for pair in self.exp.contrasts for label in pair} | set(self.exp.spec.get("arms") or {})
        for dataset in self.exp.benchmarks():
            for metric in self.metrics(dataset):
                for child in (children or self.exp.children):
                    jobs = {label: self.job(label, dataset, child) for label in used}
                    jobs = {label: job for label, job in jobs.items() if job is not None}
                    cell = {"child": child, "dataset": dataset, "metric": metric.name,
                            "arms": {label: self.value(metric, job) for label, job in jobs.items()},
                            "contrasts": {f"{b} - {a}": self.contrast(metric, jobs[a], jobs[b])
                                          for a, b in self.exp.contrasts if a in jobs and b in jobs}}
                    if "duration" in extras and dataset == "videomme":
                        cell["duration"] = self.by_duration(metric, child, jobs)
                    if "timing" in extras:
                        cell["timing"] = {label: timing(job) for label, job in jobs.items()
                                          if any(r.info.get("question_at") for r in job.rows.values())}
                    out.append(cell)
        return out

    def durations(self, child: str) -> dict:
        """Seconds per Video-MME item, from the formal offline rows of the child (info.duration)."""
        job = load_job(self.formal, "videomme", f"{child}.offline", self.catalog)
        return {} if job is None else {i: r.info.get("duration") for i, r in job.rows.items()}

    def by_duration(self, metric, child: str, jobs: dict) -> list[dict]:
        """The experiment's contrasts within each duration bin, and the longest bin minus the shortest (a bootstrap
        over videos of the difference of the two bins' paired differences)."""
        seconds = self.durations(child)
        rows = []
        for a, b in self.exp.contrasts:
            if a not in jobs or b not in jobs:
                continue
            common = sorted(set(jobs[a].rows) & set(jobs[b].rows))
            bins = [[i for i in common if seconds.get(i) is not None and lo <= seconds[i] < hi]
                    for lo, hi in DURATION_BINS]
            for (lo, hi), ids in zip(DURATION_BINS, bins):
                if ids:
                    rows.append({"contrast": f"{b} - {a}", "bin": f"[{lo}, {hi}) s",
                                 **self.contrast(metric, jobs[a], jobs[b], ids)})
            if bins[0] and bins[-1]:
                rows.append({"contrast": f"{b} - {a}", "bin": "long - short",
                             **self.bin_difference(jobs[a], jobs[b], bins[0], bins[-1])})
        return rows

    def bin_difference(self, a: Job, b: Job, short: list[str], long: list[str]) -> dict:
        """(b - a on the long bin) - (b - a on the short bin) in points, with a 95% interval from resampling the
        videos of each bin independently (the bins hold different videos)."""
        samples = self.catalog.samples(a.dataset)
        rng = np.random.default_rng(self.profile.seed)

        def clusters(ids):
            by: dict = {}
            for i in ids:
                by.setdefault(samples[i].video, []).append(100.0 * (b.rows[i].correct - a.rows[i].correct))
            return [np.array(v) for v in by.values()]

        def draws(groups):
            sums, counts = np.array([g.sum() for g in groups]), np.array([len(g) for g in groups])
            pick = rng.integers(0, len(groups), size=(self.profile.reps, len(groups)))
            return sums[pick].sum(axis=1) / counts[pick].sum(axis=1)

        s, g = clusters(short), clusters(long)
        point = np.concatenate(g).mean() - np.concatenate(s).mean()
        diff = draws(g) - draws(s)
        return {"n": len(short) + len(long), "delta": float(point),
                "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}

    def compute(self, children=None) -> dict:
        return {"experiment": self.exp.name, "title": self.exp.spec.get("title"), "paper": self.exp.spec.get("paper"),
                "cells": self.cells(children), "problems": self.problems}


def timing(job: Job) -> dict:
    """Question timing: how soon the first reply came (as a share of the video) and how the first and the last reply
    score among the questions the child answered on its own (no fallback request at the end)."""
    rows = [r for r in job.rows.values() if not r.error]
    own = [r for r in rows if not r.info.get("fallback") and r.info.get("first_reply_t") is not None]
    share = [r.info["first_reply_t"] / r.info["duration"] for r in own if r.info.get("duration")]
    return {"n": len(rows), "answered_unprompted": len(own),
            "first_reply_median_share": float(np.median(share)) if share else None,
            "replies_median": float(np.median([r.info.get("n_replies", 1) for r in own])) if own else None,
            "multi_reply": sum((r.info.get("n_replies") or 1) > 1 for r in own)}


# ---------- REPORT.md ----------

def _num(value, signed: bool = False) -> str:
    if value is None:
        return "–"
    return f"{value:+.1f}" if signed else f"{value:.1f}"


def _delta(c: dict) -> str:
    if c.get("delta") is None:
        return "–"
    if c.get("ci95") is None:
        return f"{c['delta']:+.1f}"
    return f"{c['delta']:+.1f} [{c['ci95'][0]:+.1f}, {c['ci95'][1]:+.1f}]"


def markdown(result: dict, names: dict | None = None) -> str:
    names = names or {}
    lines = [f"# {result.get('title') or result['experiment']}", ""]
    if result.get("paper"):
        lines += [f"Paper: {result['paper']}.", ""]
    lines += ["Scores are the benchmarks' own metrics on the experiment's items; Δ in points with paired "
              "cluster-bootstrap 95% intervals, + = the first arm better (ASR and leakage: + = safer).", ""]
    cells = result["cells"]
    labels = list(dict.fromkeys(label for c in cells for label in c["arms"]))
    contrasts = list(dict.fromkeys(name for c in cells for name in c["contrasts"]))
    lines += ["| child | metric | n | " + " | ".join(labels + contrasts) + " |",
              "|---|---|---|" + "---|" * (len(labels) + len(contrasts))]
    for c in cells:
        n = max((a["n"] for a in c["arms"].values()), default=0)
        row = [names.get(c["child"], c["child"]), f"{c['dataset']} {c['metric']}", str(n)]
        row += [_num(c["arms"][label]["value"]) if label in c["arms"] else "" for label in labels]
        row += [_delta(c["contrasts"][name]) if name in c["contrasts"] else "" for name in contrasts]
        lines.append("| " + " | ".join(row) + " |")
    silent = [(c, label, a) for c in cells for label, a in c["arms"].items() if a["silent"]]
    if silent:
        lines += ["", "Rows without a reply (scored as silence):", "", "| child | metric | arm | silent | n |",
                  "|---|---|---|---|---|"]
        lines += [f"| {names.get(c['child'], c['child'])} | {c['dataset']} {c['metric']} | {label} | {a['silent']} | "
                  f"{a['n']} |" for c, label, a in silent]
    if any("duration" in c for c in cells):
        lines += ["", "## By video duration", "", "| child | contrast | duration | n | Δ |", "|---|---|---|---|---|"]
        for c in cells:
            for row in c.get("duration", []):
                lines.append(f"| {names.get(c['child'], c['child'])} | {row['contrast']} | {row['bin']} | {row['n']} | "
                             f"{_delta(row)} |")
    if any(c.get("timing") for c in cells):
        lines += ["", "## Reply timing", "", "| child | arm | answered without a request at the end | first reply at "
                  "(median share of the video) | replies (median) | more than one reply |", "|---|---|---|---|---|---|"]
        for c in cells:
            for label, t in c.get("timing", {}).items():
                share = "–" if t["first_reply_median_share"] is None else f"{100 * t['first_reply_median_share']:.0f}%"
                lines.append(f"| {names.get(c['child'], c['child'])} | {label} | {t['answered_unprompted']} / {t['n']} "
                             f"| {share} | {_num(t['replies_median'])} | {t['multi_reply']} |")
    if result.get("problems"):
        lines += ["", "## Incomplete inputs", ""] + [f"- {p}" for p in result["problems"]]
    return "\n".join(lines) + "\n"
