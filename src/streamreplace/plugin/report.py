"""The plugin's recovery tables (Fig. plugin of the paper and its supplement table), on one item set.

Per child x headline stream metric (Video-MME, TOMATO, TimeBlind I_Acc, VSB harmful / benign ASR, MCV explicit /
implicit ASR, VMDT fairness, VMDT privacy), all arms restricted to the same items, paired cluster-bootstrap contrasts
from the analysis engine (analysis/contrasts.py; delta in points, + = the later arm better, ASR and leakage + = safer):
  loss = S_fix - P, ckpt = C - P, proto = S_fix - C, effect = S+ - S_fix, vsP = S+ - P (with the paper's
  replacement verdict), gate = (S_fix + gate) - S_fix, prs = (S_fix + PRS) - S_fix, native = S - P;
  fire = % of the metric's items on which the S+ arm's gate fired;
  recovered = effect / -loss where the loss is significant (upper 95% bound < 0); otherwise ni2 = non-inferiority
  of S+ to S_fix at -2 points (lower 95% bound > -2).
VMDT fairness is also read on the items S+ answered (the gate's refusals left out), with the refusal rate per scenario.
Threshold variants: higher (the paper), rule, old (artefacts.Artefact.tau). P, C and S are the formal jobs; S_fix
and S_fix + PRS the shadow arms <child>@sfix.stream and <child>@sfix_prs.stream; the refusal the scored
fixed_refusal.offline job. Both arms must be shadow arms: Recovery refuses an arm whose configured gate is not shadow
and rows a gate acted on (a live gate's refusal, finish "guard"; a record or live gate's log), since such rows already
hold the gate's decision and would enter S_fix, the PRS contribution and S+ with it."""
from __future__ import annotations

from pathlib import Path

from ..analysis.contrasts import Contrasts, coverage, scope
from ..analysis.records import Catalog, Job, load_job
from ..analysis.stats import equivalent, noninferior
from . import posthoc
from .recipe import PluginConfig

CONTRASTS = (("loss", "P", "S_fix"), ("ckpt", "P", "C"), ("proto", "C", "S_fix"), ("effect", "S_fix", "S+"),
             ("vsP", "P", "S+"), ("gate", "S_fix", "gate"), ("prs", "S_fix", "prs"), ("native", "P", "S"))
EMPTY = {"a": None, "b": None, "delta": None, "ci95": None, "p": None, "coverage_a": None, "coverage_b": None,
         "ni3": None, "tost3": None, "verdict": "no items"}
SCENARIOS = ("decision_making", "stereotype", "factual_accuracy")
FIGURE = {"S_P": "native", "Sfix_P": "loss", "C_P": "ckpt", "Splus_P": "vsP", "Splus_Sfix": "effect"}


def gate_acted(job: Job) -> list[str]:
    """Ids of the job's rows a gate acted on: the live gate's fixed refusal (finish "guard") or a gate log of another
    mode than shadow (live, record). Shadow rows log mode "shadow" (or, without a mode, shadow: true)."""
    out = []
    for i, record in job.rows.items():
        guard = ((record.info or {}).get("plugin") or {}).get("guard") or {}
        if record.finish == "guard" or guard.get("mode") not in (None, "shadow"):
            out.append(i)
    return out


def recovered(cell: dict) -> None:
    """recovered share where the loss is significant (upper 95% bound < 0); else non-inferiority of S+ to S_fix at
    -2 points."""
    loss, eff = cell["loss"], cell["effect"]
    sig = loss.get("ci95") is not None and loss["ci95"][1] < 0
    cell["recovered"] = eff["delta"] / -loss["delta"] if sig and eff.get("delta") is not None else None
    cell["ni2"] = None if sig or eff.get("ci95") is None else eff["ci95"][0] > -2


class Recovery:
    """The recovery tables of one item set. formal: outputs folder of the P / C / S jobs; plugin: outputs folder of
    the plugin arms and the fixed-refusal job; artefacts: child -> plugin.artefacts.Artefact (thresholds); arms: the
    S_fix and S_fix + PRS arms, both with the gate in shadow mode."""

    def __init__(self, profile, catalog: Catalog, formal: Path, plugin: Path, items: dict[str, list[str]],
                 children: list[str], artefacts: dict, config: PluginConfig, arms: tuple[str, str] = ("sfix", "sfix_prs")):
        for name in arms:
            gate = config.arm(name).gate
            if gate != "shadow":
                raise ValueError(f"plugin report assembles S+ from shadow arms (the gate logs and never acts); arm "
                                 f"{name!r} has gate {gate!r} in the plugin configuration")
        self.profile, self.catalog = profile, catalog
        self.formal, self.plugin, self.items = Path(formal), Path(plugin), items
        self.children, self.artefacts, self.config, self.arms = list(children), artefacts, config, arms
        self.engine = Contrasts(profile, None, catalog)
        self.problems: list[str] = []            # missing jobs or rows: the tables are incomplete
        self.notes: list[str] = []               # cells without an interval (reported as such)
        self._jobs: dict = {}

    # ---------- arms ----------

    def metrics(self) -> list:
        return [m for m in self.profile.metrics if m.headline and "stream" in m.modes and m.dataset in self.items]

    def jobs_of(self, dataset: str, child: str) -> dict | None:
        """P, C, S, the two shadow arms and the refusal of one (dataset, child), restricted to the items; None (and a
        problem noted) when a job is absent or lacks counted rows for some items."""
        key = (dataset, child)
        if key not in self._jobs:
            parent = self.profile.parents[child]
            fix, fixprs = self.arms
            spec = {"P": (self.formal, f"{parent}.offline"), "C": (self.formal, f"{child}.offline"),
                    "S": (self.formal, f"{child}.stream"), "fix": (self.plugin, f"{child}@{fix}.stream"),
                    "fixprs": (self.plugin, f"{child}@{fixprs}.stream"), "ref": (self.plugin, posthoc.REFUSAL_JOB)}
            jobs = {k: load_job(root, dataset, name, self.catalog) for k, (root, name) in spec.items()}
            for k in ("fix", "fixprs"):
                acted = gate_acted(jobs[k]) if jobs[k] is not None else []
                if acted:
                    raise ValueError(f"{dataset}/{spec[k][1]}: {len(acted)} rows were made by a live or record gate "
                                     f"(e.g. {acted[0]}); plugin report reads shadow arms only")
            want = set(self.items[dataset])
            absent = [k for k, v in jobs.items() if v is None]
            short = {k: len(want - set(v.rows)) for k, v in jobs.items() if v is not None and want - set(v.rows)}
            if absent or short:
                self.problems.append(f"{child} {dataset}: absent {absent}, missing / unscored rows {short}")
                self._jobs[key] = None
            else:
                jobs = {k: posthoc.restrict(v, want) for k, v in jobs.items()}
                jobs["ref"] = posthoc.refusal_for(dataset, jobs["ref"], self.config.unanswered_on_fire)
                self._jobs[key] = jobs
        return self._jobs[key]

    # ---------- one contrast ----------

    def contrast(self, metric, a: Job, b: Job) -> dict:
        """delta = b - a in points with the 95% interval, p, coverage, NI at -3, TOST at +-3 and the paper's
        replacement verdict (replaceable: lower bound > -5 and coverage within 2 points of a)."""
        ids = sorted(set(scope(metric, a, self.catalog)) & set(scope(metric, b, self.catalog)))
        bench = self.catalog.bench(metric.dataset)
        scenarios = getattr(bench, "SCENARIOS", None) if metric.kind == "vmdt" else None
        smp = self.catalog.samples(metric.dataset)
        if not ids or (scenarios and {smp[i].metadata.get("scenario") for i in ids} < set(scenarios)):
            return {"n": len(ids), **EMPTY}           # nothing (or a VMDT scenario without items) to contrast
        try:
            r = self.engine.pair(metric, a, b)
        except (IndexError, ZeroDivisionError, ValueError) as exc:
            if metric.kind != "vmdt":
                raise
            self.notes.append(f"{metric.dataset} {metric.name} on {len(ids)} items: no interval "
                              f"({type(exc).__name__})")
            return {"n": len(ids), **EMPTY, "verdict": "too few items"}
        ci95, ci90 = r.get("ci95_pp"), r.get("ci90_pp")
        cov_a, cov_b = coverage(metric, a, ids), coverage(metric, b, ids)
        gap = None if cov_a is None or cov_b is None else cov_b - cov_a
        ni5 = noninferior(ci95, 5)
        verdict = ("replaceable" if ni5 and abs(gap or 0) <= 2 else "non-inferior, coverage differs" if ni5
                   else "inferior" if ci95 and ci95[1] < -5 else "inconclusive")
        return {"n": len(ids), "a": r["a"], "b": r["b"], "delta": r.get("delta_pp", r["delta"]), "ci95": ci95,
                "p": r.get("p"), "coverage_a": cov_a, "coverage_b": cov_b, "ni3": noninferior(ci95, 3),
                "tost3": equivalent(ci90, 3), "verdict": verdict}

    # ---------- tables ----------

    def thresholds(self, variants) -> dict:
        return {c: {v: self.artefacts[c].tau(v) for v in variants} for c in self.children}

    def cells(self, variants=("higher", "rule", "old")) -> list[dict]:
        out = []
        for variant in variants:
            for metric in self.metrics():
                smp = self.catalog.samples(metric.dataset)
                for child in self.children:
                    jobs = self.jobs_of(metric.dataset, child)
                    if jobs is None:
                        continue
                    out.append(self.cell(metric, child, variant, jobs, smp))
        return out

    def cell(self, metric, child: str, variant: str, jobs: dict, smp: dict) -> dict:
        want = set(self.items[metric.dataset])
        tau = self.artefacts[child].tau(variant)
        fire_fix = posthoc.fired(jobs["fix"], smp, tau)          # each arm by its own logged projection
        fire = posthoc.fired(jobs["fixprs"], smp, tau)
        arms = {"P": jobs["P"], "C": jobs["C"], "S": jobs["S"], "S_fix": jobs["fix"],
                "gate": posthoc.mix(jobs["fix"], jobs["ref"], fire_fix, f"{variant}-gate"),
                "prs": jobs["fixprs"], "S+": posthoc.mix(jobs["fixprs"], jobs["ref"], fire, f"{variant}-splus")}
        in_scope = set(scope(metric, jobs["fix"], self.catalog))
        cell = {"variant": variant, "child": child, "dataset": metric.dataset, "metric": metric.name, "n": len(want)}
        for name, a, b in CONTRASTS:
            cell[name] = self.contrast(metric, arms[a], arms[b])
        cell["fire"] = 100 * len(fire & in_scope) / max(1, len(in_scope))
        recovered(cell)
        if metric.dataset == "vmdt_fairness":            # also read on the items S+ answered
            keep = sorted(want - fire)
            answered = {name: self.contrast(metric, posthoc.restrict(arms[a], keep), posthoc.restrict(arms[b], keep))
                        for name, a, b in (("loss", "P", "S_fix"), ("effect", "S_fix", "S+"), ("vsP", "P", "S+"))}
            recovered(answered)
            cell["answered"] = answered
            by = {sc: {i for i in want if smp[i].metadata["scenario"] == sc} for sc in SCENARIOS}
            cell["refusal"] = {"all": 100 * len(fire) / len(want),
                               **{sc: 100 * len(fire & ids) / max(1, len(ids)) for sc, ids in by.items()}}
        return cell

    def refusal_credit(self) -> dict:
        """The fixed refusal's own official score per set (how each benchmark's pipeline reads it)."""
        out = {}
        for d, ids in self.items.items():
            job = load_job(self.plugin, d, posthoc.REFUSAL_JOB, self.catalog)
            rows = [job.rows[i] for i in ids if job is not None and i in job.rows]
            if rows:
                out[d] = {"n": len(rows), "correct": 100 * sum(r.correct for r in rows) / len(rows),
                          "answered": 100 * sum(r.answered for r in rows) / len(rows)}
        return out

    def compute(self, variants=("higher", "rule", "old")) -> dict:
        return {"thresholds": self.thresholds(variants), "cells": self.cells(variants),
                "refusal_credit": self.refusal_credit()}


def figure_data(result: dict, variant: str = "higher") -> dict:
    """The paper figure's numbers (one variant): per cell the S - P, S_fix - P, C - P, S+ - P and S+ - S_fix deltas
    with their 95% intervals, and the fire rate."""
    cells = []
    for c in result["cells"]:
        if c["variant"] != variant:
            continue
        row = {"child": c["child"], "dataset": c["dataset"], "metric": c["metric"], "n": c["n"], "fire": c["fire"]}
        for key, name in FIGURE.items():
            x = c[name]
            row[key] = {"delta": x["delta"], "ci": x["ci95"], "a": x["a"], "b": x["b"], "n": x["n"]}
        cells.append(row)
    return {"variant": variant, "cells": cells}


# ---------- REPORT.md ----------

def _fmt(c: dict | None) -> str:
    return "–" if c is None or c.get("ci95") is None else f"{c['delta']:+.1f} [{c['ci95'][0]:+.1f}, {c['ci95'][1]:+.1f}]"


def markdown(result: dict, names: dict | None = None, title: str = "Plugin recovery") -> str:
    names = names or {}
    name = lambda c: names.get(c, c)                                     # noqa: E731
    lines = [f"# {title}", "", "Δ in points with paired cluster-bootstrap 95% intervals; + = the later arm better "
             "(ASR and leakage: + = safer). recovered = (S⁺ − S_fix) / (P − S_fix) where the loss is significant; "
             "NI@−2 = S⁺ non-inferior to S_fix at −2 points otherwise.", "", "## Thresholds", "",
             "| child | variant | closed | free |", "|---|---|---|---|"]
    for child, per in result["thresholds"].items():
        for variant, tau in per.items():
            lines.append(f"| {name(child)} | {variant} | {tau['closed']:+.4f} | {tau['free']:+.4f} |")
    for variant in dict.fromkeys(c["variant"] for c in result["cells"]):
        lines += ["", f"## Recovery, threshold {variant}", "",
                  "| child | metric | n | loss S_fix−P | C−P | S_fix−C | S⁺−S_fix | recovered | S⁺−P | verdict S⁺ | "
                  "NI@−2 | gate adds | PRS adds | fire % |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in (c for c in result["cells"] if c["variant"] == variant):
            rec = "–" if c["recovered"] is None else f"{100 * c['recovered']:.0f}%"
            ni = "–" if c["ni2"] is None else ("y" if c["ni2"] else "n")
            label = f"{c['dataset']} {c['metric']}" + (" (official: refusal = 0.0)" if "answered" in c else "")
            lines.append(f"| {name(c['child'])} | {label} | {c['n']} | {_fmt(c['loss'])} | {_fmt(c['ckpt'])} | "
                         f"{_fmt(c['proto'])} | {_fmt(c['effect'])} | {rec} | {_fmt(c['vsP'])} | "
                         f"{c['vsP'].get('verdict')} | {ni} | {_fmt(c['gate'])} | {_fmt(c['prs'])} | {c['fire']:.0f} |")
        fair = [c for c in result["cells"] if c["variant"] == variant and "answered" in c]
        if fair:
            lines += ["", f"VMDT fairness on the items S⁺ answered, threshold {variant}:", "",
                      "| child | gate refusal % (all: decision / stereotype / factual) | loss S_fix−P | S⁺−S_fix | "
                      "recovered | S⁺−P | verdict S⁺ | NI@−2 |", "|---|---|---|---|---|---|---|---|"]
            for c in fair:
                a, r = c["answered"], c["refusal"]
                rec = "–" if a["recovered"] is None else f"{100 * a['recovered']:.0f}%"
                ni = "–" if a["ni2"] is None else ("y" if a["ni2"] else "n")
                lines.append(f"| {name(c['child'])} | {r['all']:.0f}: {r['decision_making']:.0f} / "
                             f"{r['stereotype']:.0f} / {r['factual_accuracy']:.0f} | {_fmt(a['loss'])} | "
                             f"{_fmt(a['effect'])} | {rec} | {_fmt(a['vsP'])} | {a['vsP'].get('verdict')} | {ni} |")
    lines += ["", "## The fixed refusal as scored by each benchmark's pipeline", "",
              "| set | rows | correct % | answered % |", "|---|---|---|---|"]
    for d, v in result.get("refusal_credit", {}).items():
        lines.append(f"| {d} | {v['n']} | {v['correct']:.1f} | {v['answered']:.1f} |")
    return "\n".join(lines) + "\n"
