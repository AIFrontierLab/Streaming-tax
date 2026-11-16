"""Stream++'s recovery tables (Fig. 6 of the paper, Supp. D), on one item set.

Per child x headline stream metric (Video-MME, TOMATO, TimeBlind I_Acc, VSB harmful / benign ASR, MCV explicit /
implicit ASR, VMDT fairness, VMDT privacy), all arms restricted to the same items, paired cluster-bootstrap contrasts
from the analysis engine (analysis/contrasts.py; delta in points, + = the later arm better, ASR and leakage + = safer):
  native = S - P, vsS = S_plug - S (the improvement over native streaming), vsP = S_plug - P (what remains against
  the offline parent, with the paper's replacement verdict), loss = S_fix - P, ckpt = C - P, proto = S_fix - C,
  effect = S_plug - S_fix, gate = (S_fix + gate) - S_fix, prs = (S_fix + PRS) - S_fix;
  fire = % of the metric's items on which the complete plugin's gate fired.
VMDT fairness is also read on the items S_plug answered (the gate's refusals left out), with the refusal rate per
scenario. A child that answers none of a set's requests under native streaming gets no ASR or privacy score there
(silence alone would give the best value): its native and vsS contrasts read "no reply" (analysis/contrasts.py,
`no_reply`).

P, C and S are the formal jobs; S_fix and S_fix + PRS the shadow arms <child>@sfix.stream and
<child>@sfix_prs.stream; the refusal the scored fixed_refusal.offline job. Both arms must be shadow arms: Recovery
refuses an arm whose configured gate is not shadow and rows a gate acted on (a live gate's refusal, finish "guard";
a record or live gate's log), since such rows already hold the gate's decision and would enter S_fix, the PRS
contribution and S_plug with it."""
from __future__ import annotations

from pathlib import Path

from ..analysis.contrasts import Contrasts, coverage, no_reply, scope
from ..analysis.records import Catalog, Job, load_job
from ..analysis.stats import equivalent, noninferior
from . import posthoc
from .recipe import PluginConfig

CONTRASTS = (("native", "P", "S"), ("vsS", "S", "S_plug"), ("vsP", "P", "S_plug"), ("loss", "P", "S_fix"),
             ("ckpt", "P", "C"), ("proto", "C", "S_fix"), ("effect", "S_fix", "S_plug"), ("gate", "S_fix", "gate"),
             ("prs", "S_fix", "prs"))
EMPTY = {"a": None, "b": None, "delta": None, "ci95": None, "p": None, "coverage_a": None, "coverage_b": None,
         "ni3": None, "tost3": None, "verdict": "no items"}
SCENARIOS = ("decision_making", "stereotype", "factual_accuracy")
FIGURE = {"S_P": "native", "Splug_P": "vsP", "Splug_S": "vsS", "Sfix_P": "loss", "C_P": "ckpt",
          "Splug_Sfix": "effect", "gate_Sfix": "gate", "PRS_Sfix": "prs"}


def gate_acted(job: Job) -> list[str]:
    """Ids of the job's rows a gate acted on: the live gate's fixed refusal (finish "guard") or a gate log of another
    mode than shadow (live, record). Shadow rows log mode "shadow" (or, without a mode, shadow: true)."""
    out = []
    for i, record in job.rows.items():
        guard = ((record.info or {}).get("plugin") or {}).get("guard") or {}
        if record.finish == "guard" or guard.get("mode") not in (None, "shadow"):
            out.append(i)
    return out


class Recovery:
    """The recovery tables of one item set. formal: outputs folder of the P / C / S jobs; plugin: outputs folder of
    the plugin arms and the fixed-refusal job; artefacts: child -> plugin.artefacts.Artefact (thresholds); arms: the
    S_fix and S_fix + PRS arms, both with the gate in shadow mode."""

    def __init__(self, profile, catalog: Catalog, formal: Path, plugin: Path, items: dict[str, list[str]],
                 children: list[str], artefacts: dict, config: PluginConfig, arms: tuple[str, str] = ("sfix", "sfix_prs")):
        for name in arms:
            gate = config.arm(name).gate
            if gate != "shadow":
                raise ValueError(f"plugin report assembles S_plug from shadow arms (the gate logs and never acts); "
                                 f"arm {name!r} has gate {gate!r} in the plugin configuration")
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
        if no_reply(metric, a, ids) or no_reply(metric, b, ids):
            return {"n": len(ids), **EMPTY, "verdict": "no reply"}
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

    def thresholds(self) -> dict:
        return {c: self.artefacts[c].tau() for c in self.children}

    def cells(self) -> list[dict]:
        out = []
        for metric in self.metrics():
            smp = self.catalog.samples(metric.dataset)
            for child in self.children:
                jobs = self.jobs_of(metric.dataset, child)
                if jobs is not None:
                    out.append(self.cell(metric, child, jobs, smp))
        return out

    def cell(self, metric, child: str, jobs: dict, smp: dict) -> dict:
        want = set(self.items[metric.dataset])
        tau = self.artefacts[child].tau()
        fire_fix = posthoc.fired(jobs["fix"], smp, tau)          # each arm by its own logged projection
        fire = posthoc.fired(jobs["fixprs"], smp, tau)
        arms = {"P": jobs["P"], "C": jobs["C"], "S": jobs["S"], "S_fix": jobs["fix"],
                "gate": posthoc.mix(jobs["fix"], jobs["ref"], fire_fix, "gate"),
                "prs": jobs["fixprs"], "S_plug": posthoc.mix(jobs["fixprs"], jobs["ref"], fire, "plug")}
        in_scope = set(scope(metric, jobs["fix"], self.catalog))
        cell = {"child": child, "dataset": metric.dataset, "metric": metric.name, "n": len(want)}
        for name, a, b in CONTRASTS:
            cell[name] = self.contrast(metric, arms[a], arms[b])
        cell["fire"] = 100 * len(fire & in_scope) / max(1, len(in_scope))
        if metric.dataset == "vmdt_fairness":            # also read on the items S_plug answered
            keep = sorted(want - fire)
            cell["answered"] = {name: self.contrast(metric, posthoc.restrict(arms[a], keep),
                                                    posthoc.restrict(arms[b], keep))
                                for name, a, b in (("loss", "P", "S_fix"), ("effect", "S_fix", "S_plug"),
                                                   ("vsP", "P", "S_plug"))}
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

    def compute(self) -> dict:
        cells = self.cells()
        return {"thresholds": self.thresholds(), "cells": cells, "means": means(cells),
                "refusal_credit": self.refusal_credit()}


def means(cells: list[dict]) -> list[dict]:
    """Fig. 6: per metric, the unweighted mean over the children with a native streaming score of S - P and
    S_plug - P, and their difference (the mean change from native streaming)."""
    out = []
    for key in dict.fromkeys((c["dataset"], c["metric"]) for c in cells):
        rows = [c for c in cells if (c["dataset"], c["metric"]) == key
                and c["native"]["delta"] is not None and c["vsP"]["delta"] is not None]
        if rows:
            s_p = sum(c["native"]["delta"] for c in rows) / len(rows)
            plug_p = sum(c["vsP"]["delta"] for c in rows) / len(rows)
            out.append({"dataset": key[0], "metric": key[1], "children": [c["child"] for c in rows],
                        "S_P": s_p, "Splug_P": plug_p, "change": plug_p - s_p})
    return out


def figure_data(result: dict) -> dict:
    """The figures' numbers: per cell the deltas of FIGURE with their 95% intervals and the fire rate (Supp. per-child
    figures), and the per-metric means (Fig. 6)."""
    cells = []
    for c in result["cells"]:
        row = {"child": c["child"], "dataset": c["dataset"], "metric": c["metric"], "n": c["n"], "fire": c["fire"]}
        for key, name in FIGURE.items():
            x = c[name]
            row[key] = {"delta": x["delta"], "ci": x["ci95"], "a": x["a"], "b": x["b"], "n": x["n"]}
        cells.append(row)
    return {"cells": cells, "means": result["means"]}


# ---------- REPORT.md ----------

def _fmt(c: dict | None) -> str:
    if c is None or c.get("delta") is None:
        return "–"
    if c.get("ci95") is None:
        return f"{c['delta']:+.1f}"
    return f"{c['delta']:+.1f} [{c['ci95'][0]:+.1f}, {c['ci95'][1]:+.1f}]"


def markdown(result: dict, names: dict | None = None, title: str = "Stream++ recovery") -> str:
    names = names or {}
    name = lambda c: names.get(c, c)                                     # noqa: E731
    lines = [f"# {title}", "", "Δ in points with paired cluster-bootstrap 95% intervals; + = the later arm better "
             "(ASR and leakage: + = safer). S_fix = Recall + Respond; S_plug = S_fix + Refuse (PRS and the gate). "
             "– = no native streaming score (the child answers none of the set's requests).", "",
             "## Means over the children with a native score (Fig. 6)", "",
             "| metric | children | S − P | S_plug − P | change |", "|---|---|---|---|---|"]
    for m in result["means"]:
        lines.append(f"| {m['dataset']} {m['metric']} | {len(m['children'])} | {m['S_P']:+.1f} | "
                     f"{m['Splug_P']:+.1f} | {m['change']:+.1f} |")
    lines += ["", "## Per child", "",
              "| child | metric | n | S − P | S_plug − S | S_plug − P | verdict | S_fix − P | C − P | S_plug − S_fix | "
              "gate adds | PRS adds | fire % |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in result["cells"]:
        label = f"{c['dataset']} {c['metric']}"
        lines.append(f"| {name(c['child'])} | {label} | {c['n']} | {_fmt(c['native'])} | {_fmt(c['vsS'])} | "
                     f"{_fmt(c['vsP'])} | {c['vsP'].get('verdict')} | {_fmt(c['loss'])} | {_fmt(c['ckpt'])} | "
                     f"{_fmt(c['effect'])} | {_fmt(c['gate'])} | {_fmt(c['prs'])} | {c['fire']:.1f} |")
    fair = [c for c in result["cells"] if "answered" in c]
    if fair:
        lines += ["", "VMDT fairness on the items S_plug answered (the official score counts a refusal as 0):", "",
                  "| child | gate refusal % (all: decision / stereotype / factual) | S_fix − P | S_plug − S_fix | "
                  "S_plug − P |", "|---|---|---|---|---|"]
        for c in fair:
            a, r = c["answered"], c["refusal"]
            lines.append(f"| {name(c['child'])} | {r['all']:.1f}: {r['decision_making']:.1f} / "
                         f"{r['stereotype']:.1f} / {r['factual_accuracy']:.1f} | {_fmt(a['loss'])} | "
                         f"{_fmt(a['effect'])} | {_fmt(a['vsP'])} |")
    lines += ["", "## Thresholds", "", "| child | closed | free |", "|---|---|---|"]
    for child, tau in result["thresholds"].items():
        lines.append(f"| {name(child)} | {tau['closed']:+.4f} | {tau['free']:+.4f} |")
    lines += ["", "## The fixed refusal as scored by each benchmark's pipeline", "",
              "| set | rows | correct % | answered % |", "|---|---|---|---|"]
    for d, v in result.get("refusal_credit", {}).items():
        lines.append(f"| {d} | {v['n']} | {v['correct']:.1f} | {v['answered']:.1f} |")
    return "\n".join(lines) + "\n"
