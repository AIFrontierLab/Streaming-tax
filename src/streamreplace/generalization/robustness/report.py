"""`streamreplace robustness report`: the prompt-robustness analysis from the scored condition folders (the
prompt-robustness columns of Table 2, Supp. Tables eval_robustness and eval_instruction, the Sec. 4.3 sentences).

Pre-registered definitions (docs/generalization/robustness.md):
- item = a unit (make.DATASETS: a Video-MME question; an MMBench question, right only if every rotation is right).
  Errors and unparsed replies count as wrong (scoring.outcome); a unit whose edit was skipped leaves that condition.
- acc(m, c); Drop(m, c) = acc(m, clean) - acc(m, c); PDR(m, c) = 1 - acc(m, c) / acc(m, clean) (PromptRobust).
- dRob(child, c) = Drop_parent - Drop_child, per item (P_clean - P_c) - (C_clean - C_c): + = the child loses less,
  - = the child loses more. A family's value is the per-item mean over its conditions (distraction = stress and
  irrelevant).
- Primary endpoint, per child x family pooled over the benchmarks (bootstrap strata): mean, 95% stratified cluster
  bootstrap (clusters: Video-MME videos, MMBench questions), cluster sign-flip p, Holm over all children x families
  (6 x 3 = 18 tests). Statistics: the integer-index engine of analysis/stats_index.py (10,000 draws, seed 0), the
  engine these numbers were reported with. Per-benchmark family contrasts use the same functions on one benchmark
  (p uncorrected).
- Bounds and readouts (the paper's reanalysis): one-sided 95% lower bound and the Bonferroni-18 simultaneous bound of
  the same bootstrap, TOST at +-3 points (90% interval inside), and a second readout on the gold-option probability:
  the softmax of the first generated token's logits restricted to the item's option letters (info.letter_logits),
  the gold letter's share averaged over an MMBench question's rotations, contrasted like accuracy (Holm-18 apart).
  PDR-based dRob = PDR_parent - PDR_child (% of clean accuracy) on the same bootstrap draws.
- Diagnostics per (model, benchmark, condition): C->W and W->C against clean, answer change (a sample of the unit
  parsed to another answer), answered (every sample parsed), bare-letter replies, mean generated tokens.

Outputs (in <root>/report/ unless --out): ROBUSTNESS.md, results.json, items.csv (unit outcomes), primary.csv (the
pooled dRob table), contrasts.csv (pooled contrasts with bounds, both readouts; Table 2 reads the accuracy rows),
instruction.csv (instruction level, instruction.py)."""
from __future__ import annotations

import csv
from io import StringIO
import json
from pathlib import Path
import re
from typing import Callable

import numpy as np

from ...analysis.stats_index import bootstrap_ci, draws, holm, signflip_p, summary
from ...core.io import atomic_text
from .edits import skipped_units
from .make import DATASETS, Experiment, families, stem_conditions

BARE = re.compile(r"^\(?[A-Z]\)?[.:]?$")
TESTS = 18                       # the primary family: 6 children x 3 perturbation families (Bonferroni bound)
MARGIN = 3.0                     # TOST margin in points (the pre-registered resolution of the experiment)
ALPHAS = (0.05, 0.05 / TESTS)    # one-sided 95% lower bound and the Bonferroni-18 simultaneous bound


# ---------- unit outcomes ----------

def option_probs(logits: dict | None, letters) -> dict | None:
    """Softmax over the option letters present in the first-token logits; None without logits or < 2 letters."""
    if not logits:
        return None
    present = [letter for letter in letters if letter in logits]
    if len(present) < 2:
        return None
    x = np.array([logits[letter] for letter in present], dtype=float)
    e = np.exp(x - x.max())
    return dict(zip(present, e / e.sum()))


def package_rows(exp: Experiment) -> Callable[[str, str, str], dict]:
    """(condition, dataset, model) -> id -> the latest generation row joined with its score row."""
    from ...analysis.records import joined_rows

    def load(condition: str, dataset: str, model: str) -> dict:
        return joined_rows(exp.folder(condition), dataset, f"{model}.offline") or {}
    return load


class Tables:
    """Unit outcomes of every (condition, dataset, model) read once, and the units each condition skipped.

    units(c, d, m): unit -> {"cluster", "correct", "preds", "answered", "bare", "tokens"}, or None while any sample of
    the condition's subset is missing or unscored (errored rows count, as wrong). probs(c, d, m): unit -> {"p"}, the
    mean gold-option probability over the unit's samples, for complete tables; a unit with a row lacking letter logits
    or the gold letter is left out. rows: an optional row loader (verification scripts read other layouts)."""

    def __init__(self, exp: Experiment, rows: Callable[[str, str, str], dict] | None = None):
        self.exp = exp
        self._load = rows or package_rows(exp)
        self._rows: dict[tuple, dict] = {}
        self._units: dict[tuple, dict | None] = {}
        self._probs: dict[tuple, dict] = {}
        self._skips: dict[tuple, set] = {}
        self._clusters: dict[str, dict] = {}

    def units(self, condition: str, dataset: str, model: str) -> dict | None:
        k = (condition, dataset, model)
        if k not in self._units:
            self._read(*k)
        return self._units[k]

    def probs(self, condition: str, dataset: str, model: str) -> dict:
        k = (condition, dataset, model)
        if k not in self._units:
            self._read(*k)
        return self._probs.get(k, {})

    def rows(self, condition: str, dataset: str, model: str) -> dict:
        """id -> the latest row of the job (the replication check compares raw replies)."""
        k = (condition, dataset, model)
        if k not in self._rows:
            self._rows[k] = self._load(*k)
        return self._rows[k]

    def skips(self, condition: str, dataset: str) -> set[str]:
        k = (condition, dataset)
        if k not in self._skips:
            self._skips[k] = skipped_units(self.exp, condition, dataset)
        return self._skips[k]

    def cluster(self, key: tuple[str, str]) -> str:
        """The bootstrap cluster label of an item (dataset, unit): "<dataset>:<video or question>"."""
        dataset, unit = key
        if dataset not in self._clusters:
            self._clusters[dataset] = {u: f"{dataset}:{DATASETS[dataset].cluster(samples[0])}"
                                       for u, samples in self.exp.units(dataset).items()}
        return self._clusters[dataset][unit]

    def _read(self, condition: str, dataset: str, model: str) -> None:
        from ...benchmarks import registry
        from ...scoring.outcome import outcome
        d, bench = DATASETS[dataset], registry.get(dataset)
        samples = self.exp.samples(dataset, condition)
        rows = self.rows(condition, dataset, model)
        k = (condition, dataset, model)
        if any(s.id not in rows or (rows[s.id].get("correct") is None and not rows[s.id].get("error"))
               for s in samples):
            self._units[k] = None                          # not run or not scored yet
            return
        units: dict = {}
        for s in samples:
            row, u = rows[s.id], d.unit(s)
            entry = units.setdefault(u, {"cluster": f"{dataset}:{d.cluster(s)}", "correct": True, "preds": [],
                                         "answered": True, "bare": [], "tokens": []})
            prediction = row.get("judge_pred") if row.get("judge_pred") is not None else row.get("pred")
            entry["correct"] &= outcome(bench.JUDGE, row)
            entry["preds"].append(prediction)
            entry["answered"] &= prediction is not None
            entry["bare"].append(bool(BARE.match((row.get("raw") or "").strip())))
            if row.get("tokens") is not None:              # some adapters do not report token counts
                entry["tokens"].append(row["tokens"])
        self._units[k] = units
        gold, bad = {}, set()
        for s in samples:
            row, u = rows.get(s.id), d.unit(s)
            logits = ((row or {}).get("info") or {}).get("letter_logits")
            probs = option_probs(logits, list(s.choices))
            if row is None or probs is None or s.answer not in probs:
                bad.add(u)
                continue
            gold.setdefault(u, []).append(float(probs[s.answer]))
        self._probs[k] = {u: {"p": float(np.mean(v))} for u, v in gold.items() if u not in bad}


# ---------- contrasts ----------

def items(tables: Tables, datasets, conditions, model) -> list[tuple]:
    """(dataset, unit) present in clean and every condition for `model`, minus units any of the conditions skipped;
    [] while any of these tables is incomplete."""
    keys = None
    for dataset in datasets:
        units = [tables.units(c, dataset, model) for c in ["clean", *conditions]]
        if any(t is None for t in units):
            return []
        common = set.intersection(*(set(t) for t in units)) - set().union(*(tables.skips(c, dataset)
                                                                            for c in conditions))
        keys = (keys or set()) | {(dataset, u) for u in common}
    return sorted(keys or [])


def paired_keys(tables: Tables, datasets, conditions, parent, child) -> list[tuple]:
    return sorted(set(items(tables, datasets, conditions, parent)) & set(items(tables, datasets, conditions, child)))


def values(tables: Tables, kind: str, conditions, parent, child, keys) -> list[float]:
    """Per item Drop_parent - Drop_child averaged over the conditions (+ = the child loses less); kind "acc" reads
    the unit's correctness, "prob" its gold-option probability."""
    if kind == "acc":
        v = lambda m, c, k: float(tables.units(c, k[0], m)[k[1]]["correct"])
    else:
        v = lambda m, c, k: tables.probs(c, k[0], m)[k[1]]["p"]
    return [sum((v(parent, "clean", k) - v(parent, c, k)) - (v(child, "clean", k) - v(child, c, k))
                for c in conditions) / len(conditions) for k in keys]


def d_rob(tables: Tables, datasets, conditions, parent, child) -> dict | None:
    """Mean dRob in points with its 95% stratified cluster bootstrap interval and the cluster sign-flip p."""
    keys = paired_keys(tables, datasets, conditions, parent, child)
    if not keys:
        return None
    vals = values(tables, "acc", conditions, parent, child, keys)
    clusters = [tables.units("clean", k[0], parent)[k[1]]["cluster"] for k in keys]
    low, high = bootstrap_ci(vals, clusters, [k[0] for k in keys])
    return {"value": 100 * sum(vals) / len(vals), "low": 100 * low, "high": 100 * high,
            "p": signflip_p(vals, clusters), "n": len(keys), "keys": keys}


def prob_keys(tables: Tables, conditions, parent, child, keys) -> list[tuple]:
    """The accuracy items whose probability readout exists for both models in clean and every condition."""
    return [k for k in keys if all(k[1] in tables.probs(c, k[0], m) for c in ["clean", *conditions]
                                   for m in (parent, child))]


def bounds(tables: Tables, kind: str, conditions, parent, child, keys, margin: float | None = MARGIN) -> dict:
    """stats_index.summary of one readout: mean, 95% interval, sign-flip p, the one-sided 95% and Bonferroni-18
    lower bounds and, with a margin, TOST."""
    vals = values(tables, kind, conditions, parent, child, keys)
    return summary(vals, [tables.cluster(k) for k in keys], [k[0] for k in keys], alphas=ALPHAS, margin=margin)


def pdr_contrast(tables: Tables, conditions, parent, child, keys) -> dict:
    """PDR_parent - PDR_child in % of clean accuracy (+ = the child loses relatively less) with its 95% interval
    on the bootstrap draws of the accuracy contrast (a family's PDR is the mean of its conditions' PDR)."""
    acc = lambda m, c, k: float(tables.units(c, k[0], m)[k[1]]["correct"])
    columns = [[acc(m, c, k) for k in keys] for m in (parent, child) for c in ["clean", *conditions]]
    sums, _ = draws(np.column_stack(columns), [tables.cluster(k) for k in keys], [k[0] for k in keys])
    width = len(conditions) + 1

    def value(s):
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = [np.mean([1 - s[..., i * width + 1 + j] / s[..., i * width] for j in range(len(conditions))], axis=0)
                   for i in range(2)]
        return 100 * (rel[0] - rel[1])
    boot = value(sums)
    boot = boot[~np.isnan(boot)]
    point = float(value(np.column_stack(columns).sum(0)))
    return {"value": point, "low": float(np.percentile(boot, 2.5)), "high": float(np.percentile(boot, 97.5))}


def accuracy(units: dict, keys) -> float:
    return 100 * sum(units[u]["correct"] for u in keys) / len(keys)


def fmt_delta(t: dict | None, p: float | None = None) -> str:
    if t is None:
        return "–"
    shown = t["p"] if p is None else p
    return f"{t['value']:+.1f} [{t['low']:+.1f}, {t['high']:+.1f}] p={shown:.2g}"


def fmt_ci(value, low, high, digits: int = 1) -> str:
    return "–" if value is None else f"{value:+.{digits}f} [{low:+.{digits}f}, {high:+.{digits}f}]"


# ---------- the question-stem analysis ----------

def primary(exp: Experiment, tables: Tables) -> dict:
    """(child, family) -> the pooled contrasts: dRob (the saved table's statistics), the accuracy and probability
    readouts with bounds (Holm-18 within each readout) and the PDR contrast."""
    datasets, result = exp.datasets(), {}
    for pair, parent, child in exp.children:
        for family, conds in families(exp.spec).items():
            t = d_rob(tables, datasets, conds, parent, child)
            if t is None:
                result[(child, family)] = {"pair": pair, "parent": parent, "child": child, "family": family,
                                           "d_rob": None}
                continue
            keys, pkeys = t["keys"], prob_keys(tables, conds, parent, child, t["keys"])
            result[(child, family)] = {
                "pair": pair, "parent": parent, "child": child, "family": family, "d_rob": t,
                "acc": bounds(tables, "acc", conds, parent, child, keys),
                "prob": bounds(tables, "prob", conds, parent, child, pkeys),
                "pdr": pdr_contrast(tables, conds, parent, child, keys)}
    complete = {k: v for k, v in result.items() if v["d_rob"]}
    adjusted = holm({k: v["d_rob"]["p"] for k, v in complete.items()})
    for k, v in complete.items():
        v["d_rob"]["p_holm"] = adjusted[k]
    for kind in ("acc", "prob"):
        tested = {k: v[kind]["p"] for k, v in complete.items() if v[kind].get("n")}
        adjusted = holm(tested)
        for k in tested:
            complete[k][kind]["p_holm"] = adjusted[k]
    return result


def by_benchmark(exp: Experiment, tables: Tables) -> list[dict]:
    """Per child x benchmark: clean accuracies and each family's dRob on that benchmark (uncorrected p); the rows of
    Supp. Table eval_robustness (+ = the child loses less; rob_tax = -dRob)."""
    out = []
    for pair, parent, child in exp.children:
        for d in exp.datasets():
            clean_p, clean_c = tables.units("clean", d, parent), tables.units("clean", d, child)
            if clean_p is None or clean_c is None:
                continue
            entry = {"dataset": d, "pair": pair, "parent": parent, "child": child, "n_clean": len(clean_p),
                     "clean_parent": accuracy(clean_p, clean_p), "clean_child": accuracy(clean_c, clean_c)}
            for family, conds in families(exp.spec).items():
                t = d_rob(tables, [d], conds, parent, child)
                if t is None:
                    continue
                units = [u for _, u in t["keys"]]
                entry[family] = {"n": t["n"], "delta_rob_pp": t["value"], "delta_rob_ci95": [t["low"], t["high"]],
                                 "rob_tax_pp": -t["value"], "rob_tax_ci95": [-t["high"], -t["low"]],
                                 "p_uncorrected": t["p"], "matched_clean_parent": accuracy(clean_p, units),
                                 "matched_clean_child": accuracy(clean_c, units)}
            out.append(entry)
    return out


def per_condition(exp: Experiment, tables: Tables, dataset: str) -> dict:
    """Accuracy (Drop, PDR) per model and condition on the items present in clean and the condition, the per-condition
    dRob of every child, and the diagnostics."""
    conditions = [c for c in stem_conditions(exp.spec) if c != "clean"]
    accuracy_rows, contrasts, diagnostics = [], [], []
    for m in exp.models:
        clean = tables.units("clean", dataset, m)
        row = {"model": m, "clean": accuracy(clean, set(clean)) if clean else None, "conditions": {}}
        for c in conditions:
            table = tables.units(c, dataset, m)
            keys = set(clean) & set(table) - tables.skips(c, dataset) if clean and table else set()
            if keys:
                a0, a1 = accuracy(clean, keys), accuracy(table, keys)
                row["conditions"][c] = {"acc": a1, "clean": a0, "drop": a0 - a1, "pdr": 1 - a1 / a0 if a0 else None,
                                        "n": len(keys)}
        accuracy_rows.append(row)
    for pair, parent, child in exp.children:
        contrasts.append({"pair": pair, "child": child,
                          "conditions": {c: _public(d_rob(tables, [dataset], [c], parent, child)) for c in conditions}})
    for m in exp.models:
        clean = tables.units("clean", dataset, m)
        for c in stem_conditions(exp.spec):
            table = tables.units(c, dataset, m)
            if table is None or clean is None:
                continue
            keys = sorted(set(clean) & set(table) - tables.skips(c, dataset))
            if not keys:
                continue
            pct = lambda f: 100 * sum(f(u) for u in keys) / len(keys)
            bare = [b for u in keys for b in table[u]["bare"]]
            tokens = [t for u in keys for t in table[u]["tokens"]]
            diagnostics.append({
                "model": m, "condition": c,
                "c_to_w": pct(lambda u: clean[u]["correct"] and not table[u]["correct"]),
                "w_to_c": pct(lambda u: table[u]["correct"] and not clean[u]["correct"]),
                "changed": pct(lambda u: table[u]["preds"] != clean[u]["preds"]),
                "answered": pct(lambda u: table[u]["answered"]),
                "bare_letter": 100 * sum(bare) / len(bare), "tokens": sum(tokens) / len(tokens) if tokens else None})
    return {"accuracy": accuracy_rows, "d_rob": contrasts, "diagnostics": diagnostics}


def _public(t: dict | None) -> dict | None:
    return None if t is None else {k: v for k, v in t.items() if k != "keys"}


# ---------- files ----------

def csv_text(rows: list[dict], fields: list[str] | None = None) -> str:
    """CSV with the csv module's default dialect (the format the paper's tables were read from)."""
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields or list(dict.fromkeys(k for r in rows for k in r)))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def item_rows(exp: Experiment, tables: Tables) -> list[dict]:
    """items.csv: one row per (condition, dataset, model, unit) of every complete question-stem table."""
    rows = []
    for c in stem_conditions(exp.spec):
        for d in exp.datasets(c):
            for m in exp.models:
                table = tables.units(c, d, m)
                if table:
                    rows += [{"condition": c, "dataset": d, "model": m, "unit": u, "cluster": v["cluster"],
                              "correct": int(v["correct"]), "preds": "|".join(str(p) for p in v["preds"])}
                             for u, v in sorted(table.items())]
    return rows


def primary_rows(results: dict) -> list[dict]:
    """primary.csv: the pooled dRob per child x family (points, 3 decimals), Holm over the 18 tests."""
    return [{"pair": v["pair"], "parent": v["parent"], "child": v["child"], "family": v["family"],
             "d_rob_pp": round(t["value"], 3), "low": round(t["low"], 3), "high": round(t["high"], 3), "p": t["p"],
             "p_holm": t.get("p_holm"), "items": t["n"]}
            for v in results.values() if (t := v["d_rob"])]


def contrast_rows(results: dict) -> list[dict]:
    """contrasts.csv: both readouts of the pooled contrasts with their bounds, TOST and Holm-18 p."""
    return [{"pair": v["pair"], "parent": v["parent"], "child": v["child"], "family": v["family"], "readout": kind,
             **v[kind]} for v in results.values() if v["d_rob"] for kind in ("acc", "prob")]


def markdown(exp: Experiment, tables: Tables, results: dict, details: dict) -> list[str]:
    fams = families(exp.spec)
    conditions = [c for c in stem_conditions(exp.spec) if c != "clean"]
    datasets = exp.datasets()
    lines = ["# Prompt robustness (offline)", "",
             "Question stems perturbed per condition (configs/generalization/robustness.yaml, "
             "docs/generalization/robustness.md); items = units (Video-MME question, MMBench question with all "
             "rotations right). Errors / unparsed = wrong.",
             "ΔRob = Drop_parent − Drop_child in pp (Drop = acc_clean − acc_condition); **+ = the child loses less, − = "
             "the child loses more**. 95% stratified cluster bootstrap, cluster sign-flip p.", "",
             "## Completeness", "", "| condition | " + " | ".join(datasets) + " | skipped units |",
             "|---|" + "---|" * (len(datasets) + 1)]
    for c in stem_conditions(exp.spec):
        done = [sum(tables.units(c, d, m) is not None for m in exp.models) for d in datasets]
        lines.append(f"| {c} | " + " | ".join(f"{n}/{len(exp.models)} models" for n in done) + " | "
                     + ", ".join(f"{d} {len(tables.skips(c, d))}" for d in datasets) + " |")
    lines += ["", f"## Primary: ΔRob per child × family (pooled over {', '.join(datasets)}; Holm over "
              f"{sum(1 for v in results.values() if v['d_rob'])} tests, p shown Holm-adjusted)", "",
              "| pair | parent | child | " + " | ".join(f"{f} ({'+'.join(c)})" for f, c in fams.items()) + " |",
              "|---|---|---|" + "---|" * len(fams)]
    for pair, parent, child in exp.children:
        cells = [fmt_delta(results[(child, f)]["d_rob"], (results[(child, f)]["d_rob"] or {}).get("p_holm"))
                 for f in fams]
        lines.append(f"| {pair} | {parent} | {child} | " + " | ".join(cells) + " |")
    lines += ["", "## Bounds and the gold-option probability readout", "",
              f"LB95 = one-sided 95% lower bound, LB-sim = Bonferroni-{TESTS} simultaneous bound of the same bootstrap; "
              f"TOST at ±{MARGIN:g} pp; Holm over the {TESTS} tests of each readout. p_gold = the gold letter's share "
              "of the first-token softmax over the option letters (info.letter_logits), in points. ΔRob_PDR = "
              "PDR_parent − PDR_child (% of clean accuracy).", "",
              "| pair | child | family | items | ΔRob acc | LB95 | LB-sim | TOST | Holm p | ΔRob p_gold | LB95 | "
              "LB-sim | Holm p | ΔRob_PDR |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in results.values():
        if not v["d_rob"]:
            continue
        a, p, q = v["acc"], v["prob"], v["pdr"]
        lb = f"lower_{ALPHAS[1]:g}"
        prob_cells = ([fmt_ci(p["value"], p["low"], p["high"], 2), f"{p['lower_0.05']:+.2f}", f"{p[lb]:+.2f}",
                       f"{p['p_holm']:.2g}"] if p.get("n") else ["–"] * 4)
        lines.append(" | ".join([f"| {v['pair']}", v["child"], v["family"], str(a["n"]),
                                 fmt_ci(a["value"], a["low"], a["high"]), f"{a['lower_0.05']:+.2f}", f"{a[lb]:+.2f}",
                                 f"{'yes' if a['equivalent'] else 'no'} (p={a['tost_p']:.2g})", f"{a['p_holm']:.2g}",
                                 *prob_cells, fmt_ci(q["value"], q["low"], q["high"])]) + " |")
    for dataset in datasets:
        part = details[dataset]
        lines += ["", f"## {dataset}", "", "Accuracy (Drop vs clean; PDR) on the items present in clean and the "
                  "condition:", "", "| model | clean | " + " | ".join(conditions) + " |",
                  "|---|---|" + "---|" * len(conditions)]
        for row in part["accuracy"]:
            if row["clean"] is None:
                lines.append(f"| {row['model']} | incomplete |" + " |" * len(conditions))
                continue
            cells = []
            for c in conditions:
                x = row["conditions"].get(c)
                if x is None:
                    cells.append("–")
                elif x["clean"]:
                    cells.append(f"{x['acc']:.1f} ({x['drop']:+.1f}; PDR {x['pdr']:+.2f})")
                else:
                    cells.append(f"{x['acc']:.1f}")
            lines.append(f"| {row['model']} | {row['clean']:.1f} | " + " | ".join(cells) + " |")
        lines += ["", "ΔRob per condition:", "", "| pair | child | " + " | ".join(conditions) + " |",
                  "|---|---|" + "---|" * len(conditions)]
        for row in part["d_rob"]:
            lines.append(f"| {row['pair']} | {row['child']} | "
                         + " | ".join(fmt_delta(row["conditions"][c]) for c in conditions) + " |")
        lines += ["", "Diagnostics (C→W / W→C vs clean, answer change, answered, bare-letter replies, mean tokens):", "",
                  "| model | condition | C→W | W→C | changed | answered | bare letter | tokens |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in part["diagnostics"]:
            tokens = f"{r['tokens']:.1f}" if r["tokens"] is not None else "–"
            lines.append(f"| {r['model']} | {r['condition']} | {r['c_to_w']:.1f} | {r['w_to_c']:.1f} | "
                         f"{r['changed']:.1f} | {r['answered']:.1f} | {r['bare_letter']:.1f} | {tokens} |")
    return lines


def _jsonable(value):
    if isinstance(value, dict):
        return {(k if isinstance(k, str) else "/".join(map(str, k))): _jsonable(v) for k, v in value.items()
                if k != "keys"}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def write_report(exp: Experiment, out: Path | None = None, tables: Tables | None = None) -> dict:
    """Runs the whole analysis and writes the report folder; returns {name: path} of the files written."""
    from . import instruction                     # instruction.py builds on this module's tables and statistics
    out = Path(out) if out else Path(exp.root) / "report"
    tables = tables or Tables(exp)
    results = primary(exp, tables)
    details = {d: per_condition(exp, tables, d) for d in exp.datasets()}
    level = instruction.analyse(exp, tables)
    lines = markdown(exp, tables, results, details) + instruction.markdown(exp, level)
    files = {"report": out / "ROBUSTNESS.md", "results": out / "results.json", "items": out / "items.csv",
             "primary": out / "primary.csv", "contrasts": out / "contrasts.csv"}
    atomic_text(files["report"], "\n".join(lines) + "\n")
    document = {"definitions": {"delta_rob": "Drop_parent - Drop_child in points; + = the child loses less",
                                "rob_tax": "Drop_child - Drop_parent = -delta_rob",
                                "statistics": "integer-index stratified cluster bootstrap, 10000 draws, numpy seed 0; "
                                              "cluster sign-flip p; Holm over the 18 primary tests"},
                "primary": [_jsonable(v) for v in results.values()],
                "robustness_by_benchmark": _jsonable(by_benchmark(exp, tables)),
                "per_benchmark": _jsonable(details),
                "instruction_robustness": _jsonable(level["rows"]),
                "instruction_contrasts": _jsonable(level["contrasts"])}
    atomic_text(files["results"], json.dumps(document, ensure_ascii=False, indent=1) + "\n")
    if rows := primary_rows(results):
        atomic_text(files["primary"], csv_text(rows))
        atomic_text(files["contrasts"], csv_text(contrast_rows(results)))
    if rows := item_rows(exp, tables):
        atomic_text(files["items"], csv_text(rows))
    if level["contrasts"]:
        files["instruction"] = out / "instruction.csv"
        atomic_text(files["instruction"], csv_text(instruction.contrast_rows(level)))
    return {k: str(v) for k, v in files.items() if v.exists()}
