"""Instruction-level prompt robustness (Supp. Table eval_instruction; the Sec. 4.3 sentence on instruction
perturbations): the question-stem perturbations applied to the benchmark's official answer instruction instead.

Edits. A condition with `target: instruction` perturbs MMBench's single answer instruction "Answer with the option's
letter from the given choices directly." (the stem, options and hint stay byte-identical). Its `twin` is the
question-stem condition it is compared with on the same items:
    typo        the same DeepWordBug operations (perturb.typo) on the instruction
    choice      one of a fixed list of hand-written rewrites that keep the demand (the letter, directly), picked by
                int(sha256(seed), 16) % len(texts); the instruction-level counterpart of the paraphrase condition
    irrelevant  an irrelevant.txt sentence sharing no word of 4+ letters with the stem, options or instruction,
                appended to the instruction
The paper ran these on a nested, stratified 100-unit subset of the 300 MMBench questions (373 rows) and on all 300
(`seed_as`: the 100 earlier units keep exactly the same edits). Video-MME states its answer format twice (head and
tail), so it has no single instruction sentence and takes no instruction-level condition.

Analysis (per group of conditions run on the same items; the 100-unit and the 300-unit runs are separate groups,
each with its own Holm family):
- items: present in clean, the condition and its twin, minus units either skipped (the twin para condition skipped 5
  of the 100 units, leaving 95);
- per model: accuracy under the twin and the condition, level effect = Drop_instruction - Drop_stem
  (+ = the instruction edit hurts more), averaged over models as well;
- dRob_instruction = Drop_parent - Drop_child (+ = the child loses less), with the twin's dRob on the same items and
  DiD = dRob_instruction - dRob_stem; Holm over children x conditions on dRob_instruction; 95% stratified cluster
  bootstrap, cluster sign-flip p (the integer-index engine of analysis/stats_index.py);
- diagnostics (C->W, W->C, answer change, answered, bare letter) and, for a `seed_as` extension, the share of the
  earlier run's rows whose raw reply is identical and dRob on the earlier vs the new units (descriptive);
- the readouts with bounds for every instruction condition: accuracy and gold-option probability dRob with the
  one-sided 95% and Bonferroni-18 lower bounds, Holm over all instruction probability tests."""
from __future__ import annotations

from .make import DATASETS, Experiment, instruction_conditions, key


def instruction_span(dataset: str) -> str:
    """The benchmark's single official answer-instruction sentence (make.DATASETS; MMBench: its POST_PROMPT without
    the leading newline)."""
    spec = DATASETS[dataset]
    if spec.instruction is None:
        raise ValueError(f"{dataset}: instruction-level edits need a single instruction sentence (MMBench has one; "
                         "Video-MME states its answer format twice)")
    return spec.instruction()


def choice(texts: list[str], seed: str) -> str:
    """One of a fixed list of rewrites, picked by the seed string."""
    return texts[int(key(seed), 16) % len(texts)]


# ---------- analysis ----------

def groups(exp: Experiment) -> list[list[str]]:
    """Instruction-level conditions grouped by the items they ran on (datasets, units), in specification order."""
    out: dict[tuple, list[str]] = {}
    for c in instruction_conditions(exp.spec):
        v = exp.condition(c)
        out.setdefault((tuple(exp.datasets(c)), v.get("units")), []).append(c)
    return list(out.values())


def _summary(tables, values: list[float], keys: list[tuple]) -> dict | None:
    """Mean in points, 95% stratified cluster bootstrap interval and cluster sign-flip p (None without items)."""
    from ...analysis.stats_index import bootstrap_ci, signflip_p
    if not keys:
        return None
    clusters = [tables.cluster(k) for k in keys]
    low, high = bootstrap_ci(values, clusters, [k[0] for k in keys])
    return {"value": 100 * sum(values) / len(values), "low": 100 * low, "high": 100 * high,
            "p": signflip_p(values, clusters), "n": len(keys)}


def group_analysis(exp: Experiment, tables, conditions: list[str]) -> dict:
    from ...analysis.stats_index import holm
    twin = {c: exp.condition(c)["twin"] for c in conditions}
    datasets = {c: exp.datasets(c) for c in conditions}
    models = exp.models

    def keys(c, ms):
        """Items present for every model in `ms` under clean, c and its twin, minus units either skipped."""
        out = set()
        for d in datasets[c]:
            units = [tables.units(name, d, m) for name in ("clean", c, twin[c]) for m in ms]
            if any(t is None for t in units):
                return []
            common = set.intersection(*(set(t) for t in units)) - tables.skips(c, d) - tables.skips(twin[c], d)
            out |= {(d, u) for u in common}
        return sorted(out)

    ok = lambda name, m, k: float(tables.units(name, k[0], m)[k[1]]["correct"])
    drop = lambda name, m, k: ok("clean", m, k) - ok(name, m, k)
    accuracy = []
    for m in [*models, None]:
        ms = models if m is None else [m]
        row = {"model": m, "items": None, "clean": None, "conditions": {}}
        for c in conditions:
            ks = keys(c, ms)
            if not ks:
                continue
            mean = lambda f: 100 * sum(sum(f(x, k) for x in ms) / len(ms) for k in ks) / len(ks)
            row["items"], row["clean"] = len(ks), mean(lambda x, k: ok("clean", x, k))
            level = _summary(tables, [sum(ok(twin[c], x, k) - ok(c, x, k) for x in ms) / len(ms) for k in ks], ks)
            row["conditions"][c] = {"twin_acc": mean(lambda x, k: ok(twin[c], x, k)),
                                    "acc": mean(lambda x, k: ok(c, x, k)), "level": level}
        accuracy.append(row)
    results = {}
    for pair, parent, child in exp.children:
        for c in conditions:
            ks = keys(c, [parent, child])
            delta = lambda name, k: drop(name, parent, k) - drop(name, child, k)
            results[(child, c)] = {"pair": pair, "parent": parent, "child": child, "condition": c, "twin": twin[c],
                                   "instruction": _summary(tables, [delta(c, k) for k in ks], ks),
                                   "stem": _summary(tables, [delta(twin[c], k) for k in ks], ks),
                                   "did": _summary(tables, [delta(c, k) - delta(twin[c], k) for k in ks], ks),
                                   "keys": ks}
    adjusted = holm({k: v["instruction"]["p"] for k, v in results.items() if v["instruction"]})
    for k, v in results.items():
        if v["instruction"]:
            v["instruction"]["p_holm"] = adjusted[k]
    diagnostics = []
    for m in models:
        for c in conditions:
            ks = keys(c, [m])
            if not ks:
                continue
            clean = lambda k: tables.units("clean", k[0], m)[k[1]]
            table = lambda k: tables.units(c, k[0], m)[k[1]]
            pct = lambda f: 100 * sum(f(k) for k in ks) / len(ks)
            bare = [b for k in ks for b in table(k)["bare"]]
            diagnostics.append({"model": m, "condition": c,
                                "c_to_w": pct(lambda k: clean(k)["correct"] and not table(k)["correct"]),
                                "w_to_c": pct(lambda k: table(k)["correct"] and not clean(k)["correct"]),
                                "changed": pct(lambda k: table(k)["preds"] != clean(k)["preds"]),
                                "answered": pct(lambda k: table(k)["answered"]),
                                "bare_letter": 100 * sum(bare) / len(bare)})
    out = {"conditions": conditions, "twin": twin, "datasets": datasets,
           "units": exp.condition(conditions[0]).get("units"), "accuracy": accuracy, "results": results,
           "diagnostics": diagnostics, "rows": paper_rows(exp, tables, conditions, twin, datasets, results, accuracy)}
    extended = [c for c in conditions if exp.condition(c).get("seed_as")]
    if extended:
        out["replication"] = replication(exp, tables, extended, results, drop)
    return out


def paper_rows(exp: Experiment, tables, conditions, twin, datasets, results, accuracy) -> list[dict]:
    """Per child: each condition's dRob (Holm-adjusted p) with the matched clean accuracies and the condition's
    accuracy per model, keyed by the stem twin (the rows of Supp. Table eval_instruction)."""
    per_model = {row["model"]: row["conditions"] for row in accuracy if row["model"]}
    units = {c: {u for d in datasets[c] for u in exp.units(d, c)} for c in conditions}
    out = []
    for pair, parent, child in exp.children:
        names = sorted({d for c in conditions for d in datasets[c]})
        entry = {"dataset": names[0] if len(names) == 1 else names, "pair": pair, "parent": parent, "child": child,
                 "n_selected": len(units[conditions[0]])}
        for c in conditions:
            t, ks = results[(child, c)]["instruction"], results[(child, c)]["keys"]
            if t is None:
                continue
            acc = lambda m: 100 * sum(tables.units("clean", k[0], m)[k[1]]["correct"] for k in ks) / len(ks)
            entry[twin[c]] = {"condition": c, "n": len(ks), "delta_rob_pp": t["value"],
                              "delta_rob_ci95": [t["low"], t["high"]], "rob_tax_pp": -t["value"],
                              "rob_tax_ci95": [-t["high"], -t["low"]], "p_holm": t["p_holm"],
                              "matched_clean_parent": acc(parent), "matched_clean_child": acc(child),
                              "instruction_parent": per_model.get(parent, {}).get(c, {}).get("acc"),
                              "instruction_child": per_model.get(child, {}).get(c, {}).get("acc")}
        out.append(entry)
    return out


def replication(exp: Experiment, tables, conditions: list[str], results: dict, drop) -> dict:
    """For `seed_as` extensions: per model the raw replies identical to the earlier run's on the rows both ran
    (same edits there), and dRob_instruction on the earlier run's units vs the new ones (descriptive)."""
    earlier = {c: exp.condition(c)["seed_as"] for c in conditions}
    old_units = {c: {(d, DATASETS[d].unit(x)) for d in exp.datasets(c) for x in exp.samples(d, earlier[c])}
                 for c in conditions}
    identical = {}
    for m in exp.models:
        for c in conditions:
            same = total = 0
            for d in exp.datasets(c):
                now, before = tables.rows(c, d, m), tables.rows(earlier[c], d, m)
                shared = set(now) & set(before)
                same += sum(now[i].get("raw") == before[i].get("raw") for i in shared)
                total += len(shared)
            identical[(m, c)] = (same, total)
    split = {}
    for (child, c), v in results.items():
        if c not in conditions:
            continue
        parent = v["parent"]
        for name, part in (("earlier", [k for k in v["keys"] if k in old_units[c]]),
                           ("new", [k for k in v["keys"] if k not in old_units[c]])):
            split[(child, c, name)] = _summary(tables, [drop(c, parent, k) - drop(c, child, k) for k in part], part)
    return {"earlier": earlier, "identical": identical, "split": split}


def readouts(exp: Experiment, tables) -> dict:
    """(child, condition) -> accuracy and gold-option probability dRob with lower bounds for every instruction-level
    condition; Holm over all the probability tests."""
    from ...analysis.stats_index import holm, summary
    from .report import ALPHAS, values
    out = {}
    for c in instruction_conditions(exp.spec):
        twin = exp.condition(c)["twin"]
        for pair, parent, child in exp.children:
            keys = []
            for d in exp.datasets(c):
                units = [tables.units(name, d, m) for name in ("clean", c, twin) for m in (parent, child)]
                if any(t is None for t in units):
                    continue
                common = set.intersection(*(set(t) for t in units)) - tables.skips(c, d) - tables.skips(twin, d)
                keys += [(d, u) for u in sorted(common)]
            pkeys = [k for k in keys if all(k[1] in tables.probs(n, k[0], m) for n in ("clean", c)
                                            for m in (parent, child))]
            stats = lambda kind, ks: summary(values(tables, kind, [c], parent, child, ks),
                                             [tables.cluster(k) for k in ks], [k[0] for k in ks], alphas=ALPHAS)
            out[(child, c)] = {"pair": pair, "child": child, "condition": c, "acc": stats("acc", keys),
                               "prob": stats("prob", pkeys)}
    adjusted = holm({k: v["prob"]["p"] for k, v in out.items() if v["prob"]["n"]})
    for k, v in out.items():
        if k in adjusted:
            v["prob"]["p_holm"] = adjusted[k]
    return out


def analyse(exp: Experiment, tables) -> dict:
    found = [group_analysis(exp, tables, conditions) for conditions in groups(exp)]
    return {"groups": found,
            "rows": [{"conditions": g["conditions"], "units": g["units"], "rows": g["rows"]} for g in found],
            "contrasts": readouts(exp, tables) if found else {}}


def contrast_rows(level: dict) -> list[dict]:
    """instruction.csv: both readouts of every instruction-level condition."""
    return [{"pair": v["pair"], "child": v["child"], "condition": v["condition"], "readout": kind, **v[kind]}
            for v in level["contrasts"].values() for kind in ("acc", "prob")]


def markdown(exp: Experiment, level: dict) -> list[str]:
    from .report import ALPHAS, fmt_ci, fmt_delta
    several, lines = len(level["groups"]) > 1, []
    for g in level["groups"]:
        conditions, twin = g["conditions"], g["twin"]
        title = "# Instruction-level perturbations (vs the question-stem twin, same items)"
        if several:
            title += f": {', '.join(conditions)}" + (f" ({g['units']} units)" if g["units"] else "")
        lines += ["", title, "",
                  "The question-stem edit is applied to the official instruction sentence instead. "
                  + "; ".join(f"**{c}** ↔ {twin[c]} ({', '.join(g['datasets'][c])})" for c in conditions)
                  + ". Drop = acc_clean − acc_condition (pp). Level = Drop_instruction − Drop_stem, **+ = the "
                  "instruction edit hurts more**. ΔRob = Drop_parent − Drop_child, + = child loses less, − = more; "
                  "DiD = ΔRob_instruction − ΔRob_stem.", "",
                  "## Accuracy and level effect per model", "",
                  "| model | items | clean acc | " + " | ".join(f"acc {twin[c]} → {c}" for c in conditions) + " |",
                  "|---|---|---|" + "---|" * len(conditions)]
        for row in g["accuracy"]:
            cells = []
            for c in conditions:
                x = row["conditions"].get(c)
                cells.append("–" if x is None else
                             f"{x['twin_acc']:.1f} → {x['acc']:.1f} (level {fmt_delta(x['level'])})")
            clean = f"{row['clean']:.1f}" if row["clean"] is not None else "–"
            lines.append(f"| {row['model'] or '**all models (mean)**'} | {row['items'] or '–'} | {clean} | "
                         + " | ".join(cells) + " |")
        tested = sum(1 for v in g["results"].values() if v["instruction"])
        lines += ["", f"## ΔRob: instruction (Holm over {tested}) vs stem twin, and DiD", "",
                  "| pair | child | " + " | ".join(f"{c} | {twin[c]} (same items) | DiD" for c in conditions) + " |",
                  "|---|---|" + "---|---|---|" * len(conditions)]
        for pair, parent, child in exp.children:
            cells = []
            for c in conditions:
                v = g["results"][(child, c)]
                cells += [fmt_delta(v["instruction"], (v["instruction"] or {}).get("p_holm")), fmt_delta(v["stem"]),
                          fmt_delta(v["did"])]
            lines.append(f"| {pair} | {child} | " + " | ".join(cells) + " |")
        lines += ["", "Diagnostics (instruction conditions; vs clean on the same items):", "",
                  "| model | condition | C→W | W→C | changed | answered | bare letter |", "|---|---|---|---|---|---|---|"]
        for r in g["diagnostics"]:
            lines.append(f"| {r['model']} | {r['condition']} | {r['c_to_w']:.1f} | {r['w_to_c']:.1f} | "
                         f"{r['changed']:.1f} | {r['answered']:.1f} | {r['bare_letter']:.1f} |")
        if "replication" in g:
            rep, extended = g["replication"], list(g["replication"]["earlier"])
            lines += ["", "Replication of the earlier run (" + ", ".join(f"{c} extends {rep['earlier'][c]}"
                                                                          for c in extended)
                      + "): rows of the earlier run's items whose raw reply is identical (same edits there).", "",
                      "| model | " + " | ".join(extended) + " |", "|---|" + "---|" * len(extended)]
            for m in exp.models:
                cells = [f"{s}/{t}" if t else "–" for s, t in (rep["identical"][(m, c)] for c in extended)]
                lines.append(f"| {m} | " + " | ".join(cells) + " |")
            lines += ["", "ΔRob (instruction) on the earlier run's units vs the new units (descriptive):", "",
                      "| pair | child | " + " | ".join(f"{c} earlier units | {c} new units" for c in extended) + " |",
                      "|---|---|" + "---|---|" * len(extended)]
            for pair, parent, child in exp.children:
                cells = []
                for c in extended:
                    for name in ("earlier", "new"):
                        t = rep["split"][(child, c, name)]
                        cells.append(f"{fmt_delta(t)} (n={t['n']})" if t else "–")
                lines.append(f"| {pair} | {child} | " + " | ".join(cells) + " |")
    if level["contrasts"]:
        lb = f"lower_{ALPHAS[1]:g}"
        lines += ["", "## Instruction level: accuracy and gold-option probability readouts", "",
                  "Items as above; LB95 = one-sided 95% lower bound, LB-sim = Bonferroni-18 bound; Holm over every "
                  "probability test of this table.", "",
                  "| pair | child | condition | items | ΔRob acc | LB95 | LB-sim | ΔRob p_gold | LB-sim | Holm p |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for v in level["contrasts"].values():
            a, p = v["acc"], v["prob"]
            acc_cells = ([fmt_ci(a["value"], a["low"], a["high"]), f"{a['lower_0.05']:+.2f}", f"{a[lb]:+.2f}"]
                         if a["n"] else ["–"] * 3)
            prob_cells = ([fmt_ci(p["value"], p["low"], p["high"], 2), f"{p[lb]:+.2f}", f"{p['p_holm']:.2g}"]
                          if p["n"] else ["–"] * 3)
            lines.append(" | ".join([f"| {v['pair']}", v["child"], v["condition"], str(a["n"]), *acc_cells,
                                     *prob_cells]) + " |")
    return lines
