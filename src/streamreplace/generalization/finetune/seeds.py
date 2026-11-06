"""`streamreplace finetune seeds`: the training-seed replication of the fine-tuning experiment (the `seeds` block of
configs/generalization/finetune.yaml; the seed-variability statement of the supplement). Ported unchanged.

Seed 0 = the formal units and evaluations (the experiment root, only read here). Seed k = the same unit with the
training seed replaced by k: the LoRA A initialisation and, with `shuffle`, the order of the same first `budget`
training ids (epoch 0 in a seeded reshuffle). Training ids, test questions, settings, the Lingo-Judge and the zero-shot
rows (seed-independent, read from the experiment root) are the formal ones. Adapters, evaluations and this report live
in the seeds folder (<root>/<seeds.outputs>), whose subset.json holds the formal test ids (`finetune eval --seeds`
writes it).

Report: <seeds folder>/SEED_RESULTS.md and seed_tables/{seeds,summary}.csv. Accuracy in %; delta = child - parent
(+ = the child is better). Per pair x budget b x seed s: acc_P, acc_C, dAcc(b, s) with its 95% CI (the stratified
cluster bootstrap of the main report) and dGain(b, s) = dAcc(b, s) - dAcc(0). Over the seeds whose rows are all scored
(k of them): mean, between-seed SD (n - 1) and range; seed-inclusive 95% CI of the seed mean = mean +- 1.96 *
sqrt(SE_test^2 + SD^2 / k), SE_test = bootstrap SE of the per-question seed-mean value. "avg" = the mean over the
budgets every seed has (the primary endpoint's budget average; shown once two or more budgets are complete, before
that the decision rules use the one complete budget). dGain differs from dAcc by the shared zero-shot gap, so its SD
and range are the same.
Decision rules, fixed before the replication was run: SD(dAcc avg) <= 1.5 pp with the seed-inclusive CI inside
+-3.6 pp -> parity can be stated with total uncertainty; seeds more than 3 pp apart -> 'unresolved within seed noise';
the 2K and 5K units (stage 2) only if a pair's 1K SD exceeds 1.5 pp."""
from __future__ import annotations

import csv
from io import StringIO
import math
from pathlib import Path
from statistics import mean, stdev

from ...analysis.stats_index import bootstrap_ci, bootstrap_means
from ...core.io import atomic_text
from . import Experiment, label, model_id

PARITY_SD, PARITY_MARGIN, SPREAD = 1.5, 3.6, 3.0           # pp, the decision rules


def pairs(exp: Experiment) -> dict[str, list[str]]:
    """The experiment's pairs restricted to seeds.models: the parent and the children that are replicated."""
    chosen = set(exp.seeds["models"])
    out = {}
    for pair, members in exp.pairs.items():
        kept = [m for m in members if m in chosen]
        if members[0] in chosen and len(kept) > 1:
            out[pair] = kept
    return out


def outcomes(folder: Path, dataset: str, model: str, ids: list[str]) -> dict[str, int] | None:
    """id -> 0/1 of one evaluation, or None while a row is missing or unscored (the Lingo-Judge scores after the
    run)."""
    from ...analysis.records import joined_rows
    from ...scoring.outcome import final, outcome
    from .report import _bench
    rows = joined_rows(folder, dataset, f"{model}.offline") or {}
    if any(i not in rows or (final(rows[i]) is None and not rows[i].get("error")) for i in ids):
        return None
    bench = _bench(dataset)
    return {i: int(outcome(bench, rows[i])) for i in ids}


def summarize(per_seed: dict[int, list[float]], clusters: list, strata: list) -> dict:
    """Seed -> per-question values (pp scale 0-1) -> mean / SD / range over seeds and the seed-inclusive 95% CI."""
    import numpy as np
    means = {s: 100 * sum(v) / len(v) for s, v in per_seed.items()}
    k, values = len(means), list(means.values())
    out = {"k": k, "mean": sum(values) / k, "per_seed": means}
    if k < 2:
        return out
    sd = float(np.std(values, ddof=1))
    pooled = [sum(col) / k for col in zip(*per_seed.values())]
    se_test = 100 * float(np.std(bootstrap_means(pooled, clusters, strata), ddof=1))
    half = 1.96 * math.sqrt(se_test ** 2 + sd ** 2 / k)
    return {**out, "sd": sd, "range": max(values) - min(values), "se_test": se_test,
            "low": out["mean"] - half, "high": out["mean"] + half}


def write_report(exp: Experiment) -> Path:
    from .report import design
    s = exp.seeds
    if not s:
        raise ValueError("the specification has no `seeds` block (training-seed replication)")
    formal, mine = exp.root, exp.seeds_folder()
    seeds = [0, *[int(k) for k in s["seeds"]]]
    budgets = [int(b) for b in s["budgets"]]
    lines = ["# Fine-tune training-seed replication", "",
             "Accuracy in %. **Δ = child − parent: + = the streaming child is better, − = worse.** Seed 0 = the formal "
             "units (the experiment root); seeds " + ", ".join(map(str, seeds[1:])) + " = the same recipe with another "
             "seed (LoRA A init" + (" and the order of the same training ids" if s.get("shuffle") else "") + "). "
             "ΔGain(b) = ΔAcc(b) − ΔAcc(0); zero-shot rows are shared by all seeds. Per seed: 95% CI = stratified "
             "cluster bootstrap (test sampling only). Over seeds: mean, SD (n − 1), range, and the seed-inclusive 95% "
             "CI = mean ± 1.96·√(SE_test² + SD²/k). avg = mean over the budgets every seed has.", ""]
    seed_rows, summary_rows, decisions = [], [], []
    for dataset in s["datasets"]:
        samples, ids, clusters, strata = design(formal, dataset)

        def table(model: str, budget: int, seed: int):
            folder, tag = (formal, None) if budget == 0 or seed == 0 else (mine, f"s{seed}")
            return outcomes(folder, dataset, model_id(model, dataset, budget, tag), ids)

        members = [m for ms in pairs(exp).values() for m in ms]
        tables = {(m, b, k): table(m, b, k) for m in members for b in budgets for k in seeds}
        zero = {m: table(m, 0, 0) for m in members}
        lines += [f"## {dataset} ({len(ids)} test questions)", "", "Accuracy per seed (acc@0 is the shared zero-shot):", "",
                  "| model | acc@0 | " + " | ".join(f"acc@{label(b)} s{k}" for b in budgets for k in seeds)
                  + " | " + " | ".join(f"mean ± SD @{label(b)}" for b in budgets) + " |",
                  "|---|---|" + "---|" * (len(budgets) * (len(seeds) + 1))]
        pct = lambda t: 100 * sum(t.values()) / len(t)
        for m in members:
            cells = [f"{pct(zero[m]):.1f}" if zero[m] else "–"]
            cells += [f"{pct(t):.1f}" if (t := tables[(m, b, k)]) else "–" for b in budgets for k in seeds]
            for b in budgets:
                accs = [pct(t) for k in seeds if (t := tables[(m, b, k)])]
                cells.append(f"{mean(accs):.1f} ± {stdev(accs):.1f} (k={len(accs)})" if len(accs) > 1 else "–")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines += ["", "ΔAcc and ΔGain per seed, and over seeds:", "",
                  "| pair | child | budget | " + " | ".join(f"ΔAcc s{k} (95% CI)" for k in seeds)
                  + " | ΔAcc mean | SD | range | seed-incl. 95% CI | ΔGain mean (seed-incl. 95% CI) |",
                  "|---|---|---|" + "---|" * (len(seeds) + 5)]
        for pair, ms in pairs(exp).items():
            parent = ms[0]
            for child in ms[1:]:
                if not (zero[parent] and zero[child]):
                    lines.append(f"| {pair} | {child} | – | zero-shot rows missing |")
                    continue
                d0 = [zero[child][i] - zero[parent][i] for i in ids]
                per_budget = {}                          # budget -> seed -> per-question dAcc
                for b in budgets:
                    diffs = {k: [tables[(child, b, k)][i] - tables[(parent, b, k)][i] for i in ids]
                             for k in seeds if tables[(parent, b, k)] and tables[(child, b, k)]}
                    per_budget[b] = diffs
                full = [b for b in budgets if len(per_budget[b]) == len(seeds)]      # every seed scored
                rows = [(label(b), per_budget[b]) for b in budgets if per_budget[b]]
                if len(full) > 1:
                    rows.append(("avg", {k: [sum(per_budget[b][k][j] for b in full) / len(full) for j in range(len(ids))]
                                         for k in seeds}))
                decide = "avg" if len(full) > 1 else label(full[0]) if full else None
                for name, diffs in rows:
                    acc = summarize(diffs, clusters, strata)
                    gain = summarize({k: [v - z for v, z in zip(vals, d0)] for k, vals in diffs.items()}, clusters, strata)
                    cells = []
                    for k in seeds:
                        if k not in diffs:
                            cells.append("–")
                            continue
                        low, high = bootstrap_ci(diffs[k], clusters, strata)
                        cells.append(f"{acc['per_seed'][k]:+.1f} [{100 * low:+.1f}, {100 * high:+.1f}]")
                        seed_rows.append({"dataset": dataset, "pair": pair, "parent": parent, "child": child, "budget": name,
                                          "seed": k, "d_acc": acc["per_seed"][k], "low": 100 * low, "high": 100 * high,
                                          "d_gain": gain["per_seed"][k]})
                    spread = "sd" in acc
                    ci = lambda t: f"[{t['low']:+.1f}, {t['high']:+.1f}]" if "sd" in t else "–"
                    lines.append(f"| {pair} | {child} | {name} | " + " | ".join(cells) + f" | {acc['mean']:+.1f} | "
                                 + (f"{acc['sd']:.1f} | {acc['range']:.1f} | " if spread else "– | – | ")
                                 + f"{ci(acc)} | {gain['mean']:+.1f} {ci(gain)} |")
                    summary_rows.append({"dataset": dataset, "pair": pair, "child": child, "budget": name, "k": acc["k"],
                                         "d_acc_mean": acc["mean"], "d_acc_sd": acc.get("sd"), "d_acc_range": acc.get("range"),
                                         "d_acc_low": acc.get("low"), "d_acc_high": acc.get("high"),
                                         "se_test": acc.get("se_test"), "d_gain_mean": gain["mean"],
                                         "d_gain_low": gain.get("low"), "d_gain_high": gain.get("high")})
                    if name == decide:
                        first = per_budget[budgets[0]]
                        decisions.append((dataset, pair, child, name, acc,
                                          summarize(first, clusters, strata)["sd"] if len(first) == len(seeds) else None))
        lines.append("")
    lines += ["## Decision rules (fixed before the run)", "",
              f"Parity with total uncertainty needs SD(ΔAcc avg) ≤ {PARITY_SD} pp and the seed-inclusive CI inside "
              f"±{PARITY_MARGIN} pp; seeds more than {SPREAD} pp apart make the pair's statement 'unresolved within seed "
              f"noise'; stage 2 (the other budgets) only if the {label(budgets[0])} SD exceeds {PARITY_SD} pp.", "",
              "| dataset | pair | child | budgets | ΔAcc mean (SD, range, seed-incl. CI) | statement | "
              f"stage 2 ({label(budgets[0])} SD) |", "|---|---|---|---|---|---|---|"]
    for dataset, pair, child, name, acc, first_sd in decisions:
        inside = -PARITY_MARGIN <= acc["low"] and acc["high"] <= PARITY_MARGIN
        verdict = ("unresolved within seed noise" if acc["range"] > SPREAD else
                   "parity with total uncertainty" if acc["sd"] <= PARITY_SD and inside else "no parity statement")
        stage2 = "–" if first_sd is None else f"{'run' if first_sd > PARITY_SD else 'not needed'} ({first_sd:.1f} pp)"
        lines.append(f"| {dataset} | {pair} | {child} | {name} | {acc['mean']:+.1f} ({acc['sd']:.1f}, {acc['range']:.1f}, "
                     f"[{acc['low']:+.1f}, {acc['high']:+.1f}]) | {verdict} | {stage2} |")
    lines.append("")
    for name, rows in (("seeds", seed_rows), ("summary", summary_rows)):
        if rows:
            buffer = StringIO()
            writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            atomic_text(mine / "seed_tables" / f"{name}.csv", buffer.getvalue())
    target = mine / "SEED_RESULTS.md"
    atomic_text(target, "\n".join(lines) + "\n")
    return target
