"""RESULTS.md (one table per pair with paired statistics, then per-dataset details) and pairs.csv.

This is the development report written by `streamreplace report`: the headline metric of every job with its
difference to the pair's parent (+ = better than the parent; for lower-is-better headlines such as ASR the
difference is parent - candidate), an exact McNemar test per benchmark stratum, the VMDT benchmarks' own paired
bootstrap (2,000 draws), and a Details block per dataset with every metric and the completion counts. The paper's
absolute-score tables are rendered from metrics.json (eval_tables.py), not from this file; its statistics come from
`streamreplace stats` (contrasts.py)."""
from __future__ import annotations

import csv
from datetime import datetime
from io import StringIO

from ..records import outcome
from ..stats_index import mcnemar_p
from ..vmdt import official_paired

PAIR_FIELDS = ["pair", "dataset", "id", "parent", "candidate", "candidate_mode", "category",
               "parent_pred", "candidate_pred", "parent_raw", "candidate_raw"]


def fmt(value) -> str:
    return "–" if value is None else (f"{value:.1f}" if isinstance(value, float) else str(value))


def cell(result: dict | None, parent: dict | None = None, sign: int = 1) -> str:
    if not result or not result["metrics"]:
        return "–"
    key = next(iter(result["metrics"]))
    value = result["metrics"][key]
    text = fmt(value)
    if parent and parent["metrics"] and value is not None and parent["metrics"].get(key) is not None:
        text += f" ({sign * (value - parent['metrics'][key]):+.1f})"
    if result["done"] < result["expected"]:
        text += f" ⚠{result['done']}/{result['expected']}"
    return text


def _modes(profile, model: str) -> list[str]:
    return ["offline", "stream"] if model in profile.parents else ["offline"]


def _vmdt_blocks(profile, dataset: str) -> tuple[bool, str | None]:
    """(the dataset's statistics are the VMDT paired bootstrap, its resampling blocks)."""
    for m in profile.metrics:
        if m.dataset == dataset and m.kind == "vmdt":
            return True, m.blocks
    return False, None


def render(results: dict, profile, catalog, generated: datetime | None = None, smoke: bool = False) -> tuple[str, str]:
    """(RESULTS.md text, pairs.csv text) from metrics.collect() results."""
    order = list(profile.report.get("datasets", []))
    extra = sorted({k[0] for k in results} - set(order))
    datasets = [d for d in order + extra if any(k[0] == d for k in results)]
    draws = int(profile.report.get("vmdt_paired_draws", 2000))
    stamp = (generated or datetime.now()).strftime("%Y-%m-%d %H:%M")
    lines = [f"# Results ({'smoke subsets' if smoke else 'full'})", "",
             f"Generated {stamp}. Values in % unless the metric says otherwise. "
             "`(±x)` = difference to the pair's parent (offline), signed so that + = better than the parent: candidate "
             "− parent, or parent − candidate for lower-is-better headlines (ASR). ⚠ done/total = incomplete. "
             "`stream` = native streaming mode (video datasets only). Details at the bottom.", ""]
    pair_rows = []
    for pair, members in profile.pairs.items():
        parent = members[0]
        columns = [(m, mode) for m in members for mode in _modes(profile, m) if any(k[1:] == (m, mode) for k in results)]
        if not columns:
            continue
        lines += [f"## {pair}: parent `{parent}`", "",
                  "| dataset | metric | " + " | ".join(f"{m}" + (" (stream)" if mode == "stream" else "") for m, mode in columns) + " |",
                  "|---|---|" + "---|" * len(columns)]
        stats, paired = [], []
        for dataset in datasets:
            base = results.get((dataset, parent, "offline"))
            present = [results.get((dataset, m, mode)) for m, mode in columns]
            if not any(present):
                continue
            metric = next((next(iter(r["metrics"])) for r in present if r and r["metrics"]), "–")
            sign = -1 if catalog.lower_is_better(dataset) else 1
            policy = catalog.policy(dataset)
            lines.append(f"| {dataset} | {metric}{' ʲ' if any(r and r['judged'] for r in present) else ''} | " + " | ".join(
                cell(r, base if (m, mode) != (parent, "offline") else None, sign) for r, (m, mode) in zip(present, columns)) + " |")
            is_vmdt, blocks = _vmdt_blocks(profile, dataset)
            for (m, mode), r in zip(columns, present):
                if not r or not base or (m, mode) == (parent, "offline"):
                    continue
                common_ids = sorted(set(base["rows"]) & set(r["rows"]))
                strata: dict = {}
                if is_vmdt:                           # e.g. fairness gaps: the benchmark's own paired statistics
                    both = base["metric_rows"] and r["metric_rows"]
                    module = catalog.bench(dataset)
                    for row in official_paired(module, base["metric_rows"], r["metric_rows"], blocks, draws) if both else []:
                        delta = "–" if row["delta"] is None else f"{row['delta']:+.2f}"
                        interval = "–" if row["low"] is None else f"[{row['low']:+.2f}, {row['high']:+.2f}]"
                        p_value = "–" if row["p"] is None else f"{row['p']:.3g}"
                        paired.append(f"| {dataset} | {m}{' (stream)' if mode == 'stream' else ''} | {row['metric']} | "
                                      f"{fmt(row['parent'])} | {fmt(row['candidate'])} | {delta} | {interval} | {p_value} |")
                else:
                    for i in common_ids:              # one statistics row per stratum (e.g. a query split), if any
                        strata.setdefault(r["strata"].get(i), []).append(i)
                for stratum, ids in strata.items():
                    pairs = [(outcome(policy, base["rows"][i]), outcome(policy, r["rows"][i])) for i in ids]
                    b = sum(p and not c for p, c in pairs)
                    c = sum(c and not p for p, c in pairs)
                    p_value = mcnemar_p(b, c)
                    label = dataset if stratum is None else f"{dataset} [{stratum}]"
                    stats.append(f"| {label} | {m}{' (stream)' if mode == 'stream' else ''} | {len(pairs)} | {b} | {c} | "
                                 f"{100 * (c - b) / len(pairs) if pairs else 0:+.1f} | {'–' if p_value is None else f'{p_value:.3g}'} |")
                for i in common_ids:
                    pr, cr = base["rows"][i], r["rows"][i]
                    category = {(True, True): "both_correct", (True, False): "regression",
                                (False, True): "improvement", (False, False): "both_wrong"}[(outcome(policy, pr),
                                                                                             outcome(policy, cr))]
                    pair_rows.append({"pair": pair, "dataset": dataset, "id": i, "parent": parent, "candidate": m,
                                      "candidate_mode": mode, "category": category,
                                      "parent_pred": pr.get("pred"), "candidate_pred": cr.get("pred"),
                                      "parent_raw": (pr.get("raw") or "")[:300].replace("\x00", ""),   # csv cannot hold NUL
                                      "candidate_raw": (cr.get("raw") or "")[:300].replace("\x00", "")})
        if stats:
            lines += ["", f"Paired vs `{parent}` (questions answered by both; regress = parent right, candidate wrong):", "",
                      "| dataset | candidate | n | regress | improve | Δacc | McNemar p |", "|---|---|---|---|---|---|---|", *stats]
        if paired:
            lines += ["", f"Paired vs `{parent}` by the benchmark's own statistics (Δ = candidate − parent; cluster "
                      "bootstrap 95 % interval and two-sided p):", "",
                      "| dataset | candidate | metric | parent | candidate | Δ | 95 % CI | p |",
                      "|---|---|---|---|---|---|---|---|", *paired]
        lines.append("")
    lines += ["## Details", ""]
    for dataset in datasets:
        lines += [f"<details><summary>{dataset}</summary>", "",
                  "| model | mode | done | errors | no answer | forced | parse fail | truncated | latency s | metrics |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for (d, model, mode), r in sorted(results.items(), key=lambda kv: kv[0]):
            if d != dataset:
                continue
            shown = ", ".join(f"{k}={fmt(v)}" for k, v in r["metrics"].items())
            if r["judged"]:
                shown += " · rule-only: " + ", ".join(f"{k}={fmt(v)}" for k, v in list(r["rule_metrics"].items())[:3])
            lines.append(f"| {model} | {mode} | {r['done']}/{r['expected']} | {r['errors']} | {r['no_answer']} | {r['forced']} | {r['parse_fail']} | "
                         f"{r['truncated']} | {fmt(r['latency'])} | {shown} |")
        lines += ["", "</details>", ""]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=PAIR_FIELDS)
    writer.writeheader()
    writer.writerows(pair_rows)
    return "\n".join(lines) + "\n", buffer.getvalue()
