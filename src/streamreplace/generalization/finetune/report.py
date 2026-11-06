"""`streamreplace finetune report`: the fine-tuning tables from the scored rows (run `streamreplace --outputs <root>
score --judge` first, and the same for each control folder). Correctness per question is the run pipeline's own
outcome() (errors count as wrong). Ported unchanged; Supp. Tables eval_finetuning / eval_finetuning_gains and the
fine-tuning numbers of the main text come from these files.

<root>/FT_RESULTS.md and <root>/ft_tables/{accuracy,pairs}.csv (write_report):
  acc(m, d, b), b = 0 (zero-shot), 1K, 2K, 5K; Gain = acc(b) - acc(0); normalized gain (FT - ZS) / (100 - ZS)
  Delta = child - parent, + = the child is better: dAcc(b) = acc_C(b) - acc_P(b);
  dGain(b) = Gain_C(b) - Gain_P(b) = dAcc(b) - dAcc(0)
  primary endpoint per child x dataset: dGain averaged over the fine-tuned budgets; Holm over all children x datasets
  (12 cells: OpenUAV-QA is in `primary_excludes`, its tables are kept)
  95% CI: stratified cluster bootstrap (clusters = the benchmark's GROUP, strata = its STRATUM_KEYS), 10,000 resamples
  (analysis/stats_index.py, the integer-index engine); p: cluster sign-flip permutation (swapping parent and child
  within a cluster flips its contribution), 10,000 draws; exact McNemar on per-budget accuracy differences where
  questions are independent (no shared GROUP)
  A benchmark's SENSITIVITY parsers ({name: parse(sample, text)}) get an accuracy table each (EyePCR's first-round
  parser); multiple-choice sets also show the first-token letter accuracy (info.letter_logits).
  Controls (their own folders, scored there): blank frames on the zero-shot and top-budget ids (the ZS / top x real /
  blank 2x2 once every model's zero-shot blank run exists, else the top-budget table), extra units (1K x 3 epochs) and
  the image-list zero-shot anchor. The 2x2 also goes to ft_tables/blank_2x2_<dataset>.csv.
  Limitation printed with the tables: one training seed, so the intervals cover test sampling only.
  ft_tables/results.json (paper_data): the data of the paper's fine-tuning tables, read back from these files.

<root>/READOUT.md and <root>/ft_tables/{cells,pooled}.csv (write_readout): the same rows read through post-fine-tuning
accuracy. Per child x dataset: dAcc(0), dAcc_avg (mean over the fine-tuned budgets), dAcc at the top budget and
dGain_avg = dAcc_avg - dAcc(0) (the primary endpoint; the identity is exact per question), each with the 95% interval,
the sign-flip p, the one-sided 95% lower bound and the Bonferroni-12 simultaneous lower bound; Holm over the 12 primary
cells and over all 18. The worst simultaneous lower bound of dAcc_avg is the paper's parity bound (-3.5 points), and
cells.csv's LingoQA dGain_avg is the fine-tuning column of Table 2. Also: the pre-registered cross-dataset estimate
(equal weight per dataset, each dataset bootstrapped with its own seed) and LingoQA without its fixed-format
question (the one 20% of the evaluation questions share and the training data never contains; a sensitivity view).
A dGain cell whose dAcc_avg interval includes 0 is read as a zero-shot difference, not a difference in what
fine-tuning reaches (this reading was chosen after the results; it follows from the identity)."""
from __future__ import annotations

import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import re

from ...analysis.stats_index import bootstrap_ci, holm, mcnemar_p, signflip_p
from ...core.io import atomic_text
from . import Experiment, label, model_id

# LingoQA's fixed-format action/justification question, identified by the sha256 of its stripped text (no benchmark
# text is shipped)
FIXED_QUESTION_SHA256 = "fe7ec483d64ac148827d161a043050c6a5cd5e4fa367ee886747fd75fdbe72ec"
ALPHAS = (0.05, 0.05 / 12)          # one-sided 95% and Bonferroni-12 simultaneous lower bounds
READOUT_KEYS = ("d_acc_0", "d_acc_avg", "d_acc_5k", "d_gain_avg")


# ---------- scored rows ----------

def _bench(dataset: str):
    from ...benchmarks import registry
    return registry.get(dataset)


def _samples(folder: Path, dataset: str) -> list:
    """The prepared test samples restricted to the folder's subset.json, in prepared order."""
    from ...data.samples import load_samples, read_subset
    return load_samples(dataset, subset=(read_subset(folder) or {}).get(dataset))


def collect(folder: Path) -> dict:
    """(dataset, model id, mode) -> {"rows": {id: row}} for every job of an outputs folder: generation rows joined
    with their scores, restricted to the folder's samples and to counted rows (scored or errored)."""
    from ...analysis.records import joined_rows
    from ...scoring.outcome import counted
    from ...scoring.score import find_jobs
    folder = Path(folder)
    results, known = {}, {}
    if not folder.is_dir():
        return results
    for job in find_jobs(folder):
        if job.dataset not in known:
            known[job.dataset] = {s.id for s in _samples(folder, job.dataset)}
        arm = job.arm
        model = f"{arm.model}@{arm.variant}" if arm.variant else arm.model
        rows = joined_rows(folder, job.dataset, job.job) or {}
        results[(job.dataset, model, arm.mode.value)] = {
            "rows": {i: r for i, r in rows.items() if i in known[job.dataset] and counted(r)}}
    return results


def outcome(bench, row: dict) -> bool:
    from ...scoring.outcome import outcome as _outcome
    return _outcome(bench, row)


def design(folder: Path, dataset: str):
    """samples, ids, clusters (the benchmark's GROUP, else the question) and strata (its STRATUM_KEYS)."""
    bench = _bench(dataset)
    samples = _samples(folder, dataset)
    group, keys = getattr(bench.module, "GROUP", None), tuple(getattr(bench.module, "STRATUM_KEYS", ()))
    return (samples, [s.id for s in samples], [s.metadata[group] if group else s.id for s in samples],
            [tuple(s.metadata.get(k) for k in keys) for s in samples])


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields or list(dict.fromkeys(k for r in rows for k in r)))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


# ---------- FT_RESULTS.md ----------

def _outcomes(results, bench, dataset, model, labels, ids):
    """label -> {id: 0/1} for the questions answered (or errored) in that output file."""
    table = {}
    for lab in labels:
        result = results.get((dataset, f"{model}@{dataset}-{lab}", "offline"))
        if result:
            table[lab] = {i: int(outcome(bench, result["rows"][i])) for i in ids if i in result["rows"]}
    return table


def _letter_accuracy(results, dataset, model, lab, samples) -> float | None:
    result = results.get((dataset, f"{model}@{dataset}-{lab}", "offline"))
    if not result:
        return None
    hits = [max(l, key=l.get) == s.answer for s in samples
            if (l := ((result["rows"].get(s.id) or {}).get("info") or {}).get("letter_logits"))]
    return 100 * sum(hits) / len(hits) if hits else None


def _variant_accuracy(results, bench, dataset, model, lab, samples, parse) -> float | None:
    """Accuracy with another parser over the saved replies (errored or missing rows count as wrong)."""
    result = results.get((dataset, f"{model}@{dataset}-{lab}", "offline"))
    if not result or len(result["rows"]) < len(samples):
        return None
    hits = [not (row := result["rows"][s.id]).get("error") and bench.correct(s, parse(bench.view(s), row.get("raw", ""))[0])
            for s in samples]
    return 100 * sum(hits) / len(hits)


def _answered(row) -> bool:
    return not row.get("error") and bool(str(row.get("pred") or "").strip())


def _blank_2x2(exp: Experiment, results, blank, folder: Path, dataset: str, top: str) -> list[str]:
    """ZS / top budget x real / blank frames on the blank control's questions. VD = real - blank (frame dependence);
    Gain = Gain_blank + dVD exactly, Gain_blank = gain kept without frames (text / format / answer priors),
    dVD = VD@top - VD@0 = gain that needs the frames. Per pair (exploratory): dVis = dVD_child - dVD_parent."""
    bench = _bench(dataset)
    samples, ids, clusters, strata = design(folder, dataset)
    cells = {"r0": (results, "0"), "b0": (blank, "0"), "rt": (results, top), "bt": (blank, top)}
    lines = [f"Blank frames, zero-shot vs {top} ({dataset}, {len(ids)} questions; frames replaced by black ones of the same "
             f"size). VD = real − blank; Gain = Gain_blank + ΔVD; ΔVD = VD@{top} − VD@0 (gain that needs the frames); "
             "answered = no error and a non-empty parsed answer; VD|ans = VD on questions answered in both arms. "
             "95% CI = stratified cluster bootstrap.", "",
             f"| model | real@0 | blank@0 | real@{top} | blank@{top} | VD@0 | VD@{top} | Gain | Gain_blank (95% CI) | "
             f"ΔVD (95% CI) | answered% r0/b0/r{top}/b{top} | VD\\|ans @0 / @{top} |", "|---|" + "---|" * 11]
    csv_rows, deltas = [], {}
    ci = lambda v: f"{100 * sum(v) / len(v):+.1f} [{100 * (b := bootstrap_ci(v, clusters, strata))[0]:+.1f}, {100 * b[1]:+.1f}]"
    for pair, members in exp.pairs.items():
        for m in members:
            rows = {k: t[(dataset, f"{m}@{dataset}-{lab}", "offline")]["rows"] for k, (t, lab) in cells.items()}
            if not all(i in r for r in rows.values() for i in ids):
                lines.append(f"| {m} | incomplete |")
                continue
            o = {k: [int(outcome(bench, r[i])) for i in ids] for k, r in rows.items()}
            a = {k: [_answered(r[i]) for i in ids] for k, r in rows.items()}
            acc = {k: 100 * sum(v) / len(ids) for k, v in o.items()}
            gain_blank = [o["bt"][j] - o["b0"][j] for j in range(len(ids))]
            dvd = [(o["rt"][j] - o["bt"][j]) - (o["r0"][j] - o["b0"][j]) for j in range(len(ids))]
            deltas[m] = dvd

            def vd_answered(r, b):
                both = [j for j in range(len(ids)) if a[r][j] and a[b][j]]
                return 100 * sum(o[r][j] - o[b][j] for j in both) / len(both) if both else None
            ans = {k: 100 * sum(v) / len(ids) for k, v in a.items()}
            vd0, vdt = acc["r0"] - acc["b0"], acc["rt"] - acc["bt"]
            va0, vat = vd_answered("r0", "b0"), vd_answered("rt", "bt")
            fmt = lambda v: "–" if v is None else f"{v:+.1f}"
            lines.append(f"| {m} | {acc['r0']:.1f} | {acc['b0']:.1f} | {acc['rt']:.1f} | {acc['bt']:.1f} | {vd0:+.1f} | "
                         f"{vdt:+.1f} | {acc['rt'] - acc['r0']:+.1f} | {ci(gain_blank)} | {ci(dvd)} | "
                         f"{ans['r0']:.0f}/{ans['b0']:.0f}/{ans['rt']:.0f}/{ans['bt']:.0f} | {fmt(va0)} / {fmt(vat)} |")
            csv_rows.append({"dataset": dataset, "pair": pair, "model": m, "n": len(ids),
                             **{f"acc_{k}": acc[k] for k in cells}, **{f"answered_{k}": ans[k] for k in cells},
                             "vd_0": vd0, "vd_top": vdt, "gain": acc["rt"] - acc["r0"],
                             "gain_blank": 100 * sum(gain_blank) / len(ids), "dvd": 100 * sum(dvd) / len(ids),
                             "vd_answered_0": va0, "vd_answered_top": vat})
    tests, pairs = {}, {} if dataset in exp.primary_excludes else exp.pairs   # excluded: no pair test
    if pairs:
        lines += ["", f"ΔVis = ΔVD_child − ΔVD_parent ({dataset}; + = the child's FT gain needs the frames more; "
                  "exploratory, Holm over this table only):", "", "| pair | child | ΔVis (95% CI) | p | Holm p |",
                  "|---|---|---|---|---|"]
    for pair, members in pairs.items():
        for child in members[1:]:
            if members[0] in deltas and child in deltas:
                diff = [c - p for p, c in zip(deltas[members[0]], deltas[child])]
                tests[(pair, child)] = (ci(diff), signflip_p(diff, clusters))
    adjusted = holm({k: v[1] for k, v in tests.items()}) if tests else {}
    for (pair, child), (value, p) in tests.items():
        lines.append(f"| {pair} | {child} | {value} | {p:.3g} | {adjusted[(pair, child)]:.3g} |")
    lines.append("")
    if csv_rows:
        _write_csv(exp.root / "ft_tables" / f"blank_2x2_{dataset}.csv", csv_rows)
    return lines


def _controls(exp: Experiment, results) -> list[str]:
    """Blank frames (control `blank`: the top-budget adapters, plus zero-shot -> the 2x2 of _blank_2x2 once every
    model's ZS blank run exists), extra units (1K x 3 epochs) and the image-list zero-shot anchor (control `images`):
    accuracy next to the same questions' main result."""
    lines = ["## Controls", ""]
    folders = {k: exp.folder(k) for k in ("blank", "images") if k in exp.controls}
    found = {k: collect(v) for k, v in folders.items() if v.exists()}

    def acc(table, dataset, model_id, ids=None):
        result = table.get((dataset, model_id, "offline"))
        if not result:
            return None, 0
        rows = result["rows"]
        ids = [i for i in (ids if ids is not None else rows) if i in rows]
        bench = _bench(dataset)
        return (100 * sum(outcome(bench, rows[i]) for i in ids) / len(ids) if ids else None), len(ids)

    text = lambda v: "–" if v is None else f"{v:.1f}"
    top = label(exp.top_budget)
    members = [m for ms in exp.pairs.values() for m in ms]
    for dataset in exp.datasets:
        blank = found.get("blank", {})
        rows = [(m, *acc(blank, dataset, model_id(m, dataset, exp.top_budget)))
                for m in members if (dataset, f"{m}@{dataset}-{top}", "offline") in blank]
        if rows and all((dataset, f"{m}@{dataset}-0", "offline") in blank for m in members):
            lines += _blank_2x2(exp, results, blank, folders["blank"], dataset, top)
        elif rows:
            lines += [f"Blank frames ({dataset}, {top} adapters; frames replaced by black ones of the same size):", "",
                      "| model | n | acc, real frames | acc, blank frames |", "|---|---|---|---|"]
            for m, value, n in rows:
                ids = list(blank[(dataset, f"{m}@{dataset}-{top}", "offline")]["rows"])
                real, _ = acc(results, dataset, f"{m}@{dataset}-{top}", ids)
                lines.append(f"| {m} | {n} | {text(real)} | {text(value)} |")
            lines.append("")
    for extra in exp.spec.get("extra_units") or []:
        dataset, base = extra["dataset"], label(int(extra["budget"]))
        lines += [f"{base} for {extra.get('epochs')} epochs ({dataset}, tag {extra['tag']}):", "",
                  f"| model | acc@{base} | acc@{base}-{extra['tag']} |", "|---|---|---|"]
        for m in extra["models"]:
            one, _ = acc(results, dataset, f"{m}@{dataset}-{base}")
            three, _ = acc(results, dataset, f"{m}@{dataset}-{base}-{extra['tag']}")
            lines.append(f"| {m} | {text(one)} | {text(three)} |")
        lines.append("")
    for (dataset, model, mode), result in sorted(found.get("images", {}).items()):
        mine, n = acc(found["images"], dataset, model)
        video, _ = acc(results, dataset, model, list(result["rows"]))
        lines += [f"Image list ({dataset}, {model}; frames as separate images, the official vLLM path): {text(mine)} "
                  f"on {n} questions vs {text(video)} with the frames as one video", ""]
    return lines if len(lines) > 2 else []


def write_report(exp: Experiment) -> Path:
    root = exp.root
    results = collect(root)
    labels = ["0"] + [label(b) for b in exp.budgets]
    tuned = labels[1:]
    lines = ["# Fine-tune adaptability", "",
             "Accuracy in %. **Δ = child − parent: + means the streaming child is better, − worse.** ΔGain(b) = "
             "Gain_child − Gain_parent = ΔAcc(b) − ΔAcc(0). Primary endpoint: ΔGain averaged over the fine-tuned budgets, Holm-"
             "corrected over all child × dataset cells. 95% CI = stratified cluster bootstrap; p = cluster sign-flip test. "
             "One training seed: the intervals cover test sampling only.", ""]
    acc_rows, pair_rows, primary = [], [], {}
    excluded = exp.primary_excludes
    for dataset in exp.datasets:
        if not any(k[0] == dataset for k in results):
            continue
        bench = _bench(dataset)
        samples, ids, clusters, strata = design(root, dataset)
        independent = not getattr(bench.module, "GROUP", None) or len(set(clusters)) == len(clusters)
        mcq = all(s.kind == "mcq" for s in samples)
        lines += [f"## {dataset} ({len(ids)} test questions)", "",
                  "| model | " + " | ".join(f"acc@{l}" for l in labels) + " | " + " | ".join(f"gain@{l}" for l in tuned)
                  + f" | norm gain@{tuned[-1]} |" + (" letter acc@0 | letter acc@" + tuned[-1] + " |" if mcq else ""),
                  "|---|" + "---|" * (len(labels) + len(tuned) + 1 + (2 if mcq else 0))]
        if dataset in excluded:
            lines[-2:-2] = [f"**Not in the primary endpoint:** {excluded[dataset]}", ""]
        outcomes = {}
        for pair, members in exp.pairs.items():
            for model in members:
                table = _outcomes(results, bench, dataset, model, labels, ids)
                outcomes[model] = table
                acc = {l: 100 * sum(v.values()) / len(ids) if len(v) == len(ids) else None for l, v in table.items()}
                gain = {l: acc[l] - acc["0"] if acc.get(l) is not None and acc.get("0") is not None else None for l in tuned}
                norm = (gain[tuned[-1]] / (100 - acc["0"])) if gain.get(tuned[-1]) is not None and acc["0"] < 100 else None
                cells = [acc.get(l) for l in labels] + [gain.get(l) for l in tuned] + [norm]
                letters = [_letter_accuracy(results, dataset, model, l, samples) for l in ("0", tuned[-1])] if mcq else []
                text = lambda v: "–" if v is None else f"{v:.1f}" if abs(v) >= 1 or v == 0 else f"{v:.2f}"
                lines.append(f"| {model} | " + " | ".join(text(v) for v in cells + letters) + " |")
                acc_rows.append({"dataset": dataset, "pair": pair, "model": model,
                                 **{f"acc_{l}": acc.get(l) for l in labels}, **{f"gain_{l}": gain.get(l) for l in tuned},
                                 "norm_gain": norm, **({"letter_acc_0": letters[0], "letter_acc_ft": letters[1]} if mcq else {})})
        for variant, parse in (getattr(bench.module, "SENSITIVITY", None) or {}).items():
            lines += ["", f"Accuracy {variant} (sensitivity):", "", "| model | " + " | ".join(f"acc@{l}" for l in labels) + " |",
                      "|---|" + "---|" * len(labels)]
            for model in [m for ms in exp.pairs.values() for m in ms]:
                accs = [_variant_accuracy(results, bench, dataset, model, l, samples, parse) for l in labels]
                lines.append(f"| {model} | " + " | ".join("–" if a is None else f"{a:.1f}" for a in accs) + " |")
        lines += ["", "| pair | child | " + " | ".join(f"ΔAcc@{l}" for l in labels) + " | "
                  + " | ".join(f"ΔGain@{l}" for l in tuned) + " | primary (95% CI) | p |", "|---|---|" + "---|" * (len(labels) + len(tuned) + 2)]
        for pair, members in exp.pairs.items():
            parent = members[0]
            for child in members[1:]:
                P, C = outcomes[parent], outcomes[child]
                if not all(len(P.get(l, {})) == len(ids) and len(C.get(l, {})) == len(ids) for l in labels):
                    lines.append(f"| {pair} | {child} | incomplete |")
                    continue
                cell = {}
                for l in labels:
                    diff = [C[l][i] - P[l][i] for i in ids]
                    low, high = bootstrap_ci(diff, clusters, strata)
                    b = sum(P[l][i] and not C[l][i] for i in ids)
                    c = sum(C[l][i] and not P[l][i] for i in ids)
                    cell[f"d_acc_{l}"] = (100 * sum(diff) / len(ids), 100 * low, 100 * high,
                                            mcnemar_p(b, c) if independent else None)
                for l in tuned:
                    diff = [(C[l][i] - C["0"][i]) - (P[l][i] - P["0"][i]) for i in ids]
                    low, high = bootstrap_ci(diff, clusters, strata)
                    cell[f"d_gain_{l}"] = (100 * sum(diff) / len(ids), 100 * low, 100 * high, None)
                main = [sum((C[l][i] - C["0"][i]) - (P[l][i] - P["0"][i]) for l in tuned) / len(tuned) for i in ids]
                low, high = bootstrap_ci(main, clusters, strata)
                p = signflip_p(main, clusters)
                value = {"pair": pair, "value": 100 * sum(main) / len(ids), "low": 100 * low, "high": 100 * high, "p": p}
                if dataset not in excluded:
                    primary[(dataset, child)] = value
                fmt = lambda t: f"{t[0]:+.1f} [{t[1]:+.1f}, {t[2]:+.1f}]" + (f" p={t[3]:.2g}" if t[3] is not None else "")
                lines.append(f"| {pair} | {child} | " + " | ".join(fmt(cell[f'd_acc_{l}']) for l in labels) + " | "
                             + " | ".join(fmt(cell[f'd_gain_{l}']) for l in tuned)
                             + f" | {value['value']:+.1f} [{100 * low:+.1f}, {100 * high:+.1f}] | {p:.3g} |")
                pair_rows.append({"dataset": dataset, "pair": pair, "parent": parent, "child": child,
                                  **{k: v[0] for k, v in cell.items()}, **{f"{k}_ci": f"{v[1]:.2f},{v[2]:.2f}" for k, v in cell.items()},
                                  "primary": value["value"], "primary_p": p, "in_primary": dataset not in excluded})
        lines.append("")
    if primary:
        adjusted = holm({k: v["p"] for k, v in primary.items()})
        lines += ["## Primary endpoint (budget-averaged ΔGain; + = child adapts more)", "",
                  "| dataset | pair | child | ΔGain avg | 95% CI | p | Holm p |", "|---|---|---|---|---|---|---|"]
        for (dataset, child), v in primary.items():
            lines.append(f"| {dataset} | {v['pair']} | {child} | {v['value']:+.2f} | [{v['low']:+.2f}, {v['high']:+.2f}] | "
                         f"{v['p']:.3g} | {adjusted[(dataset, child)]:.3g} |")
        lines.append("")
    lines += _controls(exp, results)
    for name, rows in (("accuracy", acc_rows), ("pairs", pair_rows)):
        if rows:
            _write_csv(root / "ft_tables" / f"{name}.csv", rows)
    target = root / "FT_RESULTS.md"
    atomic_text(target, "\n".join(lines) + "\n")
    if acc_rows and pair_rows:
        atomic_text(root / "ft_tables" / "results.json", json.dumps(paper_data(exp), indent=1) + "\n")
    return target


# ---------- ft_tables/results.json: the paper's fine-tuning data ----------

def _sections(text: str) -> dict:
    """'## <dataset> (<n> test questions)' -> (n, the section's text)."""
    return {m[1]: (int(m[2]), m[3])
            for m in re.finditer(r"^## (\w+) \((\d+) test questions\)\n(.*?)(?=^## |\Z)", text, re.S | re.M)}


def _first_table(section: str) -> list[list[str]]:
    lines = section.split("| model |", 1)[1].split("\n\n", 1)[0].splitlines()
    rows = ["| model |" + lines[0]] + lines[2:]
    return [[c.strip() for c in l.strip().strip("|").split(" | ")] for l in rows if l.startswith("|")]


def paper_data(exp: Experiment) -> dict:
    """The fine-tuning entries of the paper's evaluation data, read back from FT_RESULTS.md and ft_tables/ with the
    rules that produced the paper's tables (Supp. Tables eval_finetuning and eval_finetuning_gains read the first two):
      finetuning_accuracy            {dataset: {model: {"acc@0", ..., "gain@5k", "norm gain@5k"[, letter acc]}}}: the
                                     first table of each dataset section, the numbers as printed (a missing cell: null)
      finetuning_primary             the primary-endpoint table's rows as printed (DeltaGain, CI, p, Holm)
      finetuning_max_common_budget   per pair at the top budget: zero-shot and final accuracy, gains, dGain and dAcc
                                     with their 95% intervals (2 decimals, as in pairs.csv), the budget-averaged dGain
                                     and its uncorrected p, and the accuracy / dGain curve over the budgets
    Datasets in the order primary first (readout_datasets)."""
    text = (exp.root / "FT_RESULTS.md").read_text(encoding="utf-8")
    sections = _sections(text)
    accuracy = {}
    for dataset in readout_datasets(exp)[1]:
        if dataset in sections:
            header, *rows = _first_table(sections[dataset][1])
            accuracy[dataset] = {r[0]: {k: None if v == "–" else float(v) for k, v in zip(header[1:], r[1:])}
                                 for r in rows}
    primary_section = text.split("## Primary endpoint", 1)[1].split("\n## ", 1)[0] if "## Primary endpoint" in text else ""
    primary = [dict(zip(["dataset", "pair", "child", "DeltaGain", "CI", "p", "Holm"],
                        [c.strip() for c in l.strip().strip("|").split(" | ")]))
               for l in primary_section.splitlines() if re.match(r"\| \w+ \| \w+ \| ", l) and "---" not in l
               and not l.startswith("| dataset |")]
    with open(exp.root / "ft_tables" / "accuracy.csv", newline="", encoding="utf-8") as handle:
        acc = {(r["dataset"], r["model"]): r for r in csv.DictReader(handle)}
    with open(exp.root / "ft_tables" / "pairs.csv", newline="", encoding="utf-8") as handle:
        pairs = list(csv.DictReader(handle))
    budgets = sorted(exp.budgets)
    ci = lambda s: [float(x) for x in s.split(",")]
    common = []
    for r in pairs:
        d, parent, child = r["dataset"], r["parent"], r["child"]
        top = max(b for b in budgets if r.get(f"d_gain_{label(b)}") not in (None, ""))
        a_p, a_c, t = acc[(d, parent)], acc[(d, child)], label(top)
        common.append({
            "dataset": d, "pair": r["pair"], "parent": parent, "child": child, "n_test": sections[d][0],
            "train_budget": top, "train_epochs": exp.train["epochs"], "training_seeds": 1,
            "parent_zero_shot": float(a_p["acc_0"]), "child_zero_shot": float(a_c["acc_0"]),
            "parent_final": float(a_p[f"acc_{t}"]), "child_final": float(a_c[f"acc_{t}"]),
            "parent_gain": float(a_p[f"gain_{t}"]), "child_gain": float(a_c[f"gain_{t}"]),
            "delta_gain_pp": float(r[f"d_gain_{t}"]), "delta_gain_ci95": ci(r[f"d_gain_{t}_ci"]),
            "delta_final_pp": float(r[f"d_acc_{t}"]), "delta_final_ci95": ci(r[f"d_acc_{t}_ci"]),
            "budget_average_delta_gain_pp": float(r["primary"]), "budget_average_p_uncorrected": float(r["primary_p"]),
            "in_original_primary": r["in_primary"] == "True",
            "curve": {label(b): {"parent": float(a_p[f"acc_{label(b)}"]), "child": float(a_c[f"acc_{label(b)}"]),
                                 "delta_gain": float(r[f"d_gain_{label(b)}"]),
                                 "delta_gain_ci95": ci(r[f"d_gain_{label(b)}_ci"])} for b in budgets}})
    return {"finetuning_accuracy": accuracy, "finetuning_primary": primary, "finetuning_max_common_budget": common}


# ---------- READOUT.md: post-fine-tuning accuracy ----------

def _table(header: list[str], rows: list[list]) -> list[str]:
    text = lambda v: "–" if v is None else str(v)
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(text(v) for v in r) + " |" for r in rows)]


def _ci(value, low, high, digits: int = 1) -> str:
    if value is None:
        return "–"
    if low is None:
        return f"{value:+.{digits}f}"
    return f"{value:+.{digits}f} [{low:+.{digits}f}, {high:+.{digits}f}]"


def _p(value) -> str:
    return "–" if value is None else f"{value:.2g}"


def readout_datasets(exp: Experiment) -> tuple[list[str], list[str]]:
    """(primary datasets, all datasets): primary first, as the readout orders its cells and per-dataset seeds."""
    primary = [d for d in exp.datasets if d not in exp.primary_excludes]
    return primary, primary + [d for d in exp.datasets if d in exp.primary_excludes]


def _readout_outcomes(results, bench, dataset: str, model: str, lab: str, ids: list[str]):
    """0/1 per question in `ids` order; None while a question is missing or unscored."""
    import numpy as np
    result = results.get((dataset, f"{model}@{dataset}-{lab}", "offline"))
    rows = result["rows"] if result else {}
    if any(i not in rows for i in ids):
        return None
    return np.array([float(outcome(bench, rows[i])) for i in ids])


def analyse(exp: Experiment, results: dict | None = None) -> dict:
    import numpy as np
    from ...analysis.stats_index import summary
    results = collect(exp.root) if results is None else results
    labs = ["0"] + [label(b) for b in exp.budgets]
    tuned = labs[1:]
    primary_sets, datasets = readout_datasets(exp)
    pairs = [(p, ms[0], c) for p, ms in exp.pairs.items() for c in ms[1:]]
    cells, values = {}, {}
    for dataset in datasets:
        bench = _bench(dataset)
        samples, ids, clusters, strata = design(exp.root, dataset)
        variants = {"all": np.ones(len(ids), bool)}
        if dataset == "lingoqa":
            variants["no_fixed"] = np.array([hashlib.sha256(s.question.strip().encode()).hexdigest()
                                             != FIXED_QUESTION_SHA256 for s in samples])
        cache = {}
        for pair, parent, child in pairs:
            for m in (parent, child):
                if m not in cache:
                    cache[m] = {b: _readout_outcomes(results, bench, dataset, m, b, ids) for b in labs}
            P, C = cache[parent], cache[child]
            if any(v is None for v in [*P.values(), *C.values()]):
                cells[(dataset, "all", child)] = None
                continue
            for variant, keep in variants.items():
                d = {b: (C[b] - P[b])[keep] for b in P}
                avg = np.mean([d[b] for b in tuned], axis=0)
                contrasts = {"d_acc_0": d["0"], "d_acc_avg": avg, "d_acc_5k": d[tuned[-1]], "d_gain_avg": avg - d["0"]}
                cl = [c for c, k in zip(clusters, keep) if k]
                st = [s for s, k in zip(strata, keep) if k]
                entry = {"pair": pair, "parent": parent, "child": child, "dataset": dataset, "variant": variant,
                         "n": int(keep.sum()), "acc_parent": {b: 100 * P[b][keep].mean() for b in labs},
                         "acc_child": {b: 100 * C[b][keep].mean() for b in labs}}
                for key, v in contrasts.items():
                    entry[key] = summary(v, cl, st, alphas=ALPHAS)
                cells[(dataset, variant, child)] = entry
                values[(dataset, variant, child)] = (contrasts, cl, st)
    family12 = {k: v["d_gain_avg"]["p"] for k, v in cells.items() if v and k[1] == "all" and k[0] in primary_sets}
    family18 = {k: v["d_gain_avg"]["p"] for k, v in cells.items() if v and k[1] == "all"}
    for name, family in (("holm12", family12), ("holm18", family18)):
        for k, p in holm(family).items():
            cells[k]["d_gain_avg"][name] = p
    for k, p in holm({k: v["d_acc_avg"]["p"] for k, v in cells.items()
                      if v and k[1] == "all" and k[0] in primary_sets}).items():
        cells[k]["d_acc_avg"]["holm12"] = p
    return {"cells": cells, "pooled": pooled(exp, values), "labels": labs}


def pooled(exp: Experiment, values: dict) -> dict:
    """Per child: equal-weight mean over datasets of dGain_avg, dAcc_avg and dAcc_0 (primary datasets and all)."""
    import numpy as np
    from ...analysis.stats_index import means
    primary_sets, datasets = readout_datasets(exp)
    out = {}
    for _, parent, child in [(p, ms[0], c) for p, ms in exp.pairs.items() for c in ms[1:]]:
        for name, chosen in (("primary", primary_sets), ("all", datasets)):
            if not all((d, "all", child) in values for d in chosen):
                continue
            for key in ("d_gain_avg", "d_acc_avg", "d_acc_0"):
                boots, weighted, clusters, point = [], [], [], 0.0
                for seed, d in enumerate(chosen):
                    v, cl, st = values[(d, "all", child)]
                    v = v[key]
                    boots.append(100 * means(v, cl, st, seed=seed))
                    point += 100 * v.mean() / len(chosen)
                    weighted += list(v / (len(v) * len(chosen)))
                    clusters += [f"{d}:{c}" for c in cl]
                boot = np.mean(boots, axis=0)
                out[(name, child, key)] = {"value": point, "low": float(np.percentile(boot, 2.5)),
                                           "high": float(np.percentile(boot, 97.5)),
                                           "p": signflip_p(list(map(float, weighted)), clusters)}
    return out


def reading(entry: dict, primary_sets: list[str]) -> str:
    """A Holm-significant dGain (Holm-12; outside the primary endpoint: Holm-18) whose dAcc_avg interval includes 0 is
    a zero-shot difference."""
    g, a = entry["d_gain_avg"], entry["d_acc_avg"]
    p = g.get("holm12") if entry["dataset"] in primary_sets else g.get("holm18")
    acc_zero = a["low"] <= 0 <= a["high"]
    if p is not None and p < 0.05:
        return "zero-shot difference (post-FT parity)" if acc_zero else "ΔGain and post-FT difference"
    return "no difference" if acc_zero else "post-FT difference only"


def consistency(exp: Experiment, cells: dict) -> float | None:
    """max |difference| between the readout's dGain_avg (value, p) and ft_tables/pairs.csv's primary / primary_p."""
    path = exp.root / "ft_tables" / "pairs.csv"
    if not path.is_file():
        return None
    worst = 0.0
    with open(path, encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            entry = cells.get((r["dataset"], "all", r["child"]))
            if entry:
                worst = max(worst, abs(entry["d_gain_avg"]["value"] - float(r["primary"])),
                            abs(entry["d_gain_avg"]["p"] - float(r["primary_p"])))
    return worst


def write_readout(exp: Experiment, result: dict | None = None) -> Path:
    result = analyse(exp) if result is None else result
    cells, pool = result["cells"], result["pooled"]
    primary_sets, _ = readout_datasets(exp)
    lb = f"lower_{0.05 / 12:g}"
    check = consistency(exp, cells)
    lines = ["# Fine-tuning read through post-fine-tuning accuracy", "",
             "Δ = child − parent in pp: **+ = the child is better, − = worse**. ΔGain_avg = ΔAcc_avg − ΔAcc_0 per "
             "question (the primary endpoint). 95 % stratified cluster bootstrap CI, cluster sign-flip p; LB-sim = "
             "Bonferroni-12 one-sided lower bound. "
             + (f"ΔGain_avg reproduces ft_tables/pairs.csv (max |Δ| over value and p = {check:.2g}). "
                if check is not None else "")
             + "One training seed: intervals cover test sampling only.", ""]
    for variant in ("all", "no_fixed"):
        rows = []
        for (d, v, child), e in cells.items():
            if v != variant or not e:
                continue
            fmt = lambda t: _ci(t["value"], t["low"], t["high"])
            g = e["d_gain_avg"]
            rows.append([d, e["pair"], child, e["n"], fmt(e["d_acc_0"]), fmt(e["d_acc_avg"]), f"{e['d_acc_avg'][lb]:+.1f}",
                         fmt(e["d_acc_5k"]), fmt(g), _p(g["p"]), _p(g.get("holm12")), _p(g.get("holm18")),
                         reading(e, primary_sets) if variant == "all" else "–"])
        if not rows:
            continue
        title = ("## Per child × dataset (all test questions)" if variant == "all" else
                 "## LingoQA without the fixed-format action/justification question (sensitivity)")
        lines += [title, "", *_table(["dataset", "pair", "child", "n", "ΔAcc_0", "ΔAcc_avg (1k/2k/5k)", "LB-sim ΔAcc_avg",
                                      "ΔAcc@5k", "ΔGain_avg = ΔAcc_avg − ΔAcc_0", "p", "Holm-12", "Holm-18 (pre-reg.)",
                                      "reading"], rows), ""]
    primary = [e for (d, v, _), e in cells.items() if e and v == "all" and d in primary_sets]
    if primary:
        worst = min(e["d_acc_avg"][lb] for e in primary)
        lines += [f"Non-inferiority of post-fine-tuning accuracy ({', '.join(primary_sets)}; {len(primary)} cells): "
                  f"worst Bonferroni-12 lower bound of ΔAcc_avg = {worst:+.1f} pp, i.e. every child's budget-averaged "
                  f"fine-tuned accuracy is at most {-worst:.1f} pp below its parent's, simultaneously; per dataset: "
                  + ", ".join(f"{d} {min(e['d_acc_avg'][lb] for e in primary if e['dataset'] == d):+.1f}"
                              for d in primary_sets) + ".", ""]
    rows = []
    for _, parent, child in [(p, ms[0], c) for p, ms in exp.pairs.items() for c in ms[1:]]:
        cellsrow = []
        for name in ("primary", "all"):
            for key in ("d_gain_avg", "d_acc_avg"):
                t = pool.get((name, child, key))
                cellsrow.append(f"{_ci(t['value'], t['low'], t['high'])} p={t['p']:.2g}" if t else "–")
        rows.append([child, *cellsrow])
    lines += ["## Pre-registered cross-dataset estimate (equal weight per dataset)", "",
              *_table(["child", "ΔGain_avg primary datasets", "ΔAcc_avg primary datasets", "ΔGain_avg all datasets",
                       "ΔAcc_avg all datasets"], rows), "",
              "Per-budget accuracies of every model: FT_RESULTS.md.", ""]
    flat = []
    for (d, v, child), e in cells.items():
        if e:
            for key in READOUT_KEYS:
                flat.append({"dataset": d, "variant": v, "pair": e["pair"], "child": child, "n": e["n"], "contrast": key,
                             **{k: x for k, x in e[key].items()}})
    _write_csv(exp.root / "ft_tables" / "cells.csv", flat)
    _write_csv(exp.root / "ft_tables" / "pooled.csv",
               [{"datasets": n, "child": c, "contrast": k, **v} for (n, c, k), v in pool.items()])
    target = exp.root / "READOUT.md"
    atomic_text(target, "\n".join(lines) + "\n")
    return target
