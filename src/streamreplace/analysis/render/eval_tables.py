"""Absolute-score tables of the supplement.

From metrics.json (`streamreplace report`; formal rows, no rerun overlay): eval_image, eval_video, eval_safety,
eval_vmdt. Values are the benchmarks' own metrics, rounded to 2 decimals by the benchmark and printed with 1 decimal
(|x| < 0.05 as 0.0), so e.g. AURA's streaming VSB benign ASR 24.65 prints as 75.3 after 100 - x. A video entry with
fewer completed rows than expected (generation errors) gets a double dagger.

From the prompt-robustness results.json (`streamreplace robustness report`): eval_robustness (per child x benchmark:
clean accuracies and dRob per perturbation family with its 95% interval) and eval_instruction (instruction-level
perturbations on the 100-unit MMBench selection).

From the fine-tuning results.json (`streamreplace finetune report`, ft_tables/results.json; the same schema as the
paper's evaluation data): eval_finetuning (accuracy of every model zero-shot and after each training budget, per
dataset) and eval_finetuning_gains (the primary endpoint: budget-averaged gain difference per child with its 95%
interval and Holm p, as the fine-tuning report printed them).

From the response-form audit table (paper/flags/common_response.csv; labels from an LLM audit of 400 VSB harmful
items, shipped as numbers): eval_common_response.

The four absolute-score tables come with a CSV of the unrounded values; the generalization tables are rendered from
values their reports already rounded."""
from __future__ import annotations

import csv
from pathlib import Path

from .latex import absolute, delta, tex_row, tex_table

IMAGE = [("dynamath", "avg_acc"), ("dynamath", "worst_acc"), ("mathvista", "acc"), ("mmbench", "circular_acc"),
         ("mme", "perception"), ("mme", "cognition"), ("mmmu", "acc"), ("mmstar", "acc"), ("scienceqa", "acc"),
         ("visulogic", "acc")]
VIDEO = [("videomme", "acc"), ("tomato", "acc"), ("timeblind", "I_Acc"), ("videoreasonbench", "acc")]
SAFETY = [("video_safetybench", "asr_harmful"), ("video_safetybench", "asr_benign"),
          ("mcv_safetybench", "asr_explicit"), ("mcv_safetybench", "asr_implicit")]
VMDT = [("vmdt_fairness", "fairness"), ("vmdt_fairness", "stereotype"), ("vmdt_fairness", "decision"),
        ("vmdt_fairness", "factual"), ("vmdt_privacy", "privacy"), ("vmdt_privacy", "acc/state"),
        ("vmdt_privacy", "acc/city"), ("vmdt_privacy", "acc/zipcode")]
DATASET_NAMES = {"videomme": "Video-MME", "tomato": "TOMATO", "timeblind": "TimeBlind",
                 "videoreasonbench": "VideoReasonBench", "mmbench": "MMBench"}
ROBUST_FAMILIES = ("typo", "para", "distraction")
INSTRUCTION_FAMILIES = ("typo", "para", "irrelevant")
INSTRUCTION_UNITS = 100            # the instruction-level table uses the 100-unit MMBench selection
FINETUNE_DATASETS = (("lingoqa", "LingoQA"), ("eyepcr", "EyePCR"), ("openuav_qa", "OpenUAV-QA"))
FINETUNE_BUDGETS = (("acc@0", "0"), ("acc@1k", "1K"), ("acc@2k", "2K"), ("acc@5k", "5K"))

CAPTIONS = {
    "eval_image": r"Image understanding under offline evaluation. Higher is better. Scores are percentages except MME perception (/2000) and cognition (/800). DynaMath Worst requires all ten variants of a seed problem to be correct; MMBench uses circular accuracy. Scores include the recorded judge-assisted extraction where applicable.",
    "eval_video": r"Video understanding (\%, higher is better). Video-MME uses the short subset without subtitles (900 questions); TOMATO has 1,484 questions; TimeBlind has 600 four-query instances; VideoReasonBench has 1,440 questions. $\dagger$: MMDuet streaming includes the extra forced-answer step.",
    "eval_safety": r"Safety attack success rates (\%, lower is better). HarmBench has 510 requests (400 textual and 110 multimodal), VSB has 1,132 examples per split, and MCV has 1,460 per split. The benign-query VSB split still combines the query with potentially harmful visual context. VSB and MCV are scored with Qwen2.5-72B-Instruct RJScore $>2.85$; MCV therefore differs from its original paper's judge and score threshold. $S^\dagger$: modified MMDuet streaming. Low ASR must be interpreted together with response behavior.",
    "eval_vmdt": r"VMDT results (\%). Fairness averages stereotype, decision, and factual/overkill scores over 1,063, 753, and 184 items, respectively. Privacy uses 600 questions from 200 videos; State/City/Zip are location-inference accuracies. Privacy $=100-(\mathrm{State}+2\mathrm{City}+4\mathrm{Zip})/7$. Missing or unparseable responses affect these metrics, and higher aggregate scores alone do not establish more trustworthy behavior. $S^\dagger$: modified MMDuet streaming.",
    "eval_robustness": r"Question robustness under offline inference. Clean scores are accuracy (\%) on 300 questions per benchmark (MMBench uses circular units). Perturbation columns give $\Delta_{\mathrm{Rob}}=\mathrm{Drop}_P-\mathrm{Drop}_C$ in percentage points with 95\% cluster-bootstrap intervals; positive means the child loses less accuracy. Paraphrase comparisons use matched valid edits{paraphrase_units} and their matched clean baselines. Distraction averages stress and irrelevant-content conditions within each item. These per-benchmark contrasts are secondary; the pooled primary tests are described in the text.",
    "eval_instruction": r"Instruction perturbations on the selected MMBench subset: absolute parent/child accuracy (\%) and $\Delta_{\mathrm{Rob}}$ (percentage points). Typo and irrelevant-text tests use {units} circular units; the matched paraphrase comparison uses {paraphrase}. Confidence intervals and drops use the corresponding matched clean subsets.{holm}",
    "eval_common_response": r"Refusal rates (\%) on pairwise common substantive-response subsets of the fixed 400-item VSB harmful audit. Both replies must be labeled explicit refusal, safe deflection, or harmful compliance (R/D/C); the refusal-rate measure combines R+D. Empty/broken and off-task replies are excluded. Each pair has its own common subset. The displayed exact McNemar $p$ values are unadjusted; this local-judge audit is separate from the full-benchmark RJScore evaluation.",
    "eval_finetuning": r"Downstream accuracy (\%) before and after fine-tuning on 1K, 2K, and 5K examples. Each budget uses a separately trained, one-epoch LoRA adapter with one training seed; P/C identify the initialization checkpoint, and all runs use offline inference. MMDuet uses its merged checkpoint for fine-tuning. EyePCR is sensitive to zero-shot answer formatting and text priors; OpenUAV-QA is excluded from the primary adaptability endpoint because blank-frame controls nearly saturate it.",
    "eval_finetuning_gains": r"Budget-averaged fine-tuning gain differences, $\overline{\Delta\mathrm{Gain}}=\frac{1}{3}\sum_{b\in\{1K,2K,5K\}}[(C_b-C_0)-(P_b-P_0)]$, in percentage points. The recorded primary analysis applies Holm correction jointly over all {tests} child--dataset comparisons in {datasets}{excluded}. Intervals reflect test-set sampling, with one training seed, and larger gains must be interpreted alongside starting and final accuracy and visual dependence.",
}


class Metrics:
    """Lookup over metrics.json rows."""

    def __init__(self, rows: list[dict]):
        self.rows = {(r["dataset"], r["model"], r["mode"]): r for r in rows}

    def value(self, model: str, mode: str, dataset: str, name: str):
        r = self.rows.get((dataset, model, mode))
        return None if r is None else r["metrics"].get(name)

    def row(self, model: str, mode: str, dataset: str) -> dict | None:
        return self.rows.get((dataset, model, mode))

    def incomplete(self, model: str, mode: str, dataset: str) -> bool:
        r = self.rows.get((dataset, model, mode))
        return r is not None and r["done"] < r["expected"]


def _models(profile) -> list[dict]:
    parents = set(profile.parents.values())
    return [{"id": m, "name": profile.full_name(m), "role": "P" if m in parents else "C",
             "parent": profile.parents.get(m)} for m in profile.models]


def _modes(profile, models):
    dagger = set(profile.display.get("dagger_stream", []))
    for m in models:
        yield m, "offline", m["role"]
        if m["role"] == "C":
            yield m, "stream", r"$S^\dagger$" if m["id"] in dagger else "S"


def _words(n: int) -> str:
    words = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
             "seventeen eighteen nineteen twenty").split()
    return words[n] if 0 <= n < len(words) else f"{n:,}"


def _video_caption(incomplete: list[tuple[str, str, dict]]) -> str:
    """The eval_video caption, with one sentence per entry that has incomplete coverage."""
    if not incomplete:
        return CAPTIONS["eval_video"]
    notes = []
    for name, benchmark, r in incomplete:
        missing = r["expected"] - r["done"]
        what = f"{_words(r['errors'])} {name} {benchmark} generation errors" if r["errors"] == missing else \
            f"{_words(missing)} missing or failed {name} {benchmark} rows"
        notes.append(f"{what} leave {r['done']:,}/{r['expected']:,} completed queries")
    return (CAPTIONS["eval_video"] + r" $\ddagger$: " + "; ".join(notes)
            + ("; this entry has" if len(notes) == 1 else "; these entries have") + " incomplete coverage.")


def build(metrics_rows: list[dict], profile) -> dict:
    """name -> (LaTeX text, csv rows, csv fields) of the four absolute-score tables."""
    data, models = Metrics(metrics_rows), _models(profile)
    children = [m for m in models if m["role"] == "C"]
    dagger = set(profile.display.get("dagger_stream", []))
    zero = profile.display.get("zero_below", 0.05)
    num = lambda v: absolute(v, zero)  # noqa: E731
    short = {m["id"]: profile.name_of(m["id"]) for m in models}
    out = {}

    def records(entries, columns):
        rows = []
        for model, mode, tag in entries:
            row = {"model": model, "mode": mode, "condition": tag}
            row.update({f"{d}/{k}": data.value(model, mode, d, k) for d, k in columns})
            rows.append(row)
        return rows, ["model", "mode", "condition"] + [f"{d}/{k}" for d, k in columns]

    out["eval_image"] = (tex_table(
        "ll" + "r" * 10,
        [tex_row(["Model", "Mode", r"\shortstack{DynaMath\\Avg.}", r"\shortstack{DynaMath\\Worst}",
                  "MathVista", "MMBench", r"\shortstack{MME\\Perc.}", r"\shortstack{MME\\Cog.}",
                  "MMMU", "MMStar", "ScienceQA", "VisuLogic"])],
        [tex_row([m["name"], m["role"]] + [num(data.value(m["id"], "offline", d, k)) for d, k in IMAGE]) for m in models],
        CAPTIONS["eval_image"], "tab:eval-image", colsep=1.5),
        *records([(m["id"], "offline", m["role"]) for m in models], IMAGE))
    video_rows, incomplete, video_entries = [], [], []
    for m in children:
        cells = [m["name"] + (r"$^\dagger$" if m["id"] in dagger else "")]
        for d, k in VIDEO:
            for (model, mode), tag in zip([(m["parent"], "offline"), (m["id"], "offline"), (m["id"], "stream")], "PCS"):
                text = num(data.value(model, mode, d, k))
                if data.incomplete(model, mode, d):
                    text += r"$^\ddagger$"
                    entry = (short[model], DATASET_NAMES.get(d, d), data.row(model, mode, d))
                    if entry[:2] not in [e[:2] for e in incomplete]:
                        incomplete.append(entry)
                cells.append(text)
        video_rows.append(tex_row(cells))
        video_entries += [(m["parent"], "offline", f"P of {m['id']}"), (m["id"], "offline", "C"), (m["id"], "stream", "S")]
    out["eval_video"] = (tex_table(
        "l" + "r" * 12,
        [tex_row(["Child", r"\multicolumn{3}{c}{Video-MME short}", r"\multicolumn{3}{c}{TOMATO}",
                  r"\multicolumn{3}{c}{TimeBlind $I_{\mathrm{Acc}}$}", r"\multicolumn{3}{c}{VideoReasonBench}"]),
         tex_row([""] + ["P", "C", "S"] * 4)], video_rows, _video_caption(incomplete), "tab:eval-video"),
        *records(video_entries, VIDEO))
    safety_rows, trust_rows, entries = [], [], []
    for m, mode, tag in _modes(profile, models):
        mid = m["id"]
        hb = num(data.value(mid, mode, "harmbench", "asr")) if mode == "offline" else "--"
        safety_rows.append(tex_row([m["name"], tag, hb] + [num(data.value(mid, mode, d, k)) for d, k in SAFETY]))
        trust_rows.append(tex_row([m["name"], tag] + [num(data.value(mid, mode, d, k)) for d, k in VMDT]))
        entries.append((mid, mode, tag))
    out["eval_safety"] = (tex_table(
        "llrrrrr",
        [tex_row(["Model", "Mode", "HarmBench", r"\multicolumn{2}{c}{Video-SafetyBench}", r"\multicolumn{2}{c}{MCV-SafetyBench}"]),
         tex_row(["", "", "ASR", "Harmful", "Benign-query", "Explicit", "Implicit"])], safety_rows,
        CAPTIONS["eval_safety"], "tab:eval-safety"), *records(entries, [("harmbench", "asr")] + SAFETY))
    out["eval_vmdt"] = (tex_table(
        "ll" + "r" * 8,
        [tex_row(["Model", "Mode", r"\multicolumn{4}{c}{VMDT Fairness}", r"\multicolumn{4}{c}{VMDT Privacy}"]),
         tex_row(["", "", r"Overall $\uparrow$", r"Stereo. $\uparrow$", r"Decision $\uparrow$", r"Factual $\uparrow$",
                  r"Overall $\uparrow$", r"State $\downarrow$", r"City $\downarrow$", r"Zip $\downarrow$"])], trust_rows,
        CAPTIONS["eval_vmdt"], "tab:eval-vmdt"), *records(entries, VMDT))
    return out


# ---------- prompt robustness (results.json of `streamreplace robustness report`) ----------

def _interval(value, ci, zero: float) -> str:
    return delta(value, zero) + " [" + ", ".join(delta(x, zero) for x in ci) + "]"


def robustness(document: dict, profile) -> dict:
    """name -> LaTeX text of eval_robustness and eval_instruction (whichever the document has)."""
    zero = profile.display.get("zero_below", 0.05)
    num = lambda v: absolute(v, zero)  # noqa: E731
    out = {}
    rows = document.get("robustness_by_benchmark") or []
    if rows:
        body = [tex_row([profile.full_name(r["child"]), DATASET_NAMES.get(r["dataset"], r["dataset"]),
                         num(r["clean_parent"]), num(r["clean_child"])]
                        + [_interval(r[k]["delta_rob_pp"], r[k]["delta_rob_ci95"], zero) if k in r else "--"
                           for k in ROBUST_FAMILIES]) for r in rows]
        units = {}
        for r in rows:
            if "para" in r:
                units.setdefault(r["dataset"], r["para"]["n"])
        para = (" (" + " and ".join(f"{n} {DATASET_NAMES.get(d, d)}" for d, n in units.items()) + " units)") if units else ""
        out["eval_robustness"] = tex_table(
            "llrrrrr", [tex_row(["Child", "Benchmark", "Clean P", "Clean C", "Typo", "Paraphrase", "Distraction"])],
            body, CAPTIONS["eval_robustness"].replace("{paraphrase_units}", para), "tab:eval-robustness")
    groups = [g for g in document.get("instruction_robustness") or [] if g.get("units") == INSTRUCTION_UNITS]
    if groups:
        body, holm_p, n = [], [], {}
        for r in groups[0]["rows"]:
            cells = [profile.full_name(r["child"])]
            for k in INSTRUCTION_FAMILIES:
                x = r.get(k)
                if x is None:
                    cells += ["--"] * 3
                    continue
                cells += [num(x["instruction_parent"]), num(x["instruction_child"]),
                          _interval(x["delta_rob_pp"], x["delta_rob_ci95"], zero)]
                holm_p.append(x.get("p_holm"))
                n.setdefault(k, x["n"])
            body.append(tex_row(cells))
        holm = (f" All {len(holm_p)} instruction-level parent--child tests have Holm-adjusted $p=1$."
                if holm_p and all(p == 1 for p in holm_p) else "")
        caption = (CAPTIONS["eval_instruction"].replace("{units}", str(n.get("typo", INSTRUCTION_UNITS)))
                   .replace("{paraphrase}", str(n.get("para", "--"))).replace("{holm}", holm))
        out["eval_instruction"] = tex_table(
            "l" + "r" * 9,
            [tex_row(["Child", r"\multicolumn{3}{c}{Instruction typo}", r"\multicolumn{3}{c}{Instruction paraphrase}",
                      r"\multicolumn{3}{c}{Instruction irrelevant text}"]),
             tex_row([""] + ["P", "C", r"$\Delta_{\mathrm{Rob}}$ [95\% CI]"] * 3)], body, caption,
            "tab:eval-instruction")
    return out


# ---------- fine-tuning (ft_tables/results.json of `streamreplace finetune report`) ----------

def finetune_id(model: str, profile) -> str:
    """The id a model's fine-tuning results are reported under (display.finetune_ids: MMDuet was fine-tuned as its
    LoRA-merged checkpoint), else the model id itself."""
    return (profile.display.get("finetune_ids") or {}).get(model, model)


def _count(n: int) -> str:
    """2000 -> '2{,}000' (the paper's thousands separator in math mode)."""
    return f"{n:,}".replace(",", "{,}")


def _joined(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def finetuning(document: dict, profile) -> dict:
    """name -> LaTeX text of eval_finetuning and eval_finetuning_gains (whichever the document has). The accuracies
    are the 1-decimal values the fine-tuning report printed; the gains table copies the report's ΔGain, interval and
    Holm strings unchanged."""
    zero = profile.display.get("zero_below", 0.05)
    num = lambda v: absolute(v, zero)  # noqa: E731
    out = {}
    accuracy = document.get("finetuning_accuracy") or {}
    sizes = {r["dataset"]: r["n_test"] for r in document.get("finetuning_max_common_budget") or [] if "n_test" in r}
    datasets = [(d, label) for d, label in FINETUNE_DATASETS if d in accuracy]
    if datasets:
        body = []
        for m in _models(profile):
            cells = [m["name"], m["role"]]
            for d, _ in datasets:
                x = accuracy[d].get(finetune_id(m["id"], profile)) or {}
                cells += [num(x.get(k)) for k, _ in FINETUNE_BUDGETS]
            body.append(tex_row(cells))
        groups = [r"\multicolumn{4}{c}{" + label + (f" ($n={_count(sizes[d])}$)" if d in sizes else "") + "}"
                  for d, label in datasets]
        out["eval_finetuning"] = tex_table(
            "ll" + "r" * (4 * len(datasets)),
            [tex_row(["Model", "Init."] + groups), tex_row(["", ""] + [b for _, b in FINETUNE_BUDGETS] * len(datasets))],
            body, CAPTIONS["eval_finetuning"], "tab:eval-finetuning")
    primary = document.get("finetuning_primary") or []
    tested = [(d, label) for d, label in FINETUNE_DATASETS if any(r["dataset"] == d for r in primary)]
    if tested:
        index = {(r["dataset"], r["child"]): r for r in primary}
        body = []
        for child in profile.children:
            cells = [profile.full_name(child)]
            for d, _ in tested:
                x = index.get((d, finetune_id(child, profile)))
                cells += [x["DeltaGain"], x["CI"], x["Holm"]] if x else ["--"] * 3
            body.append(tex_row(cells))
        excluded = [label for d, label in datasets if d not in dict(tested)]
        caption = (CAPTIONS["eval_finetuning_gains"].replace("{tests}", str(len(primary)))
                   .replace("{datasets}", _joined([label for _, label in tested]))
                   .replace("{excluded}", f"; {_joined(excluded)} {'is' if len(excluded) == 1 else 'are'} excluded"
                            if excluded else ""))
        gain = r"$\overline{\Delta\mathrm{Gain}}$"
        out["eval_finetuning_gains"] = tex_table(
            "l" + "r" * (3 * len(tested)),
            [tex_row(["Child"] + [r"\multicolumn{3}{c}{" + label + "}" for _, label in tested]),
             tex_row([""] + [gain, r"95\% CI", r"Holm $p$"] * len(tested))],
            body, caption, "tab:eval-finetuning-gains")
    return out


# ---------- response-form audit ----------

def common_response(path: Path, profile, dataset: str = "video_safetybench", mode: str = "offline") -> str:
    """eval_common_response from the audit table (columns dataset, split, child, mode, n, parent_rate, child_rate,
    p): one row per child, the offline (C vs P) comparison on VSB harmful."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = {(r["dataset"], r["child"], r["mode"]): r for r in csv.DictReader(f)}
    zero = profile.display.get("zero_below", 0.05)
    body = []
    for child in profile.children:
        r = rows.get((dataset, child, mode))
        if r is None:
            body.append(tex_row([profile.full_name(child), "--", "--", "--", "--"]))
            continue
        body.append(tex_row([profile.full_name(child), int(r["n"]), absolute(float(r["parent_rate"]), zero),
                             absolute(float(r["child_rate"]), zero), f'{float(r["p"]):.3g}']))
    return tex_table("lrrrr", [tex_row(["Child", "Common $n$", "Parent refusal", "Child refusal", "Exact $p$"])], body,
                     CAPTIONS["eval_common_response"], "tab:eval-common-response")
