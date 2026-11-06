"""The frozen prompt edits of the prompt-robustness conditions: generation, the per-folder override files, checks and
audit pages (what the models of Supp. Tables eval_robustness / eval_instruction were shown).

Every condition except clean gets `<root>/<condition>/prompt_overrides.jsonl`, one row per sample of the folder's
subset, in unit order:
    {"dataset", "id", "unit", "condition", "edits": [[old, new]], "skip": null | reason}
The runner applies `edits` to the benchmark's rendered prompt with data/prompts.py (each `old` must occur exactly once)
for every run on that folder, so the runner knows nothing about perturbations, and the file's fingerprint enters the
run identity. A skip row (a paraphrase that failed its checks) has no edit: the model sees the clean prompt and the
analysis leaves that unit out of the condition for every model. All samples of a unit (MMBench: the rotations of one
question) get the same edit.

Edits per condition `edit` kind (the seed string is "<seed>|<dataset>|<unit>|<condition, or its seed_as>"):
    typo        perturb.typo on the stem (rate 0.2 of the eligible words, 1 to 4 words)
    stress      the PromptBench StressTest sentence appended to the stem
    irrelevant  the first irrelevant.txt sentence, in seeded hash order, sharing no word of 4+ letters with the stem
                and the option lines (instruction level: nor with the instruction), appended to the stem
    para        the cached paraphrase of the stem (paraphrase.py); a failed paraphrase is a skip
    choice      one of the condition's fixed `texts` picked by the seed (instruction.py)
A condition with `target: instruction` applies its edit to the benchmark's answer instruction instead of the stem
(instruction.py). Because the deterministic edits depend only on the prepared samples and the seed, regenerating them
reproduces the paper's files byte for byte (their sha256 are in paper/frozen/robustness/edits.json); paraphrases
depend on a language model and are not regenerable exactly."""
from __future__ import annotations

import json
from pathlib import Path

from ...core.io import fingerprint, read_jsonl
from ...data.prompts import OVERRIDES_FILE, apply_edits, read_overrides, render_prompts
from . import instruction, perturb
from .make import DATASETS, Experiment, is_nested, key, write_once

PARAPHRASE_CACHE = "paraphrase_cache.jsonl"


def options_text(sample) -> str:
    return "\n".join(f"{k}. {v}" for k, v in (sample.choices or {}).items())


def paraphrase_key(dataset: str, unit: str, stem: str, pivot: str) -> str:
    """Cache key of a unit's paraphrase (the same key the paper's paraphrase cache used)."""
    return fingerprint({"dataset": dataset, "unit": unit, "stem": stem, "pivot": pivot})


def read_cache(exp: Experiment) -> dict[str, dict]:
    return {r["key"]: r for r in read_jsonl(Path(exp.root) / PARAPHRASE_CACHE)}


def seed_string(exp: Experiment, condition: str, dataset: str, unit: str) -> str:
    c = exp.condition(condition)
    return f"{exp.seed}|{dataset}|{unit}|{c.get('seed_as', condition)}"


def target_text(exp: Experiment, condition: str, dataset: str, sample) -> str:
    """The text a condition's edit replaces in `sample`'s prompt: the stem, or the answer instruction."""
    if exp.condition(condition).get("target") == "instruction":
        return instruction.instruction_span(dataset)
    return DATASETS[dataset].stem(sample)


def edit(exp: Experiment, condition: str, dataset: str, unit: str, samples: list, cache: dict, pool: list[str]):
    """-> ([[old, new]], None) or (None, skip reason) for one unit; every sample of a unit gets the same edits."""
    c, s = exp.condition(condition), samples[0]
    stem, seed = DATASETS[dataset].stem(s), seed_string(exp, condition, dataset, unit)
    target, avoid = stem, f"{stem}\n{options_text(s)}"
    if c.get("target") == "instruction":
        target = instruction.instruction_span(dataset)
        avoid += f"\n{target}"
    new, skip = stem_edit(c, target, seed, avoid, dataset, unit, cache, pool)
    return ([[target, new]] if new is not None else None), skip


def stem_edit(c: dict, stem: str, seed: str, avoid: str, dataset: str, unit: str, cache: dict, pool: list[str]):
    """-> (perturbed `stem`, None) or (None, skip reason); `stem` is the question stem or, at the instruction level,
    the instruction; `avoid`: the item's text a distractor must not share words with."""
    kind = c["edit"]
    if kind == "typo":
        new = perturb.typo(stem, seed, c.get("rate", 0.2), c.get("max_words", 4))
        return (new, None) if new != stem else (None, "no eligible word")
    if kind == "stress":
        return perturb.append(stem, perturb.STRESS_TEST), None
    if kind == "irrelevant":
        return perturb.append(stem, perturb.pick_irrelevant(avoid, seed, pool)), None
    if kind == "choice":
        return instruction.choice(c["texts"], seed), None
    if kind == "para":
        result = cache.get(paraphrase_key(dataset, unit, stem, c["pivot"]))
        if result is None:
            raise ValueError(f"no paraphrase for {dataset}/{unit} ({c['pivot']}); run `streamreplace robustness "
                             "paraphrase` first, or put the paper's paraphrase cache in the experiment root")
        return (result["text"], None) if result["ok"] else (None, result["reason"])
    raise ValueError(f"unknown edit kind {kind!r}")


def condition_rows(exp: Experiment, condition: str, cache: dict | None = None, pool: list[str] | None = None) -> list:
    """The override rows of one condition (not written)."""
    from ...benchmarks import registry
    if exp.subset(condition) is None:
        raise ValueError(f"{exp.folder(condition)}: no subset.json; run `streamreplace robustness subset` first")
    cache = read_cache(exp) if cache is None else cache
    pool = perturb.irrelevant_pool() if pool is None else pool
    rows = []
    for dataset in exp.datasets(condition):
        bench = registry.get(dataset)
        for unit, samples in sorted(exp.units(dataset, condition).items()):
            edits, skip = edit(exp, condition, dataset, unit, samples, cache, pool)
            edits = edits or []
            for sample in samples:
                apply_edits(bench.prompt(bench.view(sample)), edits)      # every edited span occurs exactly once
                rows.append({"dataset": dataset, "id": sample.id, "unit": unit, "condition": condition,
                             "edits": edits, "skip": skip})
    return rows


def make_edits(exp: Experiment, only: list[str] | None = None) -> dict:
    """Writes every non-clean condition folder's prompt_overrides.jsonl (once; an existing file must be identical)."""
    cache, pool, status = read_cache(exp), perturb.irrelevant_pool(), {}
    for condition, c in exp.conditions.items():
        if not c or (only and condition not in only):
            continue                                    # clean: no override file
        rows = condition_rows(exp, condition, cache, pool)
        status[condition] = {"rows": len(rows), "skipped_units": len({(r["dataset"], r["unit"]) for r in rows
                                                                      if r["skip"]}),
                             "file": write_once(exp.folder(condition) / OVERRIDES_FILE, rows)}
    return status


def skipped_units(exp: Experiment, condition: str, dataset: str) -> set[str]:
    """Units whose edit the condition skipped (none for clean)."""
    return {r["unit"] for r in read_jsonl(exp.folder(condition) / OVERRIDES_FILE)
            if r["dataset"] == dataset and r["skip"]}


def check(exp: Experiment) -> dict:
    """Every condition folder: the shared subset (instruction-level folders: nested in it); every sample has an edit
    row that applies; outside the edited span the prompt equals the clean prompt byte for byte (a stem edit leaves the
    instruction, options and hint untouched; an instruction edit leaves the stem and every option line). Returns
    per-condition statistics."""
    from ...benchmarks import registry
    subsets = {c: exp.subset(c) for c in exp.conditions}
    if missing := [c for c, s in subsets.items() if s is None]:
        raise ValueError(f"{missing}: no subset.json; run `streamreplace robustness subset`")
    restricted = {c for c, v in exp.conditions.items() if is_nested(v)}
    shared = {json.dumps(s, sort_keys=True) for c, s in subsets.items() if c not in restricted}
    if len(shared) != 1:
        raise ValueError("condition folders hold different subset.json files")
    shared_subset = json.loads(shared.pop())
    for c in restricted:
        if any(not set(ids) <= set(shared_subset.get(dataset, [])) for dataset, ids in subsets[c].items()):
            raise ValueError(f"{c}: its subset.json is not nested in the shared subset")
    report = {}
    for condition, c in exp.conditions.items():
        overrides = read_overrides(exp.folder(condition)) or {}
        if c and not overrides:
            raise ValueError(f"{condition}: no {OVERRIDES_FILE}; run `streamreplace robustness edits`")
        stats = {}
        for dataset in exp.datasets(condition):
            bench = registry.get(dataset)
            samples = exp.samples(dataset, condition)
            # raises on a missing row or an edited span not found exactly once
            prompts = render_prompts(dataset, samples, overrides.get(dataset) if c else None, bench=bench)
            changed, distance = 0, []
            for sample in samples:
                clean, span = bench.prompt(bench.view(sample)), target_text(exp, condition, dataset, sample)
                new = prompts[sample.id]
                if new == clean:
                    continue
                if clean.count(span) != 1:
                    raise ValueError(f"{condition}/{dataset}/{sample.id}: the edited span does not occur once")
                head, tail = clean.split(span)
                if not (new.startswith(head) and new.endswith(tail)):
                    raise ValueError(f"{condition}/{dataset}/{sample.id}: the edit reaches outside the "
                                     f"{c.get('target', 'stem')}")
                changed += 1
                distance.append(perturb.word_edit_ratio(span, new[len(head):len(new) - len(tail)]))
            stats[dataset] = {"samples": len(samples), "changed": changed,
                              "mean_word_edit_ratio": round(sum(distance) / len(distance), 3) if distance else 0.0}
            if c and changed == 0:
                raise ValueError(f"{condition}/{dataset}: no prompt changed")
        report[condition] = stats
    return report


def digests(exp: Experiment) -> dict:
    """Per non-clean condition: rows, sha256 of the override file and the skipped units with their reasons, as
    paper/frozen/robustness/edits.json stores them (no benchmark text)."""
    from ...core.io import file_digest
    out = {}
    for condition, c in exp.conditions.items():
        path = exp.folder(condition) / OVERRIDES_FILE
        if not c or not path.is_file():
            continue
        rows = read_jsonl(path)
        skipped: dict[str, dict[str, str]] = {}
        for r in rows:
            if r["skip"]:
                skipped.setdefault(r["dataset"], {})[r["unit"]] = r["skip"]
        out[condition] = {"rows": len(rows), "sha256": file_digest(path),
                          "skipped": {d: dict(sorted(u.items())) for d, u in skipped.items()}}
    return out


def compare_digests(exp: Experiment, frozen: dict) -> dict[str, str]:
    """condition -> "identical" | "differs" | "missing" against paper/frozen/robustness/edits.json."""
    mine = digests(exp)
    return {c: ("missing" if c not in mine else "identical" if mine[c]["sha256"] == f["sha256"] else "differs")
            for c, f in frozen.items()}


def audit(exp: Experiment, n: int = 30, only: list[str] | None = None, out: Path | None = None) -> list[str]:
    """<root>/audit/<condition>.md: `n` units per dataset (seeded hash order) with the clean and the perturbed text,
    for a manual check that the meaning is kept (typo, paraphrase) and that appended text is unrelated."""
    cache, written = read_cache(exp), []
    folder = Path(out) if out else Path(exp.root) / "audit"
    for condition, c in exp.conditions.items():
        if not c or (only and condition not in only):
            continue
        rows = {(r["dataset"], r["id"]): r for r in read_jsonl(exp.folder(condition) / OVERRIDES_FILE)}
        lines = [f"# Audit: {condition} ({json.dumps(c)})", "",
                 f"{n} units per dataset in seeded hash order. Check that the meaning is kept (typo / paraphrase) and "
                 "that the appended text is unrelated to the question (distractors).", ""]
        what = c.get("target", "stem")
        for dataset in exp.datasets(condition):
            groups = exp.units(dataset, condition)
            order = sorted(groups, key=lambda u: key(f"audit:{u}"))[:n]
            skips = sum(1 for u in groups if rows[(dataset, groups[u][0].id)]["skip"])
            lines += [f"## {dataset} ({len(groups)} units, {skips} skipped)", "",
                      f"| unit | clean {what} | perturbed {what} |", "|---|---|---|"]
            for u in order:
                sample, row = groups[u][0], rows[(dataset, groups[u][0].id)]
                new = row["edits"][0][1] if row["edits"] else f"(skipped: {row['skip']})"
                note = ""
                if c["edit"] == "para":
                    hit = cache.get(paraphrase_key(dataset, u, DATASETS[dataset].stem(sample), c["pivot"])) or {}
                    note = f"<br>method: {hit.get('method')}, equivalent: {hit.get('verdict')}"
                cell = lambda t: t.replace("|", "\\|").replace("\n", "<br>")
                lines.append(f"| {u} | {cell(target_text(exp, condition, dataset, sample))} | {cell(new)}{note} |")
            lines.append("")
        path = folder / f"{condition}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(str(path))
    return written
