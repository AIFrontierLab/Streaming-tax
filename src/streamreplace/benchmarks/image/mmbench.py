"""MMBench DEV (English, legacy TSV): 4329 rows = 1164 questions, each asked in 2-4 circular rotations of its options.

Paper use: image understanding (P and C, offline only): the absolute-score image table and the C-P contrasts.
Headline: circular accuracy, the share of questions (with all rotations present) whose every rotation is answered
correctly; vanilla accuracy uses rotation 0 only. Statistics unit: one binary unit per question (all rotations
right), group size = the question's number of rotations.

Prompt: as lmms-eval (tasks/mmbench: mmbench_doc_to_text + create_options_prompt, sys_prompt from mmbench.yaml):
"{hint} {question} There are several options:\nA. ...\n\nAnswer with the option's letter from the given choices directly."
Scoring: the common option rules per row; a reply they cannot read goes to the generic LLM extraction judge
(JUDGE fallback), not to VLMEvalKit's ChatGPT matching. docs/benchmarks/mmbench.md."""
from __future__ import annotations

import base64
import csv
import sys
from pathlib import Path

from .. import common
from ..base import JudgePolicy, MetricSpec
from ...core.schema import Sample

NAME = "mmbench"
MEDIA = "image"
JUDGE = JudgePolicy.FALLBACK
EXPECTED_COUNT = 4329
METRICS = [MetricSpec("circular_acc", "group", group="question", size="rotations", modes=("offline",))]
SYS_PROMPT = "There are several options:"
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."


def prepare(raw: Path, media: Path) -> list[Sample]:
    """raw/MMBench_DEV_EN_legacy.tsv -> 4329 rows. Row index = rotation * 1,000,000 + question number; one image per
    question (media/<question>.png). A row's image field may name another row's index instead of holding the image."""
    csv.field_size_limit(sys.maxsize)
    with open(raw / "MMBench_DEV_EN_legacy.tsv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    images = {r["index"]: r["image"] for r in rows}
    sizes = {}
    for r in rows:
        base = int(r["index"]) % 1_000_000
        sizes[base] = sizes.get(base, 0) + 1
    samples = []
    for r in rows:
        payload, seen = images[r["index"]], {r["index"]}
        while len(payload) <= 64 and payload in images and payload not in seen:   # rows can point at another row's image
            seen.add(payload)
            payload = images[payload]
        base = int(r["index"]) % 1_000_000
        image = common.save_image({"bytes": base64.b64decode(payload)}, media / f"{base}.png")
        choices = {k: r[k].strip() for k in "ABCD" if r.get(k, "").strip() and r[k].strip().lower() != "nan"}
        hint = r.get("hint", "").strip()
        hint = hint if hint and hint.lower() != "nan" else None
        question = (f"{hint}\n" if hint else "") + r["question"].strip()
        samples.append(Sample(r["index"], question, r["answer"].strip(), "mcq", choices, (str(image),),
                              metadata={"question": base, "rotation": int(r["index"]) // 1_000_000, "rotations": sizes[base],
                                        "category": r["category"], "l2": r.get("l2-category"),
                                        "hint": hint, "raw_question": r["question"]}))
    return samples


def prompt(sample) -> str:
    options = "\n".join([SYS_PROMPT] + [f"{k}. {v}" for k, v in sample.choices.items()])
    hint, question = sample.metadata["hint"], sample.metadata["raw_question"]
    return f"{hint + ' ' if hint else ''}{question} {options}\n{POST_PROMPT}"


def metrics(rows) -> dict:
    groups = {}
    for r in rows:
        groups.setdefault(r["metadata"]["question"], []).append(r)
    complete = [g for g in groups.values() if len(g) == g[0]["metadata"]["rotations"]]
    circular = {g[0]["metadata"]["question"]: all(r["correct"] for r in g) for g in complete}
    by_l2 = {}
    for g in complete:
        by_l2.setdefault(g[0]["metadata"]["l2"], []).append(circular[g[0]["metadata"]["question"]])
    return {"circular_acc": common.pct(circular.values()), "vanilla_acc": common.pct(r["correct"] for r in rows if r["metadata"]["rotation"] == 0),
            "questions": len(complete), **{f"l2/{k}": common.pct(v) for k, v in sorted(by_l2.items())}}


def smoke(samples):
    return common.smoke(samples, group="question", n=50)
