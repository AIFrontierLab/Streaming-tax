"""MME: 1187 images x 2 yes/no questions = 2374 questions in 14 subtasks (10 perception, 4 cognition).

Paper use: image understanding (P and C, offline only): the absolute-score image table and the C-P contrasts.
Headline: perception score (max 2000); cognition (max 800) is secondary. Official arithmetic: subtask score = acc +
acc+ (acc+ = share of images with both questions right), max 200 per subtask; perception and cognition are sums of
their subtasks. Statistics: image clusters within category strata (kind "mme"); deltas are reported in percentage
points of the maximum (perception / 20, cognition / 8).

Prompt: the official question verbatim (it already ends with "Please answer yes or no."), through the common prompt.
Scoring: the common yes/no rule; a reply it cannot read goes to the generic LLM extraction judge (JUDGE fallback).
docs/benchmarks/mme.md."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .. import common
from ..base import JudgePolicy, MetricSpec
from ...core.schema import Sample

NAME = "mme"
MEDIA = "image"
JUDGE = JudgePolicy.FALLBACK
EXPECTED_COUNT = 2374
METRICS = [MetricSpec("perception", "mme", scale=20.0, modes=("offline",)),
           MetricSpec("cognition", "mme", scale=8.0, modes=("offline",))]
COGNITION = {"commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"}


def prepare(raw: Path, media: Path) -> list[Sample]:
    """raw/data/test-*.parquet -> 2374 samples with ids "<question_id>#<k>" (k = 1, 2: the image's two questions);
    images to media/<sha256(question_id)[:16]>.png."""
    samples, count = [], {}
    for path in sorted((raw / "data").glob("test-*.parquet")):
        for row in common.read_parquet(path):
            key = str(row["question_id"])
            count[key] = count.get(key, 0) + 1
            image = common.save_image(row["image"], media / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.png")
            samples.append(Sample(f"{key}#{count[key]}", row["question"].strip(), row["answer"].strip().lower(), "yesno",
                                  images=(str(image),), metadata={"image": key, "category": row["category"]}))
    return samples


def metrics(rows) -> dict:
    """Subtask scores (acc + acc+, each rounded to 2 decimals), then perception and cognition as their sums.
    The sums are rounded to 1 decimal (not 2): the paper's values use this rounding."""
    per_task = {}
    for task in sorted({r["metadata"]["category"] for r in rows}):
        subset = [r for r in rows if r["metadata"]["category"] == task]
        acc_plus, _ = common.group_all(subset, "image", size=2)
        per_task[task] = round(common.pct(r["correct"] for r in subset) + (acc_plus or 0.0), 2)
    return {"perception": round(sum(v for k, v in per_task.items() if k not in COGNITION), 1),
            "cognition": round(sum(v for k, v in per_task.items() if k in COGNITION), 1),
            **{f"task/{k}": v for k, v in per_task.items()}}


def smoke(samples):
    return common.smoke(samples, group="image", n=50)
