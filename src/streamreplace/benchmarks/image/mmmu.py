"""MMMU validation: 900 questions (847 multiple choice, 53 open) over 30 subjects, up to 7 images each.

Paper use: image understanding (P and C, offline only): the absolute-score image table and the C-P contrasts.
Headline: accuracy over all 900 items; the statistics resample items.

Prompt: the official one (MMMU repository, configs/llava1.5.yaml + construct_prompt): "(A) option" lines; the images
go before the text and the "<image n>" tags stay in the question as plain text. Scoring: an exact option letter for
multiple choice (common option rules); open answers may list several accepted forms ("['$MgS$', 'MgS']"), dollar
signs are ignored and numbers get a 1% relative tolerance. A reply the rules cannot read goes to the generic LLM
extraction judge (JUDGE fallback) instead of MMMU's random-choice fallback. docs/benchmarks/mmmu.md."""
from __future__ import annotations

import ast
from pathlib import Path

from .. import common
from ..base import JudgePolicy, MetricSpec
from ...core.schema import Sample

NAME = "mmmu"
MEDIA = "image"
JUDGE = JudgePolicy.FALLBACK
EXPECTED_COUNT = 900
METRICS = [MetricSpec("acc", "acc", modes=("offline",))]


def prepare(raw: Path, media: Path) -> list[Sample]:
    """raw/<Subject>/validation-*.parquet -> 900 samples; images to media/<Subject>/<id>_<i>.png."""
    samples = []
    for path in sorted(raw.glob("*/validation-*.parquet")):
        subject = path.parent.name
        for row in common.read_parquet(path):
            images = tuple(str(common.save_image(row[f"image_{i}"], media / subject / f"{row['id']}_{i}.png"))
                           for i in range(1, 8) if row.get(f"image_{i}"))
            meta = {"subject": subject, "subfield": row.get("subfield"), "difficulty": row.get("topic_difficulty")}
            if row["question_type"] == "multiple-choice":
                choices = common.letters(ast.literal_eval(row["options"]))
                samples.append(Sample(row["id"], row["question"], row["answer"].strip(), "mcq", choices, images, metadata=meta))
            else:
                samples.append(Sample(row["id"], row["question"], row["answer"].strip(), "text", images=images, metadata=meta))
    if len(samples) != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} MMMU validation questions, got {len(samples)}")
    return samples


def prompt(sample) -> str:
    if sample.kind == "mcq":
        options = "".join(f"({k}) {v}\n" for k, v in sample.choices.items())
        return f"{sample.question}\n{options}\n{common.MCQ}"
    return f"{sample.question}\n{common.TEXT}"


def correct(sample, prediction) -> bool:
    """Open answers may list several accepted forms (\"['$MgS$', 'MgS']\"); numbers get a 1% relative tolerance."""
    if prediction is None:
        return False
    if sample.kind == "mcq":
        return prediction == sample.answer
    answers = [sample.answer]
    if sample.answer.startswith("["):
        try:
            answers = [str(a) for a in ast.literal_eval(sample.answer)] + [sample.answer]
        except (ValueError, SyntaxError):
            pass
    return any(common.equivalent(prediction.replace("$", ""), a.replace("$", ""), 1e-3, 1e-2) for a in answers)


def metrics(rows) -> dict:
    return {"acc": common.pct(r["correct"] for r in rows), **common.by(rows, "subject", "subject/")}
