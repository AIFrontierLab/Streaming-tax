"""ScienceQA-IMG test: the 2017 test questions that come with an image, 2-5 options each; the hint is given as context.

Paper use: image understanding (P and C, offline only): the absolute-score image table and the C-P contrasts.
Headline: accuracy over all 2017 items; the statistics resample items.

Prompt: as LLaVA / lmms-eval (tasks/scienceqa), rendered by the common prompt:
"Context: {hint}\n{question}\nA. ...\nAnswer with the option's letter from the given choices directly."
Scoring: the common option rules; a reply they cannot read goes to the generic LLM extraction judge (JUDGE fallback).
docs/benchmarks/scienceqa.md."""
from __future__ import annotations

from pathlib import Path

from .. import common
from ..base import JudgePolicy, MetricSpec
from ...core.schema import Sample

NAME = "scienceqa"
MEDIA = "image"
JUDGE = JudgePolicy.FALLBACK
EXPECTED_COUNT = 2017
METRICS = [MetricSpec("acc", "acc", modes=("offline",))]


def prepare(raw: Path, media: Path) -> list[Sample]:
    """raw/ScienceQA-IMG/test-*.parquet; rows without an image are skipped. Ids "<shard>-<row>", where the row number
    counts every row of the shard (also the skipped ones); images to media/<id>.png."""
    samples = []
    for path in sorted((raw / "ScienceQA-IMG").glob("test-*.parquet")):
        for number, row in enumerate(common.read_parquet(path)):
            if not row.get("image"):
                continue
            sample_id = f"{path.stem.split('-')[1]}-{number}"
            image = common.save_image(row["image"], media / f"{sample_id}.png")
            hint = (row.get("hint") or "").strip()
            question = (f"Context: {hint}\n" if hint else "") + row["question"].strip()
            samples.append(Sample(sample_id, question, chr(ord("A") + int(row["answer"])), "mcq", common.letters(row["choices"]),
                                  (str(image),), metadata={"subject": row.get("subject"), "grade": row.get("grade"),
                                                           "category": row.get("category")}))
    if len(samples) != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} ScienceQA-IMG test questions, got {len(samples)}")
    return samples


def metrics(rows) -> dict:
    return {"acc": common.pct(r["correct"] for r in rows), **common.by(rows, "subject", "subject/")}
