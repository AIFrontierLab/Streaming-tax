"""VisuLogic: 1000 visual-logic multiple-choice questions; the options are drawn inside the image (the text options
are just A-D).

Paper use: supplementary image benchmark (P and C, offline only): the supplement's image table and its C-P
contrasts (a secondary metric). Headline: accuracy over all 1000 items; the statistics resample items.

Prompt: the official default (VisuLogic-Eval, models/prompts.py SFT_PROMPT), appended to the question without a
separator exactly as upstream does (`text + self.user_prompt`). Answers are read from the last \\boxed{} by the
common option rules; a reply they cannot read goes to the generic LLM extraction judge (JUDGE fallback).
docs/benchmarks/visulogic.md."""
from __future__ import annotations

import json
from pathlib import Path

from .. import common
from ..base import JudgePolicy, MetricSpec
from ...core.schema import Sample

NAME = "visulogic"
MEDIA = "image"
JUDGE = JudgePolicy.FALLBACK
EXPECTED_COUNT = 1000
METRICS = [MetricSpec("acc", "acc", modes=("offline",))]
SFT_PROMPT = "Just Answer the question following this format: Answer:\\boxed{$LETTER}"


def prepare(raw: Path, media: Path) -> list[Sample]:
    """raw/data.jsonl + raw/images.zip (extracted into media/) -> 1000 samples."""
    common.extract_zip(raw / "images.zip", media)
    samples = []
    for line in (raw / "data.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        samples.append(Sample(row["id"], row["question"].strip(), row["label"].strip().upper(), "mcq",
                              {k: k for k in "ABCD"}, (str(media / row["image_path"]),), metadata={"category": row["tag"]}))
    if len(samples) != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} VisuLogic questions, got {len(samples)}")
    return samples


def prompt(sample) -> str:
    return f"{sample.question}{SFT_PROMPT}"     # the question already lists A-D
