"""MMStar: 1500 vision-indispensable multiple-choice questions, 6 core capabilities x 250.

Paper use: image understanding (P and C, offline only): the absolute-score image table and the C-P contrasts.
Headline: accuracy over all 1500 items (common metrics: acc, then per category); the statistics resample items.

Prompt: as lmms-eval (tasks/mmstar): the released question verbatim (its options are inline) plus the task's default
post-prompt, which has no final period. The options are parsed from the question only for scoring: the common option
rules read a letter or an option text; a reply they cannot read goes to the generic LLM extraction judge (JUDGE
fallback). docs/benchmarks/mmstar.md."""
from __future__ import annotations

from pathlib import Path
import re

from .. import common
from ..base import JudgePolicy, MetricSpec
from ...core.schema import Sample
from .dynamath import split_choices

NAME = "mmstar"
MEDIA = "image"
JUDGE = JudgePolicy.FALLBACK
EXPECTED_COUNT = 1500
METRICS = [MetricSpec("acc", "acc", modes=("offline",))]
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly"   # sic: no final period upstream


def prepare(raw: Path, media: Path) -> list[Sample]:
    """raw/mmstar.parquet -> 1500 samples; images to media/<index>.png."""
    samples = []
    for row in common.read_parquet(raw / "mmstar.parquet"):
        question = str(row["question"])
        if "Options:" in question:            # "... Options: A: x, B: y, ..."
            stem, _, options = question.partition("Options:")
            choices = {k: v.strip().rstrip(",").strip() for k, v in re.findall(r"([A-D]):\s*(.*?)(?=,\s*[A-D]:|$)", options, re.S)}
        else:                                 # MathVista-style "Hint: ...\nQuestion: ...\nChoices:\n(A) x\n(B) y"
            stem, choices = split_choices(question)
            stem = re.sub(r"^Hint:.*?\n", "", stem, flags=re.S).removeprefix("Question:").strip()
        image = common.save_image(row["image"], media / f"{row['index']}.png")
        samples.append(Sample(str(row["index"]), stem.strip(), row["answer"].strip(), "mcq", choices, (str(image),),
                              metadata={"category": row["category"], "l2": row.get("l2_category"), "raw_question": question}))
    return samples


def prompt(sample) -> str:
    return sample.metadata["raw_question"].strip() + POST_PROMPT
