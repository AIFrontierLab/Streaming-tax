"""MathVista testmini: 1000 questions (540 multiple choice, 458 integer/float, 2 list answers).

Paper use: image understanding (P and C, offline only): the absolute-score image table and the C-P contrasts.
Headline: accuracy over all 1000 items; the statistics resample items.

Prompt: the dataset's own zero-shot `query` field, verbatim ("Hint: ... at the end.\nQuestion: ...\nChoices:\n(A) ...").
Scoring: the common rules (an option letter or option text; a number within the question's precision: half a unit
of the last requested decimal for floats, exact for integers). A reply the rules cannot read goes to the generic LLM
extraction judge (JUDGE fallback), not to MathVista's own GPT extraction. docs/benchmarks/mathvista.md."""
from __future__ import annotations

from pathlib import Path

from .. import common
from ..base import JudgePolicy, MetricSpec
from ...core.schema import Sample

NAME = "mathvista"
MEDIA = "image"
JUDGE = JudgePolicy.FALLBACK
EXPECTED_COUNT = 1000
METRICS = [MetricSpec("acc", "acc", modes=("offline",))]


def prepare(raw: Path, media: Path) -> list[Sample]:
    """raw/data/testmini-*.parquet -> 1000 samples; images to media/<pid>.png."""
    samples = []
    for path in sorted((raw / "data").glob("testmini-*.parquet")):
        for row in common.read_parquet(path):
            image = common.save_image(row["decoded_image"], media / f"{row['pid']}.png")
            meta = {"task": (row.get("metadata") or {}).get("task"), "answer_type": row["answer_type"],
                    "precision": row.get("precision"), "unit": row.get("unit"), "query": row["query"]}
            if row["question_type"] == "multi_choice":
                choices = common.letters(row["choices"])
                answer = next(k for k, v in choices.items() if v == str(row["answer"]).strip())
                samples.append(Sample(row["pid"], row["question"].strip(), answer, "mcq", choices, (str(image),), metadata=meta))
            else:
                precision = row.get("precision")
                if row["answer_type"] == "float" and precision is not None:
                    meta["tolerance"] = 0.5 * 10 ** -int(precision) + 1e-9
                elif row["answer_type"] == "integer":
                    meta["tolerance"] = 0
                kind = "numeric" if row["answer_type"] in {"integer", "float"} else "text"
                samples.append(Sample(row["pid"], row["question"].strip(), str(row["answer"]).strip(), kind, images=(str(image),), metadata=meta))
    return samples


def prompt(sample) -> str:
    return sample.metadata["query"]


def metrics(rows) -> dict:
    return {"acc": common.pct(r["correct"] for r in rows), **common.by(rows, "task", "task/")}
