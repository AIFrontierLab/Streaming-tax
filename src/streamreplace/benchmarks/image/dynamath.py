"""DynaMath (sample split): 501 seed questions x 10 programmatically generated variants = 5010 image questions.

Paper use: image understanding (P and C, offline only): the absolute-score image table and the C-P contrasts.
Headline: average accuracy over all variants (`avg_acc`); worst-case accuracy (`worst_acc`, secondary) is the share of
seeds whose 10 variants are all correct. Statistics units: avg_acc resamples seeds (the 10 variants of a seed form
one cluster); worst_acc is one binary unit per seed.

Prompt: the official one of VLMEvalKit (vlmeval/dataset/dynamath.py): the raw question, then a JSON answer guide with
a 'solution' and a 'short answer' key; floats as three-digit numbers. Scoring follows upstream: floats within 1e-3,
choices by letter, free-form answers correct when the prediction is a substring of the answer (case-insensitive).
A reply the rules cannot read goes to the generic LLM extraction judge (JUDGE fallback). docs/benchmarks/dynamath.md."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .. import common
from ..base import JudgePolicy, MetricSpec
from ...core.schema import Sample

NAME = "dynamath"
MEDIA = "image"
JUDGE = JudgePolicy.FALLBACK
EXPECTED_COUNT = 5010
METRICS = [
    MetricSpec("avg_acc", "acc", cluster="seed", modes=("offline",)),
    MetricSpec("worst_acc", "group", group="seed", size=10, modes=("offline",)),
]

GUIDE = ("\n## Answer Instruction Please provide an answer to the question outlined above. Your response should adhere "
         "to the following JSON format, which includes two keys: 'solution' and 'short answer'. The 'solution' key can contain "
         "detailed steps needed to solve the question, and the 'short answer' key should provide a concise response. {INST}\n\n"
         "Example of expected JSON response format:\n\n")
EXAMPLE = json.dumps({"solution": "[Detailed step-by-step explanation]", "short answer": "[Concise Answer]"}, indent=4)
INSTRUCTIONS = {   # upstream wording, typo included
    "mcq": "Provide the corresponing choice option in the 'short answer' key, such as 'A', 'B', 'C', or 'D'.",
    "numeric": "Format the answer as a three-digit floating-point number and provide it in the 'short answer' key.",
    "text": "Float numbers in the answer should be formatted as three-digit floating-point numbers.",
}
_SHORT = re.compile(r'"short answer"\s*:\s*(?:"((?:[^"\\]|\\.)*)"|([^,}\n]+))')


def stable_id(*parts) -> str:
    """Sample id: sha256 of the parts joined by the unit separator, first 24 hex digits (seed 6 / variant 1 ->
    4632251fe0dcaf354fd88f5b). The ids key the paper's records and judge caches (not distributed) and the frozen id
    lists: keep them."""
    return hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).hexdigest()[:24]


def split_choices(question: str) -> tuple[str, dict[str, str]]:
    """(stem, {letter: option}) of a question that lists its options inline as "(A) x (B) y" or as "A. x" lines;
    the labels must run A, B, C, ... and a trailing "Choices:" is dropped from the stem. Also used by MMStar."""
    markers = list(re.finditer(r"\(([A-Z])\)\s*", question))
    if len(markers) < 2:
        markers = list(re.finditer(r"(?m)^\s*([A-Z])[.:)]\s+", question))
    labels = [m[1] for m in markers]
    if len(markers) < 2 or labels != [chr(ord("A") + i) for i in range(len(labels))]:
        raise ValueError(f"Cannot parse choices: {question[:100]}")
    stem = re.sub(r"(?i)\bchoices?\s*:\s*$", "", question[:markers[0].start()]).strip()
    return stem, {m[1]: question[m.end():markers[i + 1].start() if i + 1 < len(markers) else len(question)].strip()
                  for i, m in enumerate(markers)}


def prepare(raw: Path, media: Path) -> list[Sample]:
    """raw/data/sample{1..10}-*.parquet (one file set per variant) -> 5010 samples; images to media/variant<v>/."""
    samples = []
    kinds = {"float": "numeric", "multiple choice": "mcq", "text": "text"}
    for variant in range(1, 11):
        files = sorted((raw / "data").glob(f"sample{variant}-*.parquet"))
        if not files:
            raise ValueError(f"Missing DynaMath variant {variant}")
        for path in files:
            for row in common.read_parquet(path):
                seed, kind = str(row["id"]), kinds[str(row["answer_type"]).lower()]
                question = str(row["question"])
                stem, choices = split_choices(question) if kind == "mcq" else (question.strip(), {})
                image = common.save_image(row["decoded_image"], media / f"variant{variant}/{seed}.png")
                samples.append(Sample(
                    stable_id("dynamath", f"sample_variant{variant}/default", f"{seed}:{variant}"), stem,
                    str(row["ground_truth"]).strip(), kind, choices, (str(image),),
                    metadata={"seed": seed, "variant": variant, "subject": row.get("subject"),
                              "knowledge_level": row.get("knowledge_level"), "raw_question": question}))
    if len(samples) != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} DynaMath rows, got {len(samples)}")
    return samples


def prompt(sample) -> str:
    # "## Question\n " keeps upstream's space after the newline.
    return f"## Question\n {sample.metadata['raw_question']}" + GUIDE.format(INST=INSTRUCTIONS[sample.kind]) + EXAMPLE


def short_answer(text: str) -> str | None:
    """The JSON 'short answer' value: VLMEvalKit's preprocess (outermost braces, backslashes dropped) + json.loads,
    else the last '"short answer": ...' field (e.g. JSON broken by LaTeX in the solution)."""
    body = common.clean(text)
    start, end = body.find("{"), body.rfind("}")
    if 0 <= start < end:
        try:
            value = json.loads(body[start:end + 1].replace("\\", ""), strict=False).get("short answer")
            if value is not None and str(value).strip():
                return str(value).strip()
        except (ValueError, AttributeError):
            pass
    fields = _SHORT.findall(body)
    return next((v.strip() for v in fields[-1] if v.strip()), None) if fields else None


def parse(sample, text):
    """The JSON short answer read with the common rules (status "json_<rule>"), else the common rules on the whole
    reply. The judge's extraction comes back as "Final answer: <x>" and takes the same path."""
    answer = short_answer(text)
    if answer:
        prediction, how = common.parse(sample, answer)
        if prediction is not None:
            return prediction, f"json_{how}"
    return common.parse(sample, text)


def correct(sample, prediction) -> bool:
    """Free-form answers use the official rule (VLMEvalKit DynaMath_auxeval): the lower-cased prediction is a
    substring of the lower-cased answer. Floats (1e-3) and choices use the common rules, which match upstream."""
    if sample.kind == "text":
        return bool(prediction and prediction.strip()) and prediction.strip().lower() in sample.answer.lower()
    return common.correct(sample, prediction)


def metrics(rows) -> dict:
    """avg_acc (headline), worst_acc over seeds with all 10 variants counted, then per answer kind and subject."""
    worst, seeds = common.group_all(rows, "seed", size=10)
    return {"avg_acc": common.pct(r["correct"] for r in rows), "worst_acc": worst, "complete_seeds": seeds,
            **{f"kind/{k}": v for k, v in common.by([{**r, "metadata": {"kind": r["kind"]}} for r in rows], "kind").items()},
            **common.by(rows, "subject", "subject/")}


def smoke(samples):
    return common.smoke(samples, group="seed", n=50)
