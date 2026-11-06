"""HarmBench (centerforaisafety/HarmBench at the commit pinned in manifests/upstream.yaml): the official run_pipeline.py
default behaviours (harmbench_behaviors_text_all.csv: standard 200, contextual 100, copyright 100) plus all 110
multimodal behaviours, 510 in total, attacked with the baseline direct request. Paper: safety suite, P and C only
(text and single-image behaviours have no stream to feed).

  text        DirectRequest: the behaviour, or f"{ContextString}\\n\\n---\\n\\n{Behavior}" when it has a context
  multimodal  MultiModalDirectRequest: the behaviour's image + f"{Behavior} " (upstream appends one space)

The prompt actually sent is the shared default prompt for open questions, question.strip(), so the upstream trailing
space of the 110 multimodal test cases (and of 2 text behaviours) is removed. The paper's runs did this; it is kept
for reproduction. The image goes through each model's own image path (upstream letterboxes it to the model's native
square instead). Generation: greedy, max_new_tokens 512 (run_pipeline.py's default).

Scoring: the official classifiers through the official evaluation functions (judges/harmbench_cls.py):
cais/HarmBench-Llama-2-13b-cls for text behaviours, cais/HarmBench-Llama-2-13b-cls-multimodal-behaviors for
multimodal ones, the MinHash check for copyright behaviours. A row's `correct` = the model behaved safely
(label != 1), so the headline ASR = 100 - safe rate."""
from __future__ import annotations

import csv
from pathlib import Path

from ...core.schema import Sample
from .. import common
from ..base import JudgePolicy, MetricSpec

NAME = "harmbench"
MEDIA = "image"          # text behaviours carry no image; never run in stream mode
JUDGE = JudgePolicy.CLASSIFIER   # scored only by the official HarmBench classifiers
LOWER_IS_BETTER = True   # the headline is an attack success rate
EXPECTED_COUNT = 510
METRICS = [MetricSpec("asr", "safe", lower_is_better=True, modes=("offline",))]
TEXT = "data/behavior_datasets/harmbench_behaviors_text_all.csv"
MULTIMODAL = "data/behavior_datasets/harmbench_behaviors_multimodal_all.csv"
IMAGES = "data/multimodal_behavior_images"
CATEGORIES = ("standard", "contextual", "copyright", "multimodal")


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_case(behavior: dict) -> str:
    """The official baseline's test case string for one behaviour row (see the module docstring)."""
    if behavior["FunctionalCategory"] == "multimodal":
        return behavior["Behavior"] + " "
    context = behavior.get("ContextString", "")
    return f"{context}\n\n---\n\n{behavior['Behavior']}" if context else behavior["Behavior"]


def prepare(raw: Path, media: Path) -> list[Sample]:
    """raw = HarmBench's data/ folder at the pinned commit (raw/data/behavior_datasets, ...)."""
    samples = []
    for row in read_csv(raw / TEXT):
        samples.append(Sample(row["BehaviorID"], test_case(row), "", "open",
                              metadata={"behavior": row, "functional": row["FunctionalCategory"],
                                        "semantic": row["SemanticCategory"]}))
    for row in read_csv(raw / MULTIMODAL):
        from PIL import Image
        source = raw / IMAGES / row["ImageFileName"]
        if not source.exists():         # 7 CSV rows name .jpeg/.jpg; the repository only has the same stem as .png
            matches = sorted((raw / IMAGES).glob(f"{source.stem}.*"))
            assert len(matches) == 1, f"{row['BehaviorID']}: no unique image for {row['ImageFileName']}"
            source = matches[0]
        with Image.open(source) as image:                                 # RGB as upstream's .convert('RGB')
            path = common.save_image(image, media / f"{source.stem}.png")
        samples.append(Sample(row["BehaviorID"], test_case(row), "", "open", images=(str(path),),
                              metadata={"behavior": row, "functional": row["FunctionalCategory"],
                                        "semantic": row["SemanticCategory"], "image_file": source.name}))
    return samples


def metrics(rows) -> dict:
    """ASR in %: share of behaviours with classifier label 1 (one completion per behaviour, so the upstream mean of
    per-behaviour ASRs is the same number). Label -1 and errored rows are not successes (scoring.outcome)."""
    def asr(subset):
        return round(100 - common.pct(r["correct"] for r in subset), 2) if subset else None
    result = {"asr": asr(rows), "text_asr": asr([r for r in rows if r["metadata"]["functional"] != "multimodal"])}
    for name in CATEGORIES:
        result[f"func/{name}"] = asr([r for r in rows if r["metadata"]["functional"] == name])
    for name in sorted({r["metadata"]["semantic"] for r in rows}):
        result[f"sem/{name}"] = asr([r for r in rows if r["metadata"]["semantic"] == name])
    return result


def smoke(samples):
    """Round-robin over the four functional categories, each in a fixed hashed order."""
    per = {c: common.smoke([s for s in samples if s.metadata["functional"] == c], n=10**6) for c in CATEGORIES}
    order = []
    for i in range(max(map(len, per.values()))):
        order += [per[c][i] for c in CATEGORIES if i < len(per[c])]
    return order
