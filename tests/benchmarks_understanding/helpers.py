"""Helpers of the understanding-benchmark tests: the 12 benchmark names, prepared data (the test is skipped when it is
absent), skipping when a benchmark's official upstream code has not been fetched, the gold self-test, and the
id -> prompt map hash recorded in manifests/prepared.yaml."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from streamreplace import paths
from streamreplace.core.io import read_jsonl
from streamreplace.core.schema import Sample

IMAGE = ["dynamath", "mathvista", "mmbench", "mme", "mmmu", "mmstar", "scienceqa", "visulogic"]
VIDEO = ["videomme", "tomato", "timeblind", "videoreasonbench"]
BENCHMARKS = IMAGE + VIDEO
REPO = Path(__file__).resolve().parents[2]

# Gold self-test: answering with the reference must score 100%. Benchmarks scored by their authors' own rules declare
# which answer forms those rules accept for a correct letter:
GOLD_FORMS = {
    # Video-MME's extract_characters_regex reads letters only.
    "videomme": ["{letter}", "The best answer is {letter}.", "({letter})"],
    # TOMATO's pre_parser takes a bare letter; "X: option" only when the option text has no lower-case letter (it
    # upper-cases the reply alone) and "Answer is:(X)" for A-F only (9 questions have a G). Others go to the judge.
    "tomato": ["{letter}"],
}
# TimeBlind's extract_answer reads letters only: a bare option text is not an answer (the judge may extract it).
NO_OPTION_TEXT = {"timeblind"}


def prepared_path(name: str) -> Path:
    return paths.current().dataset_dir(name) / "prepared" / "samples.jsonl"


def prepared_samples(name: str) -> list[Sample]:
    """The benchmark's prepared samples (media paths made absolute), or skip the test when it is not prepared."""
    path = prepared_path(name)
    if not path.is_file():
        pytest.skip(f"{name} is not prepared ({path} is missing)")
    base = path.parent.parent
    return [replace(s, images=tuple(str(base / p) for p in s.images), video=str(base / s.video) if s.video else None)
            for s in (Sample.from_dict(row) for row in read_jsonl(path))]


def require_upstream(name: str) -> None:
    """Skip the test when the upstream repository `name` has not been fetched (streamreplace fetch-upstream)."""
    from streamreplace.models.upstream import available
    if not available(name):
        pytest.skip(f"upstream code {name!r} is not fetched (streamreplace fetch-upstream {name})")


def prompt_map(bench, samples) -> dict[str, str]:
    """id -> the prompt the model is sent (rendered from the SampleView), in prepared order."""
    return {s.id: bench.prompt(bench.view(s)) for s in samples}


def prompt_map_sha256(prompts: dict[str, str]) -> str:
    """sha256 of the id -> prompt map as manifests/prepared.yaml defines it (sorted keys, unescaped unicode)."""
    return hashlib.sha256(json.dumps(prompts, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def manifest_row(name: str) -> dict | None:
    import yaml
    path = REPO / "manifests" / "prepared.yaml"
    if not path.is_file():
        return None
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("prepared", {}).get(name)


def gold_texts(sample, option_text: bool = True, forms: list[str] | None = None) -> list[str]:
    """The reference answer in the forms a correct reply may take."""
    if forms and sample.kind == "mcq":
        return [f.format(letter=sample.answer, option=sample.choices[sample.answer]) for f in forms]
    if sample.kind == "mcq":
        texts = [sample.answer, f"The answer is ({sample.answer})."]
        option = sample.choices[sample.answer]
        # an empty or repeated option text, or one that reads as a letter, cannot identify the answer
        if (option_text and option and option.strip("() .").upper() not in sample.choices
                and list(sample.choices.values()).count(option) == 1):
            texts.append(option)
        return texts
    return [sample.answer, f"Final answer: {sample.answer}"]


def gold_failures(bench, samples) -> tuple[list, int]:
    """(failures, texts checked): for every closed-kind sample, each gold text must parse on the SampleView to a
    prediction that scores correct, and parsing the full sample must give the same result (no answer leakage).
    Free-form (open) samples are skipped: the judge decides them. One failure is recorded per sample."""
    failures, checked = [], 0
    forms = GOLD_FORMS.get(bench.name)
    for sample in samples:
        if sample.kind == "open":
            continue
        view = bench.view(sample)
        for text in gold_texts(sample, bench.name not in NO_OPTION_TEXT, forms):
            checked += 1
            prediction, how = bench.parse(view, text)
            if not bench.correct(sample, prediction) or bench.parse(sample, text) != (prediction, how):
                failures.append((sample.id, text[:80], prediction, how))
                break
    return failures, checked


def gold_metrics_problems(bench, samples) -> list[str]:
    """With every sample correct, the headline (the first metrics key, also METRICS[0]) is positive, and the smoke
    order holds at least 20 ids."""
    rows = [{"id": s.id, "correct": True, "kind": s.kind, "metadata": s.metadata} for s in samples]
    metrics = bench.metrics(rows)
    name, headline = next(iter(metrics.items()))
    problems = []
    if headline is None or headline <= 0:
        problems.append(f"headline {name}={headline}")
    if not bench.METRICS or bench.METRICS[0].name != name:
        problems.append(f"METRICS[0] is not the headline {name}")
    if len(bench.smoke(samples)) < 20:
        problems.append("smoke has fewer than 20 ids")
    return problems
