"""Checks on the real prepared data (skipped when a benchmark is not prepared under the data root):

- gold self-test: the reference answer, in every form a correct reply may take, parses (on the SampleView) and scores
  correct for every closed-kind sample, with the same result as parsing the full sample;
- metrics on gold: with every sample correct the headline is positive; the smoke order holds at least 20 ids;
- the prepared set matches manifests/prepared.yaml: count, and the hash of the id -> prompt map (the prompts the
  paper's runs sent; the official scorers' upstream code is needed for parsing only)."""
from __future__ import annotations

import pytest

from streamreplace.benchmarks.registry import get
from streamreplace.core.io import file_digest

from .helpers import (BENCHMARKS, VIDEO, gold_failures, gold_metrics_problems, manifest_row, prepared_path,
                      prepared_samples, prompt_map, prompt_map_sha256, require_upstream)

UPSTREAM = {"videomme": "videomme", "tomato": "tomato", "timeblind": "timeblind", "videoreasonbench": "videoreasonbench"}
CASES = [pytest.param(name, marks=pytest.mark.requires_data(name)) for name in BENCHMARKS]


@pytest.mark.parametrize("name", CASES)
def test_reference_answers_score_perfectly(name):
    samples = prepared_samples(name)
    if name in UPSTREAM:
        require_upstream(UPSTREAM[name])
    failures, checked = gold_failures(get(name), samples)
    assert not failures, f"{len(failures)} of {checked} gold answers fail, e.g. {failures[:5]}"


@pytest.mark.parametrize("name", CASES)
def test_metrics_on_gold_and_smoke(name):
    problems = gold_metrics_problems(get(name), prepared_samples(name))
    assert not problems


@pytest.mark.parametrize("name", CASES)
def test_prepared_set_matches_the_manifest(name):
    bench, samples = get(name), prepared_samples(name)
    row = manifest_row(name)
    assert row is not None, f"{name} has no manifests/prepared.yaml row"
    assert len(samples) == row["count"] == bench.EXPECTED_COUNT
    assert prompt_map_sha256(prompt_map(bench, samples)) == row["prompts_sha256"]
    assert file_digest(prepared_path(name)) == row["samples_sha256"]


@pytest.mark.parametrize("name", [pytest.param(n, marks=pytest.mark.requires_data(n)) for n in VIDEO])
def test_every_video_sample_has_a_video(name):
    """Stream jobs cover the samples with a video: for the video benchmarks that is every sample."""
    assert all(s.video for s in prepared_samples(name))
