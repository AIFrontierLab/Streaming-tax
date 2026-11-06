"""Prepared data, sample selection and prompts: prepare (idempotent, relative media paths, manifest hashes, checks),
subset and --limit in smoke order, prompt overrides as exact single-occurrence edits, the prompt fingerprint."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from streamreplace.core.io import file_digest, read_jsonl, write_jsonl
from streamreplace.data import (apply_edits, load_samples, prepared_dir, prompt_map_digest, read_overrides,
                                read_subset, render_prompts, samples_file, smoke_order)
from streamreplace.data.prepare import check_prepared, check_videos, load_expected, prepare_dataset


def test_prepare_is_idempotent_and_relative(project):
    manifest = json.loads((prepared_dir("demo", project.paths) / "manifest.json").read_text())
    assert manifest["samples"] == 6 and manifest["kinds"] == {"mcq": 2, "yesno": 2, "numeric": 1, "open": 1}
    assert manifest["samples_sha256"] == file_digest(samples_file("demo", paths=project.paths))
    assert prepare_dataset("demo", paths=project.paths)["status"] == "exists"
    row = read_jsonl(samples_file("demo", paths=project.paths))[0]
    assert row["video"] == "media/red.mp4" and not Path(row["video"]).is_absolute()
    loaded = load_samples("demo", paths=project.paths)
    assert all(Path(s.video).is_absolute() and Path(s.video).is_file() for s in loaded)
    rebuilt = prepare_dataset("demo", force=True, paths=project.paths)
    assert rebuilt["status"] == "prepared" and rebuilt["samples_sha256"] == manifest["samples_sha256"]


def test_unprepared_benchmarks_are_reported(project):
    with pytest.raises(ValueError, match="demo2 .test. is not prepared"):
        load_samples("demo2", paths=project.paths)


def test_limit_takes_the_smoke_order_then_prepared_order(project):
    samples = load_samples("demo", paths=project.paths)
    assert smoke_order("demo", samples) == ["mcq-1", "mcq-2", "yn-1", "yn-2", "num-1", "open-1"]
    assert [s.id for s in load_samples("demo", limit=1, paths=project.paths)] == ["mcq-1"]
    assert [s.id for s in load_samples("demo", limit=5, paths=project.paths)] == ["mcq-1", "mcq-2", "yn-1", "yn-2",
                                                                                 "num-1"]
    restricted = load_samples("demo", subset=["open-1", "yn-2", "unknown"], limit=1, paths=project.paths)
    assert [s.id for s in restricted] == ["yn-2"]           # subset first (prepared order), then the smoke order
    with pytest.raises(ValueError, match="positive"):
        load_samples("demo", limit=0, paths=project.paths)


def test_experiment_folder_files(tmp_path):
    assert read_subset(tmp_path) is None and read_overrides(tmp_path) is None
    (tmp_path / "subset.json").write_text(json.dumps({"demo": ["a", "b"]}))
    assert read_subset(tmp_path) == {"demo": ["a", "b"]}
    (tmp_path / "subset.json").write_text(json.dumps(["a"]))
    with pytest.raises(ValueError, match="subset.json"):
        read_subset(tmp_path)
    write_jsonl(tmp_path / "prompt_overrides.jsonl", [{"dataset": "demo", "id": "a", "edits": [["x", "y"]]},
                                                      {"dataset": "other", "id": "b", "edits": []}])
    assert read_overrides(tmp_path) == {"demo": {"a": [["x", "y"]]}, "other": {"b": []}}


def test_apply_edits_replaces_exactly_one_occurrence():
    assert apply_edits("a b c", [["b", "x"], ["x c", "y"]]) == "a y"      # in order, each on the edited text
    assert apply_edits("same", []) == "same"
    with pytest.raises(ValueError, match="occurs 0 times"):
        apply_edits("a b", [["z", "y"]])
    with pytest.raises(ValueError, match="occurs 2 times"):
        apply_edits("a a", [["a", "b"]])


def test_prompts_use_the_shared_templates_and_need_a_row_per_overridden_sample(project):
    samples = load_samples("demo", paths=project.paths)
    prompts = render_prompts("demo", samples)
    assert prompts["mcq-1"] == ("What colour dominates?\nA. red\nB. blue\n"
                                "Answer with the option's letter from the given choices directly.")
    assert prompts["yn-1"] == "Is the video red?\nAnswer with yes or no."
    assert prompts["open-1"] == "Describe the video."
    with pytest.raises(ValueError, match="no row for mcq-2"):
        render_prompts("demo", samples, {"mcq-1": []})


def test_prompt_map_digest_is_order_independent():
    assert prompt_map_digest({"a": "x", "b": "y"}) == prompt_map_digest({"b": "y", "a": "x"})
    assert prompt_map_digest({"a": "x", "b": "y"}) != prompt_map_digest({"a": "x", "b": "z"})


def test_check_prepared_compares_counts_and_hashes(project):
    result = check_prepared("demo", project.paths, expected={})
    assert result["problems"] == [] and result["count"] == 6
    assert result["notes"] and "no row in manifests/prepared.yaml" in result["notes"][0]     # nothing to compare with
    exact = {"demo": {"count": 6, "samples_sha256": result["samples_sha256"],
                      "prompts_sha256": result["prompts_sha256"]}}
    checked = check_prepared("demo", project.paths, expected=exact)
    assert checked["problems"] == [] and checked["notes"] == []
    wrong = {"demo": {"count": 7, "samples_sha256": "0" * 64, "prompts_sha256": "1" * 64}}
    assert len(check_prepared("demo", project.paths, expected=wrong)["problems"]) == 3
    assert check_videos("demo", project.paths) == {"manifest": None, "problems": []}     # no shipped manifest


def test_expected_manifest_is_strict(tmp_path):
    path = tmp_path / "prepared.yaml"
    path.write_text("version: 1\nprepared:\n  demo: {count: 6, samples_sha256: abc}\n")
    assert load_expected(path) == {"demo": {"count": 6, "samples_sha256": "abc"}}
    assert load_expected(tmp_path / "absent.yaml") == {}
    path.write_text("version: 1\nprepared:\n  demo: {cuont: 6}\n")
    with pytest.raises(ValueError, match="unknown fields"):
        load_expected(path)
    path.write_text("demo: {count: 6}\n")
    with pytest.raises(ValueError, match="expected"):
        load_expected(path)
    assert "videomme" in load_expected() or not load_expected()      # the shipped manifest parses


def test_prepare_command(project, capsys):
    assert project.cli("prepare", "demo") == 0
    out = capsys.readouterr()
    record = json.loads(out.out.strip().splitlines()[-1])
    assert record["dataset"] == "demo" and record["status"] == "exists" and record["ok"] and record["count"] == 6
    assert record["verified"] is False and "NOTE demo: no row in manifests/prepared.yaml" in out.err
    assert project.cli("prepare", "demo", "--verify") == 0
    assert project.cli("prepare", "demo2", "--verify") == 1                # not a benchmark of this project
