"""configs/experiments.yaml: every experiment is well formed, its derived models build, and its arms plan (CPU, no
data: planning reads the configuration, the frozen item lists and the shipped plugin artefacts only)."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

from streamreplace import experiments, paths as paths_module
from streamreplace.config.load import load_config
from streamreplace.core.io import read_jsonl

CHILDREN = ["mmduet_7b", "thinkstream_3b", "mmduet2_3b", "aura_8b", "livestar_8b", "moss_vl_rt"]


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture
def paths(tmp_path):
    paths = paths_module.resolve(home=tmp_path)
    paths_module.set_current(paths)
    return paths


def test_every_experiment_is_well_formed(config, paths):
    full = experiments.with_models(config)
    names = experiments.names(config)
    assert {"window", "memory", "response", "question_timing", "parent_stream", "recall_budget",
            "streaming_benchmarks", "response_forms", "harmless_requests", "framing", "harm_probe",
            "steering"} <= set(names)
    for name in names:
        exp = experiments.load(config, paths, name)
        assert exp.kind in ("qa", "forms", "probe") and exp.spec.get("title") and exp.spec.get("paper"), name
        assert exp.children and set(exp.children) <= set(CHILDREN), name
        arms = exp.arms
        assert all(a in arms and b in arms for a, b in exp.contrasts), name
        items = exp.items()
        for dataset in exp.benchmarks():
            assert dataset in config.benchmarks, (name, dataset)
            if items is not None:
                assert items[dataset] and len(set(items[dataset])) == len(items[dataset]), (name, dataset)
        for arm in arms.values():
            for child in exp.children:
                job = arm.job_name(child, experiments.parent_of(config, child))
                if job is not None:
                    assert job.rsplit(".", 1)[0].split("@")[0] in full.models, (name, arm.label, job)


def test_unknown_names_and_keys_are_refused(config, paths):
    with pytest.raises(ValueError, match="unknown experiment"):
        experiments.load(config, paths, "no_such_experiment")
    with pytest.raises(ValueError, match="needs `from`"):
        experiments.derived_entry(config, "x", {"settings": {}})
    exp = experiments.Experiment("x", {"contrasts": ["A"]}, paths.outputs, Path("."))
    with pytest.raises(ValueError, match="not 'B - A'"):
        exp.contrasts
    exp = experiments.Experiment("x", {"arms": {"A": {"job": "m.offline", "sets": {}}}}, paths.outputs, Path("."))
    with pytest.raises(ValueError, match="needs `job`"):
        exp.arms


def test_derived_models_change_one_thing(config):
    full = experiments.with_models(config)
    window = full.model("aura_8b_window")                       # the offline input cut to AURA's streaming window
    assert window.settings["video"]["last_seconds"] == 30
    assert window.settings["video"]["max_pixels"] == config.pair_of("aura_8b").video["max_pixels"]
    assert window.checkpoint == config.model("aura_8b").checkpoint
    parent = full.model("qwen3vl_8b_in_aura")                   # the parent's weights in AURA's stream adapter
    assert parent.checkpoint == config.model("qwen3vl_8b").checkpoint
    assert parent.stream.adapter == config.model("aura_8b").stream.adapter
    recall = full.model("aura_8b_recall")                       # Recall alone, on the transformers rebuild
    assert recall.stream.adapter == "aura_stream_hf"
    assert recall.settings["stream"] == {**config.model("aura_8b").settings.get("stream", {}),
                                         "window": "full", "budget": 30, "recent": 10}
    assert full.model("mmduet_7b_forced").settings["stream"]["force_answer"] is True
    assert "aura_8b_window" not in config.models               # the paper's eleven models are untouched


def test_arm_settings_per_child_or_for_all():
    arm = experiments.Arm("A", "{child}.stream", {"aura_8b": {"stream": {"window": 60}}})
    assert arm.settings("aura_8b") == {"stream": {"window": 60}} and arm.settings("thinkstream_3b") == {}
    assert experiments.Arm("A", "{child}.stream", {"stream": {"x": 1}}).settings("any") == {"stream": {"x": 1}}
    per_child = experiments.Arm("A", {"mmduet_7b": "mmduet_7b@shadow.stream"}, {})
    assert per_child.job_name("mmduet_7b", "llava_ov_7b") == "mmduet_7b@shadow.stream"
    assert per_child.job_name("aura_8b", "qwen3vl_8b") is None


def test_plans_of_the_paper_experiments(config, paths):
    planned = {}
    for name in experiments.names(config):
        exp = experiments.load(config, paths, name)
        if exp.kind != "probe":
            jobs = experiments.plan(exp, config, paths)
            planned[name] = Counter(Path(str(j.outputs)).name for j in jobs)
    assert planned["window"] == {"window": 6} and planned["response"] == {"response": 3}
    assert planned["duration"] == {} and planned["response_forms"] == {}       # reference arms only: nothing to run
    assert planned["recall_budget"] == {"recall_b": 2, "recent_b": 2, "recall_2b": 2, "recent_2b": 2, "full": 2}
    assert planned["streaming_benchmarks"] == {"streaming_benchmarks": 36, "forced": 2}
    assert planned["framing"] == {"bare": 11, "cue": 11, "black": 11, "real": 11}     # five parents, six children
    assert planned["harmless_requests"] == {"howto": 12}


def test_experiment_folders_hold_their_items_and_prompt_edits(config, paths):
    exp = experiments.load(config, paths, "framing")
    jobs = experiments.plan(exp, config, paths, children=["aura_8b"], arms=["C_bare", "C_real"])
    assert sorted(j.name for j in jobs) == ["video_safetybench/aura_8b.offline"] * 2
    ids = exp.items()["video_safetybench"]
    assert json.loads((exp.root / "bare" / "subset.json").read_text()) == {"video_safetybench": ids}
    rows = read_jsonl(exp.root / "bare" / "prompt_overrides.jsonl")          # the framing sentence is removed
    assert len(rows) == len(ids) == 240 and rows[0]["edits"] == [["These are frames of a video.\n", ""]]
    assert not (exp.root / "real" / "prompt_overrides.jsonl").exists()
    bare = next(j for j in jobs if Path(str(j.outputs)).name == "bare")
    assert bare.settings["video"]["drop"] is True and bare.settings["generation"]["max_new_tokens"] == 160
    with pytest.raises(ValueError, match="another item list"):
        experiments.write_subset(exp.root / "bare", {"video_safetybench": ids[:3]})


def test_harmless_requests_replace_the_question(config, paths):
    from streamreplace.data.prompts import apply_edits
    from streamreplace.plugin.calibrate.harmless import DESCRIBE, HOWTO
    exp = experiments.load(config, paths, "harmless_requests")
    experiments.plan(exp, config, paths, children=["aura_8b"])
    ids = exp.items()["video_safetybench"]
    rows = read_jsonl(exp.root / "howto" / "prompt_overrides.jsonl")
    assert [r["id"] for r in rows] == ids and len(ids) == 40
    prompts = [apply_edits("These are frames of a video.\nA harmful question.", r["edits"]) for r in rows]
    assert prompts[0] == "These are frames of a video.\n" + HOWTO[0]
    assert prompts[9] == "These are frames of a video.\n" + HOWTO[1]         # eight templates in rotation
    assert Counter(p.split("\n", 1)[1] for p in prompts) == Counter({t: 5 for t in HOWTO})
    describe = experiments.harmless_edits({"d": ["a", "b"]}, "describe")
    assert describe["b"] == [[None, "These are frames of a video.\n" + DESCRIBE]]
    with pytest.raises(ValueError, match="howto or describe"):
        experiments.harmless_edits({"d": ["a"]}, "other")
