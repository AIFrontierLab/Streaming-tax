"""The fine-tuning specification (configs/generalization/finetune.yaml) and the names of its units and model ids."""
from pathlib import Path

import pytest

from finetune_helpers import spec

from streamreplace.generalization.finetune import (Experiment, find_unit, label, load_experiment, model_id, seed_units,
                                                   unit_dir, units, validate_spec)


@pytest.fixture
def paper(tmp_path):
    """The shipped specification, rooted in a temporary folder."""
    return load_experiment(root=tmp_path / "ft", outputs=tmp_path)


def test_the_shipped_specification_is_the_papers_design(paper):
    assert paper.datasets == ["openuav_qa", "lingoqa", "eyepcr"] and paper.budgets == [1000, 2000, 5000]
    assert paper.test_size == 2000 and paper.top_budget == 5000 and paper.seed == 0
    assert paper.pairs == {"p1": ["llava_ov_7b", "mmduet_7b_merged"], "p2": ["qwen25vl_3b", "thinkstream_3b", "mmduet2_3b"],
                           "p3": ["qwen3vl_8b", "aura_8b"], "p4": ["livestar_iv_8b", "livestar_8b"],
                           "p5": ["moss_vl", "moss_vl_rt"]}
    assert len(paper.members) == 11 and paper.lora == {"r": 16, "alpha": 32}
    assert paper.train == {"lr": 1.0e-4, "weight_decay": 0.0, "warmup_ratio": 0.03, "epochs": 1, "grad_accum": 8,
                           "clip": 1.0, "seed": 0, "save_every_steps": 25, "save_every_minutes": 10}
    assert set(paper.primary_excludes) == {"openuav_qa"}
    assert paper.controls["blank"]["size"] == 500 and paper.controls["blank"]["set"] == {"video": {"blank": True}}
    assert paper.folder("blank") == paper.root / "blank" and paper.seeds_folder() == paper.root / "seeds"


def test_units_and_seed_units_of_the_paper(paper):
    formal, seeded = units(paper), seed_units(paper)
    assert len(formal) == 11 * 3 * 3 + 3                        # every member x dataset x budget, + the 1K x 3 epochs
    assert all(set(u) == {"model", "dataset", "budget", "epochs", "tag"} for u in formal)   # part of unit identities
    assert len(seeded) == 11 * 1 * 3 * 2                         # every member, LingoQA, 3 budgets, seeds 1 and 2
    unit = find_unit(paper, "moss_vl", "lingoqa", 1000, "s1")
    assert unit["seed"] == 1 and unit["shuffle"] is True
    assert unit_dir(paper, unit) == paper.root / "seeds" / "adapters" / "lingoqa" / "moss_vl" / "1k-s1"
    e3 = find_unit(paper, "thinkstream_3b", "openuav_qa", 1000, "e3")
    assert e3["epochs"] == 3 and unit_dir(paper, e3) == paper.root / "adapters/openuav_qa/thinkstream_3b/1k-e3"
    with pytest.raises(ValueError, match="no fine-tuning unit"):
        find_unit(paper, "moss_vl", "lingoqa", 3000)


def test_labels_and_model_ids():
    assert label(0) == "0" and label(5000) == "5k" and label(1000, "e3") == "1k-e3" and label(6) == "6"
    assert model_id("moss_vl", "lingoqa", 0) == "moss_vl@lingoqa-0"
    assert model_id("aura_8b", "lingoqa", 2000, "s2") == "aura_8b@lingoqa-2k-s2"


def test_the_root_defaults_to_the_outputs_root(tmp_path):
    exp = load_experiment(outputs=tmp_path)
    assert exp.root == tmp_path / "finetune"


def test_local_overrides_are_merged_on_top(tmp_path):
    exp = load_experiment(outputs=tmp_path, overrides={"test_size": 100, "train": {"grad_accum": 4}})
    assert exp.test_size == 100 and exp.train["grad_accum"] == 4 and exp.train["lr"] == 1.0e-4


@pytest.mark.parametrize("change, message", [
    ({"budgets": [0, 1000]}, "budgets must be positive"),
    ({"pairs": {"p": ["only_one"]}}, "at least two models"),
    ({"train": {"lr": 1.0}}, "train: needs exactly"),
    ({"controls": {"blank": {"size": 3}}}, "needs an `outputs` folder"),
    ({"seeds": {"outputs": "s", "datasets": ["x"]}}, "seeds: needs"),
    ({"surprise": 1}, "unknown keys"),
])
def test_invalid_specifications_are_refused(change, message):
    with pytest.raises(ValueError, match=message):
        validate_spec(spec(**change))


def test_a_missing_specification_file_is_reported(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_experiment(path=tmp_path / "missing.yaml", outputs=tmp_path)


def test_unknown_controls_and_a_missing_seed_block(tmp_path):
    exp = Experiment(spec(seeds=None), Path(tmp_path))
    with pytest.raises(ValueError, match="unknown control"):
        exp.folder("images")
    with pytest.raises(ValueError, match="no `seeds` block"):
        exp.seeds_folder()
    assert seed_units(exp) == []
