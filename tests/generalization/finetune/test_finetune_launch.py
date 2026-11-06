"""The evaluation model ids of the paper's experiment (model specs with options.lora and options.deterministic),
their resolved settings, the evaluation planning through the run pipeline and
the printed commands."""
import hashlib
import json

import pytest

from streamreplace import paths as paths_module
from streamreplace.config.load import load_config
from streamreplace.core.io import write_json
from streamreplace.generalization.finetune import find_unit, load_experiment, model_id, seed_units, unit_dir, units
from streamreplace.generalization.finetune import launch
from streamreplace.paths import project_root

# the original runs' resolved settings of two zero-shot ids (their meta.json files)
LLAVA_LINGOQA = {"generation": {"max_new_tokens": 256}, "image": {"max_pixels": 1003520},
                 "video": {"max_pixels": 200704, "pad_square": True, "sampler": "all"}, "stream": {"fps": 1}}
MOSS_EYEPCR = {"generation": {"max_new_tokens": 512}, "image": {"max_pixels": 1003520},
               "video": {"max_pixels": 602112, "video_max_pixels": 65536000, "nframes": 16, "sampler": "linspace_int"},
               "stream": {"fps": 1}}


@pytest.fixture
def paper(tmp_path):
    return load_experiment(root=tmp_path / "ft", outputs=tmp_path), load_config()


def test_every_evaluation_id_is_an_ordinary_model_entry(paper):
    exp, config = paper
    entries = launch.model_entries(exp, config)
    assert len(entries) == 11 * 3 + len(units(exp)) + len(seed_units(exp))
    zero, tuned = entries["moss_vl@lingoqa-0"], entries["moss_vl@lingoqa-2k"]
    assert {k for k in zero if zero[k] != tuned[k]} == {"options"}
    assert tuned["options"] == {**zero["options"], "lora": str(unit_dir(exp, find_unit(exp, "moss_vl", "lingoqa", 2000)) / "adapter")}
    assert zero["options"] == {"deterministic": True} and "stream" not in entries["thinkstream_3b@lingoqa-0"]
    merged = entries["mmduet_7b_merged@eyepcr-0"]
    assert merged["role"] == "child" and merged["checkpoint"].startswith("wangyueqian/MMDuet@")
    assert merged["base"].startswith("lmms-lab/llava-onevision") and merged["offline"]["adapter"] == "llava_offline"
    assert merged["options"] == {"merge": True, "dtype": "bfloat16", "deterministic": True}
    assert entries["llava_ov_7b@openuav_qa-5k"]["options"]["dtype"] == "bfloat16"
    assert entries["aura_8b@lingoqa-1k-s2"]["options"]["lora"].endswith("seeds/adapters/lingoqa/aura_8b/1k-s2/adapter")


def test_the_ids_resolve_the_settings_of_the_original_runs(paper):
    exp, config = paper
    ft = launch.experiment_config(exp, config)
    assert ft.settings("lingoqa", "llava_ov_7b@lingoqa-0", "offline") == LLAVA_LINGOQA
    assert ft.settings("eyepcr", "moss_vl@eyepcr-1k", "offline") == MOSS_EYEPCR
    for member in ("qwen25vl_3b", "aura_8b", "livestar_8b"):             # no override: exactly the pair member's
        for dataset in exp.datasets:
            assert ft.settings(dataset, model_id(member, dataset, 0), "offline") == config.settings(dataset, member, "offline")
    assert ft.model("qwen25vl_3b@lingoqa-0").pair is None and ft.model("qwen25vl_3b@lingoqa-0").modes == ("offline",)


def test_the_models_file_loads_as_an_extra_config(paper, tmp_path):
    exp, config = paper
    path = launch.write_models_file(exp, config, tmp_path / "ft_models.yaml")
    loaded = load_config(extra=[path])
    assert "mmduet_7b_merged@lingoqa-5k" in loaded.models and loaded.model("qwen3vl_8b@eyepcr-0").role.value == "parent"
    with pytest.raises(ValueError, match="already defines"):
        launch.experiment_config(exp, loaded)


def trained(exp, model: str, dataset: str, budget: int, tag=None, content: bytes = b"weights") -> str:
    """A fake trained unit (DONE.json + the adapter's weights file); returns the weights' sha256."""
    folder = unit_dir(exp, find_unit(exp, model, dataset, budget, tag))
    (folder / "adapter").mkdir(parents=True)
    (folder / "adapter" / "adapter_model.safetensors").write_bytes(content)
    write_json(folder / "DONE.json", {"adapter_sha256": hashlib.sha256(content).hexdigest()})
    return hashlib.sha256(content).hexdigest()


def test_evaluations_are_planned_through_the_run_pipeline(paper, tmp_path):
    exp, config = paper
    paths = paths_module.resolve(home=tmp_path / "home", outputs=tmp_path, environ={})
    with pytest.raises(ValueError, match="finetune prepare"):
        launch.plan_evaluations(exp, config, paths, ["lingoqa"], ["qwen25vl_3b"])
    write_json(exp.root / "subset.json", {"lingoqa": ["a", "b"]})
    digest = trained(exp, "qwen25vl_3b", "lingoqa", 1000)
    jobs, waiting, folder = launch.plan_evaluations(exp, config, paths, ["lingoqa"], ["qwen25vl_3b", "thinkstream_3b"])
    assert folder == exp.root and sorted(waiting) == sorted(model_id(m, "lingoqa", b) for m in ("qwen25vl_3b", "thinkstream_3b")
                                                            for b in (1000, 2000, 5000) if (m, b) != ("qwen25vl_3b", 1000))
    names = [j.job_name for j in jobs]
    assert names == ["qwen25vl_3b@lingoqa-0.offline", "qwen25vl_3b@lingoqa-1k.offline", "thinkstream_3b@lingoqa-0.offline"]
    zero, tuned = jobs[0], jobs[1]
    assert zero.lora is None and tuned.lora == unit_dir(exp, find_unit(exp, "qwen25vl_3b", "lingoqa", 1000)) / "adapter"
    assert tuned.payload["model"]["options"] == {"deterministic": True, "lora": {"sha256": digest}}
    assert zero.deterministic and tuned.deterministic and zero.subset == ["a", "b"] and zero.identity != tuned.identity
    assert zero.settings == tuned.settings and jobs[2].arm.mode.value == "offline"


def test_controls_and_seed_folders_get_their_own_questions_and_settings(paper, tmp_path):
    exp, config = paper
    paths = paths_module.resolve(home=tmp_path / "home", outputs=tmp_path, environ={})
    frozen = json.loads((project_root() / "paper" / "frozen" / "finetune" / "subset.json").read_text())
    write_json(exp.root / "subset.json", frozen)
    jobs, waiting, folder = launch.plan_evaluations(exp, config, paths, ["openuav_qa"], ["qwen25vl_3b"],
                                                    control="images")
    assert folder == exp.root / "images" and [j.job_name for j in jobs] == ["qwen25vl_3b@openuav_qa-0.offline"]
    assert jobs[0].settings["video"]["as_images"] is True and jobs[0].settings["image"] == {"max_pixels": 401408,
                                                                                            "min_pixels": 200704}
    assert json.loads((folder / "subset.json").read_text()) == {"openuav_qa": frozen["openuav_qa"]}
    trained(exp, "aura_8b", "lingoqa", 1000, "s1")
    jobs, waiting, folder = launch.plan_evaluations(exp, config, paths, seeds=True)
    assert folder == exp.root / "seeds" and [j.job_name for j in jobs] == ["aura_8b@lingoqa-1k-s1.offline"]
    assert json.loads((folder / "subset.json").read_text()) == {"lingoqa": frozen["lingoqa"]}
    assert len(waiting) == len(seed_units(exp)) - 1


def test_the_whole_experiment_as_commands(paper):
    exp, _ = paper
    lines = launch.commands(exp, gpus=[0, 1])
    ft = f"streamreplace finetune --root {exp.root}"            # every step names the experiment root
    trains = [l for l in lines if " train --model " in l]
    assert len(trains) == len(units(exp)) + len(seed_units(exp))
    assert trains[0].startswith(f"{ft} train --model ") and " --budget 5000 " in trains[0]
    assert trains[0].endswith(" --gpus 0") and trains[1].endswith(" --gpus 1")
    assert f"{ft} prepare openuav_qa lingoqa eyepcr" in lines and "streamreplace prepare openuav_qa lingoqa eyepcr" in lines
    assert f"{ft} eval --control blank --gpus 0,1" in lines and lines[-1] == f"{ft} seeds"
    assert f"streamreplace --outputs {exp.seeds_folder()} score --judge --judge-gpus 0" in lines
    assert any(l.endswith("--budgets 1000 --tags e3 --models qwen25vl_3b,thinkstream_3b,mmduet2_3b --gpus 0,1") for l in lines)
    custom = launch.commands(exp, prefix="streamreplace --config /x.yaml", finetune="FT")
    assert "FT report" in custom and f"streamreplace --config /x.yaml --outputs {exp.root} score --judge --judge-gpus 0" in custom


def test_training_units_are_listed_with_their_state(paper):
    exp, _ = paper
    trained(exp, "moss_vl", "lingoqa", 1000)
    lines = launch.describe_units(exp, launch.select_units(exp, ["lingoqa"], ["moss_vl"]))
    assert len(lines) == 3 and "trained (adapter sha256 " in lines[0] and lines[1].endswith("not started")
    assert len(launch.select_units(exp, seeds=True)) == len(seed_units(exp))
    assert [u["budget"] for u in launch.longest_first(exp, units(exp))][:3] == [5000, 5000, 5000]
