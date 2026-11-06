"""The training unit without a GPU: learning-rate schedule, epoch orders, the unit's identity and sample order (formal
and seed-replication units), the LoRA target patterns and the training payload."""
import re

import pytest

from finetune_helpers import DEMO

from streamreplace.core.io import write_json
from streamreplace.generalization.finetune import find_unit, load_experiment, seed_units, unit_dir, units
from streamreplace.generalization.finetune import data, lora, train


def test_schedule_and_epoch_order():
    values = [train.schedule(s, 125, 4) for s in range(125)]
    assert values[0] == 0.25 and values[3] == 1.0 and values[4] == 1.0 and values[-1] < 0.001
    ids = [str(i) for i in range(20)]
    assert train.epoch_order(ids, 0, 0) == ids and sorted(train.epoch_order(ids, 1, 0)) == sorted(ids)
    assert train.epoch_order(ids, 1, 0) != train.epoch_order(ids, 2, 0)
    assert train.epoch_order(ids, 0, 1) == ids and train.epoch_order(ids, 0, 1, True) != train.epoch_order(ids, 0, 2, True)
    assert train.epoch_order(ids, 1, 1, True) == train.epoch_order(ids, 1, 1)          # later epochs as before


def payload(exp, unit, tmp_path) -> dict:
    return {"version": train.PAYLOAD_VERSION, "unit": unit, "model": {"id": f"{unit['model']}@{DEMO}-0", "spec": {}},
            "adapter": "qwen_offline", "family": "qwen_vl", "settings": {"generation": {"max_new_tokens": 8}},
            "lora": exp.lora, "train": exp.train, "selection": str(exp.root), "dir": str(tmp_path / "unit"),
            "max_steps": None}


def test_a_seed_unit_changes_the_lora_seed_and_the_order_of_the_same_ids(exp, tmp_path):
    order = [f"id{i}" for i in range(2000)]
    write_json(exp.root / "train_order.json", {DEMO: order})
    write_json(exp.root / "selection.json", {DEMO: {"train_sha256": data._digest(order), "test_sha256": "t"}})
    formal = train.Unit(payload(exp, dict(units(exp)[0], budget=12), tmp_path))
    seeded = train.Unit(payload(exp, dict(next(u for u in seed_units(exp) if u["tag"] == "s2" and u["model"] == "par"),
                                          budget=12), tmp_path))
    assert (formal.seed, formal.shuffle, seeded.seed, seeded.shuffle) == (0, False, 2, True)
    assert formal.ids == seeded.ids == order[:12] and formal.hash != seeded.hash
    assert formal.steps == 6 and formal.warmup == 1                  # ceil(12 / 2) steps, ceil(0.03 * 6) warmup
    formal.by_id = seeded.by_id = {i: i for i in order[:12]}
    first = [seeded.sample(k) for k in range(12)]
    assert [formal.sample(k) for k in range(12)] == order[:12]
    assert sorted(first) == sorted(order[:12]) and first != order[:12]


def test_three_epochs_reshuffle_after_the_first(exp, tmp_path):
    order = [f"id{i}" for i in range(20)]
    write_json(exp.root / "train_order.json", {DEMO: order})
    write_json(exp.root / "selection.json", {DEMO: {"train_sha256": data._digest(order), "test_sha256": "t"}})
    unit = find_unit(exp, "par", DEMO, 6, "e3")
    trainer = train.Unit(payload(exp, unit, tmp_path))
    trainer.by_id = {i: i for i in order}
    assert trainer.samples_total == 18 and trainer.steps == 9
    seen = [trainer.sample(k) for k in range(18)]
    assert seen[:6] == order[:6] and sorted(seen[6:12]) == sorted(order[:6]) and seen[6:12] != seen[12:]


def test_the_unit_refuses_a_short_order_or_another_payload_version(exp, tmp_path):
    write_json(exp.root / "train_order.json", {DEMO: ["a", "b"]})
    write_json(exp.root / "selection.json", {DEMO: {"train_sha256": data._digest(["a", "b"]), "test_sha256": "t"}})
    with pytest.raises(ValueError, match="< budget"):
        train.Unit(payload(exp, units(exp)[0], tmp_path))
    with pytest.raises(ValueError, match="payload version"):
        train.Unit({**payload(exp, units(exp)[0], tmp_path), "version": 0})


@pytest.mark.parametrize("family, name, matches", [
    ("qwen_vl", "model.language_model.layers.3.mlp.down_proj", True),
    ("qwen_vl", "model.visual.blocks.3.mlp.down_proj", False),            # Qwen2.5-VL vision MLP has the same suffix
    ("llava_ov", "model.layers.0.self_attn.q_proj", True),
    ("llava_ov", "model.vision_tower.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj", False),
    ("livestar", "language_model.model.layers.31.attention.wqkv", True),
    ("livestar", "vision_model.encoder.layers.0.attn.qkv", False),
    ("moss", "model.language_model.layers.2.cross_attn.k_proj", True),
    ("moss", "model.language_model.layers.2.mlp.gate_proj", True),
    ("moss", "lm_head", False)])
def test_lora_targets_are_the_decoder_only(family, name, matches):
    assert bool(re.fullmatch(lora.FAMILIES[family]["targets"], name)) == matches


def test_every_fine_tunable_adapter_has_a_family():
    assert lora.family_of("qwen_offline") == "qwen_vl" and lora.FAMILIES["aura"] is lora.FAMILIES["qwen_vl"]
    assert {lora.family_of(a) for a in ("llava_offline", "internvl_offline", "moss_offline")} == {"llava_ov", "livestar", "moss"}
    with pytest.raises(ValueError, match="no LoRA family"):
        lora.family_of("aura_stream")


def test_the_training_payload_of_a_paper_unit(tmp_path):
    from streamreplace import paths as paths_module
    from streamreplace.config.load import load_config
    from streamreplace.generalization.finetune.launch import train_command, training_payload
    exp = load_experiment(root=tmp_path / "ft", outputs=tmp_path)
    config, paths = load_config(), paths_module.resolve(home=tmp_path / "home", environ={})
    unit = find_unit(exp, "moss_vl", "lingoqa", 2000)
    body = training_payload(exp, config, paths, unit)
    assert body["family"] == "moss" and body["adapter"] == "moss_offline" and body["unit"] == unit
    assert body["model"]["id"] == "moss_vl@lingoqa-0" and body["model"]["spec"]["options"] == {"deterministic": True}
    assert body["settings"]["video"]["max_pixels"] == 602112 and body["settings"]["video"]["sampler"] == "all"
    assert body["lora"] == {"r": 16, "alpha": 32} and body["train"]["grad_accum"] == 8
    assert body["selection"] == str(exp.root) and body["dir"] == str(unit_dir(exp, unit))
    command, variables, folder = train_command(exp, config, paths, unit, gpu=3, threads=2)
    assert command[1:3] == ["-m", train.__name__] and command[0].endswith("modern/bin/python")
    assert variables["CUDA_VISIBLE_DEVICES"] == "3" and variables["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert variables["FLASH_ATTENTION_DETERMINISTIC"] == "1" and (folder / "payload.json").is_file()
