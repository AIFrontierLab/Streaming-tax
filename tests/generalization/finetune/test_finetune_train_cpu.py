"""The real training unit on a CPU with a toy model (finetune_toy.py): LoRA placement and zero initialisation, the
AdamW loop with checkpoints, a resumed run bit-identical to an uninterrupted one, DONE.json and the saved adapter,
which the offline adapters' attach_lora() re-attaches exactly. Needs torch and peft (skipped without them)."""
import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("safetensors")

import finetune_toy  # noqa: E402

from streamreplace.benchmarks import registry  # noqa: E402


@pytest.fixture
def toy(tmp_path):
    yield finetune_toy.check_training(tmp_path)
    registry.unregister("ftdemo")


def test_training_resumes_bit_exactly_and_saves_the_trained_function(toy):
    assert toy["lora_modules"] == 2 * 7 and toy["zero_init_unchanged"]
    assert toy["steps"] == 6 and toy["pilot_stopped_at"] == 2 and toy["loss_steps"] == 6
    assert toy["losses_equal"] and toy["adapter_equal"]           # resumed at step 2 == never stopped
    assert toy["done_file"] and toy["no_checkpoint_left"] and toy["done_sha_is_the_file"]
    assert toy["lora_parameters"] > 0
    assert toy["reload_equals_trained"] and toy["reload_changes_the_function"]       # models.lora.attach_lora
