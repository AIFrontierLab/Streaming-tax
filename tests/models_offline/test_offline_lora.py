"""load_lora(): a fine-tuned model id's LoRA adapter attached unmerged to an offline adapter's model.

The wiring is checked with stand-ins for every adapter; attach_lora() itself runs on a toy model with the
Qwen2.5-VL / Qwen3-VL module layout when torch, PEFT and safetensors are installed (a model environment), saving the
adapter the way the fine-tuning experiment saves it and checking that the attached model computes exactly the trained
function, keeps the trained float32 values, and refuses adapters that do not fit."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from streamreplace.models import internvl_offline, llava_offline, moss_offline, qwen_offline, registry
from streamreplace.models.lora import attach_lora
from offline_fakes import FakeHead

ADAPTERS = [(qwen_offline, "QwenOffline", ("model.language_model.layers", 7)),
            (llava_offline, "LlavaOffline", ("model.layers", 7)),
            (internvl_offline, "InternVLOffline", ("language_model.model.layers", 5)),
            (moss_offline, "MossOffline", ("model.language_model.layers", 7))]


def bare(module, cls, loaded: bool = True):
    adapter = object.__new__(getattr(module, cls))
    adapter.spec = SimpleNamespace(id="demo@lingoqa-1k")
    adapter.model, adapter.head = object(), FakeHead() if loaded else None
    return adapter


@pytest.mark.parametrize("module, cls, layers", ADAPTERS)
def test_load_lora_attaches_to_the_loaded_model(monkeypatch, module, cls, layers):
    calls = []
    monkeypatch.setattr(module, "attach_lora", lambda model, layer_list, per_layer, directory: calls.append(
        (model, layer_list, per_layer, directory)) or ["wrapped"])
    adapter = bare(module, cls)
    assert getattr(module, cls).LORA_LAYERS == layers
    assert adapter.load_lora("/runs/adapters/lingoqa/demo/1k/adapter") == ["wrapped"]
    assert calls == [(adapter.model, *layers, "/runs/adapters/lingoqa/demo/1k/adapter")]


@pytest.mark.parametrize("module, cls, layers", ADAPTERS)
def test_load_lora_needs_the_weights(module, cls, layers):
    with pytest.raises(RuntimeError, match="load=False"):
        bare(module, cls, loaded=False).load_lora("adapter")


def test_a_lora_on_mmduet_needs_the_merged_model():
    adapter = bare(llava_offline, "LlavaOffline")
    adapter.model = SimpleNamespace(get_base_model=lambda: None)          # PEFT's wrapper of the unmerged LoRA
    with pytest.raises(RuntimeError, match="merge"):
        adapter.load_lora("adapter")


def test_layers_agree_with_the_fine_tuning_experiment():
    """The decoder layers each adapter attaches to are the ones the fine-tuning experiment trains (its LoRA families
    per offline adapter)."""
    lora = pytest.importorskip("streamreplace.generalization.finetune.lora")
    if not hasattr(lora, "ADAPTERS") or not hasattr(lora, "FAMILIES"):
        pytest.skip("the fine-tuning code has no adapter -> family table")
    for name, family in lora.ADAPTERS.items():
        row = lora.FAMILIES[family]
        assert registry.load_class(name).LORA_LAYERS == (row["layers"], row["per_layer"]), name


# ---------- attach_lora on a toy model (torch, PEFT, safetensors) ----------

TARGETS = r"model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)"


def toy(dtype):
    """Two decoder layers with the Qwen-VL module names, a vision MLP whose names end like the decoder's (it must not
    get a LoRA) and an LM head; base weights from seed 0."""
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn, self.mlp = nn.Module(), nn.Module()
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(self.self_attn, name, nn.Linear(8, 8, bias=False))
            for name in ("gate_proj", "up_proj", "down_proj"):
                setattr(self.mlp, name, nn.Linear(8, 8, bias=False))

        def forward(self, x):
            a, m = self.self_attn, self.mlp
            x = x + a.o_proj(a.q_proj(x) * a.k_proj(x) + a.v_proj(x))
            return x + m.down_proj(torch.tanh(m.gate_proj(x)) * m.up_proj(x))

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.language_model, self.model.visual = nn.Module(), nn.Module()
            self.model.language_model.layers = nn.ModuleList([Layer(), Layer()])
            self.model.visual.mlp = nn.Module()
            self.model.visual.mlp.gate_proj = nn.Linear(8, 8)
            self.lm_head = nn.Linear(8, 16, bias=False)

        def forward(self, x):
            for layer in self.model.language_model.layers:
                x = layer(x)
            return self.lm_head(x)

    torch.manual_seed(0)
    return Toy().to(dtype)


def trained(folder, dtype, targets: str = TARGETS):
    """A 'fine-tuned' toy: LoRA injected as the experiment injects it, set to random float32 values, saved as the
    experiment saves it (adapter_model.safetensors + adapter_config.json). Returns (model, saved tensors)."""
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    model = toy(dtype)
    config = peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none", target_modules=targets)
    peft.inject_adapter_in_model(config, model)
    generator = torch.Generator().manual_seed(1)
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.data = torch.randn(param.shape, generator=generator)
    state = {k: v.detach().cpu().contiguous() for k, v in peft.get_peft_model_state_dict(model).items()}
    folder.mkdir(parents=True, exist_ok=True)
    safetensors_torch.save_file(state, str(folder / "adapter_model.safetensors"))
    config.save_pretrained(str(folder))
    return model.eval(), state


def test_attach_reproduces_the_trained_function(tmp_path):
    torch = pytest.importorskip("torch")
    model, _ = trained(tmp_path / "adapter", torch.float32)
    fresh = toy(torch.float32)
    wrapped = attach_lora(fresh, "model.language_model.layers", 7, tmp_path / "adapter")
    assert len(wrapped) == 14 and all(n.startswith("model.language_model.layers.") for n in wrapped)
    x = torch.randn(3, 8, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        assert torch.equal(fresh(x), model(x))
    assert not fresh.training and not any(p.requires_grad for p in fresh.parameters())
    assert fresh.model.visual.mlp.gate_proj.__class__ is torch.nn.Linear              # the vision MLP is untouched


def test_attach_keeps_the_trained_float32_values_on_a_bf16_model(tmp_path):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    model, saved = trained(tmp_path / "adapter", torch.bfloat16)
    adapter = bare(qwen_offline, "QwenOffline")
    adapter.model = toy(torch.bfloat16)
    assert len(adapter.load_lora(tmp_path / "adapter")) == 14
    loaded = peft.get_peft_model_state_dict(adapter.model)
    assert loaded.keys() == saved.keys()
    assert all(v.dtype == torch.float32 and torch.equal(v, saved[k]) for k, v in loaded.items())
    assert adapter.model.lm_head.weight.dtype == torch.bfloat16
    x = torch.randn(3, 8, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)
    with torch.no_grad():
        assert torch.equal(adapter.model(x), model(x))


def test_attach_refuses_adapters_that_do_not_fit(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    trained(tmp_path / "q_only", torch.float32, targets=r"model\.language_model\.layers\.\d+\.self_attn\.q_proj")
    with pytest.raises(RuntimeError, match="LoRA on 2 modules, expected 14"):
        attach_lora(toy(torch.float32), "model.language_model.layers", 7, tmp_path / "q_only")
    _, state = trained(tmp_path / "adapter", torch.float32)
    with pytest.raises(RuntimeError, match="expected 10"):                  # another family's layout (5 per layer)
        attach_lora(toy(torch.float32), "model.language_model.layers", 5, tmp_path / "adapter")
    missing = {k: v for k, v in state.items() if ".layers.1.mlp.up_proj." not in k}
    safetensors_torch.save_file(missing, str(tmp_path / "adapter" / "adapter_model.safetensors"))
    with pytest.raises(RuntimeError, match="does not cover every injected module"):
        attach_lora(toy(torch.float32), "model.language_model.layers", 7, tmp_path / "adapter")
    extra = {**state, "model.language_model.layers.7.self_attn.q_proj.lora_A.weight": torch.zeros(4, 8)}
    safetensors_torch.save_file(extra, str(tmp_path / "adapter" / "adapter_model.safetensors"))
    with pytest.raises(RuntimeError, match="adapter keys not in the model"):
        attach_lora(toy(torch.float32), "model.language_model.layers", 7, tmp_path / "adapter")
