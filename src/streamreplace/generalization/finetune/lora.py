"""LoRA on the LLM decoder only, injected in place with PEFT 0.12 (inject_adapter_in_model): the model's own forward /
generate stay untouched (ported unchanged). Everything family-specific is one row of FAMILIES:
  targets    regex (re.fullmatch on module names) of the decoder's linear layers. Suffixes alone are not enough: Qwen2.5-VL's
             vision MLP has gate/up/down_proj and LLaVA's SigLIP tower q/k/v_proj
  layers     the decoder-layer list; matched modules must be exactly len(layers) x per_layer
  train      the submodule put in train mode (gradient checkpointing only runs in train mode), `frozen` its parts kept in
             eval mode (InternViT has drop_path 0.1; LLaVA's Qwen2Model also holds the vision tower and projector)
  checkpoint "hf": gradient_checkpointing_enable(use_reentrant=False) on the model; "lm": on `train` + input grads
             (InternLM2 hard-codes reentrant checkpointing, which needs the embeddings to require grad, as LiveStar's own
             wrap_llm_lora does)
LoRA weights are kept in fp32 (injection creates them in the base dtype), dropout 0 (PEFT's dropout modules are in
train mode after injection even in an eval-mode model; 0 also makes resumed and reloaded runs bit-identical).
Expected LoRA sizes at r 16: Qwen2.5-VL-3B 29,933,568 parameters, Qwen3-VL-8B 43,646,976, LLaVA-OneVision-7B
40,370,176, LiveStar 37,748,736, MOSS-VL 58,195,968.

ADAPTERS maps the offline model adapters (models/registry.py) to their family for training. Evaluating a fine-tuned id
is the offline adapter's own job: its load_lora(path) attaches the saved adapter unmerged on the freshly loaded weights
with models/lora.attach_lora (the same PEFT injection, and the same placement check, models/lora.check_lora, that
inject() runs here), so it computes exactly the trained function."""
from __future__ import annotations

import json
from pathlib import Path

from ...models.lora import check_lora

FAMILIES = {
    "qwen_vl": {"targets": r"model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)",
                "layers": "model.language_model.layers", "per_layer": 7,
                "train": "model.language_model", "frozen": [], "checkpoint": "hf"},
    "llava_ov": {"targets": r"model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)",
                 "layers": "model.layers", "per_layer": 7,
                 "train": "model", "frozen": ["model.vision_tower", "model.mm_projector"], "checkpoint": "hf"},
    "livestar": {"targets": r"language_model\.model\.layers\.\d+\.(attention\.(wqkv|wo)|feed_forward\.w[123])",
                 "layers": "language_model.model.layers", "per_layer": 5,
                 "train": "language_model", "frozen": [], "checkpoint": "lm"},
    "moss": {"targets": r"model\.language_model\.layers\.\d+\.((self|cross)_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)",
             "layers": "model.language_model.layers", "per_layer": 7,
             "train": "model.language_model", "frozen": [], "checkpoint": "hf"},
}
FAMILIES["aura"] = FAMILIES["qwen_vl"]          # AURA offline is the stock Qwen3-VL class

# offline adapter (models/registry.py) -> LoRA family; Qwen2.5-VL and Qwen3-VL (AURA) share one module layout
ADAPTERS = {"qwen_offline": "qwen_vl", "llava_offline": "llava_ov", "internvl_offline": "livestar",
            "moss_offline": "moss"}


def family_of(adapter: str) -> str:
    try:
        return ADAPTERS[adapter]
    except KeyError:
        raise ValueError(f"adapter {adapter!r} has no LoRA family (fine-tunable: {', '.join(ADAPTERS)})") from None


def inject(model, family: str, r: int, alpha: int, seed: int) -> list[str]:
    """Fresh LoRA (A: kaiming-uniform from `seed`, B = 0) on the decoder; returns the wrapped module names after
    checking count, placement and that only the fp32 LoRA weights train (models/lora.check_lora, which casts them to
    fp32). Seeding right before injection gives a pair's parent and child (same architecture) the same initial A."""
    import torch
    from peft import LoraConfig, inject_adapter_in_model
    torch.manual_seed(seed)
    config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none", target_modules=FAMILIES[family]["targets"])
    inject_adapter_in_model(config, model)
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name
    model.ft_lora_config = config
    row = FAMILIES[family]
    return check_lora(model, row["layers"], row["per_layer"], label=family)


def lora_parameters(model) -> dict:
    return {n: p for n, p in model.named_parameters() if "lora_" in n}


def save(model, directory: Path) -> str:
    """adapter/{adapter_model.safetensors, adapter_config.json}; returns the weights' sha256."""
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    from ...core.io import file_digest
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(model).items()}
    tmp = directory / ".adapter_model.safetensors.tmp"
    save_file(state, str(tmp))
    tmp.replace(directory / "adapter_model.safetensors")
    model.ft_lora_config.save_pretrained(str(directory))
    return file_digest(directory / "adapter_model.safetensors")


def train_mode(model, family: str):
    """Train mode for the decoder only, gradient checkpointing on; asserts no active dropout anywhere."""
    import torch
    row = FAMILIES[family]
    model.eval()
    model.get_submodule(row["train"]).train()
    for name in row["frozen"]:
        model.get_submodule(name).eval()
    if row["checkpoint"] == "hf":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        lm = model.get_submodule(row["train"])
        lm.gradient_checkpointing_enable()
        lm.enable_input_require_grads()
    active = [n for n, m in model.named_modules() if m.training and (
        (isinstance(m, torch.nn.Dropout) and m.p > 0) or float(getattr(m, "drop_prob", 0) or 0) > 0)]
    if active:
        raise RuntimeError(f"{family}: stochastic modules active in training: {active[:3]}")


def describe(model) -> dict:
    """For DONE.json / logs."""
    params = lora_parameters(model)
    return {"lora_modules": len({n.rsplit(".lora_", 1)[0] for n in params}),
            "lora_parameters": sum(p.numel() for p in params.values()),
            "config": json.loads(json.dumps(model.ft_lora_config.to_dict(), default=str)) if hasattr(model, "ft_lora_config") else None}
