"""LoRA adapters on the offline adapters' models, for the fine-tuned model ids of the generalization experiments
(never used by the paper's P/C/S runs).

A fine-tuned model id is its base checkpoint plus the LoRA adapter the fine-tuning experiment (generalization/finetune)
trained and saved in a folder: adapter_config.json + adapter_model.safetensors. Every offline adapter's
load_lora(path) attaches it right after loading with attach_lora(model, *LORA_LAYERS, path), where the adapter's
LORA_LAYERS names the decoder-layer list and how many of each decoder layer's linear layers carry a LoRA
(models/base.py). check_lora() is the placement check that both sides run: the evaluation side here, the training
side when it injects a fresh LoRA (generalization/finetune/lora.py).

Ported unchanged from the paper's fine-tuning runs: PEFT's inject_adapter_in_model with the adapter's own
configuration, LoRA left unmerged, LoRA weights in float32. torch, PEFT and safetensors are imported inside the
functions, since only the model environments have them."""
from __future__ import annotations

from pathlib import Path


def lora_modules(model) -> list[str]:
    """Names of the modules that carry a LoRA (an injected lora_A matrix)."""
    return sorted({n.rsplit(".lora_A", 1)[0] for n, _ in model.named_parameters() if ".lora_A." in n})


def check_lora(model, layers: str, per_layer: int, label: str = "") -> list[str]:
    """The LoRA-wrapped module names, after checking count and placement (exactly `per_layer` linear layers in each
    decoder layer of the list at the dotted path `layers`) and that only the float32 LoRA weights are trainable (the
    LoRA weights are cast to float32 here). `label` prefixes the error messages (the training side passes the LoRA
    family's name)."""
    import torch
    prefix = f"{label}: " if label else ""
    wrapped = lora_modules(model)
    expected = len(model.get_submodule(layers)) * per_layer
    if len(wrapped) != expected or any(not n.startswith(layers + ".") for n in wrapped):
        raise RuntimeError(f"{prefix}LoRA on {len(wrapped)} modules, expected {expected} under {layers}")
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.data = param.data.to(torch.float32)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if any("lora_" not in n for n in trainable):
        raise RuntimeError(f"{prefix}non-LoRA parameters are trainable: "
                           f"{[n for n in trainable if 'lora_' not in n][:3]}")
    return wrapped


def attach_lora(model, layers: str, per_layer: int, directory) -> list[str]:
    """Loads a saved LoRA adapter (`directory`: adapter_config.json + adapter_model.safetensors, as the fine-tuning
    experiment writes them) onto a freshly loaded model for evaluation. Ported unchanged from the paper's
    fine-tuning runs: PEFT's inject_adapter_in_model with the adapter's own configuration, then its weights; the LoRA
    stays unmerged, so the model computes exactly the trained function. The LoRA weights are cast to float32 before
    loading (injection creates them in the base dtype, and copying the trained values into bf16 would round them).
    Refuses saved keys the model does not have, a file that does not cover every injected module, and LoRA placement
    other than `per_layer` linear layers per decoder layer under `layers` (a dotted submodule path of `model`).
    Afterwards nothing requires gradients and the model is in eval mode. Returns the wrapped module names."""
    import torch
    from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    config = LoraConfig.from_pretrained(str(directory))
    inject_adapter_in_model(config, model)
    for name, param in model.named_parameters():      # fp32 before loading: copying into bf16 would round the weights
        if "lora_" in name:
            param.data = param.data.to(torch.float32)
    state = load_file(str(Path(directory) / "adapter_model.safetensors"))
    result = set_peft_model_state_dict(model, state)
    if getattr(result, "unexpected_keys", None):
        raise RuntimeError(f"adapter keys not in the model: {result.unexpected_keys[:3]}")
    if len(state) != 2 * len(lora_modules(model)):            # one A and one B matrix per wrapped module
        raise RuntimeError("adapter does not cover every injected module")
    wrapped = check_lora(model, layers, per_layer)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return wrapped
