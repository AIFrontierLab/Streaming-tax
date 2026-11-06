"""Adapter name -> "module:Class". configs/models.yaml names an adapter per model and mode; the class is imported
only when a worker loads it inside the model's own Python environment (each env can import only its own stack).

Offline adapters (conditions P and C):
- qwen_offline      Qwen2.5-VL / Qwen3-VL family: Qwen2.5-VL-3B, ThinkStream-3B, MMDuet2, Qwen3-VL-8B, AURA
- llava_offline     LLaVA-OneVision-7B and MMDuet (its LoRA kept unmerged)
- internvl_offline  LiveStar-InternVideo-8B and LiveStar-8B
- moss_offline      MOSS-VL-Instruct and MOSS-VL-Realtime
Stream adapters (condition S; plugin arms):
- mmduet_stream, thinkstream_stream, mmduet2_stream, aura_stream (vLLM server), aura_stream_hf (AURA's stream rebuilt
  in transformers: the plugin's engine and the S_HF control), livestar_stream, mossrt_stream
Tests: mock (offline and stream)."""
from __future__ import annotations

import importlib

_PACKAGE = "streamreplace.models"

ADAPTERS: dict[str, str] = {
    "qwen_offline": f"{_PACKAGE}.qwen_offline:QwenOffline",
    "llava_offline": f"{_PACKAGE}.llava_offline:LlavaOffline",
    "internvl_offline": f"{_PACKAGE}.internvl_offline:InternVLOffline",
    "moss_offline": f"{_PACKAGE}.moss_offline:MossOffline",
    "mmduet_stream": f"{_PACKAGE}.mmduet_stream:MMDuetStream",
    "thinkstream_stream": f"{_PACKAGE}.thinkstream_stream:ThinkStreamStream",
    "mmduet2_stream": f"{_PACKAGE}.mmduet2_stream:MMDuet2Stream",
    "aura_stream": f"{_PACKAGE}.aura_stream:AuraStream",
    "aura_stream_hf": f"{_PACKAGE}.aura_stream_hf:AuraStreamHF",
    "livestar_stream": f"{_PACKAGE}.livestar_stream:LiveStarStream",
    "mossrt_stream": f"{_PACKAGE}.mossrt_stream:MossRTStream",
    "mock": f"{_PACKAGE}.mock:MockModel",
}

# The modes each adapter can serve (configs are validated against this).
MODES: dict[str, frozenset[str]] = {
    **{name: frozenset({"offline"}) for name in ("qwen_offline", "llava_offline", "internvl_offline", "moss_offline")},
    **{name: frozenset({"stream"}) for name in ("mmduet_stream", "thinkstream_stream", "mmduet2_stream", "aura_stream",
                                                "aura_stream_hf", "livestar_stream", "mossrt_stream")},
    "mock": frozenset({"offline", "stream"}),
}


def names() -> list[str]:
    return list(ADAPTERS)


def target(name: str) -> tuple[str, str]:
    """(module path, class name) of an adapter."""
    try:
        module, cls = ADAPTERS[name].split(":")
    except KeyError:
        raise ValueError(f"Unknown model adapter: {name}") from None
    return module, cls


def modes(name: str) -> frozenset[str]:
    target(name)
    return MODES[name]


def load_class(name: str) -> type:
    module, cls = target(name)
    return getattr(importlib.import_module(module), cls)


def create(name: str, spec, settings: dict, upstream=None, **keywords):
    """Instantiates adapter `name` (inside the model's env): Adapter(spec, settings, upstream, **keywords), e.g.
    load=False for a CPU-only adapter without weights (models/base.py)."""
    from .upstream import Upstream
    return load_class(name)(spec, settings, upstream if upstream is not None else Upstream(), **keywords)


def available(name: str) -> bool:
    """True when the adapter's module exists (without importing it or its dependencies)."""
    from importlib.util import find_spec
    try:
        return find_spec(target(name)[0]) is not None
    except ModuleNotFoundError:
        return False
