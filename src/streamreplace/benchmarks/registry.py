"""Benchmark name -> module. Modules are imported on first use only, so listing the registry costs nothing.

Paper benchmarks (configs/benchmarks.yaml holds their settings and suites):
- image (P/C): DynaMath, MathVista, MMBench, MME, MMMU, MMStar, ScienceQA, VisuLogic (supplement);
- video (P/C/S): Video-MME, TOMATO, TimeBlind, VideoReasonBench (supplement);
- safety: HarmBench (P/C; text and image behaviours), Video-SafetyBench and MCV-SafetyBench (P/C/S);
- trust: VMDT fairness and privacy (P/C/S);
- fine-tuning experiment: LingoQA, EyePCR, OpenUAV-QA.
Tests register extra benchmarks (e.g. a synthetic demo) with register()."""
from __future__ import annotations

import importlib
from types import ModuleType

from .base import BenchmarkModule

_PACKAGE = "streamreplace.benchmarks"

BENCHMARKS: dict[str, str] = {
    **{name: f"{_PACKAGE}.image.{name}" for name in
       ("dynamath", "mathvista", "mmbench", "mme", "mmmu", "mmstar", "scienceqa", "visulogic")},
    **{name: f"{_PACKAGE}.video.{name}" for name in ("videomme", "tomato", "timeblind", "videoreasonbench")},
    **{name: f"{_PACKAGE}.safety.{name}" for name in ("harmbench", "video_safetybench", "mcv_safetybench")},
    **{name: f"{_PACKAGE}.trust.{name}" for name in ("vmdt_fairness", "vmdt_privacy")},
    **{name: f"{_PACKAGE}.finetune.{name}" for name in ("lingoqa", "eyepcr", "openuav_qa")},
}

_extra: dict[str, str] = {}
_loaded: dict[str, BenchmarkModule] = {}


def names() -> list[str]:
    return [*BENCHMARKS, *(n for n in _extra if n not in BENCHMARKS)]


def module_path(name: str) -> str:
    try:
        return _extra.get(name) or BENCHMARKS[name]
    except KeyError:
        raise ValueError(f"Unknown benchmark: {name}") from None


def register(name: str, module: str) -> None:
    """Adds (or replaces) a benchmark at runtime, e.g. the synthetic demo benchmark of the tests."""
    _extra[name] = module
    _loaded.pop(name, None)


def unregister(name: str) -> None:
    _extra.pop(name, None)
    _loaded.pop(name, None)


def load_module(name: str) -> ModuleType:
    return importlib.import_module(module_path(name))


def get(name: str) -> BenchmarkModule:
    """The benchmark module with the shared defaults filled in (cached)."""
    if name not in _loaded:
        _loaded[name] = BenchmarkModule(name, load_module(name))
    return _loaded[name]


def available(name: str) -> bool:
    """True when the benchmark's module exists (without importing it)."""
    from importlib.util import find_spec
    try:
        return find_spec(module_path(name)) is not None
    except ModuleNotFoundError:
        return False
