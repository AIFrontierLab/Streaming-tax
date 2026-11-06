"""Resolved settings of one benchmark x model x mode job: what the adapter reads (token budget, pixel budgets, frame
sampling, stream rate, system prompt) and what the run identity hashes.

Layers, merged in this order (ported from the paper's runs; the order decides every frame and budget the paper
used):
1. `defaults` (configs/defaults.yaml)
2. `benchmarks.<b>.settings` (budget, TOMATO's system prompt, ...)
3. `{"video": pairs.<p>.video}`: the pair's one video protocol, applied to every member, so parent and child see
   the same frames by construction
4. `models.<m>.settings` (optional extras; none of the paper models has any), then `extra` (e.g. the plugin's
   Recall delta, which the original runs carried in the model's settings)
5. offline mode only, and only when the benchmark has `offline_frames` (Video-SafetyBench's 16 and MCV-SafetyBench's
   33 uniform frames, the fine-tuning sets' stored frame lists): the video block loses its frame-sampling keys
   (SAMPLING_KEYS) and offline_frames is merged in; pixel budgets and padding stay the pair's
6. `--set` overrides (only keys under generation, image, video, stream, system are accepted)
7. image benchmarks: `video` and `stream` are pinned to the defaults. Image jobs never read them; the original run
   identities hashed the pinned values, so a pair's video protocol or a video override never changes an image job.

merge(): dicts merge recursively; lists and scalars are deep-copied and replace."""
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Iterable, Mapping

import yaml

if TYPE_CHECKING:
    from .schema import Config

SAMPLING_KEYS = {"fps", "nframes", "sampler", "max_frames", "min_frames", "last_seconds"}   # replaced by offline_frames
SECTIONS = ("generation", "image", "video", "stream", "system")    # the only top-level keys settings may have
PINNED_FOR_IMAGES = ("video", "stream")


def merge(*values: dict) -> dict:
    result: dict = {}
    for value in values:
        for key, item in (value or {}).items():
            result[key] = (merge(result[key], item) if isinstance(item, dict) and isinstance(result.get(key), dict)
                           else deepcopy(item))
    return result


def check_layer(value: Mapping | None, where: str) -> dict:
    """A settings layer may only set keys under SECTIONS (anything else is not a generation setting)."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{where}: settings must be a mapping")
    unknown = sorted(set(value) - set(SECTIONS))
    if unknown:
        raise ValueError(f"{where}: {', '.join(unknown)} {'is' if len(unknown) == 1 else 'are'} not generation "
                         f"settings (allowed: {', '.join(SECTIONS)}); GPUs, judge GPUs, threads, ports and shards are "
                         "command-line flags")
    for key in SECTIONS[:-1]:
        if key in value and not isinstance(value[key], Mapping):
            raise ValueError(f"{where}: {key} must be a mapping")
    return dict(value)


def parse_sets(items: Iterable[str] | None) -> dict:
    """--set video.nframes=32 -> {"video": {"nframes": 32}} (values parsed as YAML); only SECTIONS are accepted."""
    result: dict = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects field=value, got {item!r}")
        field, raw = item.split("=", 1)
        *parents, last = field.split(".")
        current = result
        for part in parents:
            current = current.setdefault(part, {})
        current[last] = yaml.safe_load(raw)
    return check_layer(result, "--set")


def compose_settings(defaults: Mapping, benchmark_settings: Mapping | None = None, pair_video: Mapping | None = None,
                     model_settings: Mapping | None = None, mode: str = "offline",
                     offline_frames: Mapping | None = None, overrides: Mapping | None = None,
                     extra: Mapping | None = None, media: str = "video", pin_image: bool = True) -> dict:
    """The layers of the module docstring, from plain dicts."""
    merged = merge(dict(defaults or {}), dict(benchmark_settings or {}),
                   {"video": dict(pair_video)} if pair_video else {}, dict(model_settings or {}), dict(extra or {}))
    override = offline_frames
    if override and mode == "offline":
        merged["video"] = merge({k: v for k, v in merged.get("video", {}).items() if k not in SAMPLING_KEYS}, override)
    merged = merge(merged, dict(overrides or {}))
    if media != "video" and pin_image:
        merged = {**{k: v for k, v in merged.items() if k not in PINNED_FOR_IMAGES},
                  **{k: deepcopy(defaults[k]) for k in PINNED_FOR_IMAGES if k in defaults}}
    return merged


def resolve_settings(config: Config, benchmark: str, model: str, mode: str, overrides: Mapping | None = None,
                     extra: Mapping | None = None, pin_image: bool = True) -> dict:
    """Settings of `model` on `benchmark` in `mode` (offline | stream) under the loaded configs.

    overrides: parsed --set values (parse_sets). extra: an additional settings layer merged after the model's own
    settings (the plugin's Recall delta). pin_image=False keeps a pair's video block on image benchmarks (not used by
    the paper's runs; image jobs never read it)."""
    bench = config.benchmark(benchmark)
    spec = config.model(model)
    mode = str(mode)
    if mode not in ("offline", "stream"):
        raise ValueError(f"unknown mode {mode!r}")
    pair = config.pair_of(model)
    return compose_settings(config.defaults, bench.settings, pair.video if pair else None, spec.settings, mode,
                            bench.offline_frames, check_layer(overrides, "overrides"), check_layer(extra, "extra"),
                            bench.media, pin_image)
