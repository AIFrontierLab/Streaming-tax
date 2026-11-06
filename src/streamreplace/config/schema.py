"""Typed, validated views of configs/models.yaml, pairs.yaml, benchmarks.yaml and defaults.yaml.

Unknown keys are errors (a typo must not silently fall back to a default that changes generations). Cross-file
checks (pair members exist and have the right roles, suites name known benchmarks, adapters exist and serve the
mode they are configured for) are done by config.load.validate()."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..core.protocol import VARIANTS, Mode, Role
from .settings import SAMPLING_KEYS, check_layer

MEDIA = ("image", "video")


def _keys(value: Any, allowed: tuple[str, ...], where: str, required: tuple[str, ...] = ()) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where}: expected a mapping, got {type(value).__name__}")
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ValueError(f"{where}: unknown keys {unknown} (allowed: {', '.join(allowed)})")
    missing = [k for k in required if k not in value]
    if missing:
        raise ValueError(f"{where}: missing {', '.join(missing)}")
    return dict(value)


@dataclass(frozen=True)
class CheckpointRef:
    """A Hugging Face checkpoint pinned to a revision, written "<org>/<name>@<revision>" in the configs."""
    repo: str
    revision: str | None = None

    @classmethod
    def parse(cls, value: Any, where: str = "checkpoint") -> CheckpointRef:
        if isinstance(value, CheckpointRef):
            return value
        if isinstance(value, Mapping):
            value = _keys(value, ("repo", "revision"), where, ("repo",))
            return cls(str(value["repo"]), str(value["revision"]) if value.get("revision") else None)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{where}: expected '<org>/<name>@<revision>', got {value!r}")
        repo, _, revision = value.strip().partition("@")
        if not repo or "/" not in repo:
            raise ValueError(f"{where}: {value!r} is not a '<org>/<name>[@<revision>]' repository reference")
        return cls(repo, revision or None)

    def __str__(self) -> str:
        return f"{self.repo}@{self.revision}" if self.revision else self.repo

    def as_dict(self) -> dict:
        return {"repo": self.repo, "revision": self.revision}


@dataclass(frozen=True)
class AdapterSpec:
    """How a model runs in one mode: the adapter (models/registry.py), its Python env, and whether the env needs the
    FFmpeg 6 shared libraries (torchcodec / qwen-vl-utils video decoding)."""
    adapter: str
    env: str
    ffmpeg: bool = False

    @classmethod
    def parse(cls, value: Any, where: str) -> AdapterSpec:
        value = _keys(value, ("adapter", "env", "ffmpeg"), where, ("adapter", "env"))
        return cls(str(value["adapter"]), str(value["env"]), bool(value.get("ffmpeg", False)))


@dataclass(frozen=True)
class DecodingSpec:
    """Offline decoding beyond greedy(): repetition_penalty, which must equal the checkpoint's generation_config
    (adapters check it). None = the adapter sets it (stream adapters use their upstream decoding)."""
    repetition_penalty: float | None = None

    @classmethod
    def parse(cls, value: Any, where: str) -> DecodingSpec:
        value = _keys(value or {}, ("repetition_penalty",), where)
        penalty = value.get("repetition_penalty")
        return cls(None if penalty is None else float(penalty))


MODEL_KEYS = ("role", "checkpoint", "base", "vision", "offline", "stream", "decoding", "settings", "options")


@dataclass(frozen=True)
class ModelSpec:
    id: str
    role: Role
    checkpoint: CheckpointRef
    offline: AdapterSpec | None = None
    stream: AdapterSpec | None = None
    base: CheckpointRef | None = None            # MMDuet: the LoRA's base model
    vision: CheckpointRef | None = None          # LLaVA-OneVision / MMDuet: the SigLIP vision tower
    decoding: DecodingSpec = DecodingSpec()
    settings: dict = field(default_factory=dict)   # optional extra settings layer (none for the paper models)
    options: dict = field(default_factory=dict)    # adapter-specific options (part of the run identity)
    pair: str | None = None                        # filled in from pairs.yaml

    @classmethod
    def parse(cls, model_id: str, value: Any) -> ModelSpec:
        where = f"models.{model_id}"
        if not model_id or "." in model_id or ("@" in model_id and model_id.rpartition("@")[2] in VARIANTS):
            raise ValueError(f"{where}: invalid model id (no '.', no '@<plugin variant>' suffix)")
        value = _keys(value, MODEL_KEYS, where, ("role", "checkpoint"))
        if value.get("offline") is None and value.get("stream") is None:
            raise ValueError(f"{where}: needs an offline and/or a stream block")
        options = value.get("options") or {}
        if not isinstance(options, Mapping):
            raise ValueError(f"{where}.options: expected a mapping")
        return cls(
            id=model_id,
            role=Role(value["role"]),
            checkpoint=CheckpointRef.parse(value["checkpoint"], f"{where}.checkpoint"),
            offline=AdapterSpec.parse(value["offline"], f"{where}.offline") if value.get("offline") else None,
            stream=AdapterSpec.parse(value["stream"], f"{where}.stream") if value.get("stream") else None,
            base=CheckpointRef.parse(value["base"], f"{where}.base") if value.get("base") else None,
            vision=CheckpointRef.parse(value["vision"], f"{where}.vision") if value.get("vision") else None,
            decoding=DecodingSpec.parse(value.get("decoding"), f"{where}.decoding"),
            settings=check_layer(deepcopy(value.get("settings")), f"{where}.settings"),
            options=deepcopy(dict(options)),
        )

    @property
    def modes(self) -> tuple[str, ...]:
        return tuple(m for m, block in (("offline", self.offline), ("stream", self.stream)) if block is not None)

    def adapter(self, mode: Mode | str) -> AdapterSpec:
        block = self.offline if Mode(mode) is Mode.OFFLINE else self.stream
        if block is None:
            raise ValueError(f"{self.id} has no {Mode(mode).value} adapter")
        return block

    def checkpoints(self) -> dict[str, CheckpointRef]:
        """Every checkpoint the model loads: checkpoint (+ base, vision)."""
        refs = {"checkpoint": self.checkpoint, "base": self.base, "vision": self.vision}
        return {k: v for k, v in refs.items() if v is not None}

    def identity_block(self, mode: Mode | str, adapter_protocol: int | None, plugin: dict | None = None) -> dict:
        """The model part of the run identity (core.identity.run_payload)."""
        block: dict[str, Any] = {k: v.as_dict() for k, v in self.checkpoints().items()}
        block.update(adapter=self.adapter(mode).adapter, adapter_protocol=adapter_protocol,
                     decoding={"repetition_penalty": self.decoding.repetition_penalty}, plugin=plugin)
        if self.options:
            block["options"] = deepcopy(self.options)
        return block


@dataclass(frozen=True)
class PairSpec:
    id: str
    parent: str
    children: tuple[str, ...]
    video: dict                                     # the pair's video protocol, applied to every member

    @classmethod
    def parse(cls, pair_id: str, value: Any) -> PairSpec:
        where = f"pairs.{pair_id}"
        value = _keys(value, ("parent", "children", "video"), where, ("parent", "children"))
        children = value["children"]
        if isinstance(children, str) or not children:
            raise ValueError(f"{where}.children: expected a non-empty list of model ids")
        video = value.get("video") or {}
        if not isinstance(video, Mapping):
            raise ValueError(f"{where}.video: expected a mapping")
        return cls(pair_id, str(value["parent"]), tuple(str(c) for c in children), deepcopy(dict(video)))

    @property
    def members(self) -> tuple[str, ...]:
        return (self.parent, *self.children)


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    media: str                                      # image | video (must equal the module's MEDIA)
    suite: str | None = None                        # grouping shown in tables (image, video, safety, trust, ...)
    settings: dict = field(default_factory=dict)    # settings layer 2
    offline_frames: dict | None = None              # offline-only frame sampling override (settings layer 5)

    @classmethod
    def parse(cls, name: str, value: Any) -> BenchmarkSpec:
        where = f"benchmarks.{name}"
        value = _keys(value, ("media", "suite", "settings", "offline_frames"), where, ("media",))
        if value["media"] not in MEDIA:
            raise ValueError(f"{where}.media: expected one of {MEDIA}, got {value['media']!r}")
        frames = value.get("offline_frames")
        if frames is not None:
            if not isinstance(frames, Mapping) or not frames:
                raise ValueError(f"{where}.offline_frames: expected a non-empty mapping")
            unknown = sorted(set(frames) - SAMPLING_KEYS)
            if unknown:
                raise ValueError(f"{where}.offline_frames: only frame-sampling keys ({', '.join(sorted(SAMPLING_KEYS))})"
                                 f" may be set; got {unknown}")
        return cls(name, value["media"], value.get("suite"), check_layer(deepcopy(value.get("settings")),
                                                                         f"{where}.settings"),
                   deepcopy(dict(frames)) if frames is not None else None)


@dataclass
class Config:
    """All loaded configuration. `raw` holds the merged core keys; `sections` the other configs/<name>.yaml files
    (judges, analysis, plugin, ...), each under its file name."""
    defaults: dict
    run: dict
    models: dict[str, ModelSpec]
    pairs: dict[str, PairSpec]
    benchmarks: dict[str, BenchmarkSpec]
    suites: dict[str, tuple[str, ...]]
    raw: dict = field(default_factory=dict)
    sections: dict[str, dict] = field(default_factory=dict)
    config_dir: Path | None = None
    sources: list[Path] = field(default_factory=list)

    def model(self, model_id: str) -> ModelSpec:
        try:
            return self.models[model_id]
        except KeyError:
            raise ValueError(f"Unknown model: {model_id}") from None

    def pair(self, pair_id: str) -> PairSpec:
        try:
            return self.pairs[pair_id]
        except KeyError:
            raise ValueError(f"Unknown pair: {pair_id}") from None

    def pair_of(self, model_id: str) -> PairSpec | None:
        pair = self.model(model_id).pair
        return self.pairs[pair] if pair else None

    def benchmark(self, name: str) -> BenchmarkSpec:
        try:
            return self.benchmarks[name]
        except KeyError:
            raise ValueError(f"Unknown benchmark: {name}") from None

    def section(self, name: str) -> dict:
        """A copy of configs/<name>.yaml (merged with local/extra overrides); {} when absent."""
        return deepcopy(self.sections.get(name, {}))

    @property
    def paths_block(self) -> dict | None:
        return deepcopy(self.raw.get("paths"))

    def expand_benchmarks(self, names: list[str] | str) -> list[str]:
        """Benchmark names and suite names ("all" = every benchmark), in order, without duplicates."""
        names = [n for n in (names.split(",") if isinstance(names, str) else names) if n]
        out: list[str] = []
        for name in names:
            if name == "all":
                out += list(self.benchmarks)
            elif name in self.suites:
                out += list(self.suites[name])
            else:
                out.append(self.benchmark(name).name)
        return list(dict.fromkeys(out))

    def expand_pairs(self, names: list[str] | str) -> list[str]:
        names = [n for n in (names.split(",") if isinstance(names, str) else names) if n]
        out: list[str] = []
        for name in names:
            out += list(self.pairs) if name == "all" else [self.pair(name).id]
        return list(dict.fromkeys(out))

    def settings(self, benchmark: str, model: str, mode: Mode | str, overrides: Mapping | None = None,
                 extra: Mapping | None = None, pin_image: bool = True) -> dict:
        """Resolved settings of one job (config.settings.resolve_settings)."""
        from .settings import resolve_settings
        return resolve_settings(self, benchmark, model, str(mode), overrides, extra, pin_image)
