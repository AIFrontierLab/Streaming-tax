"""Planning `streamreplace run`: the generation jobs a request asks for, with their resolved settings, run identities
and output files. Nothing here loads a model or writes a file.

A job is one arm (a model in one mode, optionally a plugin variant of a child's stream) on one benchmark. It writes
`<outputs>/<benchmark>/<job>.jsonl` (generation rows, append-only) and `<job>.meta.json` (identity and run segments),
where <job> is `<model>.<mode>`, or `<model>@<variant>.stream` for a plugin arm (core/protocol.py).

Conditions are always relative to a pair (configs/pairs.yaml):
    P   the parent in the offline pipeline           Arm(parent, offline)
    C   the child in the same offline pipeline       Arm(child, offline)
    S   the child through its native stream          Arm(child, stream), video benchmarks only
Parents never stream and image benchmarks are never streamed, so HarmBench (text and image behaviours) is P/C only.
All five pairs under P/C/S on the paper's two suites make the paper's 235 jobs. Plugin variants (configs/plugin.yaml
arms: sfix, sfix_prs, record, inert, hf) add Arm(child, stream, variant) on video benchmarks, for every selected
child (--variants) or one arm at a time by its name (--models aura_8b@sfix); the plugin package supplies what a
variant changes: the Recall settings delta, the engine adapter (AURA's plugin arms run on its Hugging Face rebuild)
and the identity block, and the worker attaches the arm's hooks. Models outside the pairs, such as fine-tuned
checkpoints declared in an extra config file (`--config FILE`), are selected with --models and run the conditions
of their role (a parent: P; a child: C, and S when it has a stream adapter).

Settings follow config/settings.py (the merge order of the paper's runs). A plugin arm's Recall delta is the extra
layer merged right after the model's own settings, where the original plugin runs carried it, so --set still wins.

Run identity (core/identity.py) covers the arm, the model block (checkpoints, the adapter actually run and its
PROTOCOL_VERSION, decoding, options, plugin block), the benchmark, the resolved settings, --limit, and the outputs
folder's subset and prompt overrides. A fine-tuned model names its LoRA adapter folder in `options.lora` (absolute,
or relative to the outputs folder); the identity holds the sha256 of the adapter's weights file instead of the path,
so a retrained adapter never reuses old outputs, and the worker hands the absolute path to the adapter.

Adapter constants (PROTOCOL_VERSION, LOAD_KEYS) are read from the adapter module's source, because adapter modules
import torch and upstream code that only the model's own environment has; when the source does not state them as
literals, the model environment's interpreter is asked. The worker checks them against the class it loads. A worker
process loads its model once, with the settings of its first job, so the dispatcher groups jobs of one arm into a
process only when the values of the adapter's LOAD_KEYS (dotted settings keys read at load time) agree."""
from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING, Any, Iterable

from ..core.identity import run_identity, run_payload
from ..core.io import file_digest
from ..core.protocol import VARIANTS, Arm, Mode, Role
from ..data.prompts import read_overrides
from ..data.samples import read_subset

if TYPE_CHECKING:
    from ..config.schema import Config, ModelSpec
    from ..paths import Paths

PROTOCOLS = ("P", "C", "S")
LORA_WEIGHTS = "adapter_model.safetensors"


@dataclass(frozen=True)
class AdapterConstants:
    protocol: int
    load_keys: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"PROTOCOL_VERSION": self.protocol, "LOAD_KEYS": list(self.load_keys)}


@dataclass(frozen=True)
class PluginArm:
    """What a plugin variant changes about a child's stream job."""
    settings: dict            # settings layer merged after the model's own settings (Recall)
    identity: dict            # the `plugin` block of the run identity
    adapter: str              # the stream adapter the arm runs on
    worker: dict              # what the worker needs to rebuild the arm's hooks


@dataclass
class Job:
    arm: Arm
    dataset: str
    outputs: Path
    media: str
    spec: ModelSpec
    adapter: str
    env: str
    ffmpeg: bool
    settings: dict
    payload: dict                       # the identity payload (core.identity.run_payload)
    constants: AdapterConstants | None  # None: the adapter module is missing or its constants are unknown
    deterministic: bool = False
    lora: Path | None = None
    plugin: dict | None = None
    subset: list[str] | None = None
    limit: int | None = None
    edits: dict | None = None           # the folder's prompt overrides for this benchmark

    @property
    def job_name(self) -> str:
        return self.arm.job_name()

    @property
    def name(self) -> str:
        return f"{self.dataset}/{self.job_name}"

    @property
    def mode(self) -> str:
        return self.arm.mode.value

    @property
    def folder(self) -> Path:
        return self.outputs / self.dataset

    @property
    def out(self) -> Path:
        return self.folder / f"{self.job_name}.jsonl"

    @property
    def meta(self) -> Path:
        return self.folder / f"{self.job_name}.meta.json"

    @property
    def scores(self) -> Path:
        return self.folder / f"{self.job_name}.scores.jsonl"

    @property
    def identity(self) -> str | None:
        return run_identity(self.payload) if self.constants is not None else None

    @property
    def load_values(self) -> tuple | None:
        """Values of the adapter's load-time settings keys for this job (None when the keys are unknown)."""
        if self.constants is None:
            return None
        return tuple(json.dumps(dotted(self.settings, key), sort_keys=True) for key in self.constants.load_keys)


# ---------- output-file locks (shared with the scorer) ----------

def lock_path(out: Path) -> Path:
    """The job lock of a generation file: workers hold it shared while appending; the scorer takes it exclusively
    without blocking and skips files a worker is still writing."""
    out = Path(out)
    return out.parent / f".{out.stem}.lock"


def shard_lock_path(out: Path, shard: int) -> Path:
    """Held exclusively by the worker of one shard, so a second worker never runs the same shard."""
    out = Path(out)
    return out.parent / f".{out.stem}.shard{shard}.lock"


def meta_lock_path(meta: Path) -> Path:
    """Serialises the run-segment updates of meta.json by concurrent shards."""
    meta = Path(meta)
    return meta.with_name(f".{meta.name}.lock")


# ---------- helpers ----------

def dotted(settings: dict, key: str) -> Any:
    """settings["a"]["b"] for key "a.b"; None when absent."""
    value: Any = settings
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _names(values: Iterable[str] | str | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = values.split(",")
    return [v.strip() for v in values if v and v.strip()]


def select(config: Config, pairs: Iterable[str] | str = (), models: Iterable[str] | str = (),
           protocols: Iterable[str] = PROTOCOLS, variants: Iterable[str] = ()) -> list[tuple[str, tuple, tuple]]:
    """(model, conditions, plugin variants) selections: the members of the chosen pairs ("all" = every pair; parent
    first), then the individually chosen models, all under `protocols` and `variants`. A models entry
    `<child>@<variant>` (a plugin arm's name) selects that one plugin arm only."""
    protocols, variants = tuple(protocols), tuple(variants)
    chosen: list[tuple[str, tuple, tuple]] = []
    for pair in config.expand_pairs(_names(pairs)):
        chosen += [(member, protocols, variants) for member in config.pair(pair).members]
    for entry in _names(models):
        base, at, variant = entry.rpartition("@")
        if entry not in config.models and at and variant in VARIANTS:
            chosen.append((config.model(base).id, (), (variant,)))
        else:
            chosen.append((config.model(entry).id, protocols, variants))
    return list(dict.fromkeys(chosen))


def arms_for(spec: ModelSpec, media: str, protocols: Iterable[str], variants: Iterable[str] = ()) -> list[Arm]:
    """The arms of one model on a benchmark of the given media (see the module docstring)."""
    protocols, arms = set(protocols), []
    offline = (spec.role is Role.PARENT and "P" in protocols) or (spec.role is Role.CHILD and "C" in protocols)
    if spec.offline is not None and offline:
        arms.append(Arm(spec.id, Mode.OFFLINE))
    if media == "video" and spec.role is Role.CHILD and spec.stream is not None:
        if "S" in protocols:
            arms.append(Arm(spec.id, Mode.STREAM))
        arms += [Arm(spec.id, Mode.STREAM, variant) for variant in variants]
    return arms


def spec_dict(spec: ModelSpec, lora: Path | None = None) -> dict:
    """The model's config entry (models.yaml layout), which config.schema.ModelSpec.parse() reads back; the worker
    rebuilds the spec from it. A LoRA path is written absolute."""
    out: dict[str, Any] = {"role": spec.role.value, "checkpoint": str(spec.checkpoint)}
    for name in ("base", "vision"):
        ref = getattr(spec, name)
        if ref is not None:
            out[name] = str(ref)
    for mode in ("offline", "stream"):
        block = getattr(spec, mode)
        if block is not None:
            out[mode] = {"adapter": block.adapter, "env": block.env, "ffmpeg": block.ffmpeg}
    out["decoding"] = {"repetition_penalty": spec.decoding.repetition_penalty}
    if spec.settings:
        out["settings"] = deepcopy(spec.settings)
    options = deepcopy(dict(spec.options))
    if lora is not None:
        options["lora"] = str(lora)
    if options:
        out["options"] = options
    return out


def lora_dir(spec: ModelSpec, outputs: Path) -> Path | None:
    """The absolute LoRA adapter folder of a fine-tuned model id (options.lora), or None."""
    value = spec.options.get("lora")
    if not value:
        return None
    path = Path(str(value)).expanduser()
    path = path if path.is_absolute() else (Path(outputs) / path)
    path = Path(os.path.abspath(path))
    if not (path / LORA_WEIGHTS).is_file():
        raise ValueError(f"{spec.id}: LoRA adapter {path} has no {LORA_WEIGHTS} (train it first, or fix options.lora; "
                         "a relative path is read from the outputs folder)")
    return path


# ---------- plugin arms ----------

def plugin_arm(config: Config, model: str, variant: str, default_adapter: str) -> PluginArm:
    """What plugin variant `variant` changes about `model`'s stream, from the plugin package and configs/plugin.yaml
    (the `plugin` section of the loaded configuration)."""
    try:
        from ..plugin.hooks import make_hooks, stream_adapter
        from ..plugin.recipe import load_plugin_config
    except ImportError as exc:
        raise ValueError(f"plugin arm {model}@{variant} needs the plugin package: {exc}") from exc
    section = config.section("plugin")
    plugin_config = load_plugin_config(section=section)
    hooks = make_hooks(model, variant, plugin_config)
    return PluginArm(settings=plugin_config.recall_delta(model, variant), identity=hooks.identity(),
                     adapter=stream_adapter(model, default_adapter, plugin_config),
                     worker={"child": model, "variant": variant, "section": section})


# ---------- adapter constants ----------

def _module_tree(module: str) -> ast.Module | None:
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin or not str(spec.origin).endswith(".py"):
        return None
    path = Path(spec.origin)
    return ast.parse(path.read_text(encoding="utf-8"), str(path))


def _absolute_module(module: str, target: str | None, level: int) -> str:
    if level == 0:
        return target or ""
    base = module.split(".")[:-level]
    return ".".join(base + ([target] if target else []))


_UNKNOWN = object()


def _source_constants(module: str, cls: str, depth: int = 0) -> dict | None:
    """PROTOCOL_VERSION / LOAD_KEYS of class `cls` in `module`, from literal assignments in the class body or its
    base classes (followed through the module's own classes and `from ... import` names). None if undetermined."""
    if depth > 10:
        return None
    tree = _module_tree(module)
    if tree is None:
        return None
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    imported = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            source = _absolute_module(module, node.module, node.level)
            for alias in node.names:
                imported[alias.asname or alias.name] = (source, alias.name)
    node = classes.get(cls)
    if node is None:
        return None
    found: dict[str, Any] = {}
    for stmt in node.body:
        targets: list[str] = []
        if isinstance(stmt, ast.Assign):
            targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            targets = [stmt.target.id]
        for name in targets:
            if name in ("PROTOCOL_VERSION", "LOAD_KEYS"):
                try:
                    found[name] = ast.literal_eval(stmt.value)
                except ValueError:
                    found[name] = _UNKNOWN
    for base in node.bases:
        if len(found) == 2:
            break
        inherited = None
        if isinstance(base, ast.Name) and base.id in classes:
            inherited = _source_constants(module, base.id, depth + 1)
        elif isinstance(base, ast.Name) and base.id in imported:
            inherited = _source_constants(*imported[base.id], depth + 1)
        for name, value in (inherited or {}).items():
            found.setdefault(name, value)
    if len(found) < 2 or any(v is _UNKNOWN for v in found.values()):
        return None
    return found


def _query_constants(name: str, python: Path) -> dict | None:
    """Asks the model environment's interpreter for the adapter class constants (imports the adapter module there)."""
    from .envs import src_dir
    code = ("import json, sys\nfrom streamreplace.models.registry import load_class\nc = load_class(sys.argv[1])\n"
            "print(json.dumps({'PROTOCOL_VERSION': c.PROTOCOL_VERSION, 'LOAD_KEYS': list(c.LOAD_KEYS)}))")
    env = {**os.environ, "PYTHONPATH": str(src_dir()), "CUDA_VISIBLE_DEVICES": ""}
    try:
        proc = subprocess.run([str(python), "-c", code, name], capture_output=True, text=True, timeout=900, env=env)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except ValueError:
        return None


@lru_cache(maxsize=None)
def adapter_constants(name: str, python: str | None = None) -> AdapterConstants | None:
    """PROTOCOL_VERSION and LOAD_KEYS of a registered adapter, or None when its module is missing or they cannot be
    determined. `python`: the model environment's interpreter, asked when the source does not state them."""
    from ..models import registry
    module, cls = registry.target(name)
    values = _source_constants(module, cls)
    if values is None and python is not None and Path(python).exists():
        values = _query_constants(name, Path(python))
    if values is None:
        return None
    protocol, keys = values["PROTOCOL_VERSION"], values["LOAD_KEYS"]
    if not isinstance(protocol, int) or isinstance(keys, str) or not all(isinstance(k, str) for k in keys):
        raise ValueError(f"adapter {name}: PROTOCOL_VERSION must be an int and LOAD_KEYS a tuple of dotted keys")
    return AdapterConstants(protocol, tuple(keys))


# ---------- planning ----------

def plan(config: Config, paths: Paths | None = None, *, benchmarks: Iterable[str] | str,
         pairs: Iterable[str] | str = (), models: Iterable[str] | str = (),
         protocols: Iterable[str] | str = PROTOCOLS, variants: Iterable[str] | str = (), limit: int | None = None,
         sets: dict | None = None, outputs: str | Path | None = None) -> list[Job]:
    """The jobs for the chosen pairs/models x conditions (+ plugin variants) x benchmarks, benchmark-major.

    models: model ids, or `<child>@<variant>` for one plugin arm. sets: parsed --set overrides
    (config.settings.parse_sets). outputs: the outputs folder (default: the paths' outputs root); its subset.json and
    prompt_overrides.jsonl apply to every job."""
    if paths is None:
        from .. import paths as paths_module
        paths = paths_module.current()
    outputs = Path(outputs) if outputs is not None else Path(paths.outputs)
    protocols = _names(protocols)
    unknown = sorted(set(protocols) - set(PROTOCOLS))
    if unknown:
        raise ValueError(f"unknown protocols {unknown}; expected P, C and/or S (plugin arms: --variants)")
    variants = list(dict.fromkeys(_names(variants)))
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown plugin variants {unknown}; expected some of {', '.join(VARIANTS)}")
    if limit is not None:
        limit = int(limit)
        if limit < 1:
            raise ValueError(f"--limit must be a positive number of samples, got {limit}")
    selections = select(config, pairs, models, protocols, variants)
    if not selections:
        raise ValueError("choose --pairs and/or --models")
    datasets = config.expand_benchmarks(_names(benchmarks))
    if not datasets:
        raise ValueError("choose --benchmarks (benchmark or suite names, or 'all')")
    subsets = read_subset(outputs) or {}
    overrides = read_overrides(outputs) or {}
    plugin_arms: dict[tuple[str, str], PluginArm] = {}
    loras: dict[str, tuple[Path, str] | None] = {}          # model -> (adapter folder, sha256 of its weights)

    def job(arm: Arm, dataset: str, media: str) -> Job:
        spec = config.model(arm.model)
        block = spec.adapter(arm.mode)
        adapter, plugin, extra, identity_plugin = block.adapter, None, None, None
        if arm.variant is not None:
            key = (arm.model, arm.variant)
            if key not in plugin_arms:
                plugin_arms[key] = plugin_arm(config, arm.model, arm.variant, block.adapter)
            chosen = plugin_arms[key]
            adapter, plugin, extra, identity_plugin = (chosen.adapter, chosen.worker, chosen.settings or None,
                                                       chosen.identity)
        settings = config.settings(dataset, arm.model, arm.mode, overrides=sets, extra=extra)
        constants = adapter_constants(adapter, str(paths.env_python(block.env)))
        model_block = spec.identity_block(arm.mode, constants.protocol if constants else None, plugin=identity_plugin)
        model_block["adapter"] = adapter
        if arm.model not in loras:
            folder = lora_dir(spec, outputs)
            loras[arm.model] = None if folder is None else (folder, file_digest(folder / LORA_WEIGHTS))
        lora = loras[arm.model]
        if lora is not None:
            model_block["options"]["lora"] = {"sha256": lora[1]}
        subset, edits = subsets.get(dataset), overrides.get(dataset)
        payload = run_payload(arm, model_block, dataset, settings, limit, subset, edits)
        return Job(arm=arm, dataset=dataset, outputs=outputs, media=media, spec=spec, adapter=adapter, env=block.env,
                   ffmpeg=block.ffmpeg, settings=settings, payload=payload, constants=constants,
                   deterministic=bool(spec.options.get("deterministic")), lora=lora[0] if lora else None,
                   plugin=plugin, subset=subset, limit=limit, edits=edits)

    jobs = []
    for dataset in datasets:
        media = config.benchmark(dataset).media
        arms = [arm for model, conditions, arm_variants in selections
                for arm in arms_for(config.model(model), media, conditions, arm_variants)]
        jobs += [job(arm, dataset, media) for arm in dict.fromkeys(arms)]   # an arm selected twice is planned once
    return jobs
