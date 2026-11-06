"""Loads and validates the configuration folder (configs/ in the source checkout, or --config-dir).

Files:
- defaults.yaml, models.yaml, pairs.yaml, benchmarks.yaml: the core keys `defaults`, `run`, `models`, `pairs`,
  `benchmarks`, `suites` (each defined by exactly one of these files).
- any other configs/<name>.yaml (judges, analysis, plugin, ...): kept whole as the section `<name>`
  (Config.section(name)); the module that owns it interprets it.
- configs/local.yaml (git-ignored, optional) and extra files passed with --config: merged on top, in that order.
  Their top-level keys are core keys, `paths` (see paths.py) or section names, e.g.
      paths: {home: /scratch/streamreplace}
      suites: {quick: [mme, videomme]}
      analysis: {engine: {reps: 2000}}
Merging follows config.settings.merge (dicts recursively, lists and scalars replace)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

import yaml

from .. import paths as paths_module
from .schema import BenchmarkSpec, Config, ModelSpec, PairSpec
from .settings import check_layer, merge

CORE_FILES = ("defaults", "models", "pairs", "benchmarks")
CORE_KEYS = ("defaults", "run", "models", "pairs", "benchmarks", "suites")
OVERRIDE_KEYS = (*CORE_KEYS, "paths")
LOCAL = paths_module.LOCAL_CONFIG


def read_yaml(path: Path) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: the top level must be a mapping")
    return data


def load_config(config_dir: str | Path | None = None, extra: Iterable[str | Path] = (),
                check_registry: bool = True) -> Config:
    """Reads the configuration folder plus local.yaml and `extra` files, then validates it (see validate())."""
    folder = Path(config_dir) if config_dir is not None else paths_module.default_config_dir()
    if not folder.is_dir():
        raise ValueError(f"configuration folder not found: {folder} (pass --config-dir)")
    core: dict = {}
    owner: dict[str, Path] = {}
    sections: dict[str, dict] = {}
    sources: list[Path] = []
    for path in sorted(folder.glob("*.yaml")):
        if path.name == LOCAL:
            continue
        data = read_yaml(path)
        if path.stem in CORE_FILES:
            for key, value in data.items():
                if key not in CORE_KEYS:
                    raise ValueError(f"{path}: unknown top-level key {key!r} (core keys: {', '.join(CORE_KEYS)})")
                if key in core:
                    raise ValueError(f"{path}: {key!r} is already defined in {owner[key]}")
                core[key], owner[key] = value, path
        elif path.stem in OVERRIDE_KEYS:
            raise ValueError(f"{path}: '{path.stem}' is a core key; define it in one of "
                             f"{', '.join(f + '.yaml' for f in CORE_FILES)} or in local.yaml")
        else:
            sections[path.stem] = data
        sources.append(path)
    overrides = [folder / LOCAL] if (folder / LOCAL).is_file() else []
    overrides += [Path(p) for p in extra]
    for path in overrides:
        for key, value in read_yaml(path).items():
            if key in OVERRIDE_KEYS:
                core = merge(core, {key: value})
            else:
                sections[key] = merge(sections.get(key) or {}, value if isinstance(value, dict) else {})
        sources.append(path)
    config = build(core, sections, folder, sources)
    validate(config, check_registry)
    return config


def build(core: dict, sections: dict | None = None, config_dir: Path | None = None,
          sources: list[Path] | None = None) -> Config:
    """Typed Config from the merged core keys (no cross-file validation; see validate())."""
    models = {mid: ModelSpec.parse(str(mid), value) for mid, value in (core.get("models") or {}).items()}
    pairs = {pid: PairSpec.parse(str(pid), value) for pid, value in (core.get("pairs") or {}).items()}
    benchmarks = {name: BenchmarkSpec.parse(str(name), value) for name, value in (core.get("benchmarks") or {}).items()}
    suites = {}
    for name, members in (core.get("suites") or {}).items():
        if isinstance(members, str) or not isinstance(members, list):
            raise ValueError(f"suites.{name}: expected a list of benchmark names")
        suites[str(name)] = tuple(str(m) for m in members)
    for pair in pairs.values():                     # each model knows its pair (validate() checks uniqueness)
        for member in pair.members:
            if member in models and models[member].pair is None:
                models[member] = replace(models[member], pair=pair.id)
    run = core.get("run") or {}
    if not isinstance(run, dict):
        raise ValueError("run: expected a mapping")
    return Config(defaults=check_layer(core.get("defaults"), "defaults"), run=dict(run), models=models, pairs=pairs,
                  benchmarks=benchmarks, suites=suites, raw=core, sections=dict(sections or {}),
                  config_dir=config_dir, sources=list(sources or []))


def validate(config: Config, check_registry: bool = True) -> None:
    """Cross-file checks; raises one ValueError listing every problem."""
    problems: list[str] = []
    budget = (config.defaults.get("generation") or {}).get("max_new_tokens")
    if not isinstance(budget, int):
        problems.append("defaults.generation.max_new_tokens must be an integer")
    seen: dict[str, str] = {}
    for pair in config.pairs.values():
        for member in pair.members:
            if member not in config.models:
                problems.append(f"pairs.{pair.id}: unknown model {member!r}")
                continue
            if member in seen:
                problems.append(f"model {member!r} is in pairs {seen[member]} and {pair.id} (one pair per model)")
            seen[member] = pair.id
        if pair.parent in config.models:
            parent = config.models[pair.parent]
            if parent.role.value != "parent":
                problems.append(f"pairs.{pair.id}: parent {pair.parent!r} has role {parent.role.value}")
            if parent.stream is not None:
                problems.append(f"models.{pair.parent}: parents are evaluated offline only (remove its stream block)")
            if parent.offline is None:
                problems.append(f"models.{pair.parent}: a parent needs an offline block")
        for child in pair.children:
            if child in config.models and config.models[child].role.value != "child":
                problems.append(f"pairs.{pair.id}: child {child!r} has role {config.models[child].role.value}")
    for name, members in config.suites.items():
        if name in config.benchmarks or name == "all":
            problems.append(f"suites.{name}: the name is taken by a benchmark (or reserved)")
        problems += [f"suites.{name}: unknown benchmark {m!r}" for m in members if m not in config.benchmarks]
    if check_registry:
        from ..models import registry
        for model in config.models.values():
            for mode in model.modes:
                adapter = model.adapter(mode).adapter
                if adapter not in registry.ADAPTERS:
                    problems.append(f"models.{model.id}.{mode}: unknown adapter {adapter!r}")
                elif mode not in registry.MODES[adapter]:
                    problems.append(f"models.{model.id}.{mode}: adapter {adapter!r} does not serve {mode} mode")
    if config.raw.get("paths") is not None:
        try:
            paths_module.check_paths_block(config.raw["paths"], "paths")
        except ValueError as exc:
            problems.append(str(exc))
    if problems:
        raise ValueError("invalid configuration:\n  " + "\n  ".join(problems))
