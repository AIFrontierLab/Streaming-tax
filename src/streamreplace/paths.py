"""Every filesystem location the package uses, resolved in one place. No path is hard-coded anywhere else.

There is one root setting, STREAMREPLACE_HOME (default: `workspace/` inside the source checkout). It contains

    data/         raw and prepared benchmarks, one folder per benchmark (<data>/<benchmark>)
    checkpoints/  model and judge checkpoints, one folder per Hugging Face repo (<checkpoints>/<org>/<name>)
    envs/         one Python environment per model family (<envs>/<env>/bin/python)
    outputs/      generations, scores, judge caches, reports
    cache/        derived caches (decodable frame counts, kernels' workspaces)
    upstream/     third-party code fetched at pinned commits (<upstream>/<name>)

Each root can be moved on its own. Sources, lowest to highest priority:
1. a `paths:` block in configs/local.yaml (git-ignored) or in an extra config file; relative paths are relative to
   the source checkout;
2. a paths file given with --paths FILE or STREAMREPLACE_PATHS (same keys, optionally under `paths:`); relative
   paths are relative to that file's folder;
3. environment variables STREAMREPLACE_HOME and STREAMREPLACE_{DATA,CHECKPOINTS,ENVS,OUTPUTS,CACHE,UPSTREAM};
   relative paths are relative to the current directory;
4. the command-line flags --home and --outputs.
A root set explicitly at any level wins over the default derived from home. Paths blocks and files may also map
single checkpoints or datasets elsewhere: `checkpoint_dirs: {<org>/<name>: DIR}`, `dataset_dirs: {<benchmark>: DIR}`.

Subprocesses (model workers, judges) receive the resolved roots as environment variables (Paths.environ())."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
from typing import Any, Mapping

ROOTS = ("data", "checkpoints", "envs", "outputs", "cache", "upstream")
HOME_ENV = "STREAMREPLACE_HOME"
PATHS_ENV = "STREAMREPLACE_PATHS"
ROOT_ENV = {root: f"STREAMREPLACE_{root.upper()}" for root in ROOTS}
MAP_KEYS = ("checkpoint_dirs", "dataset_dirs")
PATH_KEYS = ("home", *ROOTS, *MAP_KEYS)
LOCAL_CONFIG = "local.yaml"


def project_root() -> Path:
    """The source checkout (the folder with configs/ next to src/); else the nearest folder up from the current
    directory that has configs/defaults.yaml; else the current directory."""
    here = Path(__file__).resolve().parents[2]
    if (here / "configs").is_dir():
        return here
    cwd = Path.cwd()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "configs" / "defaults.yaml").is_file():
            return candidate
    return cwd


def default_config_dir() -> Path:
    return project_root() / "configs"


@dataclass(frozen=True)
class Paths:
    home: Path
    data: Path
    checkpoints: Path
    envs: Path
    outputs: Path
    cache: Path
    upstream: Path
    checkpoint_dirs: Mapping[str, Path] = field(default_factory=dict)
    dataset_dirs: Mapping[str, Path] = field(default_factory=dict)
    paths_file: Path | None = None
    sources: Mapping[str, str] = field(default_factory=dict)   # where each root came from (shown by `doctor`)

    def root(self, name: str) -> Path:
        if name not in ("home", *ROOTS):
            raise ValueError(f"unknown root {name!r}")
        return getattr(self, name)

    def checkpoint_dir(self, ref: Any) -> Path:
        """Local folder of a checkpoint: `<checkpoints>/<org>/<name>` unless checkpoint_dirs maps the repo id.
        `ref` is a "repo@revision" string or anything with a .repo attribute (config.schema.CheckpointRef)."""
        repo = getattr(ref, "repo", None) or str(ref).split("@", 1)[0]
        return self.checkpoint_dirs.get(repo) or self.checkpoints / repo

    def dataset_dir(self, name: str) -> Path:
        """Folder of one benchmark's data (raw/, prepared/, media/): `<data>/<name>` unless dataset_dirs maps it."""
        return self.dataset_dirs.get(name) or self.data / name

    def env_python(self, env: str) -> Path:
        """Interpreter of a model environment; env "local" is the interpreter running this code (tests, mock)."""
        if env == "local":
            return Path(sys.executable)
        return self.envs / env / "bin" / "python"

    def environ(self) -> dict[str, str]:
        """Environment variables that make a subprocess resolve exactly these roots."""
        env = {HOME_ENV: str(self.home), **{ROOT_ENV[r]: str(getattr(self, r)) for r in ROOTS}}
        if self.paths_file is not None:
            env[PATHS_ENV] = str(self.paths_file)
        return env

    def as_dict(self) -> dict[str, str]:
        return {"home": str(self.home), **{r: str(getattr(self, r)) for r in ROOTS}}


def _absolute(value: Any, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def check_paths_block(block: Mapping, where: str) -> dict:
    if not isinstance(block, Mapping):
        raise ValueError(f"{where}: paths must be a mapping")
    unknown = sorted(set(block) - set(PATH_KEYS))
    if unknown:
        raise ValueError(f"{where}: unknown path keys {unknown}; allowed: {', '.join(PATH_KEYS)}")
    for key in MAP_KEYS:
        if key in block and not isinstance(block[key], Mapping):
            raise ValueError(f"{where}: {key} must map names to folders")
    return dict(block)


def read_paths_file(path: Path) -> dict:
    """A paths file: the keys of PATH_KEYS at top level, or under a `paths:` key."""
    import yaml
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if isinstance(data, Mapping) and set(data) == {"paths"}:
        data = data["paths"]
    return check_paths_block(data, str(path))


def resolve(home: str | Path | None = None, paths_file: str | Path | None = None,
            outputs: str | Path | None = None, config_paths: Mapping | None = None,
            environ: Mapping[str, str] | None = None, project: Path | None = None) -> Paths:
    """Resolves every root from the sources listed in the module docstring (all returned paths are absolute)."""
    env = os.environ if environ is None else environ
    project = Path(project) if project is not None else project_root()
    cwd = Path.cwd()
    layers: list[tuple[str, Path, dict]] = []                      # (source label, base for relative paths, values)
    if config_paths:
        layers.append(("config paths block", project, check_paths_block(config_paths, "config paths")))
    file = paths_file or env.get(PATHS_ENV)
    file_path = _absolute(file, cwd) if file else None
    if file_path is not None:
        label = "--paths" if paths_file else PATHS_ENV
        layers.append((f"paths file ({label})", file_path.parent, read_paths_file(file_path)))
    from_env = {key: env[var] for key, var in [("home", HOME_ENV), *((r, ROOT_ENV[r]) for r in ROOTS)]
                if env.get(var)}
    if from_env:
        layers.append(("environment", cwd, from_env))
    flags = {k: v for k, v in (("home", home), ("outputs", outputs)) if v}
    if flags:
        layers.append(("command line", cwd, flags))

    def pick(key: str) -> tuple[Path, str] | None:
        for label, base, values in reversed(layers):
            if values.get(key):
                return _absolute(values[key], base), label
        return None

    sources: dict[str, str] = {}
    found = pick("home")
    home_path, sources["home"] = found if found else ((project / "workspace").resolve(), "default")
    roots = {}
    for root in ROOTS:
        found = pick(root)
        roots[root], sources[root] = found if found else (home_path / root, "home")
    maps: dict[str, dict[str, Path]] = {key: {} for key in MAP_KEYS}
    for _, base, values in layers:                                 # lowest first: later layers override entries
        for key in MAP_KEYS:
            for name, folder in (values.get(key) or {}).items():
                maps[key][str(name)] = _absolute(folder, base)
    return Paths(home=home_path, **roots, checkpoint_dirs=maps["checkpoint_dirs"], dataset_dirs=maps["dataset_dirs"],
                 paths_file=file_path, sources=sources)


def local_config_paths(config_dir: Path | None = None) -> dict | None:
    """The `paths:` block of configs/local.yaml, if that file exists."""
    path = Path(config_dir or default_config_dir()) / LOCAL_CONFIG
    if not path.is_file():
        return None
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("paths") if isinstance(data, Mapping) else None


_current: Paths | None = None


def current() -> Paths:
    """The process-wide paths: set by the command line (set_current), else resolved from the environment, the
    STREAMREPLACE_PATHS file and configs/local.yaml."""
    global _current
    if _current is None:
        _current = resolve(config_paths=local_config_paths())
    return _current


def set_current(paths: Paths | None) -> None:
    """Installs the paths the command line resolved (None forgets them, e.g. between tests)."""
    global _current
    _current = paths


def reset() -> None:
    set_current(None)


# Module-level accessors of the current paths (what library code calls when nobody passed a Paths object).

def home() -> Path:
    return current().home


def data_dir() -> Path:
    return current().data


def checkpoints_dir() -> Path:
    return current().checkpoints


def envs_dir() -> Path:
    return current().envs


def outputs_dir() -> Path:
    return current().outputs


def cache_dir() -> Path:
    return current().cache


def upstream_root() -> Path:
    return current().upstream


def dataset_dir(name: str) -> Path:
    return current().dataset_dir(name)


def checkpoint_dir(ref: Any) -> Path:
    return current().checkpoint_dir(ref)


def env_python(env: str) -> Path:
    return current().env_python(env)
