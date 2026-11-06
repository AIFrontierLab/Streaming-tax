"""Shared pytest configuration: marker skipping and a temporary-configuration fixture.

The default suite runs anywhere on CPU without data, checkpoints or fetched third-party code. Tests that need more are
marked (the markers are registered in pyproject.toml) and skipped, with the reason, when it is absent:

    requires_data(*benchmarks)  the prepared samples <data>/<benchmark>/prepared/samples.jsonl of every listed
                                benchmark (`streamreplace prepare`); without arguments, the data root itself
    requires_upstream(*names)   every listed repository under the upstream root (`streamreplace fetch-upstream`)

Roots are resolved the way the command line resolves them (paths.py: STREAMREPLACE_HOME and the per-root variables,
STREAMREPLACE_PATHS, configs/local.yaml), once per test session.

Fixtures:
    config_dir   factory: writes a configuration folder (defaults/models/pairs/benchmarks.yaml, optional section
                 files and local.yaml) under tmp_path and returns it. The default content is a small valid project:
                 a mock parent and a mock child paired on one video benchmark.
    tmp_config   factory: the same folder, loaded and validated (config.load.load_config).
Every test also gets the process-wide paths (paths.current()) restored afterwards, so a test that installs its own
roots never leaks them into the next one."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from streamreplace import paths as paths_module

DEFAULTS = {"generation": {"max_new_tokens": 16}, "image": {"max_pixels": 1003520},
            "video": {"nframes": 8, "max_pixels": 200704}, "stream": {"fps": 1}}
MODELS = {
    "parent": {"role": "parent", "checkpoint": "test/parent@0", "offline": {"adapter": "mock", "env": "local"}},
    "child": {"role": "child", "checkpoint": "test/child@0", "offline": {"adapter": "mock", "env": "local"},
              "stream": {"adapter": "mock", "env": "local"}},
}
PAIRS = {"p": {"parent": "parent", "children": ["child"], "video": {"fps": 2, "max_frames": 16}}}
BENCHMARKS = {"demo": {"media": "video", "suite": "video"}}


_roots: dict = {}


def _paths():
    """The roots of this session (resolved once), or the error that prevented resolving them."""
    if "value" not in _roots:
        try:
            _roots["value"] = paths_module.resolve(config_paths=paths_module.local_config_paths())
        except (ValueError, OSError) as exc:
            _roots["value"] = exc
    return _roots["value"]


def pytest_report_header(config):
    roots = _paths()
    if isinstance(roots, Exception):
        return f"streamreplace roots: not resolved ({roots})"
    return f"streamreplace roots: data={roots.data} upstream={roots.upstream}"


def skip_reason(item) -> str | None:
    """Why a marked test cannot run here, or None."""
    data = [m for m in item.iter_markers("requires_data")]
    upstream = [m for m in item.iter_markers("requires_upstream")]
    if not data and not upstream:
        return None
    roots = _paths()
    if isinstance(roots, Exception):
        return f"cannot resolve the data and upstream roots: {roots}"
    for marker in data:
        if not marker.args and not roots.data.is_dir():
            return f"no data root ({roots.data})"
        for name in marker.args:
            path = roots.dataset_dir(name) / "prepared" / "samples.jsonl"
            if not path.is_file():
                return f"{name} is not prepared ({path} is missing; streamreplace prepare {name})"
    for marker in upstream:
        for name in marker.args:
            if not (roots.upstream / name).exists():
                return f"upstream code {name!r} is not fetched (streamreplace fetch-upstream {name.split('/')[0]})"
    return None


def pytest_runtest_setup(item):
    reason = skip_reason(item)
    if reason:
        pytest.skip(reason)


@pytest.fixture(autouse=True)
def _restore_current_paths():
    saved = paths_module._current
    yield
    paths_module.set_current(saved)


def write_config_dir(folder: Path, defaults: dict | None = None, models: dict | None = None,
                     pairs: dict | None = None, benchmarks: dict | None = None, suites: dict | None = None,
                     run: dict | None = None, sections: dict | None = None, local: dict | None = None) -> Path:
    """A configuration folder with the given parts (None = the small default project of this module)."""
    def dump(name: str, data) -> None:
        (folder / f"{name}.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    folder.mkdir(parents=True, exist_ok=True)
    dump("defaults", {"defaults": deepcopy(DEFAULTS if defaults is None else defaults),
                      **({"run": run} if run is not None else {})})
    dump("models", {"models": deepcopy(MODELS if models is None else models)})
    dump("pairs", {"pairs": deepcopy(PAIRS if pairs is None else pairs)})
    dump("benchmarks", {"benchmarks": deepcopy(BENCHMARKS if benchmarks is None else benchmarks),
                        "suites": deepcopy(suites or {})})
    for name, data in (sections or {}).items():
        dump(name, data)
    if local is not None:
        dump("local", local)
    return folder


@pytest.fixture
def config_dir(tmp_path):
    """config_dir(**parts) -> a configuration folder under tmp_path (see write_config_dir)."""
    counter = iter(range(1_000_000))

    def make(**parts) -> Path:
        return write_config_dir(tmp_path / f"configs{next(counter)}", **parts)
    return make


@pytest.fixture
def tmp_config(config_dir):
    """tmp_config(extra=(), check_registry=True, **parts) -> the loaded, validated Config of such a folder."""
    from streamreplace.config.load import load_config

    def make(extra=(), check_registry: bool = True, **parts):
        return load_config(config_dir(**parts), extra=extra, check_registry=check_registry)
    return make
