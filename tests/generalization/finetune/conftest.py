"""Fixtures of the fine-tuning tests: the synthetic benchmark `ftdemo` (finetune_demo.py) registered with its
prepared splits under a temporary workspace, and an experiment over it (helpers in finetune_helpers.py)."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from finetune_helpers import DEMO, spec, write_prepared  # noqa: E402

from streamreplace import paths as paths_module  # noqa: E402
from streamreplace.benchmarks import registry  # noqa: E402
from streamreplace.generalization.finetune import Experiment, validate_spec  # noqa: E402


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A clean workspace (no STREAMREPLACE_* variable of the caller) with the demo benchmark registered; returns the
    resolved Paths, also installed as paths.current()."""
    for name in list(os.environ):
        if name.startswith("STREAMREPLACE_"):
            monkeypatch.delenv(name)
    registry.register(DEMO, "finetune_demo")
    paths = paths_module.resolve(home=tmp_path / "home", environ={})
    paths_module.set_current(paths)
    yield paths
    paths_module.reset()
    registry.unregister(DEMO)


@pytest.fixture
def demo(home):
    """The workspace with the demo benchmark's prepared test and training splits."""
    write_prepared(home)
    return home


@pytest.fixture
def exp(demo):
    value = spec()
    validate_spec(value)
    return Experiment(value, Path(demo.outputs) / "finetune")
