"""Fixtures of the prompt-robustness tests: the synthetic benchmark `rdemo` (robustness_demo.py) registered and
prepared under a temporary workspace, and an experiment over it (helpers in robustness_helpers.py)."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import robustness_demo  # noqa: E402
from robustness_helpers import DEMO, PAIRS, SPEC  # noqa: E402

from streamreplace import paths as paths_module  # noqa: E402
from streamreplace.benchmarks import registry  # noqa: E402
from streamreplace.core.io import write_jsonl  # noqa: E402
from streamreplace.generalization.robustness import make  # noqa: E402


@pytest.fixture
def demo(tmp_path, monkeypatch):
    """The synthetic benchmark registered, its unit definition installed and its samples prepared under
    tmp_path/home; returns the resolved Paths."""
    for name in list(os.environ):
        if name.startswith("STREAMREPLACE_"):
            monkeypatch.delenv(name)
    registry.register(DEMO, "robustness_demo")
    monkeypatch.setitem(make.DATASETS, DEMO, make.UnitSpec(
        stem=lambda s: s.metadata["raw_question"], unit=lambda s: str(s.metadata["question"]),
        cluster=lambda s: str(s.metadata["question"]), strata=("l2",),
        instruction=lambda: robustness_demo.POST_PROMPT.strip()))
    paths = paths_module.resolve(home=tmp_path / "home")
    paths_module.set_current(paths)
    write_jsonl(paths.dataset_dir(DEMO) / "prepared" / "samples.jsonl",
                [s.to_dict() for s in robustness_demo.samples()])
    yield paths
    paths_module.reset()
    registry.unregister(DEMO)


@pytest.fixture
def spec():
    return deepcopy(SPEC)


@pytest.fixture
def experiment(demo, tmp_path, spec):
    return make.Experiment(spec, tmp_path / "exp", dict(PAIRS), demo)

