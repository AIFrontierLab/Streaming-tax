"""Fixtures of the end-to-end tests: a temporary project with two mock models, a parent and a child that fails one
sample on purpose, paired on the synthetic demo benchmark (tests/fixtures/demo_benchmark.py). Runs go through the
real planner, dispatcher and worker subprocesses (env "local" = this interpreter); no GPU, no data download."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
for folder in (FIXTURES, Path(__file__).resolve().parent):      # demo_benchmark, e2e_project (any import mode)
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from e2e_project import make_project  # noqa: E402

from streamreplace import paths as paths_module  # noqa: E402
from streamreplace.benchmarks import registry  # noqa: E402

DEMOS = ("demo", "demo2")      # demo2: the same synthetic benchmark under a second name (load-time settings test)


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    """No STREAMREPLACE_* variable of the caller leaks in; the demo benchmarks are registered; paths are reset."""
    for name in list(os.environ):
        if name.startswith("STREAMREPLACE_"):
            monkeypatch.delenv(name)
    for name in DEMOS:
        registry.register(name, "demo_benchmark")
    paths_module.reset()
    yield
    paths_module.reset()
    for name in DEMOS:
        registry.unregister(name)


@pytest.fixture
def project(tmp_path):
    return make_project(tmp_path)
