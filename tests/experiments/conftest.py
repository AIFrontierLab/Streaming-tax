"""Fixtures of the experiment tests: the temporary mock project of the end-to-end tests (tests/e2e/e2e_project.py)
with the synthetic demo benchmark registered."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

TESTS = Path(__file__).resolve().parents[1]
for folder in (TESTS / "fixtures", TESTS / "e2e"):               # demo_benchmark, e2e_project
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from streamreplace import paths as paths_module  # noqa: E402
from streamreplace.benchmarks import registry  # noqa: E402


@pytest.fixture
def demo(monkeypatch):
    """The demo benchmark registered, no STREAMREPLACE_* variable of the caller, paths reset afterwards."""
    for name in list(os.environ):
        if name.startswith("STREAMREPLACE_"):
            monkeypatch.delenv(name)
    registry.register("demo", "demo_benchmark")
    paths_module.reset()
    yield
    paths_module.reset()
    registry.unregister("demo")
