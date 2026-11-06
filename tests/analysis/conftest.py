"""Fixtures of the analysis tests (the synthetic pair of analysis_fakes.py)."""
from __future__ import annotations

import pytest

from analysis_fakes import catalog, outputs, profile


@pytest.fixture
def synthetic(tmp_path):
    """(outputs folder, catalog, profile) of the synthetic pair."""
    return outputs(tmp_path / "outputs"), catalog(), profile(tmp_path)
