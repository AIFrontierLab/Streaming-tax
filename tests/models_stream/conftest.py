"""Fixtures for the stream-adapter tests (helpers live in stream_fakes.py)."""
from __future__ import annotations

import sys
import types

import pytest

from streamreplace.media import video_meta
from stream_fakes import RecordingHooks


@pytest.fixture(autouse=True)
def frame_count_cache(tmp_path, monkeypatch):
    """Decodable-frame counts of test videos go to a temporary folder, never to the user's cache."""
    monkeypatch.setattr(video_meta, "frame_count_dir", lambda: tmp_path / "frame_counts")


@pytest.fixture
def hooks():
    return RecordingHooks()


@pytest.fixture
def fake_plugin_guard(monkeypatch):
    """plugin.guard.find_layers, which the adapters call in attach (the real one belongs to the plugin)."""
    guard = types.ModuleType("streamreplace.plugin.guard")
    guard.find_layers = lambda root: ("layers of", root)
    monkeypatch.setitem(sys.modules, "streamreplace.plugin.guard", guard)
    try:
        import streamreplace.plugin  # noqa: F401
    except ImportError:
        package = types.ModuleType("streamreplace.plugin")
        package.__path__ = []
        monkeypatch.setitem(sys.modules, "streamreplace.plugin", package)
    return guard


@pytest.fixture
def fake_transformers(monkeypatch):
    """transformers.LogitsProcessorList for the HF-generate call sites when transformers is not installed."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        module = types.ModuleType("transformers")
        module.LogitsProcessorList = list
        monkeypatch.setitem(sys.modules, "transformers", module)
