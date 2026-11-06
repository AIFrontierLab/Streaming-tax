"""Fixtures for the offline-adapter and environment tests (helpers live in offline_fakes.py)."""
from __future__ import annotations

import pytest

from streamreplace.media import video_meta


@pytest.fixture(autouse=True)
def frame_count_cache(tmp_path, monkeypatch):
    """Decodable-frame counts of test videos go to a temporary folder, never to the user's cache."""
    monkeypatch.setattr(video_meta, "frame_count_dir", lambda: tmp_path / "frame_counts")
