"""Fixtures for the media tests: tiny videos generated with PyAV, frame lists, and an isolated frame-count cache."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image

from streamreplace.media import video_meta


@pytest.fixture(autouse=True)
def frame_count_cache(tmp_path, monkeypatch):
    """Decodable-frame counts of test videos go to a temporary folder, never to the user's cache."""
    cache = tmp_path / "frame_counts"
    monkeypatch.setattr(video_meta, "frame_count_dir", lambda: cache)
    return cache


def make_video(path: Path, frames: int = 12, fps: int = 4, size: int = 64) -> Path:
    """`frames` frames at `fps` (mpeg4). Frame i has a bright band over the left i/frames of the width, so every frame
    differs and its index can be read back from the mean brightness."""
    import av
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = stream.height = size
        stream.pix_fmt = "yuv420p"
        for i in range(frames):
            array = np.full((size, size, 3), 30, np.uint8)
            array[:, : (i * size) // frames] = 220
            for packet in stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def make_edit_list_video(path: Path, hidden: int = 12, total: int = 36) -> Path:
    """What a cut without re-encoding leaves: `hidden` leading frames kept for decoding but hidden by the MP4 edit
    list (negative timestamps). Needs PyAV's libx264 encoder."""
    import av
    import numpy as np
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=12)
        stream.width = stream.height = 64
        stream.pix_fmt, stream.time_base, stream.options = "yuv420p", Fraction(1, 12), {"bf": "0", "g": "100"}
        for i in range(total):
            frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), 7 * i, np.uint8), format="rgb24")
            frame.pts, frame.time_base = i - hidden, Fraction(1, 12)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def make_images(folder: Path, n: int, size=(32, 24)) -> list[Path]:
    folder.mkdir(parents=True)
    paths = []
    for i in range(n):
        path = folder / f"{i:05d}.png"
        Image.new("RGB", size, (10 * i, 0, 0)).save(path)
        paths.append(path)
    return paths


@pytest.fixture
def video(tmp_path) -> str:
    """12 frames at 4 fps (3 s), like the demo benchmark's clips."""
    return str(make_video(tmp_path / "data/media/clip.mp4"))


@pytest.fixture
def new_video():
    """make_video(path, frames=12, fps=4, size=64) -> path."""
    return make_video


@pytest.fixture
def new_edit_list_video():
    """make_edit_list_video(path, hidden=12, total=36) -> path; skips the test when libx264 is not available."""
    import av
    try:
        av.codec.Codec("libx264", "w")
    except Exception:
        pytest.skip("PyAV has no libx264 encoder")
    return make_edit_list_video


@pytest.fixture
def new_images():
    """make_images(folder, n, size=(32, 24)) -> paths; image i has red value 10 * i."""
    return make_images
