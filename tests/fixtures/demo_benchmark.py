"""Synthetic video benchmark for the tests: two tiny generated videos and one sample per answer kind. No raw data is
needed (prepare() writes the videos with PyAV). Tests register it with
streamreplace.benchmarks.registry.register("demo", "demo_benchmark"); prompts, parsing, correctness and metrics are
the shared defaults of streamreplace.benchmarks.common."""
from __future__ import annotations

from pathlib import Path

from streamreplace.core.schema import Sample

MEDIA = "video"
EXPECTED_COUNT = 6


def make_video(path: Path, color: tuple[int, int, int], seconds: int = 3, fps: int = 4):
    import av
    import numpy as np
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        for i in range(seconds * fps):
            frame = np.full((64, 64, 3), color, dtype=np.uint8)
            frame[:, : (i * 64) // (seconds * fps)] = 255 - np.array(color, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def prepare(raw: Path, media: Path) -> list[Sample]:
    red, blue = str(make_video(media / "red.mp4", (200, 30, 30))), str(make_video(media / "blue.mp4", (30, 30, 200)))
    return [
        Sample("mcq-1", "What colour dominates?", "A", "mcq", {"A": "red", "B": "blue"}, video=red,
               metadata={"group": 1}),
        Sample("mcq-2", "What colour dominates?", "B", "mcq", {"A": "red", "B": "blue"}, video=blue,
               metadata={"group": 1}),
        Sample("yn-1", "Is the video red?", "yes", "yesno", video=red, metadata={"group": 2}),
        Sample("yn-2", "Is the video red?", "no", "yesno", video=blue, metadata={"group": 2}),
        Sample("num-1", "How many seconds long is the video?", "3", "numeric", video=red, metadata={"group": 3}),
        Sample("open-1", "Describe the video.", "A red square with a growing band.", "open", video=red,
               metadata={"group": 3}),
    ]


def smoke(samples):
    return [s.id for s in samples[:4]]
