"""Helpers for the stream-adapter tests: a recording AnswerTurnHooks, a minimal torch stand-in, requests and tiny
videos. CPU only: no torch, transformers, OpenCV or checkpoints."""
from __future__ import annotations

import contextlib
from fractions import Fraction
from pathlib import Path
import types

from streamreplace.core.schema import Request


class RecordingHooks:
    """An AnswerTurnHooks that records every call (the plugin's own logic is tested with the plugin)."""
    eager_decode = False

    def __init__(self, step_result=None):
        self.calls: list[tuple] = []
        self.armed = False
        self.step_result = step_result
        self.readout = None

    def settings(self, settings):
        return settings

    def attach(self, layers, tokenizer):
        self.calls.append(("attach", layers, tokenizer))

    def arm(self, first=False):
        assert not self.armed, "armed twice"
        self.armed = True
        self.calls.append(("arm", first))

    def logits_processors(self):
        self.calls.append(("processors",))
        return ["first-step processor"]

    def on_step_logits(self, step, logits):
        assert self.armed, "LM-head logits outside the armed turn"
        self.calls.append(("step", step))
        return self.step_result

    def deadline_readout(self, fork):
        self.calls.append(("deadline",))
        self.readout = fork()

    def disarm(self):
        self.armed = False
        self.calls.append(("disarm",))

    def finish(self, req, generation):
        assert not self.armed, "finish while armed"
        self.calls.append(("finish", generation.text))
        generation.info["plugin"] = {"native": generation.text}
        return generation

    def names(self) -> list[str]:
        return [call[0] for call in self.calls]


def fake_torch(**overrides):
    """The few torch attributes the adapters' control flow touches (keyword arguments add or replace some)."""
    attributes = {"zeros": lambda *a, **k: "zeros", "long": "long", "inference_mode": contextlib.nullcontext,
                  "manual_seed": lambda seed: None}
    attributes.update(overrides)
    return types.SimpleNamespace(**attributes)


def request(kind="mcq", video="v.mp4", settings=None, sample_id="q1", prompt="Q?") -> Request:
    settings = settings or {"generation": {"max_new_tokens": 8}, "stream": {"fps": 1}}
    choices = {"A": "a", "B": "b"} if kind == "mcq" else {}
    return Request(sample_id, "demo", prompt, video=video, settings=settings, kind=kind, choices=choices)


def make_video(path: Path, frames: int = 12, fps: int = 4, size: int = 32) -> Path:
    """A tiny mpeg4 clip written with PyAV: `frames` frames at `fps`."""
    import av
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=Fraction(fps))
        stream.width = stream.height = size
        stream.pix_fmt = "yuv420p"
        for i in range(frames):
            array = np.full((size, size, 3), 10 * i % 250, np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path
