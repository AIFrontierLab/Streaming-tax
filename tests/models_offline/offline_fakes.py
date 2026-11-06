"""Helpers for the offline-adapter tests: CPU stand-ins for processors, conversation templates and frame sampling,
and the settings of the paper's pairs. No torch, transformers or checkpoints."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

from streamreplace.core.schema import Request

DEFAULTS = {"generation": {"max_new_tokens": 512}, "image": {"max_pixels": 1003520},
            "video": {"nframes": 64, "max_pixels": 200704}, "stream": {"fps": 1}}


def settings(video: dict | None = None, **extra) -> dict:
    """Resolved settings with a video block (and e.g. system=..., generation=...)."""
    out = {key: dict(value) for key, value in DEFAULTS.items()}
    if video is not None:
        out["video"] = dict(video)
    out.update(extra)
    return out


def request(prompt: str = "Q?", path: str | None = None, images=(), kind: str = "mcq", **settings_kwargs) -> Request:
    """A request for the video file `path` and/or `images`; settings_kwargs go to settings() (video=..., system=...)."""
    return Request("q1", "demo", prompt, list(images), path, settings(**settings_kwargs), kind=kind,
                   choices={"A": "a", "B": "b"} if kind == "mcq" else {})


def frames(n: int, size=(64, 48)) -> list[Image.Image]:
    return [Image.new("RGB", size, (10 * i % 255, 0, 0)) for i in range(n)]


def fake_offline_frames(n: int, fps: float = 4.0, size=(64, 48)):
    """A stand-in for media.frames.offline_frames: n frames, indices 0..n-1 at `fps`."""
    def offline_frames(path, video):
        indices = list(range(n))
        return frames(n, size), {"indices": indices, "timestamps": [i / fps for i in indices], "fps": fps,
                                 "duration": n / fps}
    return offline_frames


class FakeTokenizer:
    """Characters as tokens (an option letter is one token: its code point)."""
    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        return SimpleNamespace(input_ids=[ord(c) for c in text])

    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(i)) for i in ids)

    def batch_decode(self, batch, skip_special_tokens=False):
        return [self.decode(ids, skip_special_tokens) for ids in batch]


class FakeArray(np.ndarray):
    """A numpy array standing in for a torch tensor: .cuda() / .to(...) return it unchanged."""
    def cuda(self):
        return self

    def to(self, *args, **kwargs):
        return self


def fake_array(values) -> FakeArray:
    return np.asarray(values).view(FakeArray)


class FakeBatch(dict):
    """A processor output: a mapping of model inputs with attribute access and a no-op .to(device)."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def to(self, device):
        return self


class FakeHead:
    """The state of models/lm_head.py's KeepLast wrapper (keep, capture, captured, calls) without torch."""
    def __init__(self):
        self.keep, self.capture, self.captured, self.calls = 0, False, None, 0

    def forward(self, logits):
        """One LM-head call: the first call's logits are kept while capture is on."""
        self.calls += 1
        if self.capture and self.captured is None:
            self.captured = logits


def first_step_logits(**letters) -> list[float]:
    """First-step logits over a 128-token vocabulary (token = code point) with the given values for some letters."""
    logits = [0.0] * 128
    for letter, value in letters.items():
        logits[ord(letter)] = value
    return logits


class FakeQwenProcessor:
    """Records the chat messages and the processor call; returns input ids of the text's length."""
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.messages = None
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.messages = messages
        out = ""
        for message in messages:
            content = message["content"]
            if not isinstance(content, str):
                content = "".join(f"<{c['type']}>" if c["type"] != "text" else c["text"] for c in content)
            out += f"<{message['role']}>{content}"
        return out + "<assistant>"

    def __call__(self, text, images=None, videos=None, return_tensors=None, padding=None, **kwargs):
        self.calls.append({"text": text, "images": images, "videos": videos, "kwargs": kwargs})
        return FakeBatch(input_ids=np.zeros((1, len(text[0])), dtype=np.int64))


class FakeTemplate:
    """The internvl2_5 conversation's shape: a system turn, user / assistant turns closed by <|im_end|>."""
    def __init__(self):
        self.system_message = "DEFAULT SYSTEM"
        self.roles = ("<|im_start|>user\n", "<|im_start|>assistant\n")
        self.messages = []

    def append_message(self, role, message):
        self.messages.append((role, message))

    def get_prompt(self):
        out = f"<|im_start|>system\n{self.system_message}<|im_end|>\n"
        for role, message in self.messages:
            out += role + ("" if message is None else message + "<|im_end|>\n")
        return out


def get_conv_template(name):                # found through sys.modules[type(model).__module__], as the remote code's
    assert name == "internvl2_5"
    return FakeTemplate()


class FakeChatModel:
    template = "internvl2_5"
    system_message = "DEFAULT SYSTEM"
    num_image_token = 16
