"""Data contracts shared by every stage: prepared samples, what a model adapter receives, what it returns, and the
two row types written to the outputs folder.

The reference answer never reaches a model process: adapters get a `Request`, whose `sample` is a `SampleView`
(no answer, and without the benchmark's private metadata such as VMDT ground truth). Only the test-only mock model
receives `Request.reference`.

Output rows (one JSON object per line, see docs on the outputs layout):
- generation row, `<dataset>/<job>.jsonl`, append-only (a retry appends a new row; the last row per id wins):
  {"id", "prompt", "raw", "tokens", "finish", "latency", "info", "error", "t"}. Error rows carry raw/tokens/finish/
  info = null and the error message. `t` is the UTC end time.
- score row, `<dataset>/<job>.scores.jsonl`, one per id, rewritten atomically by scoring:
  {"id", "pred", "parse", "correct", "judge_pred", "judge_correct", "judge"}.

Standard `info` keys written by adapters: frame_indices (frame digest), frames, duration, sample_fps, frame_size,
input_tokens; streams add no_answer (the model stayed silent), forced (MMDuet's forced reply), seed (sampled
streams), schedule (digest of the stream schedule); offline mcq requests may add letter_logits. Rows of a plugin arm
add `plugin` = {"native": the child's own reply, "guard": {"proj": the gate's projection (3 decimals), "fired": bool,
...}, "readout": {option: logit} when the option readout answered}; the post-hoc S+ assembly and the analysis read
info.plugin.guard.proj."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
from typing import Any

KINDS = {"mcq", "yesno", "numeric", "text", "open"}   # open = free-form, judged by the LLM judge

# Finish reasons adapters report: eos (end of turn), length (token budget reached), stop (a stop string), guard (the
# plugin's live gate replaced the reply), readout (the plugin's option readout), unknown (the API does not say).
FINISH = ("eos", "length", "stop", "guard", "readout", "unknown")

GEN_KEYS = ("id", "prompt", "raw", "tokens", "finish", "latency", "info", "error", "t")
SCORE_KEYS = ("pred", "parse", "correct", "judge_pred", "judge_correct")
SCORE_ROW_KEYS = ("id", *SCORE_KEYS, "judge")


def seed_of(sample_id: str) -> int:
    """Per-question seed for adapters whose official decoding samples (reproducible, independent of order)."""
    return int(hashlib.sha256(str(sample_id).encode()).hexdigest()[:8], 16)


@dataclass(frozen=True)
class SampleView:
    """What prompt rendering, parsing and model adapters may see of a sample: no answer, no private metadata."""
    id: str
    question: str
    kind: str
    choices: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Sample:
    id: str
    question: str
    answer: str
    kind: str
    choices: dict[str, str] = field(default_factory=dict)   # {"A": "...", ...} for mcq
    images: tuple[str, ...] = ()                             # paths relative to the dataset dir in samples.jsonl
    video: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"{self.id}: unknown kind {self.kind}")
        if self.kind == "mcq" and self.answer not in self.choices:
            raise ValueError(f"{self.id}: answer {self.answer!r} is not an option")

    @classmethod
    def from_dict(cls, row: dict) -> Sample:
        return cls(**{**row, "images": tuple(row.get("images", ()))})

    def to_dict(self) -> dict:
        return asdict(self)

    def view(self, private: frozenset[str] | set[str] = frozenset()) -> SampleView:
        """The model-safe view: the answer is dropped and so are the metadata keys listed in `private`."""
        metadata = {k: v for k, v in self.metadata.items() if k not in private}
        return SampleView(self.id, self.question, self.kind, dict(self.choices), metadata)


@dataclass
class Request:
    """What a model adapter receives for one sample: the rendered prompt, media paths and the resolved settings."""
    id: str
    dataset: str
    prompt: str
    images: list[str] = field(default_factory=list)      # absolute paths
    video: str | None = None                             # absolute path
    settings: dict[str, Any] = field(default_factory=dict)   # resolved settings (the original runs' key layout)
    kind: str = "open"
    choices: dict[str, str] = field(default_factory=dict)
    sample: SampleView | None = None
    seed: int | None = None                              # defaults to seed_of(id)
    reference: str | None = None                         # the answer: filled in for the mock model only

    def __post_init__(self):
        if self.seed is None:
            self.seed = seed_of(self.id)


@dataclass
class Generation:
    """An adapter's result for one request. Latency is measured by the worker, not the adapter."""
    text: str | None
    tokens: int | None = None
    finish: str = "unknown"
    info: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict) -> Generation:
        """From a {"text", "tokens", "finish", "info"} dict (the adapters' return value in the paper's runs)."""
        return cls(value.get("text"), value.get("tokens"), value.get("finish") or "unknown", value.get("info") or {})

    def to_dict(self) -> dict:
        return {"text": self.text, "tokens": self.tokens, "finish": self.finish, "info": self.info}


def utc_now() -> str:
    """ISO-8601 UTC timestamp with seconds, e.g. 2026-01-31T12:00:00Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def generation_row(sample_id: str, prompt: str, generation: Generation | None, latency: float,
                   error: str | None = None, t: str | None = None) -> dict:
    """A generation row. Pass generation=None and the error message for a failed sample."""
    if generation is None or error is not None:
        return {"id": sample_id, "prompt": prompt, "raw": None, "tokens": None, "finish": None,
                "latency": latency, "info": None, "error": error or "unknown error", "t": t or utc_now()}
    return {"id": sample_id, "prompt": prompt, "raw": generation.text, "tokens": generation.tokens,
            "finish": generation.finish, "latency": latency, "info": generation.info, "error": None,
            "t": t or utc_now()}


def score_row(sample_id: str, pred: Any = None, parse: str | None = None, correct: bool | None = None,
              judge_pred: Any = None, judge_correct: bool | None = None, judge: dict | None = None) -> dict:
    """A score row; `judge` is {"name", "key"} of the judge request behind judge_pred/judge_correct, or None."""
    return {"id": sample_id, "pred": pred, "parse": parse, "correct": correct, "judge_pred": judge_pred,
            "judge_correct": judge_correct, "judge": judge}
