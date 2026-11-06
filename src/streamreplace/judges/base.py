"""Judge interface and the cache key shared with the original runs.

Four judges score what the rules cannot (configs/judges.yaml configures them; the implementations live next to this
file):
- extract        LLM answer extraction for rule-parse failures, the benchmarks' official chat extraction prompts
                 (TOMATO, TimeBlind, VMDT fallback) and VideoReasonBench's official judge;
- harmbench_cls  the official HarmBench classifiers (text and multimodal behaviours);
- rjscore        the Video-SafetyBench RJScore judge (also used for MCV-SafetyBench);
- lingo          the official Lingo-Judge (LingoQA, fine-tuning experiment).

Every judge result is cached under a key computed exactly as in the original runs, so the paper's judge caches (not
distributed) resolve every reported row and re-scoring never re-asks a judge: cache_key(fields) = first 16 hex digits
of the sha256 of json.dumps(fields, sort_keys=True, ensure_ascii=False) (default separators). The hashed field sets
are:
- extract:       {kind, question, choices, response, answer, meta}
                 (kind "extract" | "chat" | "vrb"; answer is "" except for VideoReasonBench requests)
- harmbench_cls: {classifier, num_tokens, behavior, generation}
- rjscore:       {judge, backend, goal, response, scenario} plus {policy} for MCV-SafetyBench
- lingo:         {judge, question, references, prediction}
`score` always applies cached results; new requests are sent only with --judge, and a failed request is not cached."""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from typing import Iterable, Protocol, runtime_checkable

from ..core.io import fingerprint

EXTRACT_KEY_FIELDS = ("kind", "question", "choices", "response", "answer", "meta")
HARMBENCH_KEY_FIELDS = ("classifier", "num_tokens", "behavior", "generation")
RJSCORE_KEY_FIELDS = ("judge", "backend", "goal", "response", "scenario")     # + "policy" (MCV-SafetyBench only)
LINGO_KEY_FIELDS = ("judge", "question", "references", "prediction")

JUDGES: dict[str, str] = {
    "extract": "streamreplace.judges.extraction:ExtractionJudge",
    "harmbench_cls": "streamreplace.judges.harmbench_cls:HarmBenchClassifier",
    "rjscore": "streamreplace.judges.rjscore:RJScoreJudge",
    "lingo": "streamreplace.judges.lingo:LingoJudge",
}


def cache_key(fields: dict) -> str:
    """The original runs' judge cache key (see the module docstring); byte-compatible by construction."""
    return fingerprint(fields)


@dataclass(frozen=True)
class JudgeRequest:
    judge: str                     # judge name (a key of JUDGES)
    key: str                       # cache_key() of exactly the fields the original runs hashed
    payload: dict = field(default_factory=dict)   # the hashed fields plus whatever the judge process needs


@runtime_checkable
class JudgeCache(Protocol):
    """Append-only result store of one judge configuration (judges/cache.py)."""

    def get(self, key: str) -> dict | None: ...

    def __contains__(self, key: str) -> bool: ...

    def add(self, results: Iterable[dict]) -> None: ...


@runtime_checkable
class Judge(Protocol):
    NAME: str                      # key of JUDGES
    ENV: str                       # python environment the judge process runs in

    def requests(self, bench, sample, gen: dict, score: dict) -> list[JudgeRequest]:
        """The judge requests one scored row needs (empty when the gating rules say none)."""

    def config_hash(self) -> str:
        """Hash of checkpoint repo + revision, backend and prompt version: names the cache file."""

    def run(self, reqs: list[JudgeRequest], cache: JudgeCache, gpus: list[int]) -> None:
        """Answers the requests missing from the cache (a subprocess in ENV) and appends the results."""

    def apply(self, bench, sample, score: dict, result: dict) -> dict:
        """The score row updated with a cached result."""

    def self_test(self, gpus: list[int]) -> None:
        """Refuses to judge when the judge disagrees with frozen reference results."""


def judge_names() -> list[str]:
    return list(JUDGES)


def load_judge_class(name: str) -> type:
    try:
        module, cls = JUDGES[name].split(":")
    except KeyError:
        raise ValueError(f"Unknown judge: {name}") from None
    return getattr(importlib.import_module(module), cls)
