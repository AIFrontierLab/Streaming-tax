"""Benchmark interface: one module per benchmark (benchmarks/<group>/<name>.py), listed in benchmarks/registry.py.

A benchmark module defines module-level constants and functions (the `Benchmark` protocol below). Only MEDIA and
prepare() are required; prompt / parse / correct / metrics / smoke default to the shared implementations in
benchmarks/common (the ones the paper used for every benchmark that does not override them), JUDGE defaults to
"fallback" (the LLM judge extracts an answer only when the rules could not parse one).

`registry.get(name)` returns a `BenchmarkModule`: the module with those defaults filled in. It exposes the module
constants (NAME, MEDIA, JUDGE, LOWER_IS_BETTER, METRICS, PRIVATE_METADATA, EXPECTED_COUNT), the hooks, and any other
module attribute (e.g. VMDT's scores(), Video-SafetyBench's THRESHOLD) by attribute access.

Parse status strings ("letter", "no_marker", "ambiguous_choice", "official_invalid", "no_answer", ...) are kept
verbatim for compatibility with the original outputs; judge gating uses parse_class()."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Protocol

from ..core.protocol import _StrEnum
from ..core.schema import Sample, SampleView
from .common.metrics import metrics as _default_metrics
from .common.parsing import correct as _default_correct
from .common.parsing import parse as _default_parse
from .common.prompts import prompt as _default_prompt
from .common.smoke import smoke as _default_smoke


class JudgePolicy(_StrEnum):
    NONE = "none"                 # rules only (e.g. Video-MME's official regex, VMDT privacy)
    FALLBACK = "fallback"         # the LLM judge extracts an answer only when the rule parse failed
    ALWAYS = "always"             # the judge decides correctness of every parsed answer (VideoReasonBench)
    CLASSIFIER = "classifier"     # scored only by the official HarmBench classifiers
    RJSCORE = "rjscore"           # scored only by the Video-SafetyBench RJScore judge (also MCV-SafetyBench)
    LINGO = "lingo"               # scored only by the official Lingo-Judge (LingoQA, fine-tuning experiment)

    @classmethod
    def of(cls, value) -> JudgePolicy:
        """From a module's JUDGE constant: None means NONE, strings are case-insensitive."""
        if value is None:
            return cls.NONE
        return value if isinstance(value, cls) else cls(str(value).lower())


# Under these judges an errored generation counts as safe (a reply that never happened is not an attack success);
# under every other policy an errored row counts as wrong.
SAFE_ON_ERROR = frozenset({JudgePolicy.CLASSIFIER, JudgePolicy.RJSCORE})


class ParseClass(_StrEnum):
    SILENT = "silent"             # parse "no_answer": the stream stayed silent (or an official NA reply); never judged
    PENDING = "pending"           # awaiting a judge-only scorer (classifier, RJScore, Lingo-Judge)
    UNPARSED = "unparsed"         # the rules found no answer: judged under the FALLBACK policy
    PARSED = "parsed"


def parse_class(status: str | None, pred: Any) -> ParseClass:
    if status == "no_answer":
        return ParseClass.SILENT
    if status == "pending":
        return ParseClass.PENDING
    return ParseClass.UNPARSED if pred is None else ParseClass.PARSED


@dataclass(frozen=True)
class MetricSpec:
    """Unit structure of one reported metric for the paired statistics (analysis/specs.py)."""
    name: str
    kind: Literal["acc", "group", "mme", "safe", "vmdt"]
    cluster: str | None = None             # metadata key | "video" (= Sample.video) | None (= the item)
    group: str | None = None               # all-correct group metrics: the group key ...
    size: int | str | None = None          # ... and its size (int, or a metadata key holding it)
    split: str | None = None               # keep the items whose stratum() equals this
    scale: float = 1.0
    lower_is_better: bool = False
    modes: tuple[str, ...] = ("offline", "stream")


class Benchmark(Protocol):
    """What a benchmark module may define (module level). Only MEDIA and prepare() are required."""
    NAME: str
    MEDIA: Literal["image", "video"]
    JUDGE: JudgePolicy | str | None        # default "fallback"
    EXPECTED_COUNT: int                    # prepared test-set size, checked by prepare
    LOWER_IS_BETTER: bool                  # headline is an attack success rate
    METRICS: list[MetricSpec]              # first = headline
    PRIVATE_METADATA: frozenset[str]       # metadata keys models must not see (ground truth, references)

    def prepare(self, raw: Path, media: Path) -> list[Sample]: ...
    def prompt(self, sample: SampleView) -> str: ...
    def parse(self, sample: SampleView, text: str) -> tuple[Any, str]: ...      # (pred, status); never uses the answer
    def correct(self, sample: Sample, pred) -> bool: ...
    def judge_messages(self, sample: Sample, raw: str) -> tuple[list[dict], int] | None: ...
    def judge_result(self, sample: Sample, reply: str) -> Any: ...
    def metrics(self, rows: list[dict]) -> dict: ...                           # first key = headline, % rounded to 2
    def stratum(self, sample: Sample) -> str | None: ...
    def smoke(self, samples: list[Sample]) -> list[str]: ...                   # deterministic whole-group order


_DEFAULT_HOOKS = {"prompt": _default_prompt, "parse": _default_parse, "correct": _default_correct,
                  "metrics": _default_metrics, "smoke": _default_smoke}


class BenchmarkModule:
    """A benchmark module with the shared defaults filled in (see the module docstring)."""

    def __init__(self, name: str, module: ModuleType):
        self.name, self.module = name, module
        self.NAME = getattr(module, "NAME", name)
        self.MEDIA = module.MEDIA
        if self.MEDIA not in {"image", "video"}:
            raise ValueError(f"{name}: MEDIA must be 'image' or 'video', not {self.MEDIA!r}")
        self.JUDGE = JudgePolicy.of(getattr(module, "JUDGE", "fallback"))
        self.LOWER_IS_BETTER = bool(getattr(module, "LOWER_IS_BETTER", False))
        self.METRICS = list(getattr(module, "METRICS", []))
        self.PRIVATE_METADATA = frozenset(getattr(module, "PRIVATE_METADATA", ()))
        self.EXPECTED_COUNT = getattr(module, "EXPECTED_COUNT", None)

    def __repr__(self) -> str:
        return f"BenchmarkModule({self.name!r}, {self.module.__name__})"

    def __getattr__(self, name: str):
        if name == "module":                   # not yet set (copy/pickle): avoid infinite recursion
            raise AttributeError(name)
        return getattr(self.module, name)

    # lower-case aliases of MEDIA and JUDGE
    @property
    def media(self) -> str:
        return self.MEDIA

    @property
    def judge(self) -> JudgePolicy:
        return self.JUDGE

    @property
    def lower_is_better(self) -> bool:
        return self.LOWER_IS_BETTER

    @property
    def safe_on_error(self) -> bool:
        return self.JUDGE in SAFE_ON_ERROR

    def has(self, name: str) -> bool:
        return hasattr(self.module, name)

    def _hook(self, name: str):
        return getattr(self.module, name, None) or _DEFAULT_HOOKS[name]

    def view(self, sample: Sample) -> SampleView:
        return sample.view(self.PRIVATE_METADATA)

    def prepare(self, raw: Path, media: Path) -> list[Sample]:
        return self.module.prepare(raw, media)

    @property
    def has_train(self) -> bool:
        return hasattr(self.module, "prepare_train")

    def prepare_train(self, raw: Path, media: Path) -> list[Sample]:
        return self.module.prepare_train(raw, media)

    def prompt(self, sample) -> str:
        return self._hook("prompt")(sample)

    def parse(self, sample, text):
        return self._hook("parse")(sample, text)

    def correct(self, sample, prediction) -> bool:
        return self._hook("correct")(sample, prediction)

    def metrics(self, rows) -> dict:
        """rows: counted samples, each {"id", "correct", "kind", "metadata", ...}; first key = headline metric."""
        return self._hook("metrics")(rows)

    def smoke(self, samples) -> list[str]:
        return self._hook("smoke")(samples)

    def stratum(self, sample) -> str | None:
        """Optional: the group a sample's paired statistics are reported in (e.g. a query split), or None."""
        hook = getattr(self.module, "stratum", None)
        return hook(sample) if hook else None

    def judge_messages(self, sample, text):
        """Optional: the benchmark's official LLM-extraction prompt for a response its rules could not parse ->
        (chat messages, max_tokens), or None to use the generic extraction request."""
        hook = getattr(self.module, "judge_messages", None)
        return hook(sample, text) if hook else None

    def judge_result(self, sample, reply: str):
        """Prediction from the judge's raw reply to judge_messages (the official post-processing)."""
        return self.module.judge_result(sample, reply)
