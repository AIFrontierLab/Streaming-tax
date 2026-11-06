"""Counted records of one job (dataset x model x mode): the only way the analysis reads model outputs.

A job lives in `<outputs>/<dataset>/<job>.jsonl` (generation rows, append-only) and `<job>.scores.jsonl` (one score
row per id). The rules are the ones every reported number was computed with (scoring/outcome.py holds them):

- the last generation row per id wins (retries append a new row); ids outside the prepared test split are ignored,
  and so are ids outside the folder's subset.json when the outputs folder has one (an experiment folder, e.g. the
  generation-cap reruns or a plugin item set);
- a row counts when it was scored (`correct` is not None) or when its generation errored;
- correctness = judge_correct if the judge decided, else the rule `correct`. An errored row is wrong, except for the
  safety judges (HarmBench classifier, RJScore) where a generation that never happened is not an attack success;
- stream silence was scored wrong without parsing (parse "no_answer"); RJScore judges silent replies as they are.

Flags per record: silent (parse "no_answer"), answered (a rule or judge prediction exists), judged (the rule parse
failed and the judge extracted the answer), truncated (finish "length"), forced (MMDuet's forced stream answer),
proj (the plugin gate's logged projection, info.plugin.guard.proj). Expected rows: every sample of the folder
offline, the samples with a video in stream mode.

Samples and benchmark hooks come from a `Catalog` (the package's prepared samples and benchmark registry by default;
tests and verification scripts pass their own)."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from ..benchmarks.base import SAFE_ON_ERROR, JudgePolicy
from ..core.io import latest, read_jsonl
from ..core.protocol import Arm
from ..core.schema import SCORE_KEYS
from ..scoring.outcome import counted, final, outcome

__all__ = ["Catalog", "Job", "Jobs", "Record", "counted", "expected_count", "final", "folder_subset", "job_files",
           "joined_rows", "judge_policy", "load_job", "make_record", "metric_rows", "outcome", "parse_job",
           "plugin_guard", "scoped_samples"]


# ---------- files ----------

def job_files(root: Path, dataset: str, job: str) -> tuple[Path, Path]:
    folder = Path(root) / dataset
    return folder / f"{job}.jsonl", folder / f"{job}.scores.jsonl"


def joined_rows(root: Path, dataset: str, job: str) -> dict[str, dict] | None:
    """id -> the latest generation row with its score fields (None when unscored); None when the job is absent."""
    gen_path, scores_path = job_files(root, dataset, job)
    if not gen_path.exists():
        return None
    scores = latest(read_jsonl(scores_path))
    out = {}
    for i, row in latest(read_jsonl(gen_path)).items():
        merged = dict(row)
        score = scores.get(i) or {}
        for key in SCORE_KEYS:
            merged[key] = score.get(key)
        out[i] = merged
    return out


def folder_subset(root: str | Path, dataset: str) -> list[str] | None:
    """The ids an experiment folder restricts `dataset` to (its subset.json), or None (no restriction)."""
    from ..data.samples import read_subset
    subset = read_subset(root)
    return None if subset is None else subset.get(dataset)


def parse_job(job: str) -> tuple[str, str | None, str]:
    """'<model>[@<variant>].<mode>' -> (model, variant, mode) (plugin variants as core.protocol.Arm reads them)."""
    arm = Arm.parse(job)
    return arm.model, arm.variant, arm.mode.value


# ---------- correctness ----------

def judge_policy(bench) -> JudgePolicy:
    """The benchmark's JUDGE policy (NONE when absent)."""
    return JudgePolicy.of(getattr(bench, "JUDGE", None))


def plugin_guard(row: dict) -> dict:
    """The plugin gate's log of a row: info.plugin.guard (shadow and live arms); the plugin's harmless-calibration
    rows carry it at the top level."""
    info = row.get("info") or {}
    return (info.get("plugin") or {}).get("guard") or row.get("guard") or {}


# ---------- samples and benchmark hooks ----------

def _package_samples(dataset: str) -> list:
    from ..data.samples import load_samples          # prepared test split of the package (configs + data root)
    return list(load_samples(dataset))


def _package_benchmark(dataset: str):
    from ..benchmarks.registry import get
    return get(dataset)


class Catalog:
    """Prepared samples (id -> Sample with .kind, .video, .metadata, .answer, .choices) and benchmark modules
    (JUDGE, metrics(), stratum(), LOWER_IS_BETTER; VMDT: scores()), each loaded once. The samples are the whole
    prepared test split; an outputs folder's subset.json narrows a job to part of it (scoped_samples)."""

    def __init__(self, samples: Callable[[str], list] | None = None, benchmark: Callable[[str], Any] | None = None):
        self._load_samples = samples or _package_samples
        self._load_benchmark = benchmark or _package_benchmark
        self._samples: dict[str, dict] = {}
        self._benchmarks: dict[str, Any] = {}

    def samples(self, dataset: str) -> dict:
        if dataset not in self._samples:
            self._samples[dataset] = {s.id: s for s in self._load_samples(dataset)}
        return self._samples[dataset]

    def bench(self, dataset: str):
        if dataset not in self._benchmarks:
            self._benchmarks[dataset] = self._load_benchmark(dataset)
        return self._benchmarks[dataset]

    def policy(self, dataset: str) -> JudgePolicy:
        return judge_policy(self.bench(dataset))

    def judge(self, dataset: str) -> str:
        """The judge policy as a lower-case string ('none', 'fallback', 'always', 'classifier', 'rjscore', ...)."""
        return self.policy(dataset).value

    def safe_on_error(self, dataset: str) -> bool:
        return self.policy(dataset) in SAFE_ON_ERROR

    def stratum(self, dataset: str, sample) -> str | None:
        hook = getattr(self.bench(dataset), "stratum", None)
        return hook(sample) if hook else None

    def metrics(self, dataset: str, rows: list[dict]) -> dict:
        return self.bench(dataset).metrics(rows)

    def lower_is_better(self, dataset: str) -> bool:
        return bool(getattr(self.bench(dataset), "LOWER_IS_BETTER", False))


def scoped_samples(catalog: Catalog, root: str | Path, dataset: str) -> dict:
    """id -> sample of the prepared samples a job in outputs folder `root` covers: all of them, or the folder's
    subset.json ids (unknown ids ignored), in prepared order."""
    samples = catalog.samples(dataset)
    subset = folder_subset(root, dataset)
    if subset is None:
        return samples
    keep = set(subset)
    return {i: s for i, s in samples.items() if i in keep}


# ---------- records ----------

@dataclass(frozen=True)
class Record:
    id: str
    correct: bool
    error: bool
    silent: bool
    answered: bool
    judged: bool
    truncated: bool
    forced: bool
    pred: Any                      # final prediction: judge_pred if the judge extracted one, else the rule prediction
    parse: str | None
    proj: float | None = None      # plugin gate projection logged in info.plugin.guard.proj (shadow arms)
    rule_pred: Any = None          # the score row's fields, kept for the benchmarks' metric rows
    judge_pred: Any = None
    finish: str | None = None
    info: dict = field(default_factory=dict, compare=False)


def make_record(row: dict, policy) -> Record:
    """policy: the benchmark's JudgePolicy (or the benchmark itself); it decides how an errored row counts."""
    info = row.get("info") or {}
    pred, judge_pred = row.get("pred"), row.get("judge_pred")
    return Record(id=row["id"], correct=outcome(policy, row), error=bool(row.get("error")),
                  silent=row.get("parse") == "no_answer", answered=pred is not None or judge_pred is not None,
                  judged=pred is None and judge_pred is not None, truncated=row.get("finish") == "length",
                  forced=bool(info.get("forced")), pred=judge_pred if judge_pred is not None else pred,
                  parse=row.get("parse"), proj=plugin_guard(row).get("proj"), rule_pred=pred,
                  judge_pred=judge_pred, finish=row.get("finish"), info=info)


@dataclass
class Job:
    dataset: str
    job: str                       # '<model>[@<variant>].<mode>'
    rows: dict                     # id -> Record (counted rows only), in first-appearance order
    expected: int                  # samples the job should have (stream: samples with a video)
    path: Path | None = None
    tag: str = ""                  # names the overlays applied (a derived job is a different job in caches)

    @property
    def model(self) -> str:
        return parse_job(self.job)[0]

    @property
    def variant(self) -> str | None:
        return parse_job(self.job)[1]

    @property
    def mode(self) -> str:
        return parse_job(self.job)[2]

    @property
    def complete(self) -> bool:
        return len(self.rows) == self.expected

    @property
    def uid(self) -> str:
        return f"{self.path}#{self.tag}"

    def with_overlays(self, *overlays) -> "Job":
        """A new job with each overlay (analysis.arms: Replace, ForceUnanswered, Restrict) applied in order."""
        job = self
        for overlay in overlays:
            job = replace(job, rows=overlay.apply(job.rows), tag=f"{job.tag}+{overlay.label}")
        return job


def expected_count(samples: dict, mode: str) -> int:
    return sum(1 for s in samples.values() if mode == "offline" or s.video)


def load_job(root: str | Path, dataset: str, job: str, catalog: Catalog) -> Job | None:
    """The job's counted records, or None when its generation file does not exist. In a folder with a subset.json
    only the subset's ids count (and are expected)."""
    rows = joined_rows(Path(root), dataset, job)
    if rows is None:
        return None
    smp = scoped_samples(catalog, root, dataset)
    policy = catalog.policy(dataset)
    records = {i: make_record(r, policy) for i, r in rows.items() if i in smp and counted(r)}
    return Job(dataset, job, records, expected_count(smp, parse_job(job)[2]), job_files(Path(root), dataset, job)[0])


def metric_rows(job: Job, catalog: Catalog) -> list[dict]:
    """The rows a benchmark's metrics() / scores() read (final scoring), in the job's order."""
    smp = catalog.samples(job.dataset)
    return [{"id": i, "correct": r.correct, "kind": smp[i].kind, "answered": r.answered, "pred": r.rule_pred,
             "judge_pred": r.judge_pred, "parse": r.parse, "silent": r.silent,
             "metadata": smp[i].metadata, "video": smp[i].video} for i, r in job.rows.items()]


class Jobs:
    """Loads each (root, dataset, job) once."""

    def __init__(self, root: str | Path, catalog: Catalog):
        self.root, self.catalog, self.cache = Path(root), catalog, {}

    def __call__(self, dataset: str, model: str, mode: str, root: str | Path | None = None) -> Job | None:
        key = (str(root or self.root), dataset, f"{model}.{mode}")
        if key not in self.cache:
            self.cache[key] = load_job(Path(key[0]), dataset, key[2], self.catalog)
        return self.cache[key]
