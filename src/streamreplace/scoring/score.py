"""Scoring: generation rows -> rule layer -> judge requests -> cached judge answers -> `<job>.scores.jsonl`.

For every job file `<outputs>/<dataset>/<job>.jsonl` (generation rows are never modified) the scorer writes
`<job>.scores.jsonl`, one row per id (the last generation row per id wins), atomically:
{"id", "pred", "parse", "correct", "judge_pred", "judge_correct", "judge": {"name", "key"} | null}.

Rule layer (the paper's rules, in this order; ids outside the prepared test split and errored rows get no score):
  RJSCORE benchmarks   parse "no_answer" for a silent stream row, else "pending": RJScore judges every reply as is
  silent stream row    parse "no_answer", correct False, never parsed or judged (info.no_answer)
  CLASSIFIER           parse "pending" (the HarmBench classifiers decide)
  LINGO                pred = the benchmark's parse, parse "pending" (the Lingo-Judge decides)
  otherwise            (pred, parse) = the benchmark's parse of the raw reply; correct = its correct(sample, pred)
Judge layer: each benchmark's JUDGE policy names one judge (judges/registry.py); its requests are cached under keys
byte-identical to the original runs, in `<outputs>/judges/<judge>.<config_hash>.jsonl`. Cached answers are always
applied. New requests are sent only with run_judges=True (`score --judge`); `--cache-only` instead fails, writing
nothing, when any request has no cached answer. Under the ALWAYS policy (VideoReasonBench) a row without a final
answer is judged wrong without a request, as the official evaluation does.

Each job's lock file `<dataset>/.<job>.lock` is taken without waiting: a file a generation worker is still writing is
skipped and reported, and scored on the next pass. Given the pair membership (the command line passes
configs/pairs.yaml), the scorer first checks that the offline jobs of each pair sampled identical frames for every
row (the `info.frame_indices` digests): P and C must see the same frames."""
from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping

from ..benchmarks.base import JudgePolicy
from ..core.io import file_lock, latest, read_jsonl, write_jsonl
from ..core.protocol import Arm
from ..core.schema import score_row
from .outcome import final

NOT_DATASETS = frozenset({"judges", "logs", "analysis", "tables"})     # outputs folders that hold no job files


def rule_score(bench, sample, gen: dict) -> dict:
    """The rule layer for one generation row (module docstring); sample None = an id outside the prepared split."""
    score = score_row(gen["id"])
    if sample is None or gen.get("error"):
        return score
    info = gen.get("info") or {}
    policy = JudgePolicy.of(bench.JUDGE)
    if policy is JudgePolicy.RJSCORE:            # Video-/MCV-SafetyBench: every reply (silent ones as is) is judged
        score["parse"] = "no_answer" if info.get("no_answer") else "pending"
        return score
    if info.get("no_answer"):                    # stream: the model stayed silent; wrong, never parsed or judged
        score.update(parse="no_answer", correct=False)
        return score
    if policy is JudgePolicy.CLASSIFIER:         # HarmBench: unscored until the official classifier labels it
        score["parse"] = "pending"
        return score
    if policy is JudgePolicy.LINGO:              # LingoQA: the parsed reply, unscored until Lingo-Judge scores it
        score.update(pred=bench.parse(bench.view(sample), gen["raw"])[0], parse="pending")
        return score
    prediction, how = bench.parse(bench.view(sample), gen["raw"])
    score.update(pred=prediction, parse=how, correct=bench.correct(sample, prediction))
    return score


# ---------- job files ----------

@dataclass(frozen=True)
class JobFile:
    dataset: str
    job: str                      # '<model>[@<variant>].<mode>'
    path: Path                    # <outputs>/<dataset>/<job>.jsonl

    @property
    def scores_path(self) -> Path:
        return self.path.with_name(f"{self.job}.scores.jsonl")

    @property
    def lock_path(self) -> Path:
        """The job lock generation workers hold while appending (run/plan.lock_path)."""
        from ..run.plan import lock_path
        return lock_path(self.path)

    @property
    def arm(self) -> Arm:
        return Arm.parse(self.job)


def find_jobs(outputs: Path, benchmarks: Iterable[str] | None = None, models: Iterable[str] | None = None,
              modes: Iterable[str] | None = None, jobs: Iterable[str] | None = None) -> list[JobFile]:
    """Job files under `outputs`, filtered. A folder counts when it is a known benchmark; a file when its name is a
    job name. models matches the model id with or without its plugin variant ("aura_8b" or "aura_8b@sfix")."""
    from ..benchmarks import registry
    known = set(registry.names())
    benchmarks, models, modes, jobs = (set(x) if x else None for x in (benchmarks, models, modes, jobs))
    found = []
    for folder in sorted(p for p in Path(outputs).iterdir() if p.is_dir()):
        if folder.name in NOT_DATASETS or folder.name not in known or (benchmarks and folder.name not in benchmarks):
            continue
        for path in sorted(folder.glob("*.jsonl")):
            job = path.name[:-len(".jsonl")]
            try:
                arm = Arm.parse(job)
            except ValueError:                    # scores files and anything that is not a job
                continue
            names = {arm.model, f"{arm.model}@{arm.variant}" if arm.variant else arm.model}
            if (models and not names & models) or (modes and arm.mode.value not in modes) or (jobs and job not in jobs):
                continue
            found.append(JobFile(folder.name, job, path))
    return found


# ---------- scoring ----------

class CacheMiss(RuntimeError):
    """--cache-only and some judge requests have no cached answer (nothing was written)."""

    def __init__(self, report: "Report"):
        self.report = report
        super().__init__(f"{report.total_misses} judge requests without a cached answer: "
                         + ", ".join(f"{name} {n}" for name, n in report.misses.items() if n))


@dataclass
class Report:
    jobs: list[dict] = field(default_factory=list)          # per job: dataset, job, rows, counted, unjudged, correct
    skipped: list[str] = field(default_factory=list)        # job files busy with a generation worker
    requests: dict[str, int] = field(default_factory=dict)  # judge -> distinct requests
    misses: dict[str, int] = field(default_factory=dict)    # judge -> requests without a cached answer
    missed: dict[str, dict[str, int]] = field(default_factory=dict)   # judge -> {dataset/job: rows waiting}
    caches: dict[str, str] = field(default_factory=dict)    # judge -> cache file

    @property
    def total_misses(self) -> int:
        return sum(self.misses.values())


def _package_samples(dataset: str, outputs: Path, paths=None) -> list:
    """The prepared test split, restricted to the outputs folder's subset.json when it lists the benchmark (rows of
    other ids are then left unscored, as in the original runs)."""
    from ..data.samples import load_samples, read_subset
    return load_samples(dataset, subset=(read_subset(outputs) or {}).get(dataset), paths=paths)


def frame_mismatches(jobs: list[tuple[JobFile, dict]], pair_of: Mapping[str, str]) -> list[str]:
    """The offline jobs of one pair must have sampled the same frames: for every id both rows have, the
    `info.frame_indices` digests must be equal (P and C differ only in the checkpoint). jobs: (job file, id -> latest
    generation row); pair_of: model id -> pair id. Rows without a digest (images, errors) are not compared."""
    groups: dict[tuple, dict] = defaultdict(dict)          # (dataset, pair) -> model -> {id: digest}
    for job, gens in jobs:
        arm = job.arm
        pair = pair_of.get(arm.model)
        if arm.mode.value != "offline" or arm.variant or pair is None:
            continue
        groups[(job.dataset, pair)][arm.model] = {
            i: (g.get("info") or {}).get("frame_indices") for i, g in gens.items() if not g.get("error")}
    problems = []
    for (dataset, pair), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        for i in sorted(set.intersection(*(set(d) for d in members.values()))):
            digests = {m: d[i] for m, d in members.items() if d[i] is not None}
            if len(set(digests.values())) > 1:
                problems.append(f"{dataset} {pair} {i}: {digests}")
    return problems


def score(outputs: Path | None = None, *, benchmarks: Iterable[str] | None = None,
          models: Iterable[str] | None = None, modes: Iterable[str] | None = None, jobs: Iterable[str] | None = None,
          run_judges: bool = False, cache_only: bool = False, judge_gpus: list[int] | None = None,
          only_judges: Iterable[str] | None = None, selftest: bool = True,
          samples: Callable[[str], Iterable] | None = None, judges: dict | None = None,
          judge_section: dict | None = None, pair_of: Mapping[str, str] | None = None, paths=None,
          log: Callable[[str], None] | None = print) -> Report:
    """Scores the job files under `outputs` (default: the outputs root) and returns a Report (module docstring).

    run_judges   send requests without a cached answer to their judge (on judge_gpus; default: the judge's
                 DEFAULT_GPUS, [0, 1] for the 72B RJScore judge, [0] for the others); judges listed in only_judges
                 only, when given. A judge with SELFTEST_IN_RUN (RJScore) checks itself against frozen references
                 before judging, unless selftest=False
    cache_only   apply cached answers only and raise CacheMiss, writing nothing, if any request has none
    samples      dataset -> prepared samples (default: the prepared data of the data root, restricted to the
                 outputs folder's subset.json)
    judges       judge name -> judge object (default: built from configs/judges.yaml, or judge_section)
    pair_of      model id -> pair id: when given, offline jobs of one pair must agree on every row's frame digest
                 (frame_mismatches); a disagreement raises ValueError before anything is written"""
    from ..benchmarks.registry import get
    from ..judges.registry import build, judge_name
    if run_judges and cache_only:
        raise ValueError("--judge and --cache-only exclude each other")
    if outputs is None:
        from .. import paths as paths_module
        outputs = (paths or paths_module.current()).outputs
    outputs = Path(outputs)
    load = samples or (lambda dataset: _package_samples(dataset, outputs, paths))
    judges = dict(judges or {})
    report = Report()
    say = log or (lambda message: None)

    def judge_for(name: str):
        if name not in judges:
            judges[name] = build(name, paths=paths, section=judge_section)
        return judges[name]

    by_dataset: dict[str, dict] = {}
    with ExitStack() as locks:
        loaded = []
        for job in find_jobs(outputs, benchmarks, models, modes, jobs) if outputs.is_dir() else []:
            try:
                locks.enter_context(file_lock(job.lock_path, blocking=False))
            except BlockingIOError:
                report.skipped.append(f"{job.dataset}/{job.job}")
                continue
            bench = get(job.dataset)
            if job.dataset not in by_dataset:
                by_dataset[job.dataset] = {s.id: s for s in load(job.dataset)}
            known = by_dataset[job.dataset]
            gens = latest(read_jsonl(job.path))
            scores = {i: rule_score(bench, known.get(i), g) for i, g in gens.items()}
            loaded.append((job, bench, known, gens, scores))
        if pair_of:
            problems = frame_mismatches([(job, gens) for job, _, _, gens, _ in loaded], pair_of)
            if problems:
                raise ValueError(f"{len(problems)} rows where the offline jobs of a pair sampled different frames "
                                 "(P and C must see the same frames); first: " + "; ".join(problems[:5]))

        # judge requests (one judge per benchmark, named by its JUDGE policy)
        requests: dict[str, dict] = defaultdict(dict)          # judge -> key -> JudgeRequest
        links = []                                              # (file index, id, judge, keys)
        for index, (job, bench, known, gens, scores) in enumerate(loaded):
            name = judge_name(bench)
            if name is None:
                continue
            judge = judge_for(name)
            always = JudgePolicy.of(bench.JUDGE) is JudgePolicy.ALWAYS
            for i, gen in gens.items():
                sample = known.get(i)
                if sample is None:
                    continue
                reqs = judge.requests(bench, sample, gen, scores[i])
                for request in reqs:
                    requests[name][request.key] = request
                if reqs:
                    links.append((index, i, name, [r.key for r in reqs]))
                if always and scores[i]["correct"] is not None and scores[i]["pred"] is None:
                    scores[i]["judge_correct"] = False   # no final answer at all: wrong, as in the official evaluation

        caches = {}
        for name, reqs in requests.items():
            judge = judge_for(name)
            gpus = list(judge_gpus or getattr(judge, "DEFAULT_GPUS", (0,)))     # RJScore: 0,1; the others: 0
            cache = caches[name] = judge.cache(outputs)
            report.caches[name] = str(cache.path)
            missing = cache.missing(reqs)
            if missing and run_judges and (not only_judges or name in set(only_judges)):
                if getattr(judge, "SELFTEST_IN_RUN", False):       # RJScore: self-tested inside its judge process
                    judge.run([reqs[k] for k in missing], cache, gpus, selftest=selftest)
                else:
                    judge.run([reqs[k] for k in missing], cache, gpus)
                missing = cache.refresh().missing(reqs)
            report.requests[name], report.misses[name] = len(reqs), len(missing)
        if cache_only and report.total_misses:
            for index, i, name, keys in links:
                if any(k not in caches[name] for k in keys):
                    job = loaded[index][0]
                    waiting = report.missed.setdefault(name, {})
                    waiting[f"{job.dataset}/{job.job}"] = waiting.get(f"{job.dataset}/{job.job}", 0) + 1
            raise CacheMiss(report)

        # apply cached answers, then write the score files
        unjudged: dict[int, int] = defaultdict(int)
        for index, i, name, keys in links:
            job, bench, known, gens, scores = loaded[index]
            judge, cache, row = judges[name], caches[name], scores[i]
            row["judge"] = {"name": name, "key": keys[0]}
            for key in keys:
                result = cache.get(key)
                if result is None:
                    unjudged[index] += 1
                    continue
                row = judge.apply(bench, known[i], row, result)
            scores[i] = row
        for index, (job, bench, known, gens, scores) in enumerate(loaded):
            rows = [scores[i] for i in gens]
            write_jsonl(job.scores_path, rows)
            entry = {"dataset": job.dataset, "job": job.job, "rows": len(rows),
                     "counted": sum(bool(gens[r["id"]].get("error")) or r["correct"] is not None for r in rows),
                     "unjudged": unjudged[index], "correct": sum(bool(final(r)) for r in rows)}
            report.jobs.append(entry)
            say(f"{job.dataset}/{job.job}: {entry['correct']}/{entry['rows']} correct"
                + (f", {entry['unjudged']} waiting for a judge" if entry["unjudged"] else ""))
    for name in report.skipped:
        say(f"{name}: being written by a generation worker (skipped)")
    return report
