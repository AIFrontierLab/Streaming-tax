"""Running planned jobs (run/plan.py): resume checks, then one worker process per (arm, shard) on a pool of GPU slots.

check() before anything runs:
- a job whose meta.json was written under another run identity (settings, checkpoint, adapter, plugin, subset, prompt
  overrides or --limit changed) is refused, and so is one whose stored rows carry a prompt that differs from the
  prompt rendered now (benchmark prompt code or the folder's overrides changed);
- with fresh=True exactly those jobs are archived (generation rows, meta.json and scores move to
  `<outputs>/_archive/<UTC time>/<benchmark>/`) and start over; outputs with unchanged identity and prompts always
  resume: samples that have a row are kept, samples whose last row is an error are retried;
- metas are written only after every job passed, so a refused run leaves no new metas behind.

run(): jobs with samples left are grouped per arm into worker processes; an arm's benchmarks share one process only
when the adapter's load-time settings (LOAD_KEYS) agree, because a process loads its model once, with the settings of
its first job. With shards=n each group is split into n processes, shard k taking samples[k::n] of each job, all
appending to the same output files. Processes are queued shard-major (every arm makes progress) and take GPU slots
from `gpus`: one slot per process, and repeating a GPU id packs several processes on one card (`--gpus 0,0,1,1`).
Each process runs `<env python> -m streamreplace.run.worker <payload.json>` (run/envs.py sets its environment) and
logs to `<outputs>/logs/<job>[.part<i>][.shard<k>].log`; the payload file next to the log lets one rerun a worker by
hand. The returned status maps each job (`<benchmark>/<job>`) to "complete", "<n> left (see logs/...)" or the reason
it could not start.

Requests: the dispatcher renders every request (prompt, absolute media paths, the sample's model-safe view) into the
payload, so a worker needs neither the configs nor the benchmark modules. The reference answer is included for the
test-only mock adapter and for no other."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from typing import TYPE_CHECKING, Callable, Iterable

from ..core.identity import new_meta, read_meta, write_meta
from ..core.io import atomic_text, file_digest, fingerprint, read_jsonl, read_latest
from ..core.schema import utc_now
from ..data.prompts import render_prompts
from ..data.samples import load_samples, samples_file
from .envs import default_threads, python_for, worker_env, worker_vars
from .plan import Job, spec_dict

if TYPE_CHECKING:
    from ..paths import Paths

ARCHIVE = "_archive"
LOGS = "logs"
LAUNCHERS = ("local", "print")


def _current_paths(paths: Paths | None) -> Paths:
    if paths is not None:
        return paths
    from .. import paths as paths_module
    return paths_module.current()


class Inputs:
    """Samples, prompts and requests of planned jobs, loaded once per (benchmark, subset, limit, overrides)."""

    def __init__(self, paths: Paths | None = None):
        self.paths = _current_paths(paths)
        self._samples: dict = {}
        self._prompts: dict = {}
        self._digests: dict = {}

    def bench(self, job: Job):
        from ..benchmarks import registry
        bench = registry.get(job.dataset)
        if bench.MEDIA != job.media:
            raise ValueError(f"{job.dataset}: configs/benchmarks.yaml says media {job.media!r}, the benchmark module "
                             f"says {bench.MEDIA!r}")
        return bench

    def _key(self, job: Job) -> tuple:
        return job.dataset, None if job.subset is None else tuple(job.subset), job.limit

    def samples(self, job: Job) -> list:
        """The job's selection of samples (after subset and limit), all media kinds."""
        key = self._key(job)
        if key not in self._samples:
            self._samples[key] = load_samples(job.dataset, subset=job.subset, limit=job.limit, paths=self.paths,
                                              bench=self.bench(job))
        return self._samples[key]

    def job_samples(self, job: Job) -> list:
        """The samples the job generates: stream jobs keep only the samples with a video."""
        samples = self.samples(job)
        return [s for s in samples if s.video] if job.mode == "stream" else samples

    def prompts(self, job: Job) -> dict[str, str]:
        key = self._key(job) + (None if job.edits is None else fingerprint(job.edits),)
        if key not in self._prompts:
            self._prompts[key] = render_prompts(job.dataset, self.samples(job), job.edits, bench=self.bench(job))
        return self._prompts[key]

    def requests(self, job: Job) -> list[dict]:
        """What the worker turns into core.schema.Request objects, in prepared order."""
        bench, prompts = self.bench(job), self.prompts(job)
        # video.drop (offline controls of the framing experiment): the request text without its frames
        drop = job.mode == "offline" and bool((job.settings.get("video") or {}).get("drop"))
        out = []
        for sample in self.job_samples(job):
            view = bench.view(sample)
            item = {"id": sample.id, "prompt": prompts[sample.id], "images": list(sample.images),
                    "video": None if drop else sample.video, "kind": sample.kind, "question": view.question,
                    "choices": dict(view.choices), "metadata": view.metadata}
            if job.adapter == "mock":            # the test model scores itself; real models never see the answer
                item["reference"] = sample.answer
            out.append(item)
        return out

    def samples_sha256(self, job: Job) -> str:
        if job.dataset not in self._digests:
            self._digests[job.dataset] = file_digest(samples_file(job.dataset, paths=self.paths))
        return self._digests[job.dataset]


# ---------- resume checks ----------

def _short(value) -> str:
    text = "(absent)" if value is _ABSENT else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return text if len(text) <= 60 else text[:57] + "..."


_ABSENT = object()


def payload_differences(old, new, path: str = "") -> list[str]:
    """Dotted paths where two identity payloads differ, e.g. "settings.generation.max_new_tokens: 16 -> 5"."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for key in sorted(set(old) | set(new), key=str):
            out += payload_differences(old.get(key, _ABSENT), new.get(key, _ABSENT), f"{path}.{key}" if path else key)
        return out
    return [] if old == new else [f"{path or 'payload'}: {_short(old)} -> {_short(new)}"]


def identity_change(meta: dict, job: Job) -> str:
    stored = meta.get("payload")
    if not isinstance(stored, dict):
        return f"stored identity {meta.get('identity')} has no payload to compare"
    diffs = payload_differences(stored, job.payload)
    return "; ".join(diffs[:3]) + (f"; ... ({len(diffs)} differences)" if len(diffs) > 3 else "")


def drifted_rows(job: Job, prompts: dict[str, str]) -> list[str]:
    """Ids of stored rows whose prompt differs from the prompt rendered now (rows outside the selection ignored)."""
    return [row["id"] for row in read_jsonl(job.out)
            if row["id"] in prompts and row.get("prompt") != prompts[row["id"]]]


def archive(job: Job, stamp: str) -> Path:
    """Moves the job's generation rows, meta.json and scores to <outputs>/_archive/<stamp>/<benchmark>/."""
    dest = job.outputs / ARCHIVE / stamp / job.dataset
    dest.mkdir(parents=True, exist_ok=True)
    for path in (job.out, job.meta, job.scores):
        if path.exists():
            shutil.move(str(path), str(dest / path.name))
    return dest


def check(jobs: list[Job], inputs: Inputs, fresh: bool = False, log: Callable[[str], None] = print) -> list[str]:
    """Refuses jobs whose outputs were made under another identity or with other prompts (fresh=True archives them);
    then writes the metas of jobs that have none. Returns the names of the archived jobs."""
    unknown = [f"{job.name} (adapter {job.adapter})" for job in jobs if job.constants is None]
    if unknown:
        raise ValueError("cannot determine PROTOCOL_VERSION / LOAD_KEYS (adapter module missing?) for:\n  "
                         + "\n  ".join(unknown))
    conflicts: list[tuple[Job, str]] = []
    for job in jobs:
        prompts = inputs.prompts(job)       # every prompt is rendered (and the folder's overrides applied) up front
        meta = read_meta(job.meta)
        if meta is not None and meta.get("identity") != job.identity:
            conflicts.append((job, f"identity changed: {identity_change(meta, job)}"))
        elif job.out.exists():
            drift = drifted_rows(job, prompts)
            if drift:
                conflicts.append((job, f"prompt changed for {len(drift)} stored rows, e.g. {drift[0]}"))
    if conflicts and not fresh:
        raise ValueError("existing outputs were generated under another run identity or with other prompts; rerun "
                         f"with --fresh to archive them to <outputs>/{ARCHIVE}/ and start over:\n  "
                         + "\n  ".join(f"{job.name} ({reason})" for job, reason in conflicts))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for job, reason in conflicts:
        dest = archive(job, stamp)
        log(f"[fresh] {job.name} ({reason}) -> {dest}")
    for job in jobs:
        if not job.meta.exists():
            job.meta.parent.mkdir(parents=True, exist_ok=True)
            write_meta(job.meta, new_meta(job.payload))
    return [job.name for job, _ in conflicts]


def remaining(job: Job, inputs: Inputs) -> int:
    """Samples of the job without a row, or whose last row is an error (those are retried)."""
    rows = read_latest(job.out)
    return sum(1 for s in inputs.job_samples(job) if s.id not in rows or rows[s.id].get("error"))


def job_state(job: Job, inputs: Inputs) -> dict:
    """Read-only progress of one job: {"name", "state", "expected", "done", "errors", "left", "detail"}; state is new,
    partial, complete, changed (identity), drift (prompts), unprepared or unavailable (adapter)."""
    out = {"name": job.name, "state": "", "expected": None, "done": None, "errors": None, "left": None, "detail": ""}
    if job.constants is None:
        out.update(state="unavailable", detail=f"adapter {job.adapter}: module missing or constants unknown")
        return out
    try:
        samples, prompts = inputs.job_samples(job), inputs.prompts(job)
    except Exception as exc:                 # noqa: BLE001 - data not prepared, benchmark module missing or broken
        out.update(state="unprepared", detail=f"{type(exc).__name__}: {exc}")
        return out
    rows = read_latest(job.out)
    ids = [s.id for s in samples]
    done = sum(1 for i in ids if i in rows and not rows[i].get("error"))
    errors = sum(1 for i in ids if i in rows and rows[i].get("error"))
    out.update(expected=len(ids), done=done, errors=errors, left=len(ids) - done)
    meta = read_meta(job.meta)
    if meta is not None and meta.get("identity") != job.identity:
        out.update(state="changed", detail=identity_change(meta, job))
    elif job.out.exists() and drifted_rows(job, prompts):
        out.update(state="drift", detail="stored prompts differ from the rendered prompts")
    elif meta is None and not rows:
        out["state"] = "new"
    else:
        out["state"] = "complete" if out["left"] == 0 else "partial"
    return out


# ---------- worker processes ----------

@dataclass
class Unit:
    """One worker process: jobs of one arm that share load-time settings, one shard of each."""
    jobs: list[Job]
    shard: int
    shards: int
    part: int
    parts: int

    @property
    def first(self) -> Job:
        return self.jobs[0]

    @property
    def base(self) -> str:
        return self.first.job_name + (f".part{self.part + 1}" if self.parts > 1 else "")

    @property
    def name(self) -> str:
        return self.base + (f".shard{self.shard}" if self.shards > 1 else "")

    @property
    def group(self) -> tuple:
        return group_key(self.first)


def group_key(job: Job) -> tuple:
    return job.arm, job.adapter, job.env, job.load_values


def make_units(jobs: Iterable[Job], shards: int = 1) -> list[Unit]:
    """Groups jobs per arm and load-time settings (first-appearance order), then one unit per (group, shard),
    shard-major."""
    groups: dict[tuple, list[Job]] = {}
    for job in jobs:
        groups.setdefault(group_key(job), []).append(job)
    parts = Counter(key[0] for key in groups)
    index: dict[tuple, int] = {}
    seen: Counter = Counter()
    for key in groups:
        index[key] = seen[key[0]]
        seen[key[0]] += 1
    return [Unit(groups[key], shard, shards, index[key], parts[key[0]]) for shard in range(shards) for key in groups]


def upstream_manifest_sha256() -> str | None:
    from ..paths import project_root
    path = project_root() / "manifests" / "upstream.yaml"
    return file_digest(path) if path.is_file() else None


def unit_payload(unit: Unit, inputs: Inputs) -> dict:
    """The worker's input file: the model, the adapter and its constants, and each job's settings and requests (this
    shard's slice, samples[k::n] in prepared order)."""
    first, paths = unit.first, inputs.paths
    k, n = unit.shard, unit.shards
    return {
        "version": 1,
        "unit": unit.name,
        "arm": {"model": first.arm.model, "mode": first.mode, "variant": first.arm.variant},
        "model": {"id": first.spec.id, "spec": spec_dict(first.spec, first.lora), "pair": first.spec.pair},
        "adapter": first.adapter,
        "env": first.env,
        "constants": first.constants.to_dict(),
        "deterministic": first.deterministic,
        "plugin": first.plugin,
        "plugin_identity": first.payload["model"].get("plugin"),
        "shard": [k, n],
        "paths": {"checkpoint_dirs": {k2: str(v) for k2, v in paths.checkpoint_dirs.items()},
                  "dataset_dirs": {k2: str(v) for k2, v in paths.dataset_dirs.items()}},
        "upstream_manifest_sha256": upstream_manifest_sha256(),
        "jobs": [{"dataset": job.dataset, "job": job.job_name, "out": str(job.out), "meta": str(job.meta),
                  "identity": job.identity, "settings": job.settings, "samples_sha256": inputs.samples_sha256(job),
                  "requests": inputs.requests(job)[k::n]} for job in unit.jobs],
    }


def preflight(unit: Unit, paths: Paths) -> str | None:
    """Why the unit's worker cannot start here (None when it can)."""
    from ..models import registry
    python = python_for(paths, unit.first.env)
    if not python.exists():
        return f"missing env: {python} (build it: streamreplace env build {unit.first.env})"
    if not registry.available(unit.first.adapter):
        return f"adapter module missing: {registry.target(unit.first.adapter)[0]}"
    return None


def write_payload(unit: Unit, inputs: Inputs) -> Path:
    logs = unit.first.outputs / LOGS
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / f"{unit.name}.payload.json"
    atomic_text(path, json.dumps(unit_payload(unit, inputs), ensure_ascii=False))
    return path


def worker_command(python: Path, payload: Path) -> list[str]:
    return [str(python), "-m", "streamreplace.run.worker", str(payload)]


def run(jobs: list[Job], gpus: Iterable[int], shards: int = 1, threads: int | None = None, fresh: bool = False,
        launcher: str = "local", paths: Paths | None = None, inputs: Inputs | None = None,
        log: Callable[[str], None] = print, poll: float = 0.2) -> dict[str, str]:
    """Checks, then generates every planned job's missing rows; returns {"<benchmark>/<job>": status}."""
    if launcher not in LAUNCHERS:
        raise ValueError(f"unknown launcher {launcher!r}; expected one of {', '.join(LAUNCHERS)}")
    if not jobs:
        return {}
    paths = _current_paths(paths)
    inputs = inputs or Inputs(paths)
    gpus = list(gpus)
    if not gpus:
        raise ValueError("no GPU slots: pass --gpus")
    shards = max(1, int(shards))
    check(jobs, inputs, fresh, log)
    status: dict[str, str] = {}
    todo = []
    for job in jobs:
        if remaining(job, inputs):
            todo.append(job)
        else:
            status[job.name] = "complete"
    units = make_units(todo, shards)
    threads = threads or default_threads(len(gpus))
    log(f"{len(jobs)} jobs ({len(todo)} to run) in {len(units)} worker processes on GPU slots {gpus}")
    runnable = []
    for unit in units:
        problem = preflight(unit, paths)
        if problem is None:
            runnable.append(unit)
        else:
            for job in unit.jobs:
                status[job.name] = problem
    if launcher == "print":
        return _print_commands(runnable, gpus, threads, inputs, paths, status, log)
    from ..paths import project_root
    shards_left = Counter(unit.group for unit in runnable)
    queue, free, active = deque(runnable), list(gpus), {}
    while queue or active:
        while queue and free:
            unit, slot = queue.popleft(), free.pop(0)
            first = unit.first
            python = python_for(paths, first.env)
            payload = write_payload(unit, inputs)
            handle = (first.outputs / LOGS / f"{unit.name}.log").open("a", encoding="utf-8")
            handle.write(f"\n==== {utc_now()} gpu={slot} env={first.env} python={python} "
                         f"benchmarks={[job.dataset for job in unit.jobs]} shard={unit.shard}/{unit.shards}\n")
            handle.flush()
            process = subprocess.Popen(worker_command(python, payload), stdout=handle, stderr=subprocess.STDOUT,
                                       env=worker_env(paths, first.env, slot, threads, first.ffmpeg,
                                                      first.deterministic),
                                       cwd=str(project_root()))
            active[process] = (unit, slot, handle, time.time())
            log(f"[start] {unit.name} on gpu {slot}: {', '.join(job.dataset for job in unit.jobs)}")
        time.sleep(poll)
        for process in [p for p in active if p.poll() is not None]:
            unit, slot, handle, began = active.pop(process)
            handle.close()
            free.append(slot)
            shards_left[unit.group] -= 1
            minutes = (time.time() - began) / 60
            if shards_left[unit.group] > 0:
                log(f"[done]  {unit.name} ({minutes:.1f} min); {shards_left[unit.group]} shard(s) of {unit.base} left")
                continue
            logname = f"{LOGS}/{unit.base}{'.shard*' if unit.shards > 1 else ''}.log"
            for job in unit.jobs:
                left = remaining(job, inputs)
                status[job.name] = "complete" if left == 0 else f"{left} left (see {logname})"
            complete = sum(status[job.name] == "complete" for job in unit.jobs)
            log(f"[done]  {unit.base}: {complete}/{len(unit.jobs)} benchmarks complete ({minutes:.1f} min)")
            for job in unit.jobs:
                if status[job.name] != "complete":
                    log(f"        {job.name}: {status[job.name]}")
    return status


def _print_commands(units: list[Unit], gpus: list[int], threads: int, inputs: Inputs, paths: Paths,
                    status: dict[str, str], log: Callable[[str], None]) -> dict[str, str]:
    """The `print` launcher: writes the payloads and prints one shell command per worker (e.g. for a batch
    scheduler) instead of running them; GPU slots are assigned round-robin."""
    from ..paths import project_root
    for number, unit in enumerate(units):
        first, slot = unit.first, gpus[number % len(gpus)]
        payload = write_payload(unit, inputs)
        variables = worker_vars(paths, first.env, slot, threads, first.ffmpeg, first.deterministic)
        command = " ".join([f"cd {shlex.quote(str(project_root()))} &&", "env",
                            *(f"{k}={shlex.quote(v)}" for k, v in sorted(variables.items())),
                            *(shlex.quote(part) for part in worker_command(python_for(paths, first.env), payload))])
        log(f"# {unit.name}: {', '.join(job.dataset for job in unit.jobs)}")
        log(command)
        for job in unit.jobs:
            status.setdefault(job.name, "printed")
    return status
