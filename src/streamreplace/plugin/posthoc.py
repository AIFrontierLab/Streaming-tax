"""The paper's S+, assembled after the fact from the shadow runs (and the same rule as the live gate).

The plugin arms run with the gate in shadow mode: every row logs the answer-position projection rounded to 3
decimals (info.plugin.guard.proj, read as analysis.records.Record.proj) and nothing fires. S+ on an item set is then
built per arm, each arm by its own logged projection:
1. fired = the ids whose logged projection is strictly above the item format's threshold (tau_closed for
   multiple-choice and yes/no items, tau_free for everything else); a row without a projection never fires;
2. a fired row takes the scored row of the fixed-refusal pseudo-job `fixed_refusal.offline` (the refusal text
   scored by each benchmark's official pipeline and judges);
3. on Video-MME, TOMATO and TimeBlind a fired row is unanswered and wrong instead (their parsers would otherwise
   credit the refusal: TOMATO's random fallback picks an option, TimeBlind's pairs reward a constant answer).
`S_fix + gate` is the same assembly on the sfix arm (no PRS); `S_fix + PRS` is the sfix_prs arm without the gate."""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..analysis.arms import ForceUnanswered, Replace, Restrict
from ..analysis.records import Job
from ..core.io import append_jsonl, read_latest
from ..core.protocol import FIXED_REFUSAL
from ..core.schema import Generation, generation_row
from .recipe import CLOSED_KINDS, INFO_KEY, REFUSAL

REFUSAL_JOB = f"{FIXED_REFUSAL}.offline"


def fired(job: Job, samples: dict, tau: dict) -> set[str]:
    """Ids whose logged gate projection passes the item format's threshold (closed: multiple choice / yes-no)."""
    out = set()
    for i, record in job.rows.items():
        p = record.proj
        if p is not None and p > tau["closed" if samples[i].kind in CLOSED_KINDS else "free"]:
            out.add(i)
    return out


def _digest(ids) -> str:
    return hashlib.sha256("|".join(sorted(ids)).encode()).hexdigest()[:8]


def restrict(job: Job, ids) -> Job:
    """The job on `ids` only (a distinct tag per id set keeps derived jobs apart in the analysis caches)."""
    ids = frozenset(ids)
    return job.with_overlays(Restrict(ids, label=f"restrict-{_digest(ids)}"))


def unanswered(job: Job) -> Job:
    """Every record unanswered and wrong (the refusal on the understanding sets)."""
    return job.with_overlays(ForceUnanswered(label="unanswered"))


def mix(job: Job, refusal: Job, ids, tag: str) -> Job:
    """The job with the refusal job's record in place of its own on `ids`."""
    ids = frozenset(ids)
    return job.with_overlays(Replace(ids, refusal, label=f"{tag}-{_digest(ids)}"))


def refusal_for(dataset: str, refusal: Job, unanswered_on_fire) -> Job:
    return unanswered(refusal) if dataset in unanswered_on_fire else refusal


def assemble(job: Job, refusal: Job, samples: dict, tau: dict, unanswered_on_fire=(), tag: str = "gate"):
    """(the gated job, fired ids): the arm with the gate applied post hoc. `refusal` is the scored fixed-refusal job
    of the same dataset (restricted to the same items)."""
    ids = fired(job, samples, tau)
    return mix(job, refusal_for(job.dataset, refusal, unanswered_on_fire), ids, tag), ids


# ---------- the fixed-refusal pseudo-job ----------

def refusal_rows(ids, prompts: dict[str, str], refusal: str = REFUSAL) -> list[dict]:
    """Generation rows of the fixed refusal for every id (the prompt each item was asked with)."""
    info = {"no_answer": False, INFO_KEY: {"synthetic": "the fixed refusal on every item"}}
    return [generation_row(i, prompts[i], Generation(refusal, None, "guard", dict(info)), 0.0) for i in ids]


def prompts_from(job_files, ids) -> dict[str, str]:
    """Each id's prompt from the latest rows of existing jobs of the same dataset (e.g. the plugin arms)."""
    out: dict[str, str] = {}
    for path in job_files:
        for i, row in read_latest(Path(path)).items():
            if i in ids and row.get("prompt") is not None:
                out.setdefault(i, row["prompt"])
    missing = sorted(set(ids) - set(out))
    if missing:
        raise ValueError(f"no prompt for {len(missing)} item(s), e.g. {missing[:3]}: run the plugin arms first")
    return out


def write_refusal_job(outputs: Path, dataset: str, ids, prompts: dict[str, str], refusal: str = REFUSAL) -> int:
    """<outputs>/<dataset>/fixed_refusal.offline.jsonl: adds rows for ids it lacks (existing rows are kept, so their
    scores stay valid). Returns the number of rows added; refuses a file holding another reply."""
    path = Path(outputs) / dataset / f"{REFUSAL_JOB}.jsonl"
    have = read_latest(path) if path.exists() else {}
    other = sorted(i for i, r in have.items() if r.get("raw") != refusal)
    if other:
        raise ValueError(f"{path}: rows with another reply than the fixed refusal, e.g. {other[:3]}")
    todo = [i for i in ids if i not in have]
    for row in refusal_rows(todo, prompts, refusal):          # generation rows are append-only
        append_jsonl(path, row)
    return len(todo)
