"""Qualifications of contrast cells: the marks the paper prints next to a difference instead of hiding it.

Record-derived (computed here from the arms a cell actually uses, including merged rerun arms):
  forced      the cell's stream arm contains forced answers (MMDuet's S-dagger: the response head rarely
              self-triggers, so the protocol adds one forced generation at the question frame)
  invalid     understanding metric and an arm has more than 10% of rows stopped at the generation cap
  non_answer  the stream arm's response coverage is below 90%
  chance      TimeBlind: the stream arm (S-C, S-P) or the child offline (C-P) is not above its exact bias-matched
              null models; Video-MME / TOMATO: the stream arm's accuracy on the items it answered is at or below
              the benchmark's chance level
External (paper/flags/*.csv; they need an LLM labeller or a manual audit, so they ship as tables of labels):
  non_answer  response-form verdict of the cell's transition is "response-form change", or an arm's non-answer
              share on the audited subset is >= 50% (VSB harmful, MCV explicit / implicit); VMDT privacy audit
              verdict "non-answering" for an involved arm
  artifact    VMDT fairness of a child whose likelihood replies depend on the judge (artifacts.csv); VMDT privacy
              audit verdict "extraction/format artifact" for an involved arm
Transitions per gap: C-P = P->C (arms parent offline, child offline), S-C = C->S, S-P = P->S."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .records import Job
from .stats import binom_tail

UNDERSTANDING_KINDS = {"acc", "group", "mme"}
TRANSITION = {"C-P": "P->C", "S-C": "C->S", "S-P": "P->S"}
AUDITED_SPLITS = {("video_safetybench", "harmful"), ("mcv_safetybench", "explicit"), ("mcv_safetybench", "implicit")}
TIMEBLIND_CHANCE = 6.25


def pct(values) -> float | None:
    values = list(values)
    return 100 * sum(bool(v) for v in values) / len(values) if values else None


def truncated_share(job: Job) -> float:
    return pct(r.truncated for r in job.rows.values()) or 0.0


def forced_share(job: Job) -> float:
    return pct(r.forced for r in job.rows.values()) or 0.0


def options(sample) -> int | None:
    if sample.kind == "yesno":
        return 2
    return len(sample.choices) if sample.choices else None


def chance_level(samples: dict) -> float | None:
    """Mean over items of 1/#options (yes/no = 1/2); None for free-form answers."""
    k = [options(s) for s in samples.values()]
    return None if None in k else 100 * sum(1 / x for x in k) / len(k)


def poisson_binomial_upper(probs: np.ndarray, k: int) -> float:
    """P(sum of independent Bernoulli(probs) >= k), exact by dynamic programming."""
    dist = np.zeros(len(probs) + 1)
    dist[0] = 1.0
    for p in probs:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= 1 - p
    return float(min(1.0, dist[k:].sum()))


def timeblind_nulls(job: Job, samples: dict) -> dict:
    """I_Acc against two exact null models (Poisson-binomial over instances): R = random answers at the job's own
    yes / A rate per question type (unanswered stay wrong); V = the job's own two answers to a question swapped
    between its two videos with probability 1/2. above_nulls = both p < 0.05. Instances whose four queries are not
    all counted rows of the job (a --limit run, rows not scored yet) are left out; the paper's jobs count every row.
    Without a complete instance there is nothing to test (above_nulls True, no flag)."""
    rs = job.rows
    instances: dict = {}
    for i, s in samples.items():
        instances.setdefault(s.metadata["instance"], {})[s.metadata["role"]] = i
    inst = [v for v in instances.values() if len(v) == 4 and all(i in rs for i in v.values())]
    if not inst:
        return {"I_Acc": None, "above_nulls": True}
    yes = lambda s: s.answer in ("yes", "A")  # noqa: E731
    pred = {i: (None if (r.error or r.silent) else r.pred) for i, r in rs.items()}
    rate = {}
    for kind, positive in (("yesno", "yes"), ("mcq", "A")):
        valid = [pred[i] for i in pred if samples[i].kind == kind and pred[i] is not None]
        rate[kind] = sum(p == positive for p in valid) / len(valid) if valid else 0.5
    null_r, null_v = [], []
    for v in inst:
        prob = 1.0
        for role in range(4):
            i = v[role]
            q = rate[samples[i].kind]
            prob *= 0.0 if pred.get(i) is None else (q if yes(samples[i]) else 1 - q)
        null_r.append(prob)
        hits = 0
        for swap0 in (False, True):
            for swap1 in (False, True):
                p = {r: pred.get(v[r]) for r in range(4)}
                if swap0:
                    p[0], p[1] = p[1], p[0]
                if swap1:
                    p[2], p[3] = p[3], p[2]
                hits += all(p[r] is not None and p[r] == samples[v[r]].answer for r in range(4))
        null_v.append(hits / 4)
    correct = sum(all(rs[v[r]].correct for r in range(4)) for v in inst)
    n = len(inst)
    null_r, null_v = np.array(null_r), np.array(null_v)
    out = {"I_Acc": 100 * correct / n, "null_R": 100 * null_r.mean(), "p_above_R": poisson_binomial_upper(null_r, correct),
           "null_V": 100 * null_v.mean(), "p_above_V": poisson_binomial_upper(null_v, correct),
           "p_below_chance": binom_tail(correct, n, TIMEBLIND_CHANCE / 100, upper=False),
           "p_above_chance": binom_tail(correct, n, TIMEBLIND_CHANCE / 100, upper=True)}
    out["above_nulls"] = out["p_above_R"] < 0.05 and out["p_above_V"] < 0.05
    return out


# ---------- external tables ----------

def _csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class External:
    """paper/flags/*.csv (absent files simply contribute no flags)."""

    def __init__(self, folder: Path | None):
        folder = Path(folder) if folder else None
        read = (lambda name: _csv(folder / name)) if folder else (lambda name: [])
        self.verdicts = {(r["dataset"], r["split"], r["child"], r["transition"]): r["verdict"]
                         for r in read("response_forms.csv")}
        self.nonanswer = {(r["dataset"], r["split"], r["model"], r["mode"]): float(r["nonanswer"])
                          for r in read("nonanswer_shares.csv")}
        self.privacy = {(r["child"], r["mode"]): r["verdict"] for r in read("privacy_audit.csv")}
        self.artifacts = {(r["dataset"], r["child"]) for r in read("artifacts.csv")}


# ---------- cells ----------

class Flagger:
    """Flags of contrast rows (one dict per row of contrasts.json) from the arms they used."""

    def __init__(self, profile, arms, catalog, external: External | None = None):
        self.profile, self.arms, self.catalog = profile, arms, catalog
        self.external = external or External(None)
        self.invalid_pct = float(profile.flags.get("invalid_truncated_pct", 10))
        self.coverage_pct = float(profile.flags.get("non_answer_coverage_pct", 90))
        self._nulls: dict = {}
        self._chance: dict = {}
        self.by_name = {(m.dataset, m.name): m for m in profile.metrics}

    def _job(self, row: dict, side: str) -> Job | None:
        model, mode = row[f"{side}_job"].rsplit(".", 1)
        return self.arms.job(row["dataset"], model, mode, row["child"])

    def _above_nulls(self, job: Job) -> bool:
        if job.uid not in self._nulls:
            self._nulls[job.uid] = timeblind_nulls(job, self.catalog.samples(job.dataset))
        return self._nulls[job.uid]["above_nulls"]

    def _answered_accuracy(self, row: dict) -> float | None:
        """Accuracy of the stream arm on the items it answered (not silent, not errored), on the items all three
        arms of the child counted."""
        child, dataset = row["child"], row["dataset"]
        parent = self.profile.parents[child]
        jobs = [self.arms.job(dataset, parent, "offline", child), self.arms.job(dataset, child, "offline", child),
                self.arms.job(dataset, child, "stream", child)]
        if any(j is None for j in jobs):
            return None
        p, c, s = jobs
        ids = sorted(set(p.rows) & set(c.rows) & set(s.rows))
        responded = [i for i in ids if not (s.rows[i].silent or s.rows[i].error)]
        return pct(s.rows[i].correct for i in responded)

    def chance(self, dataset: str) -> float | None:
        if dataset not in self._chance:
            self._chance[dataset] = chance_level(self.catalog.samples(dataset))
        return self._chance[dataset]

    def cell(self, row: dict) -> tuple[list[str], list[str]]:
        """(flags, notes) of one contrast row."""
        metric = self.by_name[(row["dataset"], row["metric"])]
        gap, dataset, child = row["gap"], row["dataset"], row["child"]
        a, b = self._job(row, "a"), self._job(row, "b")
        stream = b if gap in ("S-C", "S-P") else None
        flags, notes = [], []
        if stream is not None and forced_share(stream) > 0:
            flags.append("forced")
            notes.append(f"{stream.job} forced on {sum(r.forced for r in stream.rows.values())}/{len(stream.rows)} rows")
        if metric.kind in UNDERSTANDING_KINDS:
            for job in (a, b):
                share = truncated_share(job)
                if share > self.invalid_pct:
                    flags.append("invalid")
                    notes.append(f"{job.job} truncated {share:.1f}% (> {self.invalid_pct:g}%)")
        if stream is not None and row.get("coverage_b") is not None and row["coverage_b"] < self.coverage_pct:
            flags.append("non_answer")
            notes.append(f"S coverage {row['coverage_b']:.1f}% < {self.coverage_pct:g}%")
        if dataset == "timeblind":
            job = stream if stream is not None else b
            if not self._above_nulls(job):
                flags.append("chance")
                notes.append(f"{job.job} not above its bias-matched TimeBlind nulls")
        if stream is not None and metric.kind == "acc" and dataset in ("videomme", "tomato"):
            answered, level = self._answered_accuracy(row), self.chance(dataset)
            if answered is not None and level is not None and answered <= level:
                flags.append("chance")
                notes.append(f"S accuracy on answered items {answered:.1f} <= chance {level:.2f}")
        ext = self.external
        split = metric.split
        if (dataset, split) in AUDITED_SPLITS:
            transition = TRANSITION[gap]
            verdict = ext.verdicts.get((dataset, split, child, transition))
            shares = [ext.nonanswer.get((dataset, split, *self._arm_key(row, side))) for side in ("a", "b")]
            why = []
            if verdict == "response-form change":
                why.append(f"verdict {transition} = response-form change")
            if any(s is not None and 100 * s >= 50 for s in shares):
                why.append("non-answer share >= 50% in an arm (audited subset)")
            if why:
                flags.append("non_answer")
                notes.append("; ".join(why))
        if dataset == "vmdt_privacy":
            involved = {"C-P": [("C vs P", "offline")], "S-C": [("C vs P", "offline"), ("S vs P", "stream")],
                        "S-P": [("S vs P", "stream")]}[gap]
            verdicts = [(label, ext.privacy.get((child, mode))) for label, mode in involved]
            if any(v == "non-answering" for _, v in verdicts):
                flags.append("non_answer")
            if any(v == "extraction/format artifact" for _, v in verdicts):
                flags.append("artifact")
            if any(v for _, v in verdicts):
                notes.append("privacy audit: " + ", ".join(f"{k} {v}" for k, v in verdicts if v))
        if (dataset, child) in ext.artifacts:
            flags.append("artifact")
            notes.append(f"{dataset} of {child}: judge-dependent reply format (artifacts.csv)")
        return list(dict.fromkeys(flags)), notes

    def _arm_key(self, row: dict, side: str) -> tuple[str, str]:
        model, mode = row[f"{side}_job"].rsplit(".", 1)
        return model, mode

    def annotate(self, rows: list[dict]) -> list[dict]:
        """Adds 'flags' and 'flag_notes' to every row (in place) and returns the rows."""
        for r in rows:
            r["flags"], r["flag_notes"] = self.cell(r)
        return rows

    def merged_truncation(self) -> list[dict]:
        """Truncation of every merged rerun arm (what the 'invalid' flag reads for those arms)."""
        out = []
        for dataset, model, child, job in self.arms.merged_jobs():
            share = truncated_share(job)
            out.append({"dataset": dataset, "child": child, "model": model, "n": len(job.rows),
                        "truncated": share, "invalid": share > self.invalid_pct})
        return sorted(out, key=lambda c: (c["dataset"], c["child"], c["model"]))
