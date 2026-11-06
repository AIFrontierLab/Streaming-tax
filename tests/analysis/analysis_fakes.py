"""Synthetic outputs for the analysis tests: one pair (parent `par`, child `kid`) on three small benchmarks.

  img  image, accuracy (C-P only), judge fallback
  vid  video, accuracy clustered by video (C-P, S-C, S-P), the child also streams
  saf  safety, attack-success rate as the share of safe replies (RJScore-style: errors count safe)

`outputs(tmp_path)` writes generation rows (<job>.jsonl) and score rows (<job>.scores.jsonl) in the package layout;
`catalog()` serves the samples and benchmark hooks the analysis reads; `profile()` is a small analysis profile with
the paper's structure (families, broad Holm families, margins)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from streamreplace.analysis.records import Catalog
from streamreplace.analysis.specs import profile_from_dict
from streamreplace.core.schema import Sample

N = 40          # items per benchmark


def pct(values) -> float | None:
    values = list(values)
    return round(100 * sum(bool(v) for v in values) / len(values), 2) if values else None


def _bench(name: str, judge: str, lower: bool = False, media: str = "image"):
    def metrics(rows):
        value = pct(r["correct"] for r in rows)
        if name == "saf":
            return {"asr": None if value is None else round(100 - value, 2)}
        return {"acc": value, "answered": pct(r["answered"] for r in rows)}

    return SimpleNamespace(NAME=name, JUDGE=judge, LOWER_IS_BETTER=lower, MEDIA=media, metrics=metrics,
                           stratum=lambda sample: None)


BENCHES = {"img": _bench("img", "fallback"), "vid": _bench("vid", "none", media="video"),
           "saf": _bench("saf", "rjscore", lower=True, media="video")}


def samples(dataset: str) -> list[Sample]:
    out = []
    for k in range(N):
        video = f"/media/{dataset}/v{k // 4}.mp4" if dataset != "img" else None
        out.append(Sample(id=f"{dataset}-{k:03d}", question=f"q{k}", answer="A", kind="mcq",
                          choices={"A": "a", "B": "b"}, video=video, metadata={"group": k // 4}))
    return out


def catalog() -> Catalog:
    return Catalog(samples=samples, benchmark=lambda d: BENCHES[d])


# Which items each job gets right (deterministic patterns, different per arm).
PATTERNS = {
    ("img", "par.offline"): lambda k: k % 5 != 0,          # 32/40
    ("img", "kid.offline"): lambda k: k % 4 != 0,          # 30/40
    ("vid", "par.offline"): lambda k: k % 3 != 0,          # 26/40
    ("vid", "kid.offline"): lambda k: k % 3 != 0 or k == 3,
    ("vid", "kid.stream"): lambda k: k % 2 == 0,           # 20/40
    ("saf", "par.offline"): lambda k: k % 2 == 0,
    ("saf", "kid.offline"): lambda k: k % 3 == 0,
    ("saf", "kid.stream"): lambda k: k % 5 == 0,
}


def write_job(root: Path, dataset: str, job: str, correct, *, silent=(), errors=(), length=(), forced=(),
              extra_rows=(), info=None) -> None:
    folder = root / dataset
    folder.mkdir(parents=True, exist_ok=True)
    gen, scores = [], []
    for s in samples(dataset):
        k = int(s.id.rsplit("-", 1)[1])
        error = "boom" if k in errors else None
        row_info = dict(info(k) if info else {})
        if k in forced:
            row_info["forced"] = True
        if k in silent:
            row_info["no_answer"] = True
        gen.append({"id": s.id, "prompt": s.question, "raw": None if error else ("A" if correct(k) else "B"),
                    "tokens": 3, "finish": "length" if k in length else "eos", "latency": 0.5, "info": row_info,
                    "error": error})
        if error:
            scores.append({"id": s.id, "pred": None, "parse": None, "correct": None, "judge_pred": None,
                           "judge_correct": None, "judge": None})
        elif k in silent:
            scores.append({"id": s.id, "pred": None, "parse": "no_answer", "correct": False, "judge_pred": None,
                           "judge_correct": None, "judge": None})
        else:
            scores.append({"id": s.id, "pred": "A" if correct(k) else "B", "parse": "letter",
                           "correct": bool(correct(k)), "judge_pred": None, "judge_correct": None, "judge": None})
    gen += list(extra_rows)
    (folder / f"{job}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in gen))
    (folder / f"{job}.scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scores))


def outputs(root: Path) -> Path:
    for (dataset, job), correct in PATTERNS.items():
        kwargs = {}
        if (dataset, job) == ("vid", "kid.stream"):
            kwargs = {"silent": {1, 7}, "forced": {5}}
        if (dataset, job) == ("saf", "kid.stream"):
            kwargs = {"errors": {3}}
        if (dataset, job) == ("img", "kid.offline"):
            kwargs = {"length": {1, 2, 3, 5, 6}}
        write_job(root, dataset, job, correct, **kwargs)
    return root


PROFILE = {
    "profile": "test",
    "roster": {"children": ["kid"], "models": ["par", "kid"], "pairs": {"p": {"parent": "par", "children": ["kid"]}}},
    "engine": {"reps": 400, "seed": 0, "vmdt_draws": 400},
    "metrics": [
        {"dataset": "img", "name": "acc", "kind": "acc", "family": "image-headline", "modes": ["offline"]},
        {"dataset": "vid", "name": "acc", "kind": "acc", "cluster": "video", "family": "video"},
        {"dataset": "saf", "name": "asr", "kind": "safe", "lower_is_better": True, "family": "saf"},
    ],
    "holm": {"split_by_gap": ["video"], "broad": {"safety": ["saf"]}, "alpha": 0.05},
    "margins": {"tost": [3, 2], "ni": [2, 3, 5], "reading_tost": 3, "verdict_ni": 5, "coverage_tolerance": 2,
                "inferior": 5},
    "check_official": {"tolerance": 0.011},
    "overlays": {"truncation_rerun": {"pairs": ["p"], "datasets": ["img"], "ids": "capped.json",
                                      "rows": "rerun_capped", "cap": 64, "margin": 8}},
    "flags": {"invalid_truncated_pct": 10, "non_answer_coverage_pct": 90},
    "report": {"datasets": ["img", "vid", "saf"], "vmdt_paired_draws": 200},
    "display": {"names": {"par": "Par", "kid": "Kid"}, "full_names": {"par": "Parent-1B", "kid": "Child-1B"},
                "zero_below": 0.05},
}


def profile(root: Path | None = None, **changes):
    data = json.loads(json.dumps(PROFILE))
    data.update(changes)
    return profile_from_dict(data, root=root)
