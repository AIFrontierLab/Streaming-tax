"""The official LingoQA judge, Lingo-Judge (configs/judges.yaml `lingo`), for the benchmark whose JUDGE policy is
"lingo": LingoQA, the driving domain of the fine-tuning experiment (docs/generalization/finetune.md).

Lingo-Judge is a DeBERTa-v3-base classifier (wayveai/Lingo-Judge) that scores whether a student answer agrees with a
question's reference answers. It is run exactly as LingoQA's official evaluation runs it:
- the code is LingoQA's own benchmark/judge.py (github.com/wayveai/LingoQA, fetched at the pinned commit by
  `streamreplace fetch-upstream lingoqa`; the LingoQA licence does not allow redistributing it, so this repository
  never copies it), imported unchanged in its own Python environment (envs/lingo: the pins of LingoQA's
  requirements.txt, transformers 4.36.2 and torch 2.1.2), in fp32;
- `LingoJudge(pretrained_model=<Lingo-Judge folder>).eval()` on the first visible GPU (the CPU when there is none);
- per question `compute([question], [references], [prediction])`: the classifier's maximum logit over the reference
  answers, as evaluate.py's evaluate_question at its default batch size 1; an answer is correct iff that score > 0.
evaluate.py itself is not run: its input path reads the references with pandas from the downloaded parquet (and fails
on the released file, LingoQA issue #11); here the references come from the prepared samples instead. Checked against
evaluate.py (with its two known input fixes) on four prediction sets of the 500 evaluation questions: identical
decisions on every question, largest score difference 5e-11.

Score rows: the rule layer stores the cleaned reply as pred (benchmarks/finetune/lingoqa.py parse()) with parse
"pending"; a cached judge result sets judge_pred = round(score, 4), correct = (score > 0) on the unrounded score and
parse "lingo". Rows without a reply are judged too (an empty prediction), as in the paper's runs; errored rows get no
request and count as wrong.
Cache entries: {"key", "score"}; key = cache_key({judge, question, references, prediction}) (judges/base.py) with judge
= the checkpoint reference of the paper's runs, so the paper's cached scores resolve every reported row.

The judge process: `python -m streamreplace.judges.entry <job.json>` in the lingo environment calls serve()."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

from .base import JudgeRequest, cache_key
from .registry import ConfiguredJudge

JUDGE = "wayveai/Lingo-Judge@a1a6e2194fd1aedaba02bde82316a6b73753d1d5"   # the `judge` field of every cache key
BACKEND = "hf"                       # the official transformers classifier (benchmark/judge.py); the only one provided
UPSTREAM = "lingoqa/benchmark"       # <upstream root>/lingoqa/benchmark holds judge.py and constants.py


def request_key(question: str, references: list[str], prediction: str) -> str:
    return cache_key({"judge": JUDGE, "question": question, "references": references, "prediction": prediction})


def request_for(sample, prediction: str | None) -> dict:
    """The judge input of one row: the question and its reference answers (private metadata, never shown to a
    model) and the parsed reply ("" when the reply was empty)."""
    question, references, prediction = sample.question, list(sample.metadata["references"]), prediction or ""
    return {"key": request_key(question, references, prediction), "question": question, "references": references,
            "prediction": prediction}


class LingoJudge(ConfiguredJudge):
    NAME = "lingo"
    ENV = "lingo"
    PROMPT_VERSION = 1

    def __init__(self, spec, paths=None):
        super().__init__(spec, paths)
        if (spec.backend or BACKEND) != BACKEND:
            raise ValueError(f"judges.lingo.backend: only {BACKEND!r} (the official classifier) is supported")

    def identity(self) -> dict:
        return {"judge": self.NAME, "checkpoint": self.spec.checkpoint, "backend": BACKEND, "dtype": "float32",
                "batch_size": 1, "prompts": self.PROMPT_VERSION}

    def requests(self, bench, sample, gen: dict, score: dict) -> list[JudgeRequest]:
        if sample is None or score.get("parse") != "pending":
            return []
        request = request_for(sample, score.get("pred"))
        return [JudgeRequest(self.NAME, request["key"], request)]

    def apply(self, bench, sample, score: dict, result: dict) -> dict:
        return {**score, "judge_pred": round(result["score"], 4), "correct": result["score"] > 0.0, "parse": "lingo"}

    def process_spec(self) -> dict:
        from ..benchmarks.common import upstream_path
        return {"checkpoint_dir": str(self.checkpoint_dir()), "code": str(upstream_path(UPSTREAM))}

    def run(self, reqs: list[JudgeRequest], cache, gpus: list[int]) -> None:
        """One judge process on the first of `gpus` (a 184M-parameter classifier needs no more)."""
        self.launch([r.payload for r in reqs], cache, gpus[:1], self.process_spec())


# ---------- the judge process (lingo environment) ----------

def load(checkpoint_dir: str, code: str):
    """LingoQA's LingoJudge from the fetched benchmark/ folder, loaded from the local checkpoint folder, in eval mode
    on the first visible GPU (else the CPU)."""
    import torch
    sys.path.insert(0, str(Path(code).resolve()))
    from judge import LingoJudge as Official                  # benchmark/judge.py, unchanged
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return Official(pretrained_model=str(Path(checkpoint_dir).resolve())).eval().to(device)


def score(judge, request: dict) -> float:
    """evaluate_question at batch size 1: the maximum classifier logit over the references."""
    return float(judge.compute([request["question"]], [request["references"]], [request["prediction"]])[0])


def judge_all(judge, requests: list[dict], on_result: Callable[[dict], None]) -> None:
    for number, request in enumerate(requests, 1):
        on_result({"key": request["key"], "score": score(judge, request)})
        if number % 500 == 0:
            print(f"lingo: {number}/{len(requests)}", flush=True)


def serve(requests: list[dict], cache, spec: dict) -> None:
    """Judge-process side (judges/entry.py): scores the requests not yet cached, appending one line per answer."""
    judge_all(load(spec["checkpoint_dir"], spec["code"]), requests, cache.append)
