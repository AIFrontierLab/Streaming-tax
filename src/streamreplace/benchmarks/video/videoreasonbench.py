"""VideoReasonBench: 240 videos x 6 questions = 1440 free-form answers, judged by the LLM judge with the official
prompts and state checkers (github.com/llyx97/video_reason_bench, fetched at the commit pinned in
manifests/upstream.yaml).

Paper use: supplementary video benchmark, all three conditions (P, C offline; S through the child's streaming
interface, with the question at the end of the video): the supplement's video table and its contrasts (a secondary
metric family). Headline: `acc` over all 1440 questions, decided by the judge (JUDGE always: every reply with a
"Final Answer" is judged; a reply without one is wrong); the statistics resample videos.

Prompt: the question as given (it already asks for "Final Answer:"). Parsing: the official extract_final_answer, which
takes the FIRST "Final Answer" (colon optional) through the end of the reply. The rule-only fallback `correct()` (exact
match after normalisation) is reported beside the judged numbers but is not the paper's metric.
docs/benchmarks/videoreasonbench.md."""
from __future__ import annotations

import contextlib
from functools import lru_cache
import io
from pathlib import Path
import re

from .. import common
from ..base import JudgePolicy, MetricSpec
from ...core.schema import Sample

NAME = "videoreasonbench"
MEDIA = "video"
JUDGE = JudgePolicy.ALWAYS
EXPECTED_COUNT = 1440
METRICS = [MetricSpec("acc", "acc", cluster="video")]
# The video's states, coordinates and moves are what the judge's state checkers compare a reply against.
PRIVATE_METADATA = frozenset({"states", "coords", "moves"})
OFFICIAL = "videoreasonbench/utils/eval_utils.py"


def prepare(raw: Path, media: Path) -> list[Sample]:
    """raw/<task>/test-*.parquet (6 tasks) + raw/videos.zip (extracted into media/) -> 1440 samples."""
    common.extract_zip(raw / "videos.zip", media)
    samples = []
    for path in sorted(raw.glob("*/test-*.parquet")):
        task = path.parent.name
        for row in common.read_parquet(path):
            samples.append(Sample(row["id"], row["question"], str(row["answer"] or ""), "open", video=str(media / row["video"]),
                                  metadata={"task": task, "dim": row["dim"], "visible_time": row["visible_time"],
                                            "num_state": row["num_state"], "num_operation": row["num_operation"],
                                            "states": row["states"], "coords": row["coords"], "moves": row["moves"],
                                            "video": row["video"]}))
    if len(samples) != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} VideoReasonBench questions, got {len(samples)}")
    return samples


def prompt(sample) -> str:
    return sample.question            # the official question already asks for "Final Answer:"


@lru_cache(maxsize=None)
def _official_extract():
    """The official utils/eval_utils.extract_final_answer, compiled from the fetched source: eval_utils imports the
    openai / google-genai / transformers clients at module level, which the scoring environment does not have."""
    return common.upstream_functions(OFFICIAL, "extract_final_answer", scope={"re": re})["extract_final_answer"]


def official_extract(response: str):
    """The FIRST "Final Answer" (colon optional, kept in the result) through the end of the response, or False when
    there is none."""
    return _official_extract()(response)


def parse(sample, text):
    with contextlib.redirect_stdout(io.StringIO()):           # upstream prints "Final Answer not found."
        candidate = official_extract(common.clean(text))
    return (candidate, "final_answer_marker") if candidate else (None, "no_final_answer")


def correct(sample, prediction) -> bool:
    """Rule-only fallback = exact match after normalisation; the judge decides the headline number."""
    return prediction is not None and bool(sample.answer) and common.normalize(prediction) == common.normalize(sample.answer)


def metrics(rows) -> dict:
    return {"acc": common.pct(r["correct"] for r in rows), **common.by_length(rows), **common.by(rows, "task", "task/"),
            **common.by(rows, "dim", "dim/")}


def smoke(samples):
    return common.smoke(samples, group="video", n=48)
