"""TimeBlind: 600 instances x (2 videos x 2 questions) = 2400 questions; half yes/no, half A/B.

Paper use: video understanding, all three conditions (P, C offline; S through the child's streaming interface, with
the question at the end of the video): the absolute-score video table and the C-P, S-C, S-P contrasts. Headline:
instance accuracy `I_Acc`, the share of instances whose four questions are all answered correctly (complete
instances only); the statistics use one binary unit per instance (group size 4).

Instance roles (index % 4): 0 = q0/v0, 1 = q0/v1, 2 = q1/v0, 3 = q1/v1. Prompt: the official add_question_suffix
(the options stay inside the question text, as in the dataset). Scoring is the official code (fetched upstream code,
manifests/upstream.yaml): utils.extract_answer on the reply; when it finds nothing (-1), the judge gets llm_judge.py's
own system and extraction prompt (the README's fallback for outputs the regex misses; a local judge model instead
of GPT) and its reply goes through extract_answer again. docs/benchmarks/timeblind.md."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re

from .. import common
from ..base import JudgePolicy, MetricSpec
from ...core.schema import Sample

NAME = "timeblind"
MEDIA = "video"
JUDGE = JudgePolicy.FALLBACK
EXPECTED_COUNT = 2400
METRICS = [MetricSpec("I_Acc", "group", group="instance", size=4)]
# The role fixes the reference answer (yes, no, no, yes / A, B, B, A over roles 0-3), so models never see it.
PRIVATE_METADATA = frozenset({"role"})
UTILS = "timeblind/utils.py"
LLM_JUDGE = "timeblind/llm_judge.py"


def prepare(raw: Path, media: Path) -> list[Sample]:
    """raw/data.jsonl + raw/videos/ (read in place, nothing is copied to media/) + the optional raw/tags.json
    (metadata only) -> 2400 samples with ids str(index)."""
    rows = [json.loads(line) for line in (raw / "data.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    tags: dict[int, list[str]] = {}
    if (raw / "tags.json").exists():
        for tag, indices in json.loads((raw / "tags.json").read_text(encoding="utf-8")).items():
            for index in indices:
                tags.setdefault(int(index), []).append(tag)
    samples = []
    for row in rows:
        index, video = int(row["index"]), str(raw / row["video_path"].removeprefix("TimeBlind/"))
        meta = {"instance": index // 4, "role": index % 4, "type": row["type"], "tags": tags.get(index, [])}
        if row["type"] == "yes_no":
            samples.append(Sample(str(index), row["question"], row["answer"].strip().lower(), "yesno", video=video, metadata=meta))
        else:
            options = re.search(r"Option A:\s*(.*?)\s*Option B:\s*(.*)$", row["question"], re.S)
            samples.append(Sample(str(index), row["question"], row["answer"].strip().upper(), "mcq",
                                  {"A": options[1].strip(), "B": options[2].strip()}, video=video, metadata=meta))
    return samples


def prompt(sample) -> str:
    """Official add_question_suffix; options stay inside the question text as in the dataset."""
    return sample.question + (" Please output Yes or No." if sample.kind == "yesno" else " Please output A or B.")


def extract_answer(output_string: str, task_type: str) -> int:
    """Official utils.extract_answer: 1 = yes/A, 0 = no/B, -1 = invalid. It reads letters only: a bare option text
    is not an answer (the judge may extract it)."""
    return common.upstream_module(UTILS).extract_answer(output_string, task_type)


def _type(sample) -> str:
    return "yes_no" if sample.kind == "yesno" else "multiple_choice"


def _answer(sample, value: int):
    return ({1: "yes", 0: "no"} if sample.kind == "yesno" else {1: "A", 0: "B"}).get(value)


def parse(sample, text):
    value = extract_answer(common.clean(text), _type(sample))
    if value < 0:
        return None, "official_invalid"
    return _answer(sample, value), "official"


@lru_cache(maxsize=None)
def _judge_prompts():
    """(create_extraction_prompt, system message) of the official llm_judge.py, compiled from the fetched source
    (the file imports API clients at module level)."""
    prompt = common.upstream_functions(LLM_JUDGE, "create_extraction_prompt")["create_extraction_prompt"]
    system = common.upstream_assignment(LLM_JUDGE, "extract_answer_with_gpt", "system_msg")()
    return prompt, system


def judge_messages(sample, text):
    """llm_judge.py: its system message and create_extraction_prompt, max_tokens 10 (temperature 0, as upstream).
    `text` is the raw (uncleaned) reply: the judge cache keys depend on it."""
    prompt, system = _judge_prompts()
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt(text, _type(sample))}], 10


def judge_result(sample, reply: str):
    """Upstream overwrites model_output with the judge's reply and scores it with extract_answer; -1 stays wrong."""
    return _answer(sample, extract_answer(reply, _type(sample)))


def metrics(rows) -> dict:
    """Official get_scores over complete instances (question pairs = roles 0/1 and 2/3; video pairs = 0/2 and 1/3)."""
    instances: dict[int, dict[int, bool]] = {}
    for r in rows:
        instances.setdefault(r["metadata"]["instance"], {})[r["metadata"]["role"]] = r["correct"]
    complete = [v for v in instances.values() if len(v) == 4]
    n = len(complete)
    if not n:
        return {"I_Acc": None}
    q = sum((v[0] and v[1]) + (v[2] and v[3]) for v in complete)
    video = sum((v[0] and v[2]) + (v[1] and v[3]) for v in complete)
    return {"I_Acc": common.pct(all(v.values()) for v in complete), "Q_Acc": round(100 * q / (2 * n), 2),
            "V_Acc": round(100 * video / (2 * n), 2), "Acc": common.pct(x for v in complete for x in v.values()),
            "instances": n, **common.by(rows, "type", "type/")}


def smoke(samples):
    return common.smoke(samples, group="instance", n=48)
