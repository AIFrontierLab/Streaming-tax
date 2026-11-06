"""TOMATO: 1484 temporal-reasoning multiple-choice questions over 1397 distinct video files; six reasoning types
(one parquet split each); 3-7 options per question.

Paper use: video understanding, all three conditions (P, C offline; S through the child's streaming interface, with
the question at the end of the video): the absolute-score video table and the C-P, S-C, S-P contrasts. Headline:
`acc` over all 1484 questions; the statistics resample video files (questions on the same clip form one cluster).

Prompt and system prompt follow lmms-eval (tasks/tomato, from the official construct_prompt); the system prompt is
set in configs/benchmarks.yaml and only offline adapters send it. Frames come from the pair's video protocol.
Scoring is the official src/parse_result.py (fetched upstream code, manifests/upstream.yaml): pre_parser on the
reply; when it does not give one of all_choices, the judge gets gpt_parser's prompt verbatim (a local judge model
instead of gpt-4o-mini, temperature 0, max_tokens 16); a judge reply outside all_choices falls back to
random.choice(all_choices), seeded per question. Upstream retries the GPT call up to 5 times; at temperature 0 the
reply does not change. docs/benchmarks/tomato.md."""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path
import random
import re

from .. import common
from ..base import JudgePolicy, MetricSpec
from ...core.schema import Sample

NAME = "tomato"
MEDIA = "video"
JUDGE = JudgePolicy.FALLBACK
EXPECTED_COUNT = 1484
METRICS = [MetricSpec("acc", "acc", cluster="video")]
# lmms-eval construct_prompt, verbatim (trailing spaces included); {index2ans} is the Python dict repr
PROMPT = (
    'You will be provided with frames uniformly sampled from a video, the frames are provided in chronological order of the video. Analyze these frames and provide the answer to the question about the video content. Answer the multiple-choice question about the video content.\n'
    '\n'
    'You must use these frames to answer the multiple-choice question; do not rely on any externel knowledge or commonsense. \n'
    '\n'
    '<question> \n'
    '{question} \n'
    '</question>\n'
    '\n'
    '<options> \n'
    '{index2ans} \n'
    '</options>\n'
    '\n'
    'Even if the information in these separate frames is not enough to answer the question, PLEASE TRY YOUR BEST TO GUESS AN ANSWER WHICH YOU THINK WOULD BE THE MOST POSSIBLE ONE BASED ON THE QUESTION. \n'
    '\n'
    "DO NOT GENERATE ANSWER SUCH AS 'NOT POSSIBLE TO DETERMINE.' \n"
)


def prepare(raw: Path, media: Path) -> list[Sample]:
    """raw/data/<reasoning>-*.parquet + raw/videos.zip (extracted into media/) -> 1484 samples with ids
    "<reasoning>:<key>" (":<row>" appended to a repeated id). The ids seed the random fallback: keep them."""
    common.extract_zip(raw / "videos.zip", media)
    samples, seen = [], set()
    for path in sorted((raw / "data").glob("*.parquet")):
        reasoning = path.name.split("-")[0]
        for number, row in enumerate(common.read_parquet(path)):
            sample_id = f"{reasoning}:{row['key']}"
            if sample_id in seen:
                sample_id = f"{sample_id}:{number}"
            seen.add(sample_id)
            choices = common.letters(row["options"])      # question kept verbatim: the lmms-eval prompt uses it raw
            samples.append(Sample(sample_id, row["question"], chr(ord("A") + int(row["answer"])), "mcq", choices,
                                  video=str(media / "videos" / row["demonstration_type"] / f"{row['key']}.mp4"),
                                  metadata={"reasoning": reasoning, "demonstration": row["demonstration_type"],
                                            "motion": row.get("motion_type"), "video": row["key"]}))
    if len(samples) != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} TOMATO questions, got {len(samples)}")
    return samples


def prompt(sample) -> str:
    return PROMPT.format(question=sample.question, index2ans=dict(sample.choices))


OFFICIAL = "tomato/src/parse_result.py"
# The official pre_parser (used as is) reads a bare letter, "X: option" only when the option text has no lower-case
# letter (it upper-cases the reply alone), and "Answer is:(X)" for A-F only (9 questions have a G); every other
# reply goes to the judge.


@lru_cache(maxsize=None)
def _pre_parser():
    """The official pre_parser, compiled from the fetched source (the file imports API clients at module level)."""
    return common.upstream_functions(OFFICIAL, "pre_parser", scope={"re": re})["pre_parser"]


@lru_cache(maxsize=None)
def _judge_prompt():
    """The prompt f-string of the official gpt_parser, compiled from the fetched source."""
    return common.upstream_assignment(OFFICIAL, "gpt_parser", "prompt")


def _choices(sample):
    """construct_prompt: all_choices = A, B, ...; index2ans = letter -> option text."""
    return list(sample.choices), dict(sample.choices)


def parse(sample, text):
    all_choices, index2ans = _choices(sample)
    parsed = _pre_parser()(response=common.clean(text), all_choices=all_choices, index2ans=index2ans)
    return (parsed, "official_pre_parser") if parsed in all_choices else (None, "official_unparsed")


def judge_messages(sample, text):
    """gpt_parser's prompt as one user message, max_tokens 16. `text` is the raw (uncleaned) reply: the judge cache
    keys depend on it."""
    all_choices, index2ans = _choices(sample)
    content = _judge_prompt()(response=text, all_choices=all_choices, index2ans=index2ans)
    return [{"role": "user", "content": content}], 16


def judge_result(sample, reply: str):
    """The judge's letter if it is one of all_choices, else a random option seeded by the sample id (upstream draws
    from the unseeded global random module; seeding per question makes the score reproducible)."""
    all_choices, _ = _choices(sample)
    if reply in all_choices:
        return reply
    seed = int(hashlib.sha256(f"tomato-random:{sample.id}".encode()).hexdigest()[:8], 16)
    return random.Random(seed).choice(all_choices)          # upstream: random.choice(all_choices)


def metrics(rows) -> dict:
    return {"acc": common.pct(r["correct"] for r in rows), **common.by(rows, "reasoning", "reasoning/"),
            **common.by(rows, "demonstration", "demo/")}


def smoke(samples):
    """8 questions per reasoning type (types in prepared order, sha256(id) order within a type) = 48."""
    by_type = {}
    for s in samples:                              # 8 per reasoning type
        by_type.setdefault(s.metadata["reasoning"], []).append(s)
    return [s.id for group in by_type.values() for s in sorted(group, key=lambda s: hashlib.sha256(s.id.encode()).hexdigest())[:8]]
