"""Video-MME without subtitles, short split only: 300 videos x 3 multiple-choice questions = 900 (the full set has
2700 questions over short / medium / long videos).

Paper use: video understanding, all three conditions (P, C offline; S through the child's streaming interface, with
the question at the end of the video): the absolute-score video table and the C-P, S-C, S-P contrasts. Headline:
`acc` over all 900 questions; the statistics resample videos (the 3 questions of a video form one cluster).

Prompt: as lmms-eval (tasks/videomme, default model kwargs); frames come from the pair's video protocol. Scoring is
the official eval_your_results.py (fetched upstream code, manifests/upstream.yaml): extract_characters_regex, and no
judge ("The evaluation does not introduce any third-party models"). Upstream divides by the questions it could
extract a letter from; `acc` divides by all questions (silence or no letter = wrong), `acc_official` is upstream's
ratio and `answered` the share it uses. docs/benchmarks/videomme.md."""
from __future__ import annotations

from pathlib import Path
import re

from .. import common
from ..base import JudgePolicy, MetricSpec
from ...core.schema import Sample

NAME = "videomme"
MEDIA = "video"
JUDGE = JudgePolicy.NONE
EXPECTED_COUNT = 900
METRICS = [MetricSpec("acc", "acc", cluster="video")]
OFFICIAL = "videomme/eval_your_results.py"
DURATIONS = {"short"}           # only the short videos (< 2 min) are evaluated
# lmms-eval videomme_doc_to_text, verbatim; it mentions subtitles even in the task without subtitles
OPTION_PROMPT = ("Select the best answer to the following multiple-choice question based on the video and the subtitles. "
                 "Respond with only the letter (A, B, C, or D) of the correct option.")
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."


def prepare(raw: Path, media: Path) -> list[Sample]:
    """raw/videomme/test-00000-of-00001.parquet + raw/videos_chunked_*.zip and raw/subtitle.zip (extracted into
    media/) -> the 900 questions of the short split; videos at media/data/<videoID>.mp4."""
    for archive in sorted(raw.glob("videos_chunked_*.zip")) + [raw / "subtitle.zip"]:
        common.extract_zip(archive, media)
    samples = []
    for row in common.read_parquet(raw / "videomme/test-00000-of-00001.parquet"):
        if row["duration"] not in DURATIONS:
            continue
        choices = {}
        for option in row["options"]:
            label, text = re.match(r"^\s*([A-D])[.)]\s*(.*)$", option, re.S).groups()
            choices[label] = text.strip()
        samples.append(Sample(row["question_id"], row["question"].strip(), row["answer"].strip(), "mcq", choices,
                              video=str(media / "data" / f"{row['videoID']}.mp4"),
                              # Relative to the dataset folder, like the media paths. No prompt reads it: the
                              # paper's runs use no subtitles.
                              metadata={"video": row["videoID"], "duration": row["duration"], "domain": row["domain"],
                                        "task": row["task_type"],
                                        "subtitle": f"{media.name}/subtitle/{row['videoID']}.srt"}))
    if len(samples) != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} short Video-MME questions, got {len(samples)}")
    return samples


def prompt(sample) -> str:
    """Option prompt, question, raw "A. ..." option lines, then the default post_prompt (which starts with a newline)."""
    options = "\n".join(f"{k}. {v}" for k, v in sample.choices.items())
    return f"{OPTION_PROMPT}\n{sample.question}\n{options}\n{POST_PROMPT}"


def parse(sample, text):
    """Official extract_characters_regex on the cleaned reply: the first A-D letter after removing the answer
    prefixes ('' when a reply of more than 10 words has no A-D letter at all)."""
    letter = common.upstream_module(OFFICIAL).extract_characters_regex(common.clean(text))
    return (letter, "official") if letter else (None, "official_unparsed")


def metrics(rows) -> dict:
    answered = [r for r in rows if r.get("answered")]
    return {"acc": common.pct(r["correct"] for r in rows), "acc_official": common.pct(r["correct"] for r in answered),
            "answered": common.pct(r.get("answered") for r in rows),
            **common.by(rows, "duration", "duration/"), **common.by_length(rows),
            **common.by(rows, "task", "task/"), **common.by(rows, "domain", "domain/")}


def smoke(samples):
    return common.smoke(samples, group="video", n=48)
