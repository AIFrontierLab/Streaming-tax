"""LingoQA (Marcu et al., arXiv 2312.14115, github.com/wayveai/LingoQA): driving video QA, the fine-tuning experiment's
driving domain (docs/generalization/finetune.md; Supp. Tables eval_finetuning / eval_finetuning_gains).

Data: 5 frames per 4 s segment at 1 Hz (2048x1280 JPEG), from the authors' Google Drive into `raw/` (the Wayve
licence is personal and non-sublicensable and covers the evaluation code too, so neither data nor code is shipped:
the scorer is fetched at a pinned commit by `streamreplace fetch-upstream lingoqa`).
  test      the official evaluation set (evaluation/val.parquet): 500 questions x 2 reference answers, 100 segments
  train     Action (265,323 QA / 24,491 segments) + Scenery train.parquet; (segment, question) duplicates dropped
  frames    all 5 frames of the segment as a frame list at fps 1.0 (the paper's 5-frame baseline input); the
            benchmark's `offline_frames: {sampler: all}` (configs/benchmarks.yaml) reads them as stored
  prompt    the question verbatim (no official template; the shared default prompt of an open question is its
            stripped text); decoding greedy, max_new_tokens 256 (no official value)
  scoring   the official Lingo-Judge (benchmark/judge.py LingoJudge.compute, as evaluate.py's evaluate_question):
            max classifier logit over the references > 0 -> correct; run by judges/lingo.py (`score --judge`). The
            reply goes through the shared clean() first (reasoning before a </think> removed), as for every dataset.
The reference answers are private metadata: they reach the judge, never a model."""
from __future__ import annotations

import ast
from pathlib import Path

from .. import common
from ...core.schema import Sample

MEDIA = "video"
JUDGE = "lingo"               # scored only by the official Lingo-Judge (score --judge; judges/lingo.py)
EXPECTED_COUNT = 500
PRIVATE_METADATA = frozenset({"references"})
GROUP = "segment_id"          # questions of one segment share frames: one cluster in paired statistics / leakage checks
CHECK_NEAR_DUPLICATES = True  # Scenery comes from a few long drives: training frames close to evaluation frames are dropped
INTERLEAVE = "subset"         # training order alternates Action / Scenery 1:1, as the paper's baseline samples them
FPS = 1.0
TRAIN = ("action", "scenery")


def _frames(row: dict) -> list[str]:
    images = row["images"]
    return list(images) if isinstance(images, (list, tuple)) else ast.literal_eval(images)   # val.parquet: a str repr


def _video(raw: Path, media: Path, part: str, row: dict) -> str:
    from ...media.video_meta import write_frame_list
    frames = [str(raw / part / f) for f in _frames(row)]
    return str(write_frame_list(media / part / f"{row['segment_id']}.frames.json", frames, FPS))


def prepare(raw: Path, media: Path) -> list[Sample]:
    """One sample per question with its reference answers in order (evaluate.py groups by question_id/segment_id)."""
    groups: dict[str, list[dict]] = {}
    for row in common.read_parquet(raw / "evaluation/val.parquet"):
        groups.setdefault(row["question_id"], []).append(row)
    samples = []
    for question_id, rows in groups.items():
        first = rows[0]
        assert all(r["segment_id"] == first["segment_id"] and r["question"] == first["question"] for r in rows)
        references = [r["answer"] for r in rows]
        samples.append(Sample(question_id, first["question"], references[0], "open",
                              video=_video(raw, media, "evaluation", first),
                              metadata={"segment_id": first["segment_id"], "references": references}))
    return samples


def prepare_train(raw: Path, media: Path) -> list[Sample]:
    samples, seen = [], set()
    for part in TRAIN:
        videos: dict[str, str] = {}                     # one frame list per segment
        for row in common.read_parquet(raw / part / "train.parquet"):
            key = (row["segment_id"], row["question"])
            if key in seen:
                continue
            seen.add(key)
            if row["segment_id"] not in videos:
                videos[row["segment_id"]] = _video(raw, media, part, row)
            samples.append(Sample(row["question_id"], row["question"], row["answer"], "open",
                                  video=videos[row["segment_id"]],
                                  metadata={"segment_id": row["segment_id"], "subset": part}))
    return samples


def parse(sample, text):
    """The student answer sent to Lingo-Judge (the cleaned reply); correctness comes from the judge."""
    answer = common.clean(text)
    return (answer, "text") if answer else (None, "empty")


def metrics(rows) -> dict:
    """Lingo-Judge accuracy over all questions (evaluate.py: correct / evaluated), in %; errored rows count as wrong."""
    return {"acc": common.pct(r["correct"] for r in rows)}


def smoke(samples):
    return common.smoke(samples, group=GROUP, n=50)
