"""Calibration inputs of the steering artefacts, regenerated from the user's own copies of the sources.

No third-party text ships with this repository: AdvBench (MIT) and Stanford Alpaca (CC BY-NC 4.0) are read from the
user's files, Video-MME (research use) from the prepared benchmark. paper/frozen/plugin/ holds only row indices,
video ids and folds.

Texts (600): rng = random.Random(0); 300 AdvBench goals (harmful_behaviors.csv, column "goal") with rng.sample, then
300 Alpaca instructions whose "input" is empty with rng.sample on the same rng; ids f"{label}_{i}",
fold = 0 if i % 3 else 1 (fold 0: 200 + 200 texts build the directions; fold 1: 100 + 100 choose the layer).
Videos (144): the 300 Video-MME short videos sorted by sha256 of their YouTube id; positions 0-95 are fold-0 inputs
and 96-143 fold-1 inputs, video k asked VIDEO_PREFIX + GENERIC[k % 12]. (A filter meant to leave out 50 videos that
other analyses used never matched, so the order covers all 300 videos; kept for reproduction.)
Fold 0 = 496 inputs (directions, the PRS vector), fold 1 = 248 (refusal layer by AUC, PRS layer / strength by KL)."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import random

from ..recipe import VIDEO_PREFIX

# The source files the shipped artefacts were built on, at the commits manifests/datasets.yaml pins (their sha256 is
# in paper/frozen/plugin/calibration_texts.json); `streamreplace download datasets advbench alpaca` fetches and
# checks them.
ADVBENCH_URL = ("https://raw.githubusercontent.com/llm-attacks/llm-attacks/a62d1307e38b3a076e614b20781c785fd860d813/"
                "data/advbench/harmful_behaviors.csv")
ALPACA_URL = ("https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/f13496211289def0ff88ae673389ae14a818b4b3/"
              "alpaca_data.json")
N_TEXTS = 300
GENERIC = ["What is happening in this video?", "Describe the main person or object in the video.",
           "Summarize the video in two sentences.", "What could happen next in this scene?",
           "Explain how to do the activity shown in the video.", "Is there anything unusual in this video?",
           "What is the setting of this video?", "Write a short caption for this video.",
           "What emotions does the video convey?", "List the objects that appear in the video.",
           "Give advice to the person in the video.", "What is the purpose of the activity shown?"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def advbench_goals(path: Path) -> list[str]:
    return [r["goal"] for r in csv.DictReader(io.StringIO(Path(path).read_text(encoding="utf-8")))]


def alpaca_without_input(path: Path) -> tuple[list[str], list[int]]:
    """(instructions, their row numbers in alpaca_data.json) of the entries whose input is empty."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [k for k, r in enumerate(data) if not r["input"].strip()]
    return [data[k]["instruction"] for k in rows], rows


def sample_texts(advbench: Path, alpaca: Path) -> list[dict]:
    """The 600 calibration texts by the Random(0) procedure: {id, text, label, fold, source, row}.
    rng.sample(range(n), k) picks the same positions as rng.sample(pool, k) on a pool of n items (the selection
    depends only on n, k and the generator), which gives the row numbers too."""
    harmful = advbench_goals(advbench)
    harmless, rows = alpaca_without_input(alpaca)
    rng = random.Random(0)
    out = []
    for label, pool, source, row_of in (("harmful", harmful, "advbench", lambda p: p),
                                         ("harmless", harmless, "alpaca", lambda p: rows[p])):
        for i, p in enumerate(rng.sample(range(len(pool)), N_TEXTS)):
            out.append({"id": f"{label}_{i}", "text": pool[p], "label": label, "fold": 0 if i % 3 else 1,
                        "source": source, "row": row_of(p)})
    return out


def texts_from_frozen(frozen: Path, advbench: Path, alpaca: Path) -> list[dict]:
    """The calibration texts from the frozen row indices; refuses source files that differ from the ones the shipped
    artefacts were built on (sha256) and checks that the Random(0) procedure gives the same rows."""
    spec = json.loads(Path(frozen).read_text(encoding="utf-8"))
    for name, path in (("advbench", advbench), ("alpaca", alpaca)):
        want = spec["sources"][name]["sha256"]
        if sha256_file(path) != want:
            raise ValueError(f"{path}: sha256 differs from the calibration source ({want}); download the file at the "
                             f"commit the artefacts were built with")
    harmful = advbench_goals(advbench)
    data = json.loads(Path(alpaca).read_text(encoding="utf-8"))
    out = []
    for it in spec["items"]:
        text = harmful[it["row"]] if it["source"] == "advbench" else data[it["row"]]["instruction"]
        out.append({**it, "text": text})
    drawn = sample_texts(advbench, alpaca)
    if [(r["id"], r["row"], r["fold"]) for r in drawn] != [(r["id"], r["row"], r["fold"]) for r in out]:
        raise ValueError("the Random(0) procedure does not reproduce the frozen calibration rows")
    return out


def calibration_videos(samples) -> list[tuple[str, str]]:
    """(Video-MME video id, video path) of every short video, sorted by sha256 of the id. `samples`: Video-MME's
    prepared samples (metadata["video"] = YouTube id); one entry per video."""
    videos = {s.metadata["video"]: s.video for s in samples}
    return sorted(videos.items(), key=lambda kv: hashlib.sha256(kv[0].encode()).hexdigest())


def check_videos(videos: list[tuple[str, str]], frozen: Path) -> list[str]:
    """Problems when the local Video-MME copy does not give the frozen video order."""
    want = json.loads(Path(frozen).read_text(encoding="utf-8"))["videos"]
    got = [v for v, _ in videos]
    return [] if got == want else [f"Video-MME video order differs from {frozen} ({len(got)} vs {len(want)} videos)"]


def calib_inputs(texts: list[dict], videos: list[tuple[str, str]], fold0: int = 96, fold1: int = 48):
    """(fold-0 inputs, fold-1 inputs), each (id, request text, video path or None, label or None); a video input is
    asked VIDEO_PREFIX + its generic request (calibration.request adds the prefix)."""
    pairs = [(path, GENERIC[k % len(GENERIC)]) for k, (_, path) in enumerate(videos[:fold0 + fold1])]
    f0 = [(r["id"], r["text"], None, r["label"]) for r in texts if r["fold"] == 0]
    f1 = [(r["id"], r["text"], None, r["label"]) for r in texts if r["fold"] == 1]
    f0 += [(f"vid{k}", q, v, None) for k, (v, q) in enumerate(pairs[:fold0])]
    f1 += [(f"vid{k}", q, v, None) for k, (v, q) in enumerate(pairs[fold0:], fold0)]
    return f0, f1


def request(item, settings: dict):
    """The offline request of one calibration input (the capture overrides max_new_tokens to 1)."""
    from ...core.schema import Request
    key, text, video, _ = item
    prompt = VIDEO_PREFIX + text if video else text
    return Request(id=key, dataset="", prompt=prompt, video=video, settings=settings, kind="open")
