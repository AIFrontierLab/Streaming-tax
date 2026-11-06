"""OpenUAV-QA (arXiv 2607.12477; annotations HF choucsan/OpenUAV-QA @121d55c3, frames ModelScope choucisan/OpenUAV-QA):
4-option video QA on egocentric drone trajectories (task_type path_planning), train 11,476 / test 3,895 (seen maps
1,900 + unseen maps 1,995). An optional benchmark of the fine-tuning experiment: its tables are reported, but it is
left out of the primary endpoint (fine-tuned models reach 98-100% with blank frames: the options' text gives the
answer away; docs/generalization/finetune.md).

The official code (Self-in-Space @df44978, Apache-2.0) is fetched by `streamreplace fetch-upstream openuav_qa`:
  frames    the trajectory's frame files, sorted; more than 32 -> np.linspace(0, n - 1, 32, dtype=int)
            (uav_dataset.py SISBenchDataset._load_images); prepare writes them as a frame list, fed as one video with
            fps 2.0 (the official LoRA-eval path, lora/src/eval/dataset_utils.py); pixel floor 256*28*28 (run_uav.py,
            configs/benchmarks.yaml)
  prompt    UAV_PROMPT_TEMPLATES["default"] with SISBenchDataset._format_options ("A. text", sorted keys)
  decoding  greedy, max_new_tokens 128 (eval/run_eval.sh)
  scoring   run_uav.py extract_answer (strip <think>, regex cascade) == the answer letter; no judge
Frames are chosen from raw/modelscope_tree.jsonl (every trajectory's file list, written by the downloader), so prepare
does not need them on disk; only the chosen frames of the selected trajectories have to be downloaded. Trajectories
without frames on ModelScope are skipped, as SISBenchDataset._validate_frames does; so are those with a chosen frame
ModelScope lists but serves empty (raw/modelscope_unavailable.json), since upstream could not read them either."""
from __future__ import annotations

from functools import cache
import json
from pathlib import Path
import re

from .. import common
from ...core.schema import Sample

MEDIA = "video"
JUDGE = None                 # official scoring is the regex extractor only
ACCEPTS_OPTION_TEXT = False  # extract_answer reads letters only
EXPECTED_COUNT = 3893
GROUP = "source_traj"        # one trajectory per question; the key of the train/test disjointness check
STRATUM_KEYS = ("source_map",)   # the <= 2K test subset keeps every map's share
DATASET = "openuav_qa/eval/src/uav/vqa/uav_dataset.py"
RUN = "openuav_qa/eval/src/uav/vqa/run_uav.py"
MAX_FRAMES = 32              # eval/run_eval.sh MAX_FRAMES
FPS = 2.0                    # lora/src/eval/dataset_utils.py: frames passed as one video at fps 2.0
SUFFIXES = (".jpg", ".jpeg", ".png")


def select_frames(names: list[str]) -> list[str]:
    """SISBenchDataset._load_images for a frame directory: the frame files sorted, linspace-subsampled above MAX_FRAMES
    (checked against the compiled upstream method in the tests)."""
    import numpy as np
    files = sorted(n for n in names if Path(n).suffix.lower() in SUFFIXES)
    if len(files) > MAX_FRAMES:
        files = [files[i] for i in np.linspace(0, len(files) - 1, MAX_FRAMES, dtype=int)]
    return files


def listing(raw: Path) -> dict[str, list[str]]:
    """Trajectory folder -> its frame file names on ModelScope."""
    if not (raw / ".listing_complete").exists():
        raise ValueError(f"{raw}: ModelScope file list incomplete; download the OpenUAV-QA file listing first")
    return {e["folder"]: [f["name"] for f in e["files"]]
            for e in map(json.loads, (raw / "modelscope_tree.jsonl").read_text().splitlines())}


def unavailable(raw: Path) -> set[str]:
    """"<folder>/<file>" that ModelScope lists but serves empty (recorded by the frame download)."""
    path = raw / "modelscope_unavailable.json"
    return set(json.loads(path.read_text())) if path.exists() else set()


def read_split(raw: Path, split: str) -> list[dict]:
    return [json.loads(line) for line in (raw / f"TravelUAV_{split}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _samples(raw: Path, media: Path, split: str) -> list[Sample]:
    from ...media.video_meta import write_frame_list
    files, missing = listing(raw), unavailable(raw)
    samples, skipped, short = [], [], 0
    for item in read_split(raw, split):
        names = [n for n in files.get(item["video_name"], []) if Path(n).suffix.lower() in SUFFIXES]
        frames = select_frames(names)
        if not frames or any(f"{item['video_name']}/{f}" in missing for f in frames):
            skipped.append(item["id"])
            continue
        listed = len(names)
        short += listed != item["total_frames"]
        folder = raw / "frames" / item["video_name"]
        video = write_frame_list(media / f"{item['id']}.frames.json", [str(folder / f) for f in frames], FPS)
        samples.append(Sample(item["id"], item["question"], item["answer"], "mcq", dict(sorted(item["options"].items())),
                              video=str(video),
                              metadata={"split": item.get("split", "train"), "source_map": item["source_map"],
                                        "source_traj": item["source_traj"], "total_frames": item["total_frames"],
                                        "frames_listed": listed, "frames": len(frames)}))
    print(f"openuav_qa {split}: {len(samples)} samples; skipped {len(skipped)} without frames "
          f"{skipped[:30]}; {short} with frame count != total_frames", flush=True)
    return samples


def prepare(raw: Path, media: Path) -> list[Sample]:
    return _samples(raw, media, "test")


def prepare_train(raw: Path, media: Path) -> list[Sample]:
    return _samples(raw, media, "train")


@cache
def _official():
    """The upstream prompt template, option formatter and answer extractor, compiled from the fetched source once."""
    method = "SISBenchDataset._format_options"
    return (common.upstream_constant(DATASET, "UAV_PROMPT_TEMPLATES")["default"],
            common.upstream_functions(DATASET, method)[method],
            common.upstream_functions(RUN, "extract_answer", scope={"re": re})["extract_answer"])


def prompt(sample) -> str:
    template, format_options, _ = _official()
    return template.format(question=sample.question, options_str=format_options(None, sample.choices))


def extract_answer(text: str):
    return _official()[2](text)


def parse(sample, text):
    letter = extract_answer(common.clean(text))
    return (letter, "official") if letter else (None, "official_invalid")


def metrics(rows) -> dict:
    return {"acc": common.pct(r["correct"] for r in rows), **common.by(rows, "split", "split/")}


def smoke(samples):
    return common.smoke(samples, n=48)
