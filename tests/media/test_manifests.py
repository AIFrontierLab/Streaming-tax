"""The shipped manifests/video_meta/<benchmark>.jsonl and the paper's offline frames recomputed from them.

goldens/paper_digests.json lists, for a few videos of each video benchmark (the first, the shortest, the longest and an
edit-list video) and each pair, what the paper's parent run recorded in its row: the frame digest
(info.frame_indices), the frame count given to the model, the duration and, for the Qwen pairs, the sample fps. Every
member of a pair recorded the same digest. These tests recompute all of it on a CPU from the manifest values alone.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import random

import pytest
import yaml

from streamreplace.media.digest import frame_digest
from streamreplace.media.frames import offline_indices
from streamreplace.media.qwen_sampling import element, fetch_video_selection
from streamreplace.media.video_meta import (compare, dataset_videos, default_manifest_dir, library_versions,
                                            load_manifest, manifest_info, measure)
from streamreplace.paths import project_root

VIDEOS = {"videomme": 300, "tomato": 1397, "timeblind": 1200, "videoreasonbench": 240, "video_safetybench": 1132,
          "mcv_safetybench": 2920, "vmdt_fairness": 2000, "vmdt_privacy": 200}
EDIT_LIST_VIDEOS = {"tomato": 172, "timeblind": 58}          # clips cut without re-encoding (docs/frames.md)

# The paper's pair protocols (configs/pairs.yaml) and offline frame overrides (configs/benchmarks.yaml).
DEFAULT_VIDEO = {"nframes": 64, "max_pixels": 200704}
PAIRS = {"p1": {"fps": 2, "max_frames": 160, "pad_square": True},
         "p2": {"sampler": "qwen_vl_utils", "fps": 2, "min_pixels": 200704, "max_pixels": 401408},
         "p3": {"sampler": "qwen_vl_utils", "fps": 2, "max_pixels": 602112},
         "p4": {"fps": 1},
         "p5": {"fps": 1, "max_pixels": 16777216, "video_max_pixels": 65536000}}
OFFLINE_FRAMES = {"video_safetybench": {"nframes": 16, "sampler": "linspace_int"},
                  "mcv_safetybench": {"nframes": 33, "sampler": "linspace_int"}}
SAMPLING_KEYS = {"fps", "nframes", "sampler", "max_frames", "min_frames", "last_seconds"}
QWEN_PAIRS = {"p2", "p3"}                                    # every member is a Qwen2.5-VL / Qwen3-VL architecture

GOLDENS = json.loads((Path(__file__).parent / "goldens" / "paper_digests.json").read_text())["cases"]


def offline_video(benchmark: str, pair: str) -> dict:
    """The merged offline `video` settings: defaults, the pair's block, then the benchmark's offline_frames, which
    replace the sampling keys only."""
    video = {**DEFAULT_VIDEO, **PAIRS[pair]}
    if benchmark in OFFLINE_FRAMES:
        video = {**{k: v for k, v in video.items() if k not in SAMPLING_KEYS}, **OFFLINE_FRAMES[benchmark]}
    return video


def recorded_info(benchmark: str, pair: str, row: dict) -> dict:
    """The frame info an offline row of `pair` records for the manifest `row` (the adapters' rules)."""
    video = offline_video(benchmark, pair)
    if video.get("sampler") == "qwen_vl_utils":
        torchcodec = row["torchcodec"]
        selected = fetch_video_selection(element(video), torchcodec["frames"], torchcodec["fps"])
        indices = selected["frames_indices"]
        return {"digest": frame_digest({"indices": indices}), "frames": len(indices),
                "duration": round(selected["total_num_frames"] / selected["fps"], 2),
                "sample_fps": round(selected["sample_fps"], 4)}
    info = manifest_info(row)
    indices = offline_indices(info, video)
    result = {"digest": frame_digest({"indices": indices}), "frames": len(indices),
              "duration": round(info["duration"], 2)}
    if pair in QWEN_PAIRS:                                   # an odd count is padded with the last frame
        result["frames"] += result["frames"] % 2
        result["sample_fps"] = round(result["frames"] / max(info["duration"], 1e-6), 4)
    return result


def test_there_is_one_manifest_per_video_benchmark():
    assert {p.stem for p in default_manifest_dir().glob("*.jsonl")} == set(VIDEOS)


@pytest.mark.parametrize("benchmark", sorted(VIDEOS))
def test_manifest_rows_are_complete_relative_and_sorted(benchmark):
    rows = [json.loads(line) for line in (default_manifest_dir() / f"{benchmark}.jsonl").read_text().splitlines()]
    videos = [row["video"] for row in rows]
    assert len(rows) == VIDEOS[benchmark] and videos == sorted(set(videos))
    for row in rows:
        path = PurePosixPath(row["video"])
        assert not path.is_absolute() and ".." not in path.parts and "\\" not in row["video"] and ":" not in row["video"]
        assert set(row) == {"video", "frames", "fps", "header_frames", "torchcodec"}
        assert isinstance(row["frames"], int) and row["frames"] >= 1 and row["fps"] > 0
        assert row["header_frames"] >= row["frames"]
        torchcodec = row["torchcodec"]
        assert set(torchcodec) == {"frames", "fps", "duration"} and torchcodec["fps"] > 0
        assert torchcodec["frames"] == row["frames"]         # FFmpeg's exact scan and the PyAV decode agree
    assert sum(row["header_frames"] > row["frames"] for row in rows) == EDIT_LIST_VIDEOS.get(benchmark, 0)


@pytest.mark.parametrize("case", GOLDENS, ids=lambda c: f"{c['benchmark']}-{c['pair']}-{c['video']}")
def test_paper_frames_are_recomputed_from_the_manifest(case):
    row = load_manifest(case["benchmark"])[case["video"]]
    expected = {key: case[key] for key in ("digest", "frames", "duration", "sample_fps") if key in case}
    assert recorded_info(case["benchmark"], case["pair"], row) == expected


def test_goldens_cover_every_benchmark_pair_and_sampler():
    assert {c["benchmark"] for c in GOLDENS} == set(VIDEOS)
    assert {(c["benchmark"], c["pair"]) for c in GOLDENS} == {(b, p) for b in VIDEOS for p in PAIRS}
    edit_list = {(c["benchmark"], c["video"]) for c in GOLDENS
                 if load_manifest(c["benchmark"])[c["video"]]["header_frames"]
                 > load_manifest(c["benchmark"])[c["video"]]["frames"]}
    assert {b for b, _ in edit_list} == set(EDIT_LIST_VIDEOS)
    assert any(c["pair"] == "p1" and c["frames"] == 160 for c in GOLDENS)         # p1's 160-frame cap
    assert any(c["pair"] == "p2" and c["frames"] == 4 for c in GOLDENS)           # qwen-vl-utils' 4-frame floor


def test_protocols_above_are_the_shipped_configuration():
    configs = project_root() / "configs"
    pairs = yaml.safe_load((configs / "pairs.yaml").read_text())["pairs"]
    assert {name: pair["video"] for name, pair in pairs.items()} == PAIRS
    benchmarks = yaml.safe_load((configs / "benchmarks.yaml").read_text())["benchmarks"]
    assert {name: benchmarks[name]["offline_frames"] for name in OFFLINE_FRAMES} == OFFLINE_FRAMES
    defaults = yaml.safe_load((configs / "defaults.yaml").read_text())["defaults"]
    assert defaults["video"] == DEFAULT_VIDEO


def test_the_qwen_pairs_read_video_with_qwen_vl_utils_and_ffmpeg():
    """Every member of p2 and p3 is a Qwen2.5-VL / Qwen3-VL architecture. Offline it runs the qwen_offline adapter,
    whose videos come from qwen-vl-utils with torchcodec, so its environment needs the FFmpeg libraries: without them
    the library silently falls back to torchvision (other frames), which the adapter refuses."""
    configs = project_root() / "configs"
    pairs = yaml.safe_load((configs / "pairs.yaml").read_text())["pairs"]
    models = yaml.safe_load((configs / "models.yaml").read_text())["models"]
    for pair in sorted(QWEN_PAIRS):
        assert pairs[pair]["video"]["sampler"] == "qwen_vl_utils"
        for model in (pairs[pair]["parent"], *pairs[pair]["children"]):
            offline = models[model]["offline"]
            assert (offline["adapter"], offline.get("ffmpeg")) == ("qwen_offline", True), model


# With the data (`streamreplace prepare <benchmark>`) -------------------------------------------------------------------

def _dataset_dir(benchmark: str) -> Path:
    from streamreplace import paths
    return paths.dataset_dir(benchmark)


@pytest.mark.parametrize("benchmark", [pytest.param(b, marks=pytest.mark.requires_data(b)) for b in sorted(VIDEOS)])
def test_prepared_videos_are_the_manifest_videos(benchmark):
    assert dataset_videos(_dataset_dir(benchmark)) == sorted(load_manifest(benchmark))


@pytest.mark.parametrize("benchmark", [pytest.param(b, marks=pytest.mark.requires_data(b)) for b in sorted(VIDEOS)])
def test_local_libraries_measure_the_manifest_values(benchmark):
    """Two random videos decoded again with the local libraries (PyAV count; decord and torchcodec when installed).
    `streamreplace video-meta verify` does this for every video."""
    readers = library_versions()
    manifest = load_manifest(benchmark)
    for video in random.Random(0).sample(sorted(manifest), 2):
        measured = measure(str(_dataset_dir(benchmark) / video), reader="decord" if readers["decord"] else None,
                           torchcodec=readers["torchcodec"] is not None, recount=True)
        assert compare(manifest[video], measured) == [], video
