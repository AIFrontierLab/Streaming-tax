"""Video metadata (media/video_meta.py): frame-list videos, the values measured on a video file, and the video_meta
manifest helpers. Videos are generated with PyAV (conftest.py); decord and torchcodec checks run when installed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from streamreplace.media import video_meta
from streamreplace.media.digest import frame_digest
from streamreplace.media.frames import offline_frames, sample_frames
from streamreplace.media.video_meta import (compare, dataset_videos, default_manifest_dir, is_frame_list,
                                            library_versions, load_manifest, manifest_info, manifest_path, measure,
                                            measure_videos, pyav_meta, read_frame_list, reader_meta, video_info,
                                            write_frame_list, write_manifest)

READERS = library_versions()


def test_library_versions_name_the_three_readers():
    assert set(READERS) == {"av", "decord", "torchcodec"} and READERS["av"]


# Frame lists -----------------------------------------------------------------------------------------------------------

def test_frame_list_stores_paths_relative_to_the_list(tmp_path, new_images):
    images = new_images(tmp_path / "data/media/clip", 5)
    path = write_frame_list(tmp_path / "data/media/clip.frames.json", images, 1)
    assert json.loads(path.read_text()) == {"fps": 1.0, "frames": [f"clip/{i:05d}.png" for i in range(5)]}
    absolute, stored, fps = read_frame_list(str(path))
    assert [Path(p).resolve() for p in absolute] == [p.resolve() for p in images]
    assert stored == [f"clip/{i:05d}.png" for i in range(5)] and fps == 1.0
    before = path.stat().st_mtime_ns
    write_frame_list(path, images, 1.0)                    # same content: not rewritten, cached counts stay valid
    assert path.stat().st_mtime_ns == before
    assert is_frame_list(str(path)) and not is_frame_list("media/clip.mp4")


def test_frame_list_in_another_folder_survives_moving_the_tree(tmp_path, new_images):
    """`prepare` writes lists under media/ for frames left under raw/: the stored paths climb out of the list's folder
    and still resolve after the dataset tree is copied or moved."""
    images = new_images(tmp_path / "ds/raw/frames/traj1", 5)
    path = write_frame_list(tmp_path / "ds/media/traj1.frames.json", images, fps=2.0)
    stored = json.loads(path.read_text())
    assert stored == {"fps": 2.0, "frames": [f"../raw/frames/traj1/{i:05d}.png" for i in range(5)]}
    shutil.copytree(tmp_path / "ds", tmp_path / "moved")
    absolute, relative, fps = read_frame_list(str(tmp_path / "moved/media/traj1.frames.json"))
    assert relative == stored["frames"] and fps == 2.0
    assert all(Path(p).resolve().is_relative_to((tmp_path / "moved").resolve()) for p in absolute)
    frames, meta = offline_frames(str(tmp_path / "moved/media/traj1.frames.json"), {"sampler": "all"})
    assert [f.getpixel((0, 0))[0] for f in frames] == [10 * i for i in range(5)]
    assert meta["timestamps"] == [0.0, 0.5, 1.0, 1.5, 2.0] and meta["duration"] == 2.5
    assert not list(path.parent.glob("*.tmp")) and not list(path.parent.glob(".*.tmp"))


def test_frame_list_frames_are_read_as_stored_and_digested_by_file(tmp_path, new_images):
    images = new_images(tmp_path / "media/clip", 5)
    path = str(write_frame_list(tmp_path / "media/clip.frames.json", images, 2.0))
    assert video_info(path) == {"frames": 5, "header_frames": 5, "fps": 2.0, "duration": 2.5}
    frames, meta = sample_frames(path, nframes=3)
    files = ["clip/00000.png", "clip/00002.png", "clip/00004.png"]
    assert meta["indices"] == [0, 2, 4] and meta["files"] == files
    assert [f.getpixel((0, 0))[0] for f in frames] == [0, 20, 40]           # image i has red value 10 * i
    assert frame_digest(meta) == "3:" + hashlib.sha256(",".join(files).encode()).hexdigest()[:12]
    assert frame_digest(meta) != frame_digest({"indices": [0, 2, 4]})
    frames, meta = offline_frames(path, {"sampler": "all", "fps": 2, "nframes": 64})
    assert len(frames) == 5 and meta["indices"] == [0, 1, 2, 3, 4]


# Values measured on a video --------------------------------------------------------------------------------------------

def test_reader_values_of_a_video(video):
    assert pyav_meta(video) == (12, 4.0)
    assert measure(video, reader="auto") == {"frames": 12, "fps": 4.0, "header_frames": 12}
    assert measure(video, reader=None) == {"frames": 12}
    if READERS["decord"]:
        assert measure(video, reader="decord") == {"frames": 12, "fps": 4.0, "header_frames": 12}


def test_without_decord_the_fps_comes_from_the_pyav_header(video, monkeypatch):
    def missing(path):
        raise ImportError("decord is not installed")
    monkeypatch.setattr(video_meta, "decord_meta", missing)
    assert reader_meta(video) == (12, 4.0)
    assert video_info(video) == {"frames": 12, "header_frames": 12, "fps": 4.0, "duration": 3.0}


def test_measure_records_frames_hidden_by_an_edit_list(tmp_path, new_edit_list_video):
    row = measure(str(new_edit_list_video(tmp_path / "cut.mp4")), reader="auto")
    assert (row["frames"], row["header_frames"], row["fps"]) == (24, 36, 12.0)


@pytest.mark.skipif(READERS["torchcodec"] is None, reason="torchcodec (with its FFmpeg libraries) is not available")
def test_torchcodec_exact_mode_counts_the_decodable_frames(tmp_path, video, new_edit_list_video):
    assert video_meta.torchcodec_meta(video) == {"frames": 12, "fps": 4.0, "duration": 3.0}
    assert video_meta.torchcodec_meta(str(new_edit_list_video(tmp_path / "cut.mp4")))["frames"] == 24
    assert measure(video, reader=None, torchcodec=True)["torchcodec"]["frames"] == 12


def test_recount_neither_reads_nor_writes_the_cache(video, frame_count_cache):
    assert measure(video, reader=None, recount=True) == {"frames": 12}
    assert not frame_count_cache.exists() or not list(frame_count_cache.glob("*.json"))


def test_measure_videos_in_parallel_keeps_the_order(tmp_path, new_video):
    root = tmp_path / "demo"
    new_video(root / "media/a.mp4", frames=12, fps=4)
    new_video(root / "media/b.mp4", frames=6, fps=2)
    videos = ["media/b.mp4", "media/a.mp4"]
    seen = []
    serial = measure_videos(root, videos, progress=lambda done, total: seen.append((done, total)),
                            cache_dir=tmp_path / "counts", reader="auto")
    parallel = measure_videos(root, videos, workers=2, cache_dir=tmp_path / "counts2", reader="auto")
    expected = [{"video": "media/b.mp4", "frames": 6, "fps": 2.0, "header_frames": 6},
                {"video": "media/a.mp4", "frames": 12, "fps": 4.0, "header_frames": 12}]
    assert serial == parallel == expected
    assert seen == [(1, 2), (2, 2)]
    assert len(list((tmp_path / "counts").glob("*.json"))) == 2              # counts cached where asked


# Manifests -------------------------------------------------------------------------------------------------------------

def test_dataset_videos_are_the_unique_videos_of_the_prepared_samples(tmp_path):
    root = tmp_path / "demo"
    with pytest.raises(FileNotFoundError):
        dataset_videos(root)
    (root / "prepared").mkdir(parents=True)
    samples = [{"id": "1", "video": "media/b.mp4"}, {"id": "2", "video": "media/a.mp4"},
               {"id": "3", "video": "media/b.mp4"}, {"id": "4", "images": ["media/x.png"]}, {"id": "5", "video": None}]
    (root / "prepared/samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    assert dataset_videos(root) == ["media/a.mp4", "media/b.mp4"]


def test_manifest_rows_are_written_sorted_and_read_back(tmp_path):
    rows = [{"video": "media/b.mp4", "frames": 8, "fps": 2.0, "header_frames": 8},
            {"video": "media/a.mp4", "frames": 12, "fps": 4.0, "header_frames": 14,
             "torchcodec": {"frames": 12, "fps": 4.0, "duration": 3.0}}]
    path = write_manifest(manifest_path("demo", tmp_path), rows)
    assert path == tmp_path / "demo.jsonl"
    assert [json.loads(line)["video"] for line in path.read_text().splitlines()] == ["media/a.mp4", "media/b.mp4"]
    assert load_manifest("demo", tmp_path) == {row["video"]: row for row in rows}
    assert load_manifest("absent", tmp_path) == {}
    assert manifest_info(rows[0]) == {"frames": 8, "header_frames": 8, "fps": 2.0, "duration": 4.0}


def test_the_default_manifests_are_those_of_the_checkout():
    from streamreplace.paths import project_root
    assert default_manifest_dir() == project_root() / "manifests" / "video_meta"
    assert manifest_path("videomme").is_file() and load_manifest("videomme")


def test_compare_reports_each_differing_value():
    row = {"video": "v.mp4", "frames": 10, "fps": 25.0, "header_frames": 12,
           "torchcodec": {"frames": 10, "fps": 25.0, "duration": 0.4}}
    assert compare(row, dict(row)) == []
    assert compare(row, {"frames": 10}) == []              # values a missing library could not measure are skipped
    measured = {"frames": 11, "fps": 25.000001, "header_frames": 12,
                "torchcodec": {"frames": 10, "fps": 25.0, "duration": 0.44}}
    assert compare(row, measured) == ["frames: manifest 10, measured 11", "fps: manifest 25.0, measured 25.000001",
                                      "torchcodec.duration: manifest 0.4, measured 0.44"]
