"""Golden tests of the frame-sampling rules (media/frames.py), the frame digest (media/digest.py) and the
decodable-frame count (media/video_meta.py)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from streamreplace.media.digest import frame_digest
from streamreplace.media.frames import (frame_indices, offline_frames, offline_indices, offline_sampling,
                                        sample_frames)
from streamreplace.media.video_meta import decodable_frames, video_info


def test_uniform_nframes_on_a_12_frame_video():
    assert frame_indices({"frames": 12, "fps": 4.0}, nframes=4) == [0, 4, 7, 11]


def test_fixed_step_takes_one_frame_every_1_over_fps_seconds():
    info = {"frames": 12, "fps": 4.0}
    assert frame_indices(info, fps=1) == [0, 4, 8]                    # timestamps 0, 1, 2 s
    assert frame_indices(info, fps=2) == [0, 2, 4, 6, 8, 10]
    ntsc = {"frames": 300, "fps": 29.97}                                  # step 14.985 frames at 2 fps, truncated
    assert frame_indices(ntsc, fps=2) == [0, 14, 29, 44, 59, 74, 89, 104, 119, 134, 149, 164, 179, 194, 209, 224,
                                          239, 254, 269, 284]
    assert frame_indices({"frames": 3, "fps": 30.0}, fps=1) == [0]       # shorter than 1/fps: still one frame


def test_fps_wins_over_nframes():
    assert offline_sampling({"nframes": 4, "fps": 1}) == {"fps": 1.0, "max_frames": None}
    info = {"frames": 12, "fps": 4.0}
    assert offline_indices(info, {"nframes": 4, "fps": 1}) == offline_indices(info, {"fps": 1}) == [0, 4, 8]


def test_max_frames_switches_to_uniform_frames_over_the_whole_video():
    info = {"frames": 93 * 24, "fps": 24.0}
    assert len(frame_indices(info, fps=2)) == 186
    capped = frame_indices(info, fps=2, max_frames=160)
    gaps = {b - a for a, b in zip(capped, capped[1:])}
    assert len(capped) == 160 and capped[0] == 0 and capped[-1] == info["frames"] - 1 and max(gaps) - min(gaps) <= 1
    assert frame_indices(info, fps=2, max_frames=400) == frame_indices(info, fps=2)   # under the cap: untouched
    assert offline_indices(info, {"fps": 2, "max_frames": 160, "pad_square": True}) == capped


def test_uniform_positions_round_half_to_even():
    """Python's round(): position 2.5 -> 2 and 1.5 -> 2. A half-up rounding would give [0, 3, 5] and [0, 2, 3]."""
    assert frame_indices({"frames": 6, "fps": 1.0}, nframes=3) == [0, 2, 5]
    assert frame_indices({"frames": 4, "fps": 1.0}, nframes=3) == [0, 2, 3]


@pytest.mark.parametrize("total, count", [(125, 16), (313, 16), (16, 16), (7, 16)])
def test_linspace_int_truncates_like_numpy(total, count):
    """The Video-SafetyBench (16) and MCV-SafetyBench (33) offline overrides."""
    expected = np.linspace(0, total - 1, min(count, total), dtype=int).tolist()
    assert frame_indices({"frames": total, "fps": 30.0}, count, floor=True) == expected
    assert offline_indices({"frames": total, "fps": 30.0}, {"nframes": count, "sampler": "linspace_int"}) == expected


def test_sampler_all_and_no_settings_give_every_frame():
    info = {"frames": 5, "fps": 2.0}
    assert frame_indices(info) == [0, 1, 2, 3, 4]
    assert offline_sampling({"sampler": "all", "fps": 2}) == {}
    assert offline_indices(info, {"sampler": "all"}) == [0, 1, 2, 3, 4]


def test_offline_indices_refuses_the_qwen_sampler():
    with pytest.raises(ValueError):
        offline_indices({"frames": 10, "fps": 2.0}, {"sampler": "qwen_vl_utils", "fps": 2})


def test_digest_is_the_frame_count_and_a_hash_of_the_indices():
    assert frame_digest({"indices": [0, 4, 7, 11]}) == "4:aa99d842319d"        # sha256("0,4,7,11")[:12]
    assert frame_digest({"indices": [0, 4, 7, 11], "timestamps": [0.0, 1.0, 1.75, 2.75]}) == "4:aa99d842319d"


def test_digest_is_taken_before_model_padding(video):
    """Qwen pads an odd frame count with the last frame; the row's digest is that of the sampled frames."""
    frames, meta = sample_frames(video, fps=1)
    assert frame_digest(meta) == "3:" + hashlib.sha256(b"0,4,8").hexdigest()[:12]
    padded = {**meta, "indices": meta["indices"] + meta["indices"][-1:]}
    assert frame_digest(padded) != frame_digest(meta)


def test_frame_sampling_on_a_video_is_uniform_and_deterministic(video):
    frames, meta = sample_frames(video, nframes=4)
    assert len(frames) == 4 and meta["indices"] == [0, 4, 7, 11]
    frames, meta = sample_frames(video, fps=1)
    assert len(frames) == 3 and meta["timestamps"] == [0.0, 1.0, 2.0] and meta["duration"] == 3.0
    assert sample_frames(video, nframes=4)[1] == sample_frames(video, nframes=4)[1]
    brightness = [np.asarray(f).mean() for f in sample_frames(video)[0]]
    assert len(brightness) == 12 and brightness == sorted(brightness)    # the band grows: frames come in order


def test_offline_frames_follow_the_settings_and_blank_keeps_count_and_size(video):
    frames, meta = offline_frames(video, {"nframes": 4, "fps": 1, "max_pixels": 1})   # pixel keys: the adapter's
    assert meta["timestamps"] == [0.0, 1.0, 2.0] and len(frames) == 3
    frames, meta = offline_frames(video, {"nframes": 4, "sampler": "linspace_int"})
    assert meta["indices"] == [0, 3, 7, 11]
    blank, blank_meta = offline_frames(video, {"nframes": 4, "blank": True})
    assert len(blank) == 4 and blank[0].size == (64, 64) and blank[0].getextrema() == ((0, 0), (0, 0), (0, 0))
    assert blank_meta["blank"] and blank_meta["indices"] == [0, 4, 7, 11]


def test_video_info_counts_decodable_frames_and_caches_them(video, frame_count_cache):
    info = video_info(video)
    assert (info["frames"], info["fps"], info["duration"]) == (12, 4.0, 3.0)
    entries = list(frame_count_cache.glob("*.json"))
    assert len(entries) == 1
    assert json.loads(entries[0].read_text()) == {"path": str(Path(video).resolve()), "frames": 12}
    assert video_info(video) == info and len(list(frame_count_cache.glob("*"))) == 1


def test_frame_count_cache_key_is_the_original_format(video, frame_count_cache):
    """sha256("<resolved path>|<size>|<mtime_ns>")[:24]: a frame-count cache folder of the paper's runs is reused."""
    resolved = Path(video).resolve()
    stat = resolved.stat()
    key = hashlib.sha256(f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:24]
    frame_count_cache.mkdir(parents=True, exist_ok=True)
    (frame_count_cache / f"{key}.json").write_text(json.dumps({"path": str(resolved), "frames": 7}))
    assert decodable_frames(video) == 7                                   # read from the cache, not decoded
    assert decodable_frames(video, recount=True) == 12                    # recount ignores the cache ...
    assert json.loads((frame_count_cache / f"{key}.json").read_text())["frames"] == 7   # ... and does not write it
    other = frame_count_cache.parent / "elsewhere"
    assert decodable_frames(video, other) == 12 and len(list(other.glob("*.json"))) == 1


def test_frame_count_ignores_frames_hidden_by_the_edit_list(tmp_path, new_edit_list_video, frame_count_cache):
    video = str(new_edit_list_video(tmp_path / "cut.mp4"))
    info = video_info(video)
    assert (info["header_frames"], info["frames"], info["duration"]) == (36, 24, 2.0)   # the header counts them
    frames, meta = sample_frames(video, fps=12)
    assert meta["indices"] == list(range(24))
    means = [round(np.asarray(f.convert("L")).mean()) for f in frames]
    assert len(set(means)) == 24                                         # no repeated tail frames
    assert video_info(video) == info and len(list(frame_count_cache.glob("*.json"))) == 1
