"""The CPU mirror of qwen-vl-utils 0.0.14 frame selection (media/qwen_sampling.py), the offline sampler of pairs p2
and p3.

goldens/qwen_selection.json holds what the real library computes, recorded with qwen-vl-utils 0.0.14 and torch 2.8
on an AVX-512 CPU: calculate_video_frame_range + smart_nframes + torch.linspace(start, end, n).round() (the frame
selection of its torchcodec reader, without decoding) for real and synthetic (frame count, fps, element) cases, the
exact float32 values of torch.linspace, and positions where float64 arithmetic would pick another frame.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from streamreplace.media.qwen_sampling import (_to_float32, calculate_video_frame_range, element,
                                               fetch_video_selection, linspace_f32, smart_nframes)

GOLDENS = json.loads((Path(__file__).parent / "goldens" / "qwen_selection.json").read_text())


def _sha(indices: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, indices)).encode()).hexdigest()


def _case_id(case: dict) -> str:
    return f"{case['total']}@{case['fps']:.4f}-{json.dumps(case['element'], separators=(',', ':'))}"


@pytest.mark.parametrize("case", GOLDENS["selection"], ids=_case_id)
def test_selection_equals_the_real_library(case):
    if "error" in case:
        with pytest.raises(ValueError):
            fetch_video_selection(case["element"], case["total"], case["fps"])
        return
    got = fetch_video_selection(case["element"], case["total"], case["fps"])
    indices = got["frames_indices"]
    assert len(indices) == case["n"]
    assert got["total_num_frames"] == case["total_num_frames"]
    assert got["sample_fps"] == case["sample_fps"]                   # the same float, not just close
    assert (got["fps"], got["video_backend"]) == (case["fps"], "torchcodec")
    if "indices" in case:
        assert indices == case["indices"]
    else:
        assert _sha(indices) == case["indices_sha256"] and [indices[0], indices[-1]] == case["first_last"]


@pytest.mark.parametrize("case", GOLDENS["linspace"], ids=lambda c: f"{c['start']}-{c['end']}-{c['steps']}")
def test_linspace_f32_is_bit_identical_to_torch(case):
    values = np.array(linspace_f32(case["start"], case["end"], case["steps"]), dtype=np.float32)
    assert hashlib.sha256(values.tobytes()).hexdigest() == case["float32_sha256"]


@pytest.mark.parametrize("case", GOLDENS["float64_differs"], ids=lambda c: f"{c['end'] + 1}-{c['steps']}")
def test_float32_positions_pick_frames_that_float64_would_not(case):
    """Long videos at 768 frames: torch's float32 positions round to another frame than exact arithmetic would."""
    mirror = [round(v) for v in linspace_f32(case["start"], case["end"], case["steps"])]
    float64 = np.round(np.linspace(case["start"], case["end"], case["steps"])).astype(int).tolist()
    for position, torch_index, float64_index in zip(case["positions"], case["torch"], case["float64"]):
        assert mirror[position] == torch_index
        assert float64[position] == float64_index != torch_index


def test_fma_and_separate_rounding_agree_on_the_frames_of_typical_videos():
    """Rounding the product and the sum separately (code without fused multiply-add) can change the last bit of a
    position; the rounded frame indices of these cases do not change."""
    for total, steps in [(2228, 148), (4320, 33), (899, 60), (17981, 768)]:
        fused = [round(v) for v in linspace_f32(0, total - 1, steps)]
        separate = [round(v) for v in linspace_f32(0, total - 1, steps, fma=False)]
        assert fused == separate


def test_smart_nframes_follows_the_duration_at_2_fps():
    assert smart_nframes({"fps": 2}, total_frames=2228, video_fps=29.97002997002997) == 148      # 74.3 s -> 148.7
    assert smart_nframes({"fps": 2}, total_frames=45, video_fps=30.0) == 4                     # 1.5 s: at least 4
    assert smart_nframes({"fps": 2}, total_frames=54001, video_fps=25.0) == 768                 # 36 min: at most 768
    assert smart_nframes({"fps": 2}, total_frames=301, video_fps=1.0) == 300                   # at most total, even
    assert smart_nframes({"fps": 2, "max_frames": 41}, total_frames=2228, video_fps=29.97) == 40
    assert smart_nframes({"fps": 2, "min_frames": 7}, total_frames=60, video_fps=30.0) == 8


def test_smart_nframes_with_nframes_rounds_half_to_even():
    assert smart_nframes({"nframes": 9}, total_frames=100, video_fps=30.0) == 8                # round(4.5) = 4
    assert smart_nframes({"nframes": 11}, total_frames=100, video_fps=30.0) == 12               # round(5.5) = 6
    with pytest.raises(ValueError):
        smart_nframes({"nframes": 64}, total_frames=30, video_fps=30.0)
    with pytest.raises(AssertionError):
        smart_nframes({"nframes": 8, "fps": 2}, total_frames=100, video_fps=30.0)


def test_frame_range_of_a_time_window():
    assert calculate_video_frame_range({}, 300, 30.0) == (0, 299, 300)
    assert calculate_video_frame_range({"video_start": 1.0, "video_end": 5.5}, 300, 30.0) == (30, 165, 136)
    assert calculate_video_frame_range({"video_start": 2.0}, 300, 30.0) == (60, 299, 240)
    assert calculate_video_frame_range({"video_end": 99.0}, 300, 30.0) == (0, 299, 300)       # clamped
    with pytest.raises(ValueError):
        calculate_video_frame_range({"video_start": 5.0, "video_end": 5.0}, 300, 30.0)      # empty window
    window = fetch_video_selection({"fps": 2, "video_start": 1.0, "video_end": 5.5}, 300, 30.0)
    assert window["frames_indices"][0] == 30 and window["frames_indices"][-1] == 165
    assert window["total_num_frames"] == 136


def test_element_takes_the_selection_keys_of_the_pair_settings():
    p2 = {"nframes": 64, "max_pixels": 401408, "sampler": "qwen_vl_utils", "fps": 2, "min_pixels": 200704}
    assert element(p2) == {"fps": 2}                                     # nframes (from defaults) is ignored
    assert element({"nframes": 16, "max_pixels": 1}) == {"nframes": 16}
    assert element({"fps": 2, "min_frames": 6, "max_frames": 40}) == {"fps": 2, "min_frames": 6, "max_frames": 40}


def test_float32_rounding_of_exact_rationals():
    assert _to_float32(Fraction(1, 3)) == float(np.float32(1 / 3))
    assert _to_float32(Fraction(2 ** 24 + 1)) == 2.0 ** 24                # a tie rounds to the even significand
    assert _to_float32(Fraction(2 ** 24 + 3)) == 2.0 ** 24 + 4
    assert _to_float32(Fraction(-5, 2)) == -2.5 and _to_float32(Fraction(0)) == 0.0
    assert linspace_f32(5, 5, 1) == [5.0]
