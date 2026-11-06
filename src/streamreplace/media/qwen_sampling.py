"""CPU mirror of the frame selection of qwen-vl-utils 0.0.14 `fetch_video` (torchcodec backend).

The Qwen pairs p2 (Qwen2.5-VL-3B, ThinkStream-3B, MMDuet2-3B) and p3 (Qwen3-VL-8B, AURA-8B) sample offline video
(conditions P and C) with Qwen's official qwen-vl-utils: an even number of frames at about 2 fps, uniform over the
whole video. The model adapters call the real library. This module recomputes the same selection (frame indices,
sample fps, frame count) from two numbers torchcodec reports for a video, its exact-mode frame count and average fps
(recorded in the video_meta manifests), with neither torch nor decoding. It is used to verify the frame digests of
the Qwen pairs' rows on any CPU.

What the library does (torchcodec backend, which the adapters require):
- `calculate_video_frame_range`: the whole video unless `video_start`/`video_end` are given;
- `smart_nframes`: n = floor_to_even(clamp(total / fps_src * fps, [min_frames, max_frames])), at most `total`,
  with min_frames 4 and max_frames min(768, total) by default; or round_to_even(nframes) when `nframes` is given;
- positions `torch.linspace(start, end, n)` in float32, rounded half to even (`.round().long()`);
- `sample_fps = n / total * fps_src`.
The float32 positions matter: near a .5 boundary a float32 error can move an index, so `linspace_f32` reproduces
torch's CPU kernel bit for bit (see its docstring). The functions marked "verbatim" are copied from qwen-vl-utils
0.0.14 (Apache-2.0, https://github.com/QwenLM/Qwen2.5-VL), minus logging.
"""
from __future__ import annotations

from fractions import Fraction
import math
from typing import Any

import numpy as np

FPS = 2.0
FRAME_FACTOR = 2
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


def round_by_factor(number: int, factor: int) -> int:
    """Verbatim: the closest integer to `number` divisible by `factor` (Python round, half to even)."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Verbatim: the smallest integer >= `number` divisible by `factor`."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Verbatim: the largest integer <= `number` divisible by `factor`."""
    return math.floor(number / factor) * factor


def smart_nframes(ele: dict[str, Any], total_frames: int, video_fps: int | float) -> int:
    """Verbatim: the number of frames fetch_video samples, from `nframes` or from `fps` (+ min_frames/max_frames)."""
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def calculate_video_frame_range(ele: dict[str, Any], total_frames: int, video_fps: float) -> tuple[int, int, int]:
    """Verbatim: (start frame, end frame, frame count) of the `video_start`/`video_end` window (seconds)."""
    if video_fps <= 0:
        raise ValueError("video_fps must be a positive number")
    if total_frames <= 0:
        raise ValueError("total_frames must be a positive integer")
    video_start = ele.get("video_start", None)
    video_end = ele.get("video_end", None)
    if video_start is None and video_end is None:
        return 0, total_frames - 1, total_frames
    max_duration = total_frames / video_fps
    if video_start is not None:
        video_start_clamped = max(0.0, min(video_start, max_duration))
        start_frame = math.ceil(video_start_clamped * video_fps)
    else:
        start_frame = 0
    if video_end is not None:
        video_end_clamped = max(0.0, min(video_end, max_duration))
        end_frame = math.floor(video_end_clamped * video_fps)
        end_frame = min(end_frame, total_frames - 1)
    else:
        end_frame = total_frames - 1
    if start_frame >= end_frame:
        raise ValueError(
            f"Invalid time range: Start frame {start_frame} (at {video_start_clamped if video_start is not None else 0}s) "
            f"exceeds end frame {end_frame} (at {video_end_clamped if video_end is not None else max_duration}s). "
            f"Video duration: {max_duration:.2f}s ({total_frames} frames @ {video_fps}fps)"
        )
    return start_frame, end_frame, end_frame - start_frame + 1


def _to_float32(value: Fraction) -> float:
    """The float32 nearest to an exact rational (ties to even), as a Python float. Exact for the normal range."""
    if value == 0:
        return 0.0
    sign = -1.0 if value < 0 else 1.0
    num, den = abs(value.numerator), value.denominator
    exponent = num.bit_length() - den.bit_length()              # floor(log2 |value|) is exponent or exponent - 1
    if (num << max(0, -exponent)) < (den << max(0, exponent)):
        exponent -= 1                                           # now 2**exponent <= |value| < 2**(exponent + 1)
    shift = 23 - exponent                                       # 24-bit significand
    scaled_num, scaled_den = (num << shift, den) if shift >= 0 else (num, den << -shift)
    significand, remainder = divmod(scaled_num, scaled_den)
    if 2 * remainder > scaled_den or (2 * remainder == scaled_den and significand % 2):
        significand += 1
    return sign * math.ldexp(significand, -shift)


def linspace_f32(start: int, end: int, steps: int, fma: bool = True) -> list[float]:
    """torch.linspace(start, end, steps) in float32 (torch's default dtype), as torch 2.8 computes it on a CPU.

    step = float32(float32(end - start) / float32(steps - 1)); element i is start + step * i for i < steps // 2 and
    end - step * (steps - 1 - i) otherwise. fma=True evaluates each element as one fused multiply-add rounded once to
    float32: bit-identical to torch 2.8 on the x86 AVX-512 CPU it was checked on (tests/media). fma=False rounds the
    product and the sum separately, as code compiled without fused multiply-add does. Both select the same frames on
    every video of the paper's benchmarks (docs/frames.md).
    """
    if steps == 1:
        return [float(np.float32(start))]
    first, last = np.float32(start), np.float32(end)
    step = np.float32(np.float32(last - first) / np.float32(steps - 1))
    halfway = steps // 2
    exact_step, exact_first, exact_last = Fraction(float(step)), Fraction(float(first)), Fraction(float(last))
    values = []
    for i in range(steps):
        if fma:
            if i < halfway:
                values.append(_to_float32(exact_first + exact_step * i))
            else:
                values.append(_to_float32(exact_last - exact_step * (steps - 1 - i)))
        elif i < halfway:
            values.append(float(np.float32(first + np.float32(step * np.float32(i)))))
        else:
            values.append(float(np.float32(last - np.float32(step * np.float32(steps - 1 - i)))))
    return values


def fetch_video_selection(ele: dict[str, Any], total_frames: int, video_fps: float, fma: bool = True) -> dict:
    """What fetch_video(ele, return_video_sample_fps=True, return_video_metadata=True) selects with the torchcodec
    backend for a video whose VideoDecoder metadata gives `total_frames` (num_frames) and `video_fps` (average_fps):
    {"frames_indices", "total_num_frames", "fps", "video_backend"} (its video_metadata) plus "sample_fps".
    Frame decoding and resizing are not reproduced; pixel keys of `ele` are ignored."""
    start_frame, end_frame, total_frames = calculate_video_frame_range(ele, total_frames, video_fps)
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    # torch's .round() rounds half to even, as Python's round() does.
    indices = [round(value) for value in linspace_f32(start_frame, end_frame, nframes, fma=fma)]
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    return {"frames_indices": indices, "total_num_frames": total_frames, "fps": video_fps,
            "video_backend": "torchcodec", "sample_fps": sample_fps}


def element(video: dict) -> dict:
    """The frame-selection keys of fetch_video's element as the Qwen adapters build it from the merged `video`
    settings: fps, min_frames and max_frames when set, and nframes only when there is no fps."""
    ele = {key: video[key] for key in ("fps", "min_frames", "max_frames") if key in video}
    if "fps" not in ele:
        ele["nframes"] = video["nframes"]
    return ele
