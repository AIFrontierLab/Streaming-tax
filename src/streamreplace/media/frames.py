"""Deterministic frame sampling shared by every model of a pair (paper conditions P, C and S).

Offline (P and C), every model with the framework sampler reads its frames through `offline_frames` with the pair's
`video` settings block, so all members of a pair see the same frames by construction:
- pairs p1 (2 fps, at most 160 frames), p4 and p5 (1 fps): one frame every 1/fps seconds from frame 0;
- Video-SafetyBench (16 frames) and MCV-SafetyBench (33 frames), for every pair: uniform frames over the whole video
  (`sampler: linspace_int`), the benchmarks' official sampling.
The Qwen pairs p2/p3 sample their other offline videos with Qwen's own qwen-vl-utils instead (qwen_sampling.py).
The p4/p5 streams and the MMDuet2 stream read their frames with `sample_frames` at the stream fps, which gives the same
indices as offline at 1 fps.

The formulas are those of the paper's runs and must not change: a different index changes what every model of the
pair sees. The frame count is the decodable count and the fps is decord's (video_meta.py).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from .video_meta import is_frame_list, read_frame_list, video_info


def open_rgb(path: str) -> Image.Image:
    with Image.open(Path(path)) as image:
        return image.convert("RGB")


def frame_indices(info: dict, nframes: int | None = None, fps: float | None = None,
                  max_frames: int | None = None, floor: bool = False) -> list[int]:
    """Frame indices for a video with `info` = {"frames": decodable count, "fps": source fps}.

    - fps: one frame every 1/fps seconds, i = 0, 1, ...: index min(total - 1, int(i * src_fps / fps)), and
      max(1, int(total / step)) frames, so a trailing part shorter than 1/fps is not sampled. fps wins over nframes.
      If that gives more than `max_frames` frames, `max_frames` uniform frames over the whole video are used instead.
    - nframes (or the max_frames cap): uniform frames from the first to the last, position i * (total - 1) / (n - 1),
      rounded with Python's round() (half to even: 2.5 -> 2), or truncated with floor=True ("linspace_int", which
      matches np.linspace(0, total - 1, n, dtype=int) on the paper's videos).
    - neither: every frame.
    """
    total = info["frames"]
    if fps:
        step = info["fps"] / fps
        indices = [min(total - 1, int(i * step)) for i in range(max(1, int(total / step)))]
        if not (max_frames and len(indices) > max_frames):
            return indices
        nframes = max_frames
    count = max(1, min(nframes or total, total))
    if floor:
        return [int(i * (total - 1) / max(count - 1, 1)) for i in range(count)]
    # round() is half-to-even; numpy's or a "+0.5" rounding would move some indices. Kept: it made the paper's frames.
    return [round(i * (total - 1) / max(count - 1, 1)) for i in range(count)]


def sample_frames(path: str, nframes: int | None = None, fps: float | None = None,
                  max_frames: int | None = None, floor: bool = False) -> tuple[list[Image.Image], dict]:
    """RGB frames at frame_indices() and {"indices", "timestamps", "fps", "duration"} (+ "files" for a frame list).

    Frames are read with decord's get_batch (as in the paper's runs), else decoded with PyAV."""
    info = video_info(path)
    indices = frame_indices(info, nframes, fps, max_frames, floor)
    times = [i / info["fps"] for i in indices]
    meta = {"indices": indices, "timestamps": times, "fps": info["fps"], "duration": info["duration"]}
    if is_frame_list(path):
        paths, stored, _ = read_frame_list(path)
        return [open_rgb(paths[i]) for i in indices], {**meta, "files": [stored[i] for i in indices]}
    try:
        from decord import VideoReader, cpu
        reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        frames = [Image.fromarray(a) for a in reader.get_batch(indices).asnumpy()]
    except ImportError:
        import av
        wanted, frames = {}, []
        for position, index in enumerate(indices):
            wanted.setdefault(index, []).append(position)
        slots = [None] * len(indices)
        with av.open(str(path)) as container:
            for number, frame in enumerate(container.decode(video=0)):
                if number in wanted:
                    image = frame.to_image().convert("RGB")
                    for position in wanted[number]:
                        slots[position] = image
                if number >= indices[-1]:
                    break
        frames = [f for f in slots if f is not None]
    return frames, meta


def offline_sampling(video: dict) -> dict:
    """The sample_frames() arguments for the merged `video` settings of an offline job:
    `sampler: all` = every frame (frame lists whose frames `prepare` already chose by the dataset's official rule);
    else `fps` (one frame every 1/fps s, capped by `max_frames`) wins over `nframes` (uniform; `sampler: linspace_int`
    truncates the positions instead of rounding them). Pixel keys and `pad_square` are the adapter's business."""
    if video.get("sampler") == "all":
        return {}
    if video.get("fps"):
        return {"fps": float(video["fps"]), "max_frames": video.get("max_frames")}
    return {"nframes": int(video["nframes"]), "floor": video.get("sampler") == "linspace_int"}


def offline_indices(info: dict, video: dict) -> list[int]:
    """The indices offline_frames() samples for `video` settings, from video_info() or manifest_info() values,
    without decoding. Not for `sampler: qwen_vl_utils` (the Qwen pairs' own sampler, see qwen_sampling.py)."""
    if video.get("sampler") == "qwen_vl_utils":
        raise ValueError("sampler qwen_vl_utils: use qwen_sampling.fetch_video_selection")
    return frame_indices(info, **offline_sampling(video))


def offline_frames(path: str, video: dict) -> tuple[list[Image.Image], dict]:
    """Offline frames from the merged `video` settings (see offline_sampling) and their meta (see sample_frames).
    `blank: true` (a control of the fine-tuning experiments) replaces the frames by black frames of the same number
    and size."""
    frames, meta = sample_frames(path, **offline_sampling(video))
    if video.get("blank"):
        frames, meta = [Image.new("RGB", f.size) for f in frames], {**meta, "blank": True}
    return frames, meta
