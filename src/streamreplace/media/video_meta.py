"""Video metadata for frame sampling: decodable frame count, reader fps, frame-list videos, and the per-benchmark
video_meta manifests.

Frame sampling (frames.py) needs two numbers per video, measured as in the paper's runs:
- `frames`: the number of frames that actually decode. PyAV (FFmpeg) decodes the whole video stream once, and the
  count is cached in `<cache root>/frame_counts/` (cache root: STREAMREPLACE_CACHE, see paths.py), keyed by the
  file's resolved path, size and modification time. The container header can report more frames: a clip cut without
  re-encoding keeps the frames from the previous keyframe up to the cut point, and the MP4 edit list hides them.
  FFmpeg (PyAV, and torchcodec in exact mode) skips the hidden frames; decord's len() counts them and pads indices
  past the real end with the last frame.
- `fps`: decord's `VideoReader.get_avg_fps()`. Every model environment of the paper's runs had decord 0.6.0. Without
  decord, PyAV's `average_rate` (or 30) is used, which can differ for some files (docs/frames.md).
`duration = frames / fps`.

A video can also be a frame list, `<id>.frames.json` = {"fps": f, "frames": [image paths relative to the file]}, which
`prepare` writes for datasets released as frame sequences (LingoQA, OpenUAV-QA). Its frames are read as stored.

Manifests. `manifests/video_meta/<benchmark>.jsonl` records the values the paper's runs used for every video of a
benchmark, one row per video, keyed by the video path relative to the benchmark's dataset directory (the `video`
field of prepared/samples.jsonl):
    {"video": "media/001.mp4", "frames": 2228, "fps": 29.97002997002997, "header_frames": 2228,
     "torchcodec": {"frames": 2228, "fps": 29.97..., "duration": 74.34...}}
`header_frames` is decord's len(); `torchcodec` is the exact-mode metadata that qwen-vl-utils reads for the Qwen pairs
(qwen_sampling.py). With a manifest, frame indices can be computed and checked on a CPU without decoding anything, and
`streamreplace video-meta verify` checks that a local copy of the data, decoded with the local libraries, gives the
paper's values.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import threading
from typing import Callable, Iterable

from ..core.io import read_jsonl, write_jsonl

FRAME_LIST = ".frames.json"
TORCHCODEC_KEYS = ("frames", "fps", "duration")


# Frame lists ---------------------------------------------------------------------------------------------------------

def is_frame_list(path: str) -> bool:
    return str(path).endswith(FRAME_LIST)


def write_frame_list(path: Path, frames: list[str], fps: float) -> Path:
    """Frame paths are stored relative to the list's folder, so the dataset tree can move (or be a symlink).
    An identical existing list is left untouched (same mtime, so cached frame counts stay valid)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    relative = [os.path.relpath(os.path.abspath(f), os.path.abspath(path.parent)) for f in frames]
    text = json.dumps({"fps": float(fps), "frames": relative}) + "\n"
    if not path.exists() or path.read_text() != text:
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(text)
        tmp.replace(path)
    return path


def read_frame_list(path: str) -> tuple[list[str], list[str], float]:
    """(absolute frame paths, the paths as stored - identical in every checkout, fps)."""
    data = json.loads(Path(path).read_text())
    base = Path(path).parent
    return [str(base / f) for f in data["frames"]], list(data["frames"]), float(data["fps"])


# Decodable frames and fps --------------------------------------------------------------------------------------------

def frame_count_dir() -> Path:
    """Where decodable-frame counts are cached: `<cache root>/frame_counts`. The cache root is resolved by
    streamreplace.paths, the one module that resolves the user's directories."""
    from ..paths import cache_dir
    return Path(cache_dir()) / "frame_counts"


def _decode_count(path: Path) -> int:
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        return sum(1 for _ in container.decode(stream))


def decodable_frames(path: str, cache_dir: Path | None = None, recount: bool = False) -> int:
    """Frames that actually decode (PyAV), cached per file in `cache_dir` (default: frame_count_dir()).

    The cache key is sha256("<resolved path>|<size>|<mtime_ns>")[:24] and each entry is {"path", "frames"}: the
    format of the paper's runs' frame-count cache, so such a cache folder can be reused as is. recount=True decodes
    the file again and neither reads nor writes the cache (verification)."""
    resolved = Path(path).resolve()
    if recount:
        return _decode_count(resolved)
    stat = resolved.stat()
    key = hashlib.sha256(f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:24]
    root = Path(cache_dir) if cache_dir is not None else frame_count_dir()
    cached = root / f"{key}.json"
    if cached.exists():
        return json.loads(cached.read_text())["frames"]
    count = _decode_count(resolved)
    root.mkdir(parents=True, exist_ok=True)
    tmp = cached.with_suffix(f".{os.getpid()}-{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps({"path": str(resolved), "frames": count}))
    tmp.replace(cached)
    return count


def decord_meta(path: str) -> tuple[int, float]:
    """(header frame count, fps) as decord reports them: len(VideoReader) and get_avg_fps()."""
    from decord import VideoReader, cpu
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    return len(reader), float(reader.get_avg_fps())


def pyav_meta(path: str) -> tuple[int, float]:
    """(header frame count, fps) from PyAV's stream header: the fallback when decord is not installed."""
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return stream.frames, float(stream.average_rate or 30)


def reader_meta(path: str) -> tuple[int, float]:
    """(header frame count, fps) from decord, or from PyAV when decord is not installed (as in the paper's code)."""
    try:
        return decord_meta(path)
    except ImportError:
        return pyav_meta(path)


def video_info(path: str, cache_dir: Path | None = None) -> dict:
    """{"frames": decodable frames (not the header count), "header_frames", "fps", "duration": frames / fps}."""
    if is_frame_list(path):
        frames, _, fps = read_frame_list(path)
        return {"frames": len(frames), "header_frames": len(frames), "fps": fps, "duration": len(frames) / fps}
    header, fps = reader_meta(path)
    count = decodable_frames(path, cache_dir)
    return {"frames": count, "header_frames": header, "fps": fps, "duration": count / fps if fps else 0.0}


def torchcodec_meta(path: str) -> dict:
    """torchcodec's metadata as qwen-vl-utils 0.0.14 reads it: VideoDecoder's default seek_mode="exact" scans the file,
    so `frames` (num_frames) excludes edit-list hidden frames and `fps` (average_fps) = frames / scanned duration."""
    from torchcodec.decoders import VideoDecoder
    meta = VideoDecoder(str(path), seek_mode="exact").metadata
    return {"frames": int(meta.num_frames), "fps": float(meta.average_fps), "duration": float(meta.duration_seconds)}


def library_versions() -> dict[str, str | None]:
    """Version of each video library this process can use; None if it is missing or cannot load (torchcodec raises
    at import when the FFmpeg libraries are not found)."""
    versions: dict[str, str | None] = {}
    for name in ("av", "decord", "torchcodec"):
        try:
            versions[name] = str(getattr(importlib.import_module(name), "__version__", "unknown"))
        except Exception:
            versions[name] = None
    return versions


# Manifests -----------------------------------------------------------------------------------------------------------

def default_manifest_dir() -> Path:
    """manifests/video_meta of the source checkout (located like manifests/prepared.yaml, by paths.project_root)."""
    from ..paths import project_root
    return project_root() / "manifests" / "video_meta"


def manifest_path(benchmark: str, manifest_dir: Path | None = None) -> Path:
    return Path(manifest_dir if manifest_dir is not None else default_manifest_dir()) / f"{benchmark}.jsonl"


def load_manifest(benchmark: str, manifest_dir: Path | None = None) -> dict[str, dict]:
    """Video path (relative to the dataset directory) -> manifest row. Empty if the benchmark has no manifest."""
    return {row["video"]: row for row in read_jsonl(manifest_path(benchmark, manifest_dir))}


def write_manifest(path: Path, rows: Iterable[dict]) -> Path:
    """Rows sorted by video path, atomically written."""
    write_jsonl(path, sorted(rows, key=lambda row: row["video"]))
    return Path(path)


def manifest_info(row: dict) -> dict:
    """video_info() as recorded in a manifest row (what frame_indices() needs: frames and fps)."""
    fps = row["fps"]
    return {"frames": row["frames"], "header_frames": row["header_frames"], "fps": fps,
            "duration": row["frames"] / fps if fps else 0.0}


def dataset_videos(dataset_dir: Path) -> list[str]:
    """The videos of a prepared benchmark: unique `video` paths of prepared/samples.jsonl (relative to the dataset
    directory), sorted."""
    samples = Path(dataset_dir) / "prepared" / "samples.jsonl"
    if not samples.exists():
        raise FileNotFoundError(f"{samples} not found: prepare the benchmark first")
    return sorted({row["video"] for row in read_jsonl(samples) if row.get("video")})


def measure(path: str, cache_dir: Path | None = None, reader: str | None = "decord", torchcodec: bool = False,
            recount: bool = False) -> dict:
    """The manifest values of one video (without "video"), measured with the installed libraries.

    reader: "decord" (fps and header_frames from decord, as in the paper's runs), "auto" (decord, else PyAV's header:
    what video_info() uses), or None (skip both). torchcodec=True adds the torchcodec metadata. recount=True decodes
    again instead of using the frame-count cache."""
    if is_frame_list(path):
        info = video_info(path)
        return {"frames": info["frames"], "fps": info["fps"], "header_frames": info["header_frames"]}
    row: dict = {"frames": decodable_frames(path, cache_dir, recount=recount)}
    if reader is not None:
        header, fps = decord_meta(path) if reader == "decord" else reader_meta(path)
        row.update(fps=fps, header_frames=header)
    if torchcodec:
        row["torchcodec"] = torchcodec_meta(path)
    return row


def _measure_job(job: tuple) -> dict:
    path, kwargs = job
    return measure(path, **kwargs)


def measure_videos(dataset_dir: Path, videos: list[str], workers: int = 1,
                   progress: Callable[[int, int], None] | None = None, **kwargs) -> list[dict]:
    """measure() for each video (paths relative to `dataset_dir`) in parallel; rows in the order of `videos`.
    kwargs go to measure(); the frame-count cache folder is resolved here, so every worker uses the same one."""
    if kwargs.get("cache_dir") is None:
        kwargs["cache_dir"] = frame_count_dir()
    jobs = [(str(Path(dataset_dir) / video), kwargs) for video in videos]
    rows = []
    if workers > 1:
        # Workers start in fresh interpreters ("spawn"): forking a process whose decoding libraries already run
        # threads can deadlock the child. multiprocessing.Pool rather than ProcessPoolExecutor: on Python 3.10 the
        # executor's map() can deadlock once its pending submits fill the wakeup pipe (CPython gh-105829), which
        # happens with a few thousand videos.
        with get_context("spawn").Pool(workers) as pool:
            for number, values in enumerate(pool.imap(_measure_job, jobs, chunksize=4), 1):
                rows.append({"video": videos[number - 1], **values})
                if progress:
                    progress(number, len(jobs))
    else:
        for number, job in enumerate(jobs, 1):
            rows.append({"video": videos[number - 1], **_measure_job(job)})
            if progress:
                progress(number, len(jobs))
    return rows


def compare(expected: dict, measured: dict) -> list[str]:
    """Differences between a manifest row and a measurement of the same video. Values are compared exactly (one fps
    digit can move a frame index); fields absent from either side are not compared."""
    diffs = []
    for key in ("frames", "fps", "header_frames"):
        if key in expected and key in measured and expected[key] != measured[key]:
            diffs.append(f"{key}: manifest {expected[key]!r}, measured {measured[key]!r}")
    if "torchcodec" in expected and "torchcodec" in measured:
        for key in TORCHCODEC_KEYS:
            if expected["torchcodec"][key] != measured["torchcodec"][key]:
                diffs.append(f"torchcodec.{key}: manifest {expected['torchcodec'][key]!r}, "
                             f"measured {measured['torchcodec'][key]!r}")
    return diffs
