"""AURA's native stream through a local vLLM server (paper condition S for AURA-8B, pair p3).

AURA ships official benchmark adapters (AURA_bench_eval, OVO-Bench and StreamingBench) that ask a question at time t
against a vLLM server. The adapter re-implements that question-answering procedure in AURA's official evaluation
environment (vLLM 0.11.0 with the transformers patch from AURA's README, applied by the aura environment):

- server: `vllm serve <checkpoint> --max-model-len 32768 --gpu-memory-utilization 0.9 --media-io-kwargs
  '{"video": {"num_frames": -1}}' --limit-mm-per-prompt '{"video": 50}' --served-model-name aura` on a free local
  port (AURA_bench_eval/deploy_aura_vllm.sh), started by the adapter on the worker's GPU (and stopped again when it
  does not come up).
- frames and memory: the last min(30, floor(d)) one-second segments of the video, aligned to its end (d = decodable
  frames / decord fps); a video shorter than one second is one segment (0, d). Each segment is cut with
  imageio-ffmpeg's FFmpeg (the aura environment's; no other build) to an mp4 with exactly two frames (`-vf fps=2`,
  lossless x264, no audio), the 2-fps chunks AURA was trained on, so each segment is one temporal patch. `schedule()`
  gives the segments. (Settings `stream.window` selects more segments; the server accepts at most 50 videos per
  request, so "full" is meant for the transformers rebuild.)
- question: every segment is one user turn (a base64 video); every earlier segment is followed by a prefilled
  assistant "<|silent|>" turn, and the last segment's turn carries the benchmark prompt. AURA's live-stream system
  prompt; the dataset's system prompt is not used.
- decoding: one request, temperature 0, max_tokens = the benchmark's budget, mm_processor_kwargs {fps 1.0, size
  {shortest_edge 4096, longest_edge 602112}}. With fps 1.0 the prompt stamps each two-frame segment "<0.5 seconds>"
  although the clips are 2 fps: kept as AURA's official request does it.
- answer: the reply with <|silent|> removed (`readouts.visible`); an empty reply is silence (`info.no_answer`,
  scored wrong). There is no forcing.
- nondeterminism: 8 requests are in flight (`concurrency`, settings `stream.aura_concurrency`), and vLLM batching is
  not batch-invariant, so a greedy reply can differ from a one-at-a-time run in rare cases.
- deliberate deviations from the official eval: official segments are re-encoded at the source fps with audio and
  their length comes from ffprobe; the official max_tokens is 128 (here the benchmark's budget).

The plugin cannot run on vLLM (no hidden states): its engine is aura_stream_hf.py, the same procedure rebuilt in
transformers, which shares this module's segments, system prompt and spans.
"""
from __future__ import annotations

import base64
import hashlib
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

from .. import paths
from ..core.io import fingerprint
from ..core.schema import Generation, Request
from ..media.video_meta import video_info
from .base import AnswerTurnHooks, BaseAdapter
from .readouts import AURA_SILENT, visible

ADAPTER = "aura_stream"
# AURA's live-stream system prompt, verbatim from AURA_bench_eval/OVO-Bench/models/AURA.py (SYSTEM_PROMPT) at
# https://github.com/aurateam2026/AURA commit 82d78f0 (listed in NOTICE).
SYSTEM = ("You are receiving a live video stream where the final frame is the present moment. Respond only when a "
          "response is needed based on the user's message or the visual context. Otherwise, output `<|silent|>` to "
          "signify silence.")
WINDOW = 30                  # official: the last 30 one-second segments
SEGMENT_FPS = 2
MM_PROCESSOR_KWARGS = {"fps": 1.0, "size": {"shortest_edge": 4096, "longest_edge": 602112}}
SERVER_ARGS = ["--max-model-len", "32768", "--gpu-memory-utilization", "0.9",
               "--media-io-kwargs", '{"video": {"num_frames": -1}}', "--limit-mm-per-prompt", '{"video": 50}',
               "--served-model-name", "aura"]
CONCURRENCY = 8
SERVER_WAIT = (600, 2.0)     # attempts x seconds while the server loads


def spans(duration: float, window=None) -> list[tuple[float, float]]:
    """AURA's one-second segments (start, length), aligned to the video's end. window: None = the official last
    min(30, floor(d)); "full" = every segment (the plugin's Recall); an int N = the last min(N, floor(d)).
    A video shorter than one second is the single span (0, d)."""
    window = window or WINDOW
    count = math.floor(duration) if window == "full" else min(int(window), math.floor(duration))
    return [(0.0, duration)] if count < 1 else [(duration - count + i, 1.0) for i in range(count)]


def schedule(req: Request, meta: dict | None = None, adapter: str = ADAPTER) -> dict:
    """CPU-only: the segments AURA is shown (the same for the vLLM stream and its transformers rebuild, `adapter`).
    `meta` = {"frames" (decodable), "fps" (decord)} as in a video_meta manifest row; measured from the file when None.
    Settings stream.window selects the window (see spans)."""
    meta = video_info(req.video) if meta is None else meta
    duration = meta["frames"] / meta["fps"] if meta["fps"] else 0.0
    window = req.settings.get("stream", {}).get("window")
    kept = spans(duration, window)
    plan = {"adapter": adapter, "duration": duration, "segments": len(kept), "spans": [list(s) for s in kept],
            "window_start": round(kept[0][0], 2), "evicted": max(0, math.floor(duration) - len(kept)),
            "frames_per_segment": SEGMENT_FPS, "question_segment": len(kept) - 1, "question_time": duration,
            "window": window or WINDOW}
    plan["digest"] = fingerprint(plan)
    return plan


def ffmpeg_executable() -> str:
    """The FFmpeg binary of imageio-ffmpeg (pinned in envs/aura/requirements.lock), which cut the paper's segments.
    There is deliberately no fallback to another FFmpeg: a different build can pick and convert the two frames
    differently, and the segment cache does not record which build cut a file."""
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError:
        raise RuntimeError("AURA's segments are cut with imageio-ffmpeg, which this environment lacks: run AURA in "
                           "the aura environment (envs/aura/requirements.lock pins it)") from None
    return get_ffmpeg_exe()


def segment(video: str, start: float, length: float) -> Path:
    """[start, start + length] of the video as an mp4 holding exactly two frames (fps=2: about +0.0 s and +0.5 s),
    cached under <cache>/aura-segments-2fps/<sha256(resolved video path)[:16]>/<start>_<length>.mp4. Upstream
    re-encodes at the source fps and lets the patched processor keep the first and last frame, which is also one
    temporal patch per segment."""
    folder = paths.cache_dir() / "aura-segments-2fps" / hashlib.sha256(os.path.realpath(video).encode()).hexdigest()[:16]
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{start:.2f}_{length:.2f}.mp4"
    target = folder / name
    if not target.exists():
        partial = folder / f".{name}.{os.getpid()}.{threading.get_ident()}.partial.mp4"
        subprocess.run([ffmpeg_executable(), "-v", "error", "-y", "-ss", f"{start:.2f}", "-i", video, "-t",
                        f"{length:.2f}", "-vf", f"fps={SEGMENT_FPS}", "-an", "-c:v", "libx264", "-crf", "0",
                        "-preset", "fast", str(partial)], check=True)
        partial.rename(target)
    return target


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class AuraStream(BaseAdapter):
    """Condition S of AURA-8B (env aura: vLLM 0.11.0, patched transformers 4.57.1). load=False builds a
    schedule-only adapter (no server)."""
    PROTOCOL_VERSION = 1
    LOAD_KEYS: tuple[str, ...] = ("stream.aura_concurrency",)

    def __init__(self, spec, settings: dict, upstream, load: bool = True):
        super().__init__(spec, settings, upstream)
        self.concurrency = int(settings.get("stream", {}).get("aura_concurrency", CONCURRENCY))
        self.server = None
        self._log = None
        if load:
            self._load()

    def _load(self) -> None:
        from openai import OpenAI
        port = free_port()                   # shards on other GPUs each run their own server
        log_dir = paths.cache_dir() / "aura-vllm"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log = (log_dir / f"{port}.log").open("a")
        try:
            self.server = subprocess.Popen(
                ["vllm", "serve", str(paths.checkpoint_dir(self.spec.checkpoint)), "--port", str(port), *SERVER_ARGS],
                stdout=self._log, stderr=subprocess.STDOUT,
                env={**os.environ, "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"})
            self.client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="local")
            self._wait_until_ready()
        except BaseException:
            # The run worker never closes an adapter whose construction failed: a server still loading (or stuck)
            # would keep its share of the GPU after the worker moved on, and the next AURA job would start another.
            self.close()
            raise

    def _wait_until_ready(self) -> None:
        attempts, pause = SERVER_WAIT
        for _ in range(attempts):
            try:
                self.client.models.list()
                return
            except Exception:
                if self.server.poll() is not None:
                    raise RuntimeError(f"AURA vLLM server exited; see {self._log.name}") from None
                time.sleep(pause)
        raise RuntimeError(f"AURA vLLM server did not come up; see {self._log.name}")

    def schedule(self, req: Request) -> dict:
        return schedule(req)

    def stream(self, req: Request, hooks: AnswerTurnHooks | None = None) -> Generation:
        if hooks is not None:
            raise ValueError("AURA's vLLM stream exposes no hidden states: the plugin runs on aura_stream_hf")
        plan = schedule(req)
        kept = [tuple(s) for s in plan["spans"]]
        messages = [{"role": "system", "content": SYSTEM}]
        for index, (start, length) in enumerate(kept):
            data = base64.b64encode(segment(req.video, start, length).read_bytes()).decode()
            user = [{"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + data}}]
            if index == len(kept) - 1:
                user.append({"type": "text", "text": req.prompt})
            messages.append({"role": "user", "content": user})
            if index < len(kept) - 1:
                messages.append({"role": "assistant", "content": AURA_SILENT})
        response = self.client.chat.completions.create(
            model="aura", messages=messages, temperature=0, max_tokens=req.settings["generation"]["max_new_tokens"],
            extra_body={"mm_processor_kwargs": MM_PROCESSOR_KWARGS})
        text, literal = visible(response.choices[0].message.content)
        info = {"segments": len(kept), "window_start": plan["window_start"], "duration": round(plan["duration"], 2),
                "prompt_tokens": response.usage.prompt_tokens, "no_answer": not text}
        if literal:                          # a literal <|silent|> in the text (a checkpoint without AURA's token)
            info["silent_text"] = True
        info["schedule"] = plan["digest"]
        return Generation(text, response.usage.completion_tokens, response.choices[0].finish_reason, info)

    def close(self) -> None:
        """Stops the vLLM server (killed when it does not exit within 60 s of SIGTERM)."""
        server, self.server = self.server, None
        if server is not None:
            server.terminate()
            try:
                server.wait(60)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(60)
        log, self._log = self._log, None
        if log is not None:
            log.close()
