"""AURA's stream rebuilt in Hugging Face transformers: the plugin's engine for AURA and the S_HF control.

AURA's native stream (aura_stream.py) runs in a vLLM server, which exposes no hidden states, so the plugin cannot act
on it. This adapter rebuilds the same question-answering procedure in transformers (Qwen3VLForConditionalGeneration,
bf16, flash-attn 2 when installed, else sdpa), in AURA's official evaluation environment (vLLM 0.11's OpenCV and the
README's patched transformers, which also accept the one-frame segment of a sub-second video as vLLM did):

- segments: aura_stream.spans(): the last min(30, floor(d)) one-second segments aligned to the video's end, or every
  segment with settings `stream.window: full`; the same ffmpeg two-frame cuts (aura_stream.segment), decoded with
  OpenCV as vLLM's video loader does. Stream++'s Recall (settings `stream.window: full`, `stream.budget: B`,
  `stream.recent: k`; read by this adapter only) keeps plugin.recall.aura_spans(d, B, recent=k): every segment while
  floor(d) <= B, else the newest k plus a hash-priority reservoir of B - k older ones (stream id: the video's file
  name; `stream.memory_seed` varies the reservoir); the schedule then records "budget" and "recent" and the row
  `info.kept_starts`. An int `stream.window` N keeps AURA's own last-N window, as on the vLLM stream: the recent-only
  memory of the budget controls, and the S_HF control at the default 30.
- prompt: AURA's live-stream system prompt; one user turn per segment, each earlier segment followed by an assistant
  "<|silent|>" turn, the last carrying the benchmark prompt.
- processor: video metadata {fps 1.0, frames_indices range(n), total_num_frames n, duration n} per segment, so each
  segment is stamped "<0.5 seconds>" exactly as vLLM wrote it (mm_processor_kwargs fps 1.0); do_sample_frames False,
  do_resize True with size {shortest_edge 4096, longest_edge 602112}.
- decoding: one greedy generate up to the benchmark's budget; the reply is decoded without special tokens and
  "<|silent|>" removed; empty = silence (`info.no_answer`).
- diagnostic settings (never part of S or of the plugin): `stream.system: none` drops the system turn;
  `stream.input: offline` shows the video as the offline pipeline does (Supp. "Parent controls", TimeBlind): the
  whole clip from 0 s cut at 2 fps as ONE video in one user turn, every frame resized to the offline per-frame
  budget (smart_resize to at most 602,112 pixels) and stamped with its time in the video.
- fidelity: on the smoke items the rebuild's prompt token counts equal vLLM's (18/18) and 15/18 replies are identical;
  the job with this adapter and no plugin (variant `hf`) is the S_HF control that separates the engine change from
  the plugin's effect.

Plugin hook points (hooks=None is the S_HF control):
- attach: once, with the language model's decoder layers and the tokenizer.
- answer turn: the single generate call. `hooks.arm()` runs immediately before it (the gate reads the last token of
  the full prompt's prefill) and `hooks.disarm()` right after; the call gets
  `logits_processor=hooks.logits_processors()` (first-step capture, the <|silent|> / <|im_end|> ban).
- `hooks.finish(req, generation)` makes the final row.
"""
from __future__ import annotations

from pathlib import Path

from .. import paths
from ..core.io import fingerprint
from ..core.schema import Generation, Request
from . import aura_stream
from .aura_stream import AURA_SILENT, segment, system_prompt
from .base import AnswerTurnHooks, BaseAdapter
from .readouts import visible

ADAPTER = "aura_stream_hf"
SIZE = {"shortest_edge": 4096, "longest_edge": 602112}


def schedule(req: Request, meta: dict | None = None) -> dict:
    """CPU-only: aura_stream.schedule() of this adapter. With Recall's budget (settings stream.budget B with
    stream.window full) the segments are plugin.recall.aura_spans(d, B, recent, seed) and the plan records "budget"
    and "recent"; with stream.input offline the plan is one span covering the whole clip."""
    plan = aura_stream.schedule(req, meta, adapter=ADAPTER)
    stream = req.settings.get("stream") or {}
    if stream.get("input", "segments") not in ("segments", "offline"):
        raise ValueError(f"stream.input must be segments or offline, not {stream['input']!r}")
    if stream.get("budget") is None and stream.get("input") != "offline":
        return plan
    plan = {k: v for k, v in plan.items() if k != "digest"}
    if stream.get("input") == "offline":
        plan.update(spans=[[0.0, round(plan["duration"], 2)]], segments=1, window_start=0.0, evicted=0,
                    question_segment=0, input="offline")
    else:
        from ..plugin.recall import aura_policy, aura_spans
        budget, recent, seed = aura_policy(req.settings)     # raises unless stream.window is full
        kept, evicted = aura_spans(plan["duration"], budget, recent, Path(req.video).name, seed)
        plan.update(spans=[list(s) for s in kept], segments=len(kept), window_start=round(kept[0][0], 2),
                    evicted=evicted, question_segment=len(kept) - 1, budget=budget, recent=recent)
        if seed:
            plan["memory_seed"] = seed
    plan["digest"] = fingerprint(plan)
    return plan


def stamped(times: list[float]) -> dict:
    """Video metadata that make Qwen3-VL stamp each two-frame patch with its time in the video: the processor writes
    "<t seconds>" with t the mean of the patch's first and last frame index over fps, so the indices are the frames'
    times in ms at fps 1000 (fps only enters those stamps; frames are passed already sampled)."""
    index = [int(round(1000 * t)) for t in times]
    return {"fps": 1000.0, "frames_indices": index, "total_num_frames": index[-1] + 1, "duration": times[-1] + 0.5}


def decode(path, length: float = 1.0):
    """A segment's frames (two for a one-second segment, one for a segment shorter than a frame interval), RGB,
    decoded with OpenCV as vLLM's video loader does."""
    import cv2
    import numpy as np
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if (length <= 1.0 and len(frames) not in (1, 2)) or not frames:
        raise RuntimeError(f"{path}: {len(frames)} frames, expected 2 (or 1 for a sub-second video)")
    return np.stack(frames)


class AuraStreamHF(BaseAdapter):
    """AURA-8B's stream on transformers (env aura). load=False builds a schedule-only adapter (no model)."""
    PROTOCOL_VERSION = 1
    LOAD_KEYS: tuple[str, ...] = ()
    concurrency = 1

    def __init__(self, spec, settings: dict, upstream, load: bool = True):
        super().__init__(spec, settings, upstream)
        self._attached: AnswerTurnHooks | None = None
        if load:
            self._load()

    def _load(self) -> None:
        import torch
        import transformers
        from transformers import AutoProcessor
        self.torch = torch
        try:
            import flash_attn  # noqa: F401
            attention = "flash_attention_2"
        except ImportError:                  # the aura environment has vLLM's own kernels only
            attention = "sdpa"
        checkpoint = str(paths.checkpoint_dir(self.spec.checkpoint))
        self.processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
        # no device_map (the aura environment has no accelerate): built on the CPU, then moved
        self.model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            checkpoint, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation=attention
        ).to("cuda:0").eval()

    def schedule(self, req: Request) -> dict:
        return schedule(req)

    def inputs(self, req: Request, kept: list[tuple[float, float]]):
        """The processor inputs of the rebuilt prompt (CPU)."""
        import numpy as np
        system = system_prompt(req.settings)
        offline = (req.settings.get("stream") or {}).get("input") == "offline"
        messages = [{"role": "system", "content": system}] if system else []
        videos, metas = [], []
        for index, (start, length) in enumerate(kept):
            frames = decode(segment(req.video, start, length), length)
            videos.append(frames)
            n = len(frames)
            metas.append(stamped([start + k / 2 for k in range(n)]) if offline else
                         {"fps": 1.0, "frames_indices": list(range(n)), "total_num_frames": n, "duration": float(n)})
            content = [{"type": "video"}]
            if index == len(kept) - 1:
                content.append({"type": "text", "text": req.prompt})
            messages.append({"role": "user", "content": content})
            if index < len(kept) - 1:
                messages.append({"role": "assistant", "content": AURA_SILENT})
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if offline:                          # every frame at the offline per-frame resolution
            from PIL import Image
            from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
            h, w = smart_resize(videos[0].shape[1], videos[0].shape[2], factor=32, min_pixels=SIZE["shortest_edge"],
                                max_pixels=SIZE["longest_edge"])
            videos = [np.stack([np.asarray(Image.fromarray(f).resize((w, h), resample=3)) for f in v]) for v in videos]
            return self.processor(text=[text], videos=videos, video_metadata=metas, do_sample_frames=False,
                                  do_resize=False, return_tensors="pt")
        return self.processor(text=[text], videos=videos, video_metadata=metas, do_sample_frames=False, do_resize=True,
                              size=SIZE, return_tensors="pt")

    def _attach(self, hooks: AnswerTurnHooks) -> None:
        if hooks is self._attached:
            return
        from ..plugin.guard import find_layers
        hooks.attach(find_layers(self.model), self.processor.tokenizer)
        self._attached = hooks

    def stream(self, req: Request, hooks: AnswerTurnHooks | None = None) -> Generation:
        plan = self.schedule(req)
        kept = [tuple(s) for s in plan["spans"]]
        x = self.inputs(req, kept).to("cuda:0")
        budget = int(req.settings["generation"]["max_new_tokens"])
        extra = {}
        if hooks is not None:
            from transformers import LogitsProcessorList
            self._attach(hooks)
            extra["logits_processor"] = LogitsProcessorList(hooks.logits_processors())
            hooks.arm()
        try:
            with self.torch.inference_mode():
                out = self.model.generate(**x, do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None,
                                          max_new_tokens=budget, **extra)
        finally:
            if hooks is not None:
                hooks.disarm()
        ids = out[0, x["input_ids"].shape[1]:]
        text = visible(self.processor.tokenizer.decode(ids, skip_special_tokens=True))[0]
        info = {"segments": len(kept), "evicted": plan["evicted"], "window_start": plan["window_start"],
                "duration": round(plan["duration"], 2), "prompt_tokens": int(x["input_ids"].shape[1]),
                "no_answer": not text, "engine": "hf", "schedule": plan["digest"]}
        if "budget" in plan:                 # Recall: the kept segments' start times
            info["kept_starts"] = [round(start, 2) for start, _ in kept]
        generation = Generation(text, int(len(ids)), "length" if len(ids) >= budget else "eos", info)
        return hooks.finish(req, generation) if hooks is not None else generation

    def close(self) -> None:
        self.model = None
        if hasattr(self, "torch"):
            self.torch.cuda.empty_cache()
