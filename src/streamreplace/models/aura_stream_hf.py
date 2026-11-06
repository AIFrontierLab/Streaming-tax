"""AURA's stream rebuilt in Hugging Face transformers: the plugin's engine for AURA and the S_HF control.

AURA's native stream (aura_stream.py) runs in a vLLM server, which exposes no hidden states, so the plugin cannot act
on it. This adapter rebuilds the same question-answering procedure in transformers (Qwen3VLForConditionalGeneration,
bf16, flash-attn 2 when installed, else sdpa), in AURA's official evaluation environment (vLLM 0.11's OpenCV and the
README's patched transformers, which also accept the one-frame segment of a sub-second video as vLLM did):

- segments: aura_stream.spans(): the last min(30, floor(d)) one-second segments aligned to the video's end, or every
  segment with settings `stream.window: full` (the plugin's Recall); the same ffmpeg two-frame cuts
  (aura_stream.segment), decoded with OpenCV as vLLM's video loader does. Recall with a budget (settings
  `stream.window: full` and `stream.budget: B`, read by this adapter only; no shipped arm sets it) keeps
  plugin.recall.aura_spans(d, B): every segment while floor(d) <= B, else the last min(30, B) plus a hash-priority
  reservoir of the older ones (stream id: the video's file name); the schedule then records "budget". An int
  `stream.window` N keeps AURA's own last-N window, as on the vLLM stream, so the S_HF control stays comparable.
- prompt: AURA's live-stream system prompt; one user turn per segment, each earlier segment followed by an assistant
  "<|silent|>" turn, the last carrying the benchmark prompt.
- processor: video metadata {fps 1.0, frames_indices range(n), total_num_frames n, duration n} per segment, so each
  segment is stamped "<0.5 seconds>" exactly as vLLM wrote it (mm_processor_kwargs fps 1.0); do_sample_frames False,
  do_resize True with size {shortest_edge 4096, longest_edge 602112}.
- decoding: one greedy generate up to the benchmark's budget; the reply is decoded without special tokens and
  "<|silent|>" removed; empty = silence (`info.no_answer`).
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
from .aura_stream import AURA_SILENT, SYSTEM, segment
from .base import AnswerTurnHooks, BaseAdapter
from .readouts import visible

ADAPTER = "aura_stream_hf"
SIZE = {"shortest_edge": 4096, "longest_edge": 602112}


def schedule(req: Request, meta: dict | None = None) -> dict:
    """CPU-only: aura_stream.schedule() of this adapter, plus Recall's budget (settings stream.budget B with
    stream.window full): the segments of plugin.recall.aura_spans(d, B), with "budget" in the plan. Without
    stream.budget the plan is aura_stream's own."""
    plan = aura_stream.schedule(req, meta, adapter=ADAPTER)
    if (req.settings.get("stream") or {}).get("budget") is None:
        return plan
    from ..plugin.recall import aura_policy, aura_spans
    budget = aura_policy(req.settings)                   # raises unless stream.window is full
    kept, evicted = aura_spans(plan["duration"], budget, stream_id=Path(req.video).name)
    plan = {k: v for k, v in plan.items() if k != "digest"}
    plan.update(spans=[list(s) for s in kept], segments=len(kept), window_start=round(kept[0][0], 2),
                evicted=evicted, question_segment=len(kept) - 1, budget=budget)
    plan["digest"] = fingerprint(plan)
    return plan


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
        messages = [{"role": "system", "content": SYSTEM}]
        videos, metas = [], []
        for index, (start, length) in enumerate(kept):
            frames = decode(segment(req.video, start, length), length)
            videos.append(frames)
            n = len(frames)
            metas.append({"fps": 1.0, "frames_indices": list(range(n)), "total_num_frames": n, "duration": float(n)})
            content = [{"type": "video"}]
            if index == len(kept) - 1:
                content.append({"type": "text", "text": req.prompt})
            messages.append({"role": "user", "content": content})
            if index < len(kept) - 1:
                messages.append({"role": "assistant", "content": AURA_SILENT})
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
        generation = Generation(text, int(len(ids)), "length" if len(ids) >= budget else "eos", info)
        return hooks.finish(req, generation) if hooks is not None else generation

    def close(self) -> None:
        self.model = None
        if hasattr(self, "torch"):
            self.torch.cuda.empty_cache()
