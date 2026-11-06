"""Offline adapter of MOSS-VL-Instruct-0708 and MOSS-VL-Realtime (conditions P and C of pair p5). Both checkpoints ship
their own model code (trust_remote_code); their offline API (`offline_video_generate` and the query path behind it)
is the same in both.

- video (pair p5): the framework's frames (1 fps from frame 0, the same indices as the MOSS-VL-Realtime stream) are
  handed to MOSS's video processor through a temporary override of its `fetch_videos` (frames_as_video): it returns
  the frames as one uint8 TCHW tensor with VideoMetadata(total_num_frames = round(duration * fps), fps = the source
  fps, duration, frames_indices, actual_timestamps = index / source fps) instead of decoding the video file itself.
  The call is offline_video_generate(prompt, longest_edge = video.max_pixels, video_max_pixels =
  video.video_max_pixels (the pair's MOSS-VL-Realtime budget: 16,777,216 pixels per frame, 65,536,000 per video),
  max_frames = the number of frames, [shortest_edge = video.min_pixels], [system_prompt = the benchmark's system
  prompt]) with the API's defaults for everything else (shortest_edge 4096, video_fps 1.0, vision_chunked_length 64,
  no system message).
- images: one query with media_kwargs max_pixels = image.max_pixels and the image processor's size
  {shortest_edge 4096, longest_edge image.max_pixels}; no system message.
Decoding: greedy (do_sample False) with the benchmark's token budget. The API's sampling defaults (temperature 1.0,
top_k 50, top_p 1.0) are inert under greedy decoding; repetition_penalty is the API default 1.0 (the checkpoints'
generation_config.json sets none), passed explicitly from configs/models.yaml. bf16; FlashAttention-2 when installed.

build_inputs() runs MOSS's own offline query path on a weightless instance of the model class (the code's
build_offline_prepare_helper) whose final generate() is replaced by a recorder, so the CPU inputs are exactly what
generate() receives. Fine-tuning: encode(req, answer) / train_forward(enc), and load_lora(path) for a fine-tuned id's
LoRA."""
from __future__ import annotations

import contextlib
import inspect
from pathlib import Path
import threading

from .. import paths
from ..core.schema import Generation, Request
from ..media.digest import frame_digest
from ..media.frames import offline_frames, open_rgb
from .base import EOT, BaseAdapter, PreparedInputs, supervised_length
from .decoding import check_repetition_penalty, checkpoint_repetition_penalty
from .lm_head import counted_stop, head_wrapper, letter_logits
from .lora import attach_lora
from .qwen_offline import attention_implementation

DEVICE = "cuda:0"
FRAMES = "streamreplace_frames"        # placeholder video argument: the overridden fetch_videos ignores it
IMAGE_SHORTEST_EDGE = 4096             # the API's default pixel floor
VISION_CHUNKED_LENGTH = 64             # the API's default for its offline generate
DECODING = ("max_new_tokens", "temperature", "top_k", "top_p", "repetition_penalty", "do_sample",
            "vision_chunked_length")


class MossOffline(BaseAdapter):
    """Conditions P and C of MOSS-VL-Instruct and MOSS-VL-Realtime (env modern). load=False builds the processor and
    a weightless instance of the model class: build_inputs() and generation_kwargs() work on a CPU."""
    PROTOCOL_VERSION = 1
    LOAD_KEYS: tuple[str, ...] = ()
    CAPTURES_LETTER_LOGITS = True
    # decoder layers; q, k, v, o of the self- or cross-attention and gate, up, down get a LoRA
    LORA_LAYERS = ("model.language_model.layers", 7)

    def __init__(self, spec, settings: dict, upstream, load: bool = True):
        super().__init__(spec, settings, upstream)
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        self.torch = torch
        checkpoint = Path(paths.checkpoint_dir(spec.checkpoint))
        penalty = spec.decoding.repetition_penalty
        check_repetition_penalty(penalty, checkpoint, spec.id)
        self.repetition_penalty = penalty if penalty is not None else checkpoint_repetition_penalty(checkpoint)
        self.attention = attention_implementation()
        self.processor = AutoProcessor.from_pretrained(str(checkpoint), trust_remote_code=True, local_files_only=True)
        self.model = self.head = None
        if load:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(checkpoint), trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16,
                device_map=DEVICE, attn_implementation=self.attention).eval()
            self.head = self.model.lm_head = head_wrapper(self.model.lm_head)
            cls = type(self.model)
        else:
            config = AutoConfig.from_pretrained(str(checkpoint), trust_remote_code=True, local_files_only=True)
            cls = get_class_from_dynamic_module(config.auto_map["AutoModelForCausalLM"], str(checkpoint))
        self._lock = threading.Lock()
        self._state: dict = {}
        self._helper = self._prepare_helper(cls)
        self.recorded_generate_kwargs: dict | None = None   # what the last build_inputs() saw generate() receive

    def _prepare_helper(self, cls):
        """A weightless instance of the model class whose offline query path stops at generate(): the prepared CPU
        inputs and generate()'s keyword arguments are recorded in self._state instead of running a model. With an
        answer to train on, the training inputs are built there too, under the same processor overrides and frames."""
        helper = cls.build_offline_prepare_helper()
        if self.model is not None:                  # one lock for the processor overrides of both instances
            helper._offline_processor_lock = self.model._offline_processor_lock
        state, prepare_query = self._state, cls.offline_prepare_query_cpu

        def prepare(processor, query, *args, **kwargs):
            prepared = prepare_query(helper, processor, query, *args, **kwargs)
            state.update(query=query, prepared=prepared)
            return prepared

        def generate(**kwargs):
            state["generate_kwargs"] = {k: kwargs[k] for k in DECODING if k in kwargs}
            if state.get("answer") is not None:
                prepared, answer = state["prepared"], state["answer"]
                text = prepared["input_text"]
                state["n"] = supervised_length(
                    lambda t: self.processor.tokenizer(t, add_special_tokens=False).input_ids, text, answer)
                media = dict(state["query"].get("media_kwargs") or {})
                images, videos = helper._offline_collect_media(prepared["working_messages"])
                state["inputs"] = helper._offline_run_processor(
                    self.processor, helper._offline_build_processor_kwargs(text + answer + EOT, images, videos, media),
                    media)
            ids = kwargs["input_ids"]
            return self.torch.cat([ids, ids[:, -1:]], dim=1)     # one placeholder "generated" token

        helper.offline_prepare_query_cpu = prepare
        helper._offline_move_inputs_to_devices = lambda inputs: dict(inputs)   # stay on the CPU, no dtype cast
        helper.generate = generate
        return helper

    # ---------- inputs ----------
    @contextlib.contextmanager
    def frames_as_video(self, frames, meta: dict):
        """Within the block, MOSS's video processor returns these frames and their metadata instead of decoding the
        video file named in the query."""
        from transformers.video_utils import VideoMetadata
        import numpy as np
        tensor = self.torch.from_numpy(np.stack([np.asarray(f) for f in frames])).permute(0, 3, 1, 2)
        metadata = VideoMetadata(total_num_frames=round(meta["duration"] * meta["fps"]), fps=meta["fps"],
                                 duration=meta["duration"], height=tensor.shape[2], width=tensor.shape[3],
                                 frames_indices=list(meta["indices"]), video_backend=FRAMES)
        metadata.actual_timestamps = list(meta["timestamps"])
        processor = self.processor.video_processor
        shadowed = "fetch_videos" in vars(processor)
        original = processor.fetch_videos
        processor.fetch_videos = lambda videos, **kwargs: ([tensor], [metadata])
        try:
            yield
        finally:
            if shadowed:
                processor.fetch_videos = original
            else:
                del processor.fetch_videos

    def _video_call(self, req: Request, frames: list) -> dict:
        """offline_video_generate's arguments for these frames: the pair's per-frame and total pixel budgets, the
        benchmark's pixel floor (video.min_pixels; the API's shortest_edge otherwise), greedy, the token budget."""
        s = req.settings
        per_frame = s["video"]["max_pixels"]
        call = dict(prompt=req.prompt, video=FRAMES, longest_edge=per_frame,
                    video_max_pixels=s["video"].get("video_max_pixels", per_frame * len(frames)),
                    max_frames=len(frames), max_new_tokens=s["generation"]["max_new_tokens"], do_sample=False,
                    repetition_penalty=self.repetition_penalty)
        if s["video"].get("min_pixels"):
            call["shortest_edge"] = s["video"]["min_pixels"]
        if s.get("system"):
            call["system_prompt"] = s["system"]
        return call

    def _image_query(self, req: Request, images: list) -> tuple[dict, dict]:
        """The image query and the image processor overrides (no system message for images, as in the paper's
        runs)."""
        s = req.settings
        query = {"prompt": req.prompt, "images": images, "videos": [],
                 "media_kwargs": {"max_pixels": s["image"]["max_pixels"]},
                 "generate_kwargs": {"max_new_tokens": s["generation"]["max_new_tokens"], "do_sample": False,
                                     "vision_chunked_length": VISION_CHUNKED_LENGTH,
                                     "repetition_penalty": self.repetition_penalty}}
        overrides = {"size": {"shortest_edge": IMAGE_SHORTEST_EDGE, "longest_edge": s["image"]["max_pixels"]}}
        return query, overrides

    def _run(self, model, req: Request) -> tuple[str, dict]:
        """The paper's offline call on `model` (the loaded model or the weightless helper): (reply, info)."""
        s = req.settings
        if req.video:
            frames, meta = offline_frames(req.video, s["video"])
            with self.frames_as_video(frames, meta):
                text = model.offline_video_generate(self.processor, **self._video_call(req, frames))
            return text, {"frames": len(frames), "duration": round(meta["duration"], 2),
                          "frame_indices": frame_digest(meta)}
        images = [open_rgb(p) for p in req.images]
        query, overrides = self._image_query(req, images)
        text = model._offline_generate_one_with_processor_overrides(self.processor, query,
                                                                    image_processor_overrides=overrides)
        return text, {"images": len(images)}

    def _encode(self, req: Request, answer: str | None = None) -> dict:
        with self._lock:
            self._state.clear()
            self._state["answer"] = answer
            _, info = self._run(self._helper, req)
            state = dict(self._state)
        self.recorded_generate_kwargs = state["generate_kwargs"]
        inputs = state["inputs"] if answer is not None else state["prepared"]["inputs_cpu"]
        return {"inputs": inputs, "n": state.get("n", 0),
                "info": {**info, "input_tokens": int(inputs["input_ids"].shape[1])}}

    def build_inputs(self, req: Request) -> PreparedInputs:
        enc = self._encode(req)
        return PreparedInputs(enc["inputs"], enc["info"])

    def generation_kwargs(self, req: Request) -> dict:
        """The decoding keyword arguments MOSS's offline API hands to generate() for this request: the call's values,
        the API's defaults for the rest, normalized by the API's own _offline_build_call_kwargs."""
        cls = type(self._helper)
        if req.video:
            defaults = inspect.signature(cls.offline_video_generate).parameters
            call = self._video_call(req, [])
            requested = {k: call.get(k, defaults[k].default) for k in DECODING}
        else:
            requested = self._image_query(req, [])[0]["generate_kwargs"]
        return {k: v for k, v in cls._offline_build_call_kwargs(requested).items() if k in DECODING}

    # ---------- generation ----------
    def generate(self, req: Request) -> Generation:
        budget = req.settings["generation"]["max_new_tokens"]
        self.head.capture, self.head.captured, self.head.calls = req.kind == "mcq", None, 0
        try:
            with self.torch.inference_mode():
                text, info = self._run(self.model, req)
        finally:
            self.head.capture = False
        if (letters := letter_logits(self.head, self.processor.tokenizer, req)):
            info["letter_logits"] = letters
        stop = counted_stop(self.head, budget)
        return Generation(str(text).strip(), stop["tokens"], stop["finish"], {**info, "attention": self.attention})

    # ---------- fine-tuning (generalization experiments) ----------
    def encode(self, req: Request, answer: str | None = None) -> dict:
        return self._encode(req, answer)

    def train_forward(self, enc: dict):
        """Logits of the last n + 1 positions; vision_chunked_length 64 as the official offline generate."""
        inputs = self.model._offline_move_inputs_to_devices(enc["inputs"])
        self.head.keep = enc["n"] + 1
        try:
            return self.model(**inputs, vision_chunked_length=VISION_CHUNKED_LENGTH, use_cache=False).logits
        finally:
            self.head.keep = 0

    def load_lora(self, path) -> list[str]:
        """A fine-tuned model id (`options.lora`): its saved LoRA adapter attached unmerged (attach_lora). The run
        worker calls it right after construction."""
        if self.head is None:
            raise RuntimeError(f"{self.spec.id}: load_lora needs the model weights (the adapter was built with "
                               "load=False)")
        return attach_lora(self.model, *self.LORA_LAYERS, path)

    def close(self) -> None:
        if self.model is not None:
            self.model = self.head = None
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
