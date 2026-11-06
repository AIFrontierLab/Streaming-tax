"""Offline adapter of the Qwen2.5-VL / Qwen3-VL family (conditions P and C): Qwen2.5-VL-3B, ThinkStream-3B and
MMDuet2-3B (pair p2), Qwen3-VL-8B and AURA-8B (pair p3).

The children run the stock pipeline of their architecture exactly as their parent does (the stock transformers class
and chat template, no think tokens, no proactive prompt); within a pair only the checkpoint differs.

Inputs (build_inputs() builds them on the CPU):
- images: each image resized with Qwen's smart_resize (both sides multiples of 28 for Qwen2.5, 32 for Qwen3; area in
  [image.min_pixels or 3136, image.max_pixels]) with PIL bicubic; the processor is told not to resize again.
- video with `video.sampler: qwen_vl_utils` (pairs p2 and p3 on every video benchmark without offline_frames): Qwen's
  qwen-vl-utils 0.0.14 `fetch_video` at `video.fps` (an even number of frames, uniform over the whole video) with its
  own resize within [video.min_pixels, video.max_pixels]. It must read the video with torchcodec, which needs the
  FFmpeg 6 shared libraries: qwen-vl-utils silently falls back to torchvision when torchcodec cannot load, so any
  other backend is an error. The processor gets do_sample_frames=False, do_resize=False and the sampling rate the
  library reports (Qwen2.5: fps=[sample_fps]; Qwen3: the video metadata).
- video with the framework's frames (benchmarks with offline_frames: Video-SafetyBench's 16 and MCV-SafetyBench's 33
  uniform frames, frame lists): media.frames.offline_frames; an odd count is padded with the last frame (temporal
  patch size 2); each frame is smart_resized within [video.min_pixels or 3136, video.max_pixels]; Qwen2.5 gets
  fps = frames / duration, Qwen3 the frame metadata.
- the message: images or the video, then the prompt; the benchmark's system prompt (TOMATO) as a system turn,
  otherwise the template's default (Qwen2.5: "You are a helpful assistant."; Qwen3: none).
- `video.as_images` (a fine-tuning control, never used for P/C/S): the sampled frames as separate images.

Decoding: greedy (models/decoding.py) with the model's repetition_penalty from configs/models.yaml, checked against
the checkpoint's generation_config.json (1.05 for the p2 models, 1.0 for the p3 models; the original runs inherited
it from there). bf16; FlashAttention-2 when installed, otherwise sdpa (info.attention records which). The reply is
decoded with special tokens and stripped of <|im_end|> / <|endoftext|>. Qwen2.5 checkpoints get the temporal-position
fix of patches/qwen25_rope.py. Multiple-choice requests record the first token's option-letter logits
(info.letter_logits) for the prompt-robustness analysis.

Fine-tuning (generalization experiments): encode(req, answer) builds the training inputs with the same code as the
evaluation inputs, train_forward(enc) returns the logits of the supervised span, and load_lora(path) attaches a
fine-tuned model id's saved LoRA adapter with models/lora.attach_lora (shared by every offline adapter; each one names
its decoder layers in LORA_LAYERS)."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .. import paths
from ..core.schema import Generation, Request
from ..media.digest import frame_digest
from ..media.frames import offline_frames, open_rgb
from .base import EOT, BaseAdapter, PreparedInputs, supervised_length
from .decoding import check_repetition_penalty, checkpoint_repetition_penalty, greedy
from .lm_head import head_wrapper, letter_logits
from .lora import attach_lora

DEVICE = "cuda:0"          # the worker exposes exactly one GPU per process (CUDA_VISIBLE_DEVICES)
MIN_PIXELS = 3136          # qwen-vl-utils' default pixel floor (4 * 28 * 28)
BICUBIC = 3                # PIL.Image.BICUBIC


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Qwen's resize rule: both sides divisible by `factor`, area within [min_pixels, max_pixels], aspect ratio kept
    (sides rounded to the nearest multiple, floored when shrinking, ceiled when growing)."""
    h, w = max(factor, round(height / factor) * factor), max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h, w = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    return h, w


def attention_implementation() -> str:
    """FlashAttention-2 when the package imports (as in the paper's runs), else PyTorch's sdpa."""
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return "sdpa"
    return "flash_attention_2"


class QwenOffline(BaseAdapter):
    """Conditions P and C of the Qwen2.5-VL / Qwen3-VL models (env modern). load=False builds the processor only:
    build_inputs() and generation_kwargs() work on a CPU without the weights."""
    PROTOCOL_VERSION = 1
    LOAD_KEYS: tuple[str, ...] = ()
    CAPTURES_LETTER_LOGITS = True
    LORA_LAYERS = ("model.language_model.layers", 7)    # decoder layers; q, k, v, o, gate, up, down get a LoRA

    def __init__(self, spec, settings: dict, upstream, load: bool = True):
        super().__init__(spec, settings, upstream)
        import torch
        import transformers
        from transformers import AutoProcessor
        from .patches import qwen25_rope
        self.torch = torch
        checkpoint = Path(paths.checkpoint_dir(spec.checkpoint))
        model_type = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))["model_type"]
        self.qwen3 = model_type == "qwen3_vl"
        self.factor = 32 if self.qwen3 else 28            # patch 16 / 14 x spatial merge 2
        penalty = spec.decoding.repetition_penalty
        check_repetition_penalty(penalty, checkpoint, spec.id)
        self.repetition_penalty = penalty if penalty is not None else checkpoint_repetition_penalty(checkpoint)
        self.attention = attention_implementation()
        # Qwen2.5 uses the slow image processor, as in the paper's runs
        self.processor = AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True,
                                                       **({} if self.qwen3 else {"use_fast": False}))
        # Qwen2.5 rows record whether the temporal-position fix is (load=False: would be) applied
        self.rope_fix = None if self.qwen3 else qwen25_rope.needs_fix()
        self.model = self.head = None
        if not load:
            return
        cls = transformers.Qwen3VLForConditionalGeneration if self.qwen3 else \
            transformers.Qwen2_5_VLForConditionalGeneration
        self.model = cls.from_pretrained(str(checkpoint), local_files_only=True, torch_dtype=torch.bfloat16,
                                         device_map=DEVICE, attn_implementation=self.attention).eval()
        if not self.qwen3:
            self.rope_fix = qwen25_rope.apply(self.model.model)
        self.head = self.model.lm_head = head_wrapper(self.model.lm_head)

    # ---------- inputs ----------
    def _fetch_video(self, path: str, video: dict):
        """Qwen's official offline sampling (qwen-vl-utils fetch_video, as in the Qwen2.5-VL / Qwen3-VL model cards),
        read with torchcodec (exact frame count)."""
        from ..media.video_meta import is_frame_list
        if is_frame_list(path):
            raise ValueError(f"{path} is a frame list: the benchmark needs offline_frames (e.g. sampler: all)")
        from qwen_vl_utils import fetch_video
        keys = ("fps", "min_frames", "max_frames", "min_pixels", "max_pixels")
        ele = {"video": path, **{k: video[k] for k in keys if k in video}}
        if "fps" not in ele:
            ele["nframes"] = video["nframes"]
        (frames, metadata), sample_fps = fetch_video(ele, image_patch_size=self.factor // 2,
                                                     return_video_sample_fps=True, return_video_metadata=True)
        if metadata.get("video_backend") != "torchcodec":
            raise RuntimeError(f"qwen-vl-utils read {path} with {metadata.get('video_backend')}, not torchcodec "
                               "(FFmpeg 6 libraries missing? the model needs ffmpeg: true in configs/models.yaml)")
        return frames, metadata, sample_fps

    def _resize(self, image, max_pixels: int, min_pixels: int):
        h, w = smart_resize(image.height, image.width, self.factor, min_pixels, max_pixels)
        return image.resize((w, h), resample=BICUBIC)

    def _encode(self, req: Request, answer: str | None = None) -> dict:
        """The processor inputs (CPU) of generate(); with an answer, prompt + answer + <|im_end|> for a training step
        (n = the supervised tokens)."""
        s = req.settings
        content, images, videos, kwargs, info = [], [], [], {}, {}
        image_pixels = (s["image"]["max_pixels"], s["image"].get("min_pixels", MIN_PIXELS))
        for path in req.images:
            images.append(self._resize(open_rgb(path), *image_pixels))
            content.append({"type": "image"})
        if req.video and s["video"].get("as_images"):     # fine-tuning control: the frames as separate images
            frames, meta = offline_frames(req.video, s["video"])
            for frame in frames:
                images.append(self._resize(frame, *image_pixels))
                content.append({"type": "image"})
            info.update(frame_indices=frame_digest(meta), frames=len(frames), as_images=True)
        elif req.video and s["video"].get("sampler") == "qwen_vl_utils":
            video, metadata, sample_fps = self._fetch_video(req.video, s["video"])
            videos.append(video)
            content.append({"type": "video"})
            kwargs = {"do_sample_frames": False, "do_resize": False}
            if self.qwen3:
                kwargs["video_metadata"] = [metadata]
            else:
                kwargs["fps"] = [sample_fps]
                info["rope_fix"] = self.rope_fix
            info.update(frame_indices=frame_digest({"indices": metadata["frames_indices"]}), frames=int(video.shape[0]),
                        frame_size=[int(video.shape[3]), int(video.shape[2])], sample_fps=round(sample_fps, 4),
                        duration=round(metadata["total_num_frames"] / metadata["fps"], 2), sampler="qwen_vl_utils")
        elif req.video:
            import numpy as np
            frames, meta = offline_frames(req.video, s["video"])
            info["frame_indices"] = frame_digest(meta)    # before padding: the same digest for every model of the pair
            if len(frames) % 2:                           # temporal patch size 2
                frames.append(frames[-1])
                meta["indices"].append(meta["indices"][-1])
            video_pixels = (s["video"]["max_pixels"], s["video"].get("min_pixels", MIN_PIXELS))
            frames = [self._resize(f, *video_pixels) for f in frames]
            videos.append(np.stack([np.asarray(f) for f in frames]))
            content.append({"type": "video"})
            sample_fps = len(frames) / max(meta["duration"], 1e-6)
            kwargs = {"do_sample_frames": False, "do_resize": False}
            if self.qwen3:
                kwargs["video_metadata"] = [{"fps": meta["fps"], "frames_indices": meta["indices"],
                                             "total_num_frames": round(meta["duration"] * meta["fps"]),
                                             "duration": meta["duration"]}]
            else:
                kwargs["fps"] = [sample_fps]
                info["rope_fix"] = self.rope_fix
            info.update(frames=len(frames), frame_size=list(frames[0].size), sample_fps=round(sample_fps, 4),
                        duration=round(meta["duration"], 2))
        content.append({"type": "text", "text": req.prompt})
        messages = ([{"role": "system", "content": s["system"]}] if s.get("system") else []) + \
            [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        n = 0
        if answer is not None:
            n = supervised_length(lambda t: self.processor.tokenizer(t, add_special_tokens=False).input_ids, text,
                                  answer)
            text += answer + EOT
        inputs = self.processor(text=[text], images=images or None, videos=videos or None, return_tensors="pt",
                                padding=True, **({"do_resize": False} if images and not videos else {}), **kwargs)
        info.update(input_tokens=int(inputs.input_ids.shape[1]), attention=self.attention)
        if images:
            info["image_sizes"] = [list(i.size) for i in images]
        return {"inputs": inputs, "n": n, "info": info}

    def build_inputs(self, req: Request) -> PreparedInputs:
        enc = self._encode(req)
        return PreparedInputs(enc["inputs"], enc["info"])

    def generation_kwargs(self, req: Request) -> dict:
        """generate() keyword arguments: greedy() with the model's repetition_penalty."""
        return {**greedy(req.settings), "repetition_penalty": self.repetition_penalty}

    # ---------- generation ----------
    def generate(self, req: Request) -> Generation:
        enc = self._encode(req)
        inputs, info = enc["inputs"].to(DEVICE), enc["info"]
        self.head.capture, self.head.captured = req.kind == "mcq", None
        try:
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, **self.generation_kwargs(req))
        finally:
            self.head.capture = False
        if (letters := letter_logits(self.head, self.processor.tokenizer, req)):
            info["letter_logits"] = letters
        ids = output[0, inputs.input_ids.shape[1]:]
        budget = req.settings["generation"]["max_new_tokens"]
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        finish = "eos" if len(ids) and int(ids[-1]) in eos else ("length" if len(ids) >= budget else "unknown")
        text = self.processor.tokenizer.decode(ids, skip_special_tokens=False)
        text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
        return Generation(text.strip(), int(len(ids)), finish, info)

    # ---------- fine-tuning (generalization experiments) ----------
    def encode(self, req: Request, answer: str | None = None) -> dict:
        return self._encode(req, answer)

    def train_forward(self, enc: dict):
        """Logits of the last n + 1 positions (they predict the answer + <|im_end|>)."""
        self.head.keep = enc["n"] + 1
        try:
            return self.model(**enc["inputs"].to(DEVICE), use_cache=False).logits
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
