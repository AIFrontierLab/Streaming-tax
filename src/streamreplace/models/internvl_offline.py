"""Offline adapter of LiveStar-InternVideo-8B and LiveStar-8B (conditions P and C of pair p4): InternVL-2.5-style chat
models whose released checkpoints ship weights and config only.

Loading: the model code, tokenizer files and streaming helpers come from upstream LiveStar's inference/ folder
(`streamreplace fetch-upstream livestar`). An overlay folder in the cache root joins them with the checkpoint:
symlinks to the checkpoint's files and the code files, plus a model.safetensors.index.json rebuilt from the shard
headers (the release has none); it is the same overlay as the LiveStar stream adapter's (livestar_stream.overlay).
The model is loaded with trust_remote_code in bf16; the checkpoint config's `template`
(internvl2_5) and, when present, `system_message` replace the code's defaults; the language model's generation config
is rebuilt from the checkpoint's llm_config (the release has no generation_config.json).

Inputs (build_inputs() builds them on the CPU):
- tiles: one 448x448 tile per frame or image (max_dynamic_patch 1, upstream's dynamic_preprocess and transform), merged
  to 16 tokens by the model (token merging).
- video (pair p4): the framework's frames (1 fps from frame 0), "Frame{i}: <image>\\n" per frame (1-based), then the
  prompt. Images: "<image>\\n" for one image, "Image-{i}: <image>\\n" for several. Text-only prompts have no tiles.
- template: internvl2_5 with the template's own default system message (in upstream's conversation.py; this code does
  not restate it); the benchmark's system prompt (TOMATO) replaces it.
Decoding: the model's chat() with greedy decoding (models/decoding.py) and repetition_penalty 1.05, the official demo's
decoding, which the original runs passed explicitly; generation stops at <|im_end|>.

Fine-tuning: encode(req, answer) builds the ids chat() builds plus the answer; train_forward(enc) runs the model's
streaming_infer forward (its plain forward needs a torch.distributed process group); load_lora(path) attaches a
fine-tuned id's LoRA."""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import sys

from .. import paths
from ..core.schema import Generation, Request
from ..media.digest import frame_digest
from ..media.frames import offline_frames, open_rgb
from .base import EOT, BaseAdapter, PreparedInputs, supervised_length
from .decoding import check_repetition_penalty, greedy
from .livestar_stream import overlay
from .lm_head import counted_stop, head_wrapper, letter_logits
from .lora import attach_lora

CODE = "livestar/inference"     # upstream folder: remote model code, tokenizer files, streaming_infer.py
TILE = 448
IMG_CONTEXT = "<IMG_CONTEXT>"


class InternVLOffline(BaseAdapter):
    """Conditions P and C of the LiveStar models (env livestar). load=False builds the tokenizer and an empty model
    skeleton (no weights, meta device): build_inputs() and generation_kwargs() work on a CPU."""
    PROTOCOL_VERSION = 1
    LOAD_KEYS: tuple[str, ...] = ()
    CAPTURES_LETTER_LOGITS = True
    LORA_LAYERS = ("language_model.model.layers", 5)    # InternLM2 decoder layers; wqkv, wo, w1, w2, w3 get a LoRA

    def __init__(self, spec, settings: dict, upstream, load: bool = True):
        super().__init__(spec, settings, upstream)
        import torch
        self.torch = torch
        self._paths = ExitStack()
        # streaming_infer and the remote code import their siblings (tshm, conversation), so the folder stays on
        # sys.path for the adapter's lifetime
        self._paths.enter_context(upstream.use(CODE))
        try:
            self._setup(load)
        except BaseException:
            self._paths.close()
            raise

    def _setup(self, load: bool) -> None:
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer, GenerationConfig
        spec, upstream = self.spec, self.upstream
        from streaming_infer import build_transform, dynamic_preprocess
        self._transform, self._dynamic_preprocess = build_transform(input_size=TILE), dynamic_preprocess
        code = upstream.path(CODE)
        checkpoint = Path(paths.checkpoint_dir(spec.checkpoint))
        if (checkpoint / "generation_config.json").exists():
            check_repetition_penalty(spec.decoding.repetition_penalty, checkpoint, spec.id)
        # The releases have no generation config: the official demo's 1.05, which the original runs passed explicitly
        self.repetition_penalty = spec.decoding.repetition_penalty if spec.decoding.repetition_penalty is not None \
            else 1.05
        folder = overlay(checkpoint, code)
        self.tokenizer = AutoTokenizer.from_pretrained(str(code), trust_remote_code=True, local_files_only=True)
        if load:
            self.model = AutoModel.from_pretrained(str(folder), trust_remote_code=True, local_files_only=True,
                                                   torch_dtype=torch.bfloat16).cuda().eval()
        else:
            from accelerate import init_empty_weights
            config = AutoConfig.from_pretrained(str(folder), trust_remote_code=True, local_files_only=True)
            with init_empty_weights():
                self.model = AutoModel.from_config(config, trust_remote_code=True, torch_dtype=torch.bfloat16)
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        for key in ("template", "system_message"):
            if key in config:
                setattr(self.model, key, config[key])
        self.max_tiles = int(config.get("max_dynamic_patch", 1))
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT)   # chat() sets it too
        self.head = None
        if not load:
            return
        # the release has no generation_config: the language model's defaults (incl. BOS) from llm_config
        lm_config = self.model.language_model.config.__class__.from_dict(config["llm_config"])
        self.model.language_model.generation_config = GenerationConfig.from_model_config(lm_config)
        if torch.are_deterministic_algorithms_enabled():   # fine-tuning: InternLM2 calls flash-attn directly (and
            import functools                               # re-imports it in every forward), so its backward is
            import flash_attn                              # deterministic only with the flag set at the source
            for name in ("flash_attn_func", "flash_attn_varlen_func"):
                function = getattr(flash_attn, name)
                if not isinstance(function, functools.partial):
                    setattr(flash_attn, name, functools.partial(function, deterministic=True))
        lm = self.model.language_model
        self.head = lm.output = head_wrapper(lm.output)

    # ---------- inputs ----------
    def _pixels(self, images) -> list:
        """One stack of 448x448 tiles per frame or image (upstream's dynamic_preprocess with use_thumbnail=True, which
        adds nothing when there is a single tile)."""
        return [self.torch.stack([self._transform(t) for t in self._dynamic_preprocess(
            i, image_size=TILE, max_num=self.max_tiles, use_thumbnail=True)]) for i in images]

    def _media(self, req: Request) -> tuple[list, str, dict]:
        """Tiles per frame or image, the question prefix that places them, and the frame info."""
        info = {}
        if req.video:
            frames, meta = offline_frames(req.video, req.settings["video"])
            prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(frames)))
            info.update(frames=len(frames), duration=round(meta["duration"], 2), frame_indices=frame_digest(meta))
        else:
            frames = [open_rgb(p) for p in req.images]
            prefix = ("" if not frames else "<image>\n" if len(frames) == 1
                      else "".join(f"Image-{i + 1}: <image>\n" for i in range(len(frames))))
        tiles = self._pixels(frames)
        return tiles, prefix, {**info, "tiles": sum(len(t) for t in tiles)}

    def _query(self, req: Request, tiles: list, prefix: str, answer: str | None = None) -> tuple[str, int]:
        """The text chat() builds for a first turn (history=None): template + system message, the user turn, an open
        assistant turn, each <image> from the right replaced by <img> + <IMG_CONTEXT> x num_image_token x tiles +
        </img>; with an answer, + answer + <|im_end|> for a training step (n = the supervised tokens)."""
        question = prefix + req.prompt
        if tiles and "<image>" not in question:
            question = "<image>\n" + question
        template = sys.modules[type(self.model).__module__].get_conv_template(self.model.template)
        template.system_message = req.settings.get("system") or self.model.system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        for num_patches in [len(t) for t in tiles]:
            image_tokens = "<img>" + IMG_CONTEXT * self.model.num_image_token * num_patches + "</img>"
            index = query.rfind("<image>")
            query = query[:index] + image_tokens + query[index + len("<image>"):]
        n = 0
        if answer is not None:
            n = supervised_length(lambda t: self.tokenizer(t).input_ids, query, answer)
            query += answer + EOT
        return query, n

    def _encode(self, req: Request, answer: str | None = None) -> dict:
        tiles, prefix, info = self._media(req)
        query, n = self._query(req, tiles, prefix, answer)
        tokens = self.tokenizer(query, return_tensors="pt")
        pixels = self.torch.cat(tiles) if tiles else None
        return {"inputs": {"pixel_values": pixels, "input_ids": tokens.input_ids,
                           "attention_mask": tokens.attention_mask},
                "n": n, "info": {**info, "input_tokens": int(tokens.input_ids.shape[1])}}

    def build_inputs(self, req: Request) -> PreparedInputs:
        enc = self._encode(req)
        return PreparedInputs(enc["inputs"], enc["info"])

    def generation_kwargs(self, req: Request) -> dict:
        """The generation config chat() receives: greedy() with repetition_penalty 1.05 (chat() adds the
        <|im_end|> stop token)."""
        return {**greedy(req.settings), "repetition_penalty": self.repetition_penalty}

    # ---------- generation ----------
    def generate(self, req: Request) -> Generation:
        s, torch = req.settings, self.torch
        tiles, prefix, info = self._media(req)
        info["input_tokens"] = len(self.tokenizer(self._query(req, tiles, prefix)[0]).input_ids)
        pixels = torch.cat(tiles).to("cuda", dtype=torch.bfloat16) if tiles else None   # None: text-only chat
        default = self.model.system_message
        if s.get("system"):
            self.model.system_message = s["system"]
        self.head.capture, self.head.captured, self.head.calls = req.kind == "mcq", None, 0
        try:
            with torch.inference_mode():
                text = self.model.chat(self.tokenizer, pixels, prefix + req.prompt, self.generation_kwargs(req),
                                       num_patches_list=[len(t) for t in tiles], history=None, return_history=False)
        finally:
            self.model.system_message = default
            self.head.capture = False
        if (letters := letter_logits(self.head, self.tokenizer, req)):
            info["letter_logits"] = letters
        stop = counted_stop(self.head, s["generation"]["max_new_tokens"])
        return Generation(text.strip(), stop["tokens"], stop["finish"], info)

    # ---------- fine-tuning (generalization experiments) ----------
    def encode(self, req: Request, answer: str | None = None) -> dict:
        return self._encode(req, answer)

    def train_forward(self, enc: dict):
        """Logits of the last n + 1 positions, through streaming_infer (forward() needs a torch.distributed process
        group)."""
        x, torch = enc["inputs"], self.torch
        self.head.keep = enc["n"] + 1
        try:
            return self.model.streaming_infer(
                pixel_values=x["pixel_values"].to("cuda", dtype=torch.bfloat16), input_ids=x["input_ids"].cuda(),
                attention_mask=x["attention_mask"].cuda(),
                image_flags=torch.ones(x["pixel_values"].shape[0], dtype=torch.long, device="cuda"),
                use_cache=False).logits
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
        if self.head is not None:
            self.model = self.head = None
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        self._paths.close()
