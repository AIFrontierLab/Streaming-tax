"""Offline adapter of LLaVA-OneVision-7B and MMDuet-7B (conditions P and C of pair p1), on the LLaVA-NeXT code.

MMDuet is LLaVA-OneVision with a LoRA on the language model and a retrained projector. Offline it runs in LLaVA-NeXT's
pipeline exactly as its parent: the LoRA (PEFT adapter "mmduet") and its projector are loaded onto the parent's
weights and kept unmerged; MMDuet's two score heads (informative, relevance) are not part of an offline forward and
are not loaded. Every loaded LoRA and projector tensor is checked to equal the file.

Inputs (build_inputs() builds them on the CPU):
- video (pair p1): the framework's frames (2 fps from frame 0; 160 uniform frames when there would be more), each
  padded to a black 384x384 square exactly like MMDuet's own video loader (pad_to_square), then the SigLIP processor:
  196 tokens per frame plus one newline token. The text is "<image>\\n" + the benchmark prompt.
- images: LLaVA-NeXT's process_images with the checkpoint's anyres_max_9 setting; one "<image>" per image, a newline,
  the prompt. Text-only prompts use LLaVA's plain text path.
- template: conv_templates["qwen_1_5"]; the benchmark's system prompt (TOMATO) replaces the template's system line,
  otherwise it stays "You are a helpful assistant.".

Decoding: greedy (models/decoding.py) with repetition_penalty 1.05, the parent's generation_config.json value (MMDuet's
LoRA folder has no generation config, so MMDuet uses the parent's, as in the original runs); fp16, sdpa (the llava
environment has no FlashAttention). For video requests only the last position of the prefill is projected through the
LM head (models/lm_head.py): its logits are unchanged and a 160-frame prompt fits in 48 GB.

The SigLIP tower is loaded from a local folder whose name must contain "siglip" (LLaVA-NeXT picks the tower class by
name; see the patch of llava/model/multimodal_encoder/builder.py in manifests/upstream.yaml).

Model options (configs/models.yaml `options`; the fine-tuning experiment only): `dtype` (float16, the builder's
default and the paper's evaluation, or bfloat16) and `merge` (true: the LoRA folded into the weights and the projector
swapped in). Fine-tuning: encode(req, answer) / train_forward(enc), built by the same code as the evaluation inputs;
load_lora(path) attaches a fine-tuned id's LoRA (on MMDuet only with `merge`, the model the experiment trained)."""
from __future__ import annotations

from contextlib import ExitStack
import copy
from pathlib import Path

from .. import paths
from ..core.schema import Generation, Request
from ..media.digest import frame_digest
from ..media.frames import offline_frames, open_rgb
from .base import EOT, BaseAdapter, PreparedInputs, supervised_length
from .decoding import check_repetition_penalty, checkpoint_repetition_penalty, greedy, missing_checkpoint
from .lm_head import head_wrapper, letter_logits
from .lora import attach_lora

MODEL_NAME = "llava-onevision-qwen2-7b-ov"     # LLaVA-NeXT's builder selects the Qwen LLaVA class from this name
TEMPLATE = "qwen_1_5"
DEVICE = "cuda:0"
RESOLUTION = 384                               # SigLIP input side and MMDuet's square frame


def pad_to_square(frame, resolution: int = RESOLUTION):
    """MMDuet demo/liveinfer.py load_video's frame geometry, verbatim: the long side resized to `resolution` (OpenCV
    bilinear, short side truncated to an integer), then centered black padding to resolution x resolution.
    `frame`: an HxWx3 uint8 RGB array."""
    import cv2
    height, width = frame.shape[:2]
    if width > height:
        new_width, new_height = resolution, int((height / width) * resolution)
    else:
        new_height, new_width = resolution, int((width / height) * resolution)
    resized = cv2.resize(frame, (new_width, new_height))
    return cv2.copyMakeBorder(resized, top=(resolution - new_height) // 2, bottom=(resolution - new_height + 1) // 2,
                              left=(resolution - new_width) // 2, right=(resolution - new_width + 1) // 2,
                              borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))


class LlavaOffline(BaseAdapter):
    """Conditions P and C of LLaVA-OneVision-7B and MMDuet-7B (env llava). load=False builds the tokenizer, the SigLIP
    processor and the model config only: build_inputs() and generation_kwargs() work on a CPU without the weights."""
    PROTOCOL_VERSION = 1
    LOAD_KEYS: tuple[str, ...] = ()
    CAPTURES_LETTER_LOGITS = True
    LORA_LAYERS = ("model.layers", 7)       # Qwen2 decoder layers; q, k, v, o, gate, up, down get a LoRA

    def __init__(self, spec, settings: dict, upstream, load: bool = True):
        super().__init__(spec, settings, upstream)
        import torch
        self.torch = torch
        self._paths = ExitStack()
        # LLaVA-NeXT's modules import each other lazily, so the package stays importable for the adapter's lifetime
        self._paths.enter_context(upstream.use("llava"))
        try:
            self._setup(load)
        except BaseException:
            self._paths.close()
            raise

    def _setup(self, load: bool) -> None:
        torch, spec = self.torch, self.spec
        from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN
        base_ref = spec.base or spec.checkpoint
        base = Path(paths.checkpoint_dir(base_ref))
        # checked first: without its folder the penalty check below would report a configuration mismatch
        if not (base / "config.json").is_file():
            what = f"{'the LoRA base ' if spec.base else ''}checkpoint {base_ref.repo}"
            raise missing_checkpoint(spec.id, base, what, "no config.json")
        self.vision = str(paths.checkpoint_dir(spec.vision))
        if "siglip" not in self.vision.lower():
            raise ValueError(f"{spec.id}: the vision tower folder {self.vision} must have 'siglip' in its name "
                             "(LLaVA-NeXT chooses the tower class by name)")
        penalty = spec.decoding.repetition_penalty       # MMDuet's LoRA uses its base's generation config
        check_repetition_penalty(penalty, base, spec.id)
        self.repetition_penalty = penalty if penalty is not None else checkpoint_repetition_penalty(base)
        dtype = spec.options.get("dtype", "float16")
        if dtype not in ("float16", "bfloat16"):
            raise ValueError(f"{spec.id}: options.dtype must be float16 or bfloat16, got {dtype!r}")
        self.model = self.head = None
        if not load:
            from transformers import AutoTokenizer
            from llava.model.language_model.llava_qwen import LlavaQwenConfig
            from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
            # what load_pretrained_model builds for this model, without the weights
            self.tokenizer = AutoTokenizer.from_pretrained(str(base))
            self.config = LlavaQwenConfig.from_pretrained(str(base))
            self.config.mm_vision_tower = self.vision
            if getattr(self.config, "mm_use_im_patch_token", True):
                self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            if getattr(self.config, "mm_use_im_start_end", False):
                self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.processor = SigLipImageProcessor()
            self.info = {"dtype": str(getattr(torch, dtype))}
            return
        from llava.model.builder import load_pretrained_model
        self.tokenizer, self.model, self.processor, _ = load_pretrained_model(
            str(base), None, MODEL_NAME, device_map=DEVICE, attn_implementation="sdpa", torch_dtype=dtype,
            overwrite_config={"mm_vision_tower": self.vision})
        if dtype == "bfloat16":                          # the builder casts the vision tower to fp16 regardless
            self.model.get_vision_tower().to(dtype=torch.bfloat16)
        self.info = {"dtype": str(self.model.dtype)}
        if spec.base is not None:
            self._apply_mmduet(Path(paths.checkpoint_dir(spec.checkpoint)))
            if spec.options.get("merge"):                # LoRA folded into the weights (and the projector swapped in)
                self.model = self.model.merge_and_unload()
                self.info["mmduet_merged"] = True
        self.model.eval()
        self.config = self.model.config
        self._wrap_lm_head()

    def _wrap_lm_head(self) -> None:
        """transformers 4.44's Qwen2 projects (and casts to fp32) the logits of every prompt position although
        generate() reads only the last one; a 160-frame prompt (~31k tokens x 152k vocabulary) needs ~18 GB for that
        alone. Video requests project only the last position (head.keep = 1, the same logits for it)."""
        inner = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        self.head = inner.lm_head = head_wrapper(inner.lm_head)

    def _apply_mmduet(self, path: Path) -> None:
        """MMDuet's LoRA + mm_projector through PEFT (adapter "mmduet", unmerged); the informative / relevance heads
        are not part of an offline forward. Every other tensor of the file must have been loaded exactly."""
        import peft
        from safetensors.torch import load_file
        weights = load_file(str(path / "adapter_model.safetensors"))
        self.model = peft.PeftModel.from_pretrained(self.model, str(path), adapter_name="mmduet", is_trainable=False,
                                                    autocast_adapter_dtype=False)
        state = self.model.state_dict()
        mapped = excluded = 0
        for source, tensor in weights.items():
            if ".informative_head." in source or ".relevance_head." in source:
                excluded += 1
                continue
            target = (source.replace(".lora_A.weight", ".lora_A.mmduet.weight")
                      .replace(".lora_B.weight", ".lora_B.mmduet.weight")
                      .replace(".mm_projector.", ".mm_projector.modules_to_save.mmduet."))
            if target not in state or not self.torch.equal(state[target].cpu(), tensor.to(state[target].dtype)):
                raise RuntimeError(f"MMDuet weight not loaded exactly: {source}")
            mapped += 1
        self.info["mmduet_weights"] = {"mapped": mapped, "excluded_heads": excluded}

    # ---------- inputs ----------
    def _encode(self, req: Request, answer: str | None = None) -> dict:
        """generate()'s inputs on the CPU; with an answer, prompt + answer + <|im_end|> for a training step."""
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token
        s, info = req.settings, dict(self.info)
        if req.video:
            import numpy as np
            frames, meta = offline_frames(req.video, s["video"])
            arrays = [np.asarray(f) for f in frames]
            if s["video"].get("pad_square"):             # the frame geometry of MMDuet's stream input
                arrays = [pad_to_square(a) for a in arrays]
            pixels = self.processor.preprocess(np.stack(arrays), return_tensors="pt")["pixel_values"]
            images, sizes, modalities = [pixels], [arrays[0].shape[1::-1]], ["video"]
            text = DEFAULT_IMAGE_TOKEN + "\n" + req.prompt
            info.update(frames=len(frames), duration=round(meta["duration"], 2), frame_indices=frame_digest(meta),
                        pad_square=bool(s["video"].get("pad_square")))
        elif req.images:
            pil = [open_rgb(p) for p in req.images]
            images = list(process_images(pil, self.processor, self.config))
            sizes, modalities = [i.size for i in pil], ["image"] * len(pil)
            text = DEFAULT_IMAGE_TOKEN * len(pil) + "\n" + req.prompt
        else:                                            # text-only prompt: LLaVA's plain-embedding path
            images = sizes = modalities = None
            text = req.prompt
        conversation = copy.deepcopy(conv_templates[TEMPLATE])
        if s.get("system"):
            conversation.system = "<|im_start|>system\n" + s["system"]
        conversation.append_message(conversation.roles[0], text)
        conversation.append_message(conversation.roles[1], None)
        prompt, n = conversation.get_prompt(), 0
        if answer is not None:      # tokenizer_image_token tokenizes the text after the last image token as one chunk
            n = supervised_length(lambda t: self.tokenizer(t).input_ids, prompt.rsplit(DEFAULT_IMAGE_TOKEN, 1)[-1],
                                  answer)
            prompt += answer + EOT
        ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        return {"inputs": {"input_ids": ids, "images": images, "image_sizes": sizes, "modalities": modalities},
                "n": n, "info": {**info, "input_tokens": int(ids.shape[1])}}

    def build_inputs(self, req: Request) -> PreparedInputs:
        enc = self._encode(req)
        return PreparedInputs(enc["inputs"], enc["info"])

    def generation_kwargs(self, req: Request) -> dict:
        """generate() keyword arguments: greedy() with the model's repetition_penalty."""
        return {**greedy(req.settings), "repetition_penalty": self.repetition_penalty}

    def _model_inputs(self, enc: dict) -> dict:
        x = enc["inputs"]
        images = [t.to("cuda", dtype=self.model.dtype) for t in x["images"]] if x["images"] is not None else None
        return {"images": images, "image_sizes": x["image_sizes"], "modalities": x["modalities"]}

    # ---------- generation ----------
    def generate(self, req: Request) -> Generation:
        enc = self._encode(req)
        ids, info = enc["inputs"]["input_ids"].cuda(), enc["info"]
        budget = req.settings["generation"]["max_new_tokens"]
        self.head.keep = 1 if req.video else 0          # image prompts keep the unmodified full projection
        self.head.capture, self.head.captured = req.kind == "mcq", None
        try:
            with self.torch.inference_mode():
                output = self.model.generate(ids, **self.generation_kwargs(req), **self._model_inputs(enc))
        finally:
            self.head.keep, self.head.capture = 0, False
        if (letters := letter_logits(self.head, self.tokenizer, req)):
            info["letter_logits"] = letters
        tokens = int(output.shape[1])                  # generate() on embeddings returns only the new tokens
        text = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
        return Generation(text, tokens, "length" if tokens >= budget else "eos", info)

    # ---------- fine-tuning (generalization experiments) ----------
    def encode(self, req: Request, answer: str | None = None) -> dict:
        return self._encode(req, answer)

    def train_forward(self, enc: dict):
        """Logits of the last n + 1 positions (they predict the answer + <|im_end|>)."""
        self.head.keep = enc["n"] + 1
        try:
            return self.model(input_ids=enc["inputs"]["input_ids"].cuda(), use_cache=False,
                              **self._model_inputs(enc)).logits
        finally:
            self.head.keep = 0

    def load_lora(self, path) -> list[str]:
        """A fine-tuned model id (`options.lora`): its saved LoRA adapter attached unmerged (attach_lora). The run
        worker calls it right after construction. MMDuet must be loaded with `merge: true`: the fine-tuning experiment
        trained on the merged model, and PEFT's wrapper of the unmerged MMDuet LoRA has other module names."""
        if self.head is None:
            raise RuntimeError(f"{self.spec.id}: load_lora needs the model weights (the adapter was built with "
                               "load=False)")
        if hasattr(self.model, "get_base_model"):
            raise RuntimeError(f"{self.spec.id}: a fine-tuning LoRA on MMDuet needs options.merge: true")
        return attach_lora(self.model, *self.LORA_LAYERS, path)

    def close(self) -> None:
        if self.model is not None:
            self.model = self.head = None
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        self._paths.close()
