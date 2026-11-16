# Offline adapters (conditions P and C)

Conditions **P** (the parent) and **C** (the child) run in one shared offline pipeline: the model receives the
whole image set or the whole sampled video at once, followed by the benchmark question, and answers with greedy
decoding. Within a pair, the parent and its children receive *the same frames* (the pair's single video protocol,
`configs/pairs.yaml`) and *the same prompt text*; only the checkpoint differs. C - P therefore measures what the
streaming fine-tune did to the weights, independently of any streaming machinery.

Four adapter modules in `src/streamreplace/models/` serve the eleven models:

| Adapter | Models (pair) | Environment | Third-party code |
|---|---|---|---|
| `qwen_offline` | Qwen2.5-VL-3B, ThinkStream-3B, MMDuet2-3B (p2); Qwen3-VL-8B, AURA-8B (p3) | `modern` | transformers' stock Qwen2.5-VL / Qwen3-VL classes; qwen-vl-utils 0.0.14 |
| `llava_offline` | LLaVA-OneVision-7B, MMDuet-7B (p1) | `llava` | LLaVA-NeXT (`streamreplace fetch-upstream llava`) |
| `internvl_offline` | LiveStar-InternVideo-8B, LiveStar-8B (p4) | `livestar` | LiveStar's `inference/` model code (`streamreplace fetch-upstream livestar`) |
| `moss_offline` | MOSS-VL-Instruct-0708, MOSS-VL-Realtime (p5) | `modern` | the checkpoints' own model code (`trust_remote_code`) |

The streaming children are evaluated offline with their **architecture's stock pipeline**: ThinkStream and MMDuet2
run exactly as Qwen2.5-VL (no thinking tokens, no proactive prompt), AURA exactly as Qwen3-VL, MMDuet exactly as
LLaVA-OneVision (its score heads are not used), LiveStar exactly as LiveStar-InternVideo, and MOSS-VL-Realtime
through the same offline API as MOSS-VL-Instruct.

## Rules shared by every offline adapter

- **Frames.** Video frames come from the pair's protocol (`docs/protocol.md`, `docs/frames.md`):
  - p1, p4, p5: the framework's fixed-step sampler (2, 1 and 1 fps from frame 0; p1 switches to 160 uniform frames
    above 160 frames).
  - p2, p3: Qwen's own qwen-vl-utils `fetch_video` at 2 fps (an even number of frames, uniform over the whole
    video), read with torchcodec.
  - Video-SafetyBench (16 frames) and MCV-SafetyBench (33 frames) replace only the frame sampling, for every pair,
    by their official uniform frames (`offline_frames` in `configs/benchmarks.yaml`); pixel budgets and padding stay
    the pair's.

  Every video row records `info.frame_indices`, a digest of the sampled frame indices taken *before* any
  model-specific padding, so it is identical for all members of a pair and can be checked against other runs.
- **Prompt.** The benchmark's rendered prompt, byte for byte. Only TOMATO has a system prompt ("You are an expert in
  understanding dynamics of objects."); it replaces the template's default system message. Without one, each model
  keeps its template default (Qwen2.5: "You are a helpful assistant."; Qwen3: none; LLaVA's `qwen_1_5` template:
  "You are a helpful assistant."; LiveStar: the default of its `internvl2_5` template in upstream's
  `conversation.py`; MOSS: none).
- **Decoding.** Greedy: `do_sample=False, num_beams=1, temperature/top_p/top_k=None` with the benchmark's
  `generation.max_new_tokens` (`models/decoding.py: greedy()`), plus the model's `repetition_penalty` from
  `configs/models.yaml`:

  | Value | Models | Where the paper's runs took it from |
  |---|---|---|
  | 1.05 | LLaVA-OneVision, MMDuet, Qwen2.5-VL-3B, ThinkStream-3B, MMDuet2 | the checkpoint's `generation_config.json` (MMDuet's LoRA has none and uses its base's) |
  | 1.05 | LiveStar-InternVideo, LiveStar | passed by the adapter (the official demo's decoding; the releases have no generation config) |
  | 1.0 | Qwen3-VL-8B, AURA | the checkpoint's `generation_config.json` (no penalty) |
  | 1.0 | MOSS-VL-Instruct, MOSS-VL-Realtime | MOSS's offline API default |

  Each adapter checks the configured value against the checkpoint's `generation_config.json` when it has one, so a
  different checkpoint cannot silently change the decoding. A checkpoint folder that is missing (for MMDuet also its
  LLaVA-OneVision base) is reported as missing, with the `streamreplace download models <id>` command, not as a
  mismatch of the penalty. Sampling parameters set to `None` are inert under greedy
  decoding; the original runs did not pass them for LLaVA and LiveStar and inherited the checkpoint's values, which
  greedy decoding also ignores (the verification below checks that both give the same decoding configuration).
- **Budget.** `finish` is `length` when the reply used the whole token budget, `eos` when it ended on its own.
  Replies that hit the budget are the ones the paper reran with an 8,192-token budget (the truncation reruns of
  MathVista, MMMU and MMStar).
- **First-token letter logits.** For multiple-choice requests every adapter records `info.letter_logits`:
  `{letter: logit}` of the first generated token for each option letter of the request that is a single token of the
  model's tokenizer (rounded to 4 decimals), taken from the LM head's first call during generation (the last prompt
  position), so no extra forward pass is made. The prompt-robustness analysis reads them (the probability readout:
  the softmax over the option letters), and the fine-tuning report shows the letter accuracy; they never influence the
  reply. The capture is the one of the paper's runs (`models/lm_head.py`), unchanged.
- **Row info.** `frames`, `duration`, `frame_indices`, `input_tokens` and family-specific keys (`sample_fps`,
  `frame_size`, `rope_fix`, `sampler`, `pad_square`, `tiles`, `image_sizes`, `images`, `dtype`, `attention`).

### Settings of the controls

None of these is part of P or C. The diagnostic experiments ([../experiments.md](../experiments.md)) set them:

| Setting | Effect | Experiment |
|---|---|---|
| `video.last_seconds: N` (`qwen_offline`, qwen-vl-utils sampler) | the same sampler on the last N seconds of the video only: what a streaming window of N seconds keeps | the window controls C_W and P_W (`window`, `window_300`) |
| `video.blank: true` | the frame count and size are kept and every frame is black | `framing` (black frames); the fine-tuning blank control |
| `video.drop: true` | the request is sent as text, without its frames | `framing` (the bare request, the framing sentence alone) |

## Qwen2.5-VL / Qwen3-VL family (`qwen_offline`)

- **Classes.** `Qwen2_5_VLForConditionalGeneration` / `Qwen3VLForConditionalGeneration` from transformers 4.57.3
  (chosen by the checkpoint's `model_type`), bf16, FlashAttention-2 when installed (otherwise sdpa;
  `info.attention` records which). Qwen2.5 uses the slow image processor (`use_fast=False`), as in the paper's runs.
- **Images.** Each image is resized with Qwen's `smart_resize` (both sides multiples of 28 for Qwen2.5 and 32 for
  Qwen3, area within [3,136, 1,003,520] pixels, PIL bicubic); the processor is told not to resize again
  (`info.image_sizes`).
- **Video, p2/p3.** `fetch_video({video, fps: 2, [min_pixels,] max_pixels}, image_patch_size=14|16,
  return_video_sample_fps=True, return_video_metadata=True)`. qwen-vl-utils silently falls back to torchvision when
  torchcodec cannot load; the adapter refuses that (torchcodec needs the FFmpeg 6 libraries of the `ffmpeg`
  environment). The processor gets `do_sample_frames=False, do_resize=False` and, for Qwen2.5, `fps=[sample_fps]`;
  for Qwen3, the video metadata. A 720p video becomes 840x448 frames at p2 and 1024x576 at p3.
- **Video, framework frames** (Video-SafetyBench, MCV-SafetyBench). An odd frame count is padded by repeating the last
  frame (temporal patch size 2); each frame is `smart_resize`d within the pair's pixel range; Qwen2.5 gets
  `fps = frames / duration`, Qwen3 the frames' metadata.
- **Qwen2.5 temporal positions.** transformers 4.57's `Qwen2_5_VLModel.get_rope_index` converts the seconds per
  temporal grid step to an integer before scaling, which collapses or compresses temporal positions whenever that
  value is not a whole number. The paper's runs re-bound the method with the conversion done to float32
  (`models/patches/qwen25_rope.py`); every p2 video row records `info.rope_fix = true`. The patch is version-checked
  and raises if the installed transformers does not contain the exact expression it fixes.
- **Reply.** Decoded with special tokens, then `<|im_end|>` and `<|endoftext|>` removed and whitespace stripped.

## LLaVA-OneVision and MMDuet (`llava_offline`)

- **Loading.** LLaVA-NeXT's `load_pretrained_model(<LLaVA-OneVision folder>, None, "llava-onevision-qwen2-7b-ov",
  attn_implementation="sdpa", torch_dtype=float16, overwrite_config={"mm_vision_tower": <SigLIP folder>})`. The SigLIP
  folder's name must contain "siglip" (LLaVA-NeXT picks the vision tower class by name; the fetched LLaVA-NeXT
  carries the small dispatch fix listed in `manifests/upstream.yaml`).
- **MMDuet.** The LoRA and MMDuet's projector are loaded with PEFT (adapter name `mmduet`) onto the LLaVA-OneVision
  weights and kept **unmerged**. The two score heads (informative, relevance) belong to MMDuet's streaming decision
  and are not loaded; every other tensor of the LoRA file is checked to equal what was loaded
  (`info.mmduet_weights`).
- **Video (p1).** Frames at 2 fps (160 uniform frames above 160), each resized so its long side is 384 and padded
  to a black 384x384 square exactly as MMDuet's own video loader does (`pad_square`), then the SigLIP processor: 196
  tokens per frame plus one newline token. The text is `<image>\n` + the prompt.
- **Images.** LLaVA-NeXT's `process_images` with the checkpoint's `anyres_max_9` setting; one `<image>` per image,
  a newline, then the prompt. Text-only prompts (HarmBench text behaviours) use LLaVA's text path.
- **Template.** `conv_templates["qwen_1_5"]`.
- **Memory.** For video requests only the last prompt position is projected through the LM head
  (`models/lm_head.py`); its logits are unchanged, and a 160-frame prompt then fits in 48 GB.
- **Reply.** `batch_decode(skip_special_tokens=True)`, stripped.

## LiveStar-InternVideo and LiveStar (`internvl_offline`)

- **Loading.** The released checkpoints contain weights and config only. The adapter builds an overlay folder in the
  cache root (`<cache>/overlays/<checkpoint folder>`): symlinks to the checkpoint's files and to the model code and
  tokenizer from upstream LiveStar's `inference/` folder, plus a `model.safetensors.index.json` rebuilt from the shard
  headers. The model loads with `trust_remote_code` in bf16; the language model's generation config is rebuilt from
  the checkpoint's `llm_config`.
- **Tiles.** One 448x448 tile per frame or image (`max_dynamic_patch` 1), reduced by the model's token merging to
  16 visual tokens.
- **Video (p4).** Frames at 1 fps; the question is `Frame1: <image>\n...FrameN: <image>\n` followed by the prompt.
- **Images.** `<image>\n` for one image, `Image-1: <image>\nImage-2: <image>\n...` for several.
- **Decoding.** The model's `chat()` with greedy decoding and repetition penalty 1.05; it stops at `<|im_end|>`.
  The token count of a reply is the number of LM-head calls of the generation.

## MOSS-VL-Instruct and MOSS-VL-Realtime (`moss_offline`)

- **Video (p5).** The framework's 1 fps frames (the same indices the MOSS-VL-Realtime stream receives) are handed to
  MOSS's video processor by temporarily overriding its `fetch_videos` (`frames_as_video`), together with metadata
  that describes them (source fps, duration, frame indices, timestamps = index / source fps). The call is MOSS's
  `offline_video_generate` with the pair's pixel budget, MOSS-VL-Realtime's official one (at most 16,777,216 pixels
  per frame and 65,536,000 per video: native resolution unless a video exceeds the total), `max_frames` = the number
  of frames, the benchmark's system prompt when there is one, and the API's defaults otherwise (`shortest_edge` 4096,
  `video_fps` 1.0, `vision_chunked_length` 64).
- **Images.** One query with `max_pixels` = 1,003,520 and the image processor's size {shortest edge 4096, longest
  edge 1,003,520}; no system message.
- **Decoding.** `do_sample=False` with the token budget; the API's own sampling defaults are inert under greedy
  decoding and its repetition penalty is 1.0.

## Inputs on a CPU and verification

Every offline adapter accepts `load=False`: it then builds the processors and tokenizers (and, for LiveStar and MOSS,
an empty model skeleton without weights) but no model. `build_inputs(request)` returns the exact processor inputs
and the row information that `generate()` would produce, and `generation_kwargs(request)` the decoding arguments,
without a GPU. MOSS's inputs are produced by MOSS's own offline query path on a weightless instance of the model class,
with the final generation step replaced by a recorder, so they are exactly what its `generate()` receives.

With the processor files of each checkpoint (`streamreplace download models --processors-only`), these were checked on
a CPU against the paper's original rows: for 20 rows of every one of the 187 offline jobs (11 models x 17
benchmarks), and for all 900 Video-MME rows of the five parents, the input token count, number of frames, frame size,
sampling rate, frame digest, duration, tiles and image sizes recorded by the original runs are reproduced, the
resolved settings equal the original runs' settings, and the decoding arguments are greedy plus the checkpoint's
repetition penalty (with the same effective decoding configuration as the original calls). For LiveStar the text
tokenized by the model's own `chat()` was checked to equal the adapter's input text. On the prompt-robustness rows,
the option letters whose first-token logits an adapter records are the ones the original rows recorded. The
fine-tuning inputs (`encode`) were checked to be the evaluation input followed by exactly the supervised tokens, and
`load_lora`'s attachment was run on weightless models of every architecture with the fine-tuning experiment's own
adapters (LoRA configuration, every saved tensor's name and shape, placement on all decoder layers).

## Fine-tuning (generalization experiments)

The adapters also provide what the LoRA fine-tuning experiments need, built by the same code as the evaluation
inputs:

- `encode(request, answer)`: the evaluation input followed by `answer + <|im_end|>`, and `n`, the number of
  supervised tokens. An answer whose tokenization would merge with the end of the prompt is refused, so the trained
  prompt is exactly the evaluated one.
- `train_forward(enc)`: the logits of the last `n + 1` positions (only those are projected through the LM head).
- `load_lora(path)`: evaluating a fine-tuned model. A fine-tuned model is declared in an extra config file with
  `options.lora` (the adapter folder written by the training: `adapter_config.json` and
  `adapter_model.safetensors`); the run worker calls `load_lora` right after the base checkpoint is loaded, and
  refuses an adapter without it, so a fine-tuned model can never silently evaluate its base weights. The LoRA is
  injected with PEFT (0.12) from the folder's own configuration and loaded **unmerged**, so the model computes
  exactly the trained function; its weights stay in float32 as during training (they are cast before loading, so the
  trained values are not rounded to bf16). The adapter refuses a folder with tensors the model does not have, a
  folder that does not cover every injected module, and any placement other than the experiment's: a LoRA on every
  linear layer of every decoder layer and nowhere else (`LORA_LAYERS`):

  | Adapter | Decoder layers | LoRA-wrapped linear layers per decoder layer |
  |---|---|---|
  | `qwen_offline` | `model.language_model.layers` | q, k, v, o, gate, up, down (7) |
  | `llava_offline` | `model.layers` | q, k, v, o, gate, up, down (7) |
  | `internvl_offline` | `language_model.model.layers` | wqkv, wo, w1, w2, w3 (5) |
  | `moss_offline` | `model.language_model.layers` | q, k, v, o of the self- or cross-attention, gate, up, down (7) |

  Rows of a fine-tuned model are otherwise ordinary rows; the run identity records the sha256 of the adapter's
  weights.

The fine-tuning experiment ran the LLaVA-OneVision pair in bf16 and with MMDuet's LoRA merged; these are the model
options `dtype: bfloat16` and `merge: true` of `llava_offline` (a fine-tuning LoRA on MMDuet needs `merge: true`, the
model it was trained on). LiveStar's flash-attention calls are made deterministic when PyTorch's deterministic mode is
on, as the training requires (the worker switches it on for models with `options.deterministic`).

## Deviations from the models' official offline evaluations

These are properties of the shared protocol, kept for the comparison to be fair within each pair:

- Frames follow the pair's protocol rather than each model's own default (for example LLaVA-OneVision receives
  frames at 2 fps padded to a square like MMDuet's, and MOSS-VL-Instruct receives MOSS-VL-Realtime's pixel budget).
- Prompts are the benchmark harness's prompts, not model-specific prompt templates.
- Decoding is greedy with the benchmark's budget for every model.
