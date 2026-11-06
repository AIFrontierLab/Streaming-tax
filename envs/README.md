# Environments

Every model family needs its own torch / transformers stack, so the evaluation runs in several Python
environments. Each folder here defines one:

- `requirements.lock`: the exact packages of the paper's runs. `streamreplace env build <name>` installs exactly this
  list with `uv pip sync` (nothing is resolved anew).
- `requirements.in`: the direct requirements the lock was resolved from (edit it and re-lock with
  `uv pip compile` only if you deliberately leave the paper's setup).
- `python-version`: the interpreter version of the paper's runs (uv downloads it if needed).

`patches/` holds the one patch of an installed package that an environment needs (AURA's transformers patch); `env
build` applies it and checks the file's sha256 before and after.

| Environment | Python | Key packages | Runs |
|---|---|---|---|
| `core` | 3.12.14 | numpy 2.5.3, PyYAML, Pillow, PyAV 13.1.0, decord 0.6.0, pyarrow, huggingface_hub 0.36.2, gdown 5.2.0 | the `streamreplace` command: prepare, run planning, scoring, statistics, tables, downloads |
| `modern` | 3.12.14 | torch 2.8.0, transformers 4.57.3, qwen-vl-utils 0.0.14, torchcodec 0.7.0, flash-attn 2.8.3, flashinfer 0.6.7 | Qwen2.5-VL-3B, ThinkStream-3B, MMDuet2-3B, Qwen3-VL-8B, AURA-8B offline; ThinkStream stream; MOSS-VL-Instruct and MOSS-VL-Realtime offline; MOSS-VL-Realtime stream |
| `llava` | 3.10.14 | torch 2.1.2, transformers 4.44.2, peft 0.12.0 (no FlashAttention: sdpa) | LLaVA-OneVision-7B and MMDuet-7B offline; MMDuet stream |
| `livestar` | 3.10.14 | torch 2.5.1, transformers 4.37.2, flash-attn 2.7.4.post1 | LiveStar-InternVideo-8B and LiveStar-8B offline; LiveStar stream |
| `mmduet2` | 3.10.14 | torch 2.4.0, transformers 4.49.0, qwen-vl-utils 0.0.8, flash-attn 2.6.3 | MMDuet2 stream |
| `aura` | 3.11.16 | vLLM 0.11.0, transformers 4.57.1 (patched) | AURA stream (vLLM server) |
| `judge` | 3.12.14 | vLLM 0.30.0+cu129, torch 2.13.0+cu129, transformers 5.17.0, spacy 3.8.16 + en_core_web_sm 3.8.0, datasketch 1.6.5, google-genai | extraction judge, HarmBench classifiers, RJScore judge (and the benchmarks' official evaluation code they import) |
| `lingo` | 3.10.14 | torch 2.1.2, transformers 4.36.2 (LingoQA's own pins) | the Lingo-Judge of the fine-tuning experiment |
| `ffmpeg` | - | FFmpeg 6.1.2 shared libraries (conda-forge) | torchcodec in `modern`, `mmduet2` and `aura` (models with `ffmpeg: true`) |

Every model environment pins decord 0.6.0 and PyAV 13.1.0: video fps comes from decord and decodable frame counts
from PyAV in every model's frame sampling, so these two must not change.

GPU support: every environment runs on NVIDIA GPUs of compute capability 8.0-9.0 (Ampere, Ada Lovelace, Hopper);
`modern` and `aura` also on Blackwell. The pinned flash-attn wheels are built for CUDA 12 and Linux x86_64; the
`judge` builds target CUDA 12.8-12.9 drivers.

See docs/environments.md for building, checking and the CPU-only use of the environments.
