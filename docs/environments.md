# Environments

The models of the study need incompatible software stacks (transformers 4.37 for LiveStar, 4.44 for LLaVA-NeXT,
4.49 for MMDuet2's streaming client, 4.57 for the Qwen and MOSS models, vLLM for AURA's stream and for the judges),
so `streamreplace` runs each model in its own Python environment. The `streamreplace` command itself runs in a small
`core` environment and starts the model environments' interpreters as worker processes, with this repository's
`src/` on their `PYTHONPATH` (nothing from this repository is installed into a model environment).

| Environment | Python | For |
|---|---|---|
| `core` | 3.12.14 | the `streamreplace` command: prepare, run, score, statistics, tables, plugin post-processing, downloads |
| `modern` | 3.12.14 | Qwen2.5-VL-3B, ThinkStream-3B, MMDuet2-3B, Qwen3-VL-8B, AURA-8B offline; ThinkStream stream; MOSS-VL-Instruct and MOSS-VL-Realtime offline and stream |
| `llava` | 3.10.14 | LLaVA-OneVision-7B and MMDuet-7B offline; MMDuet stream |
| `livestar` | 3.10.14 | LiveStar-InternVideo-8B and LiveStar-8B offline; LiveStar stream |
| `mmduet2` | 3.10.14 | MMDuet2 stream |
| `aura` | 3.11.16 | AURA stream (a local vLLM server) |
| `judge` | 3.12.14 | the extraction judge, the HarmBench classifiers, the RJScore judge |
| `lingo` | 3.10.14 | the Lingo-Judge of the fine-tuning experiment (LingoQA's own pins: torch 2.1.2, transformers 4.36.2) |
| `ffmpeg` | - | FFmpeg 6 shared libraries for torchcodec (Qwen video sampling, ThinkStream, MOSS, MMDuet2, AURA) |

The fine-tuning experiment trains and evaluates in the model environments above (`modern`, `llava`, `livestar`, each
with PEFT 0.12.0); only its judge has an environment of its own.

`configs/models.yaml` names each model's environment per mode (`offline.env`, `stream.env`) and whether it needs the
FFmpeg libraries (`ffmpeg: true`); `configs/judges.yaml` names the judges' environment.

## Where environments live

Environments are built under the envs root: `$STREAMREPLACE_ENVS`, by default `$STREAMREPLACE_HOME/envs`
(`workspace/envs` in the checkout). A model environment's interpreter is `<envs root>/<name>/bin/python`; the FFmpeg
libraries are `<envs root>/ffmpeg/lib`, which the run command puts on `LD_LIBRARY_PATH` for `ffmpeg: true` models.

## Building

Requirements: Linux x86_64, [uv](https://docs.astral.sh/uv/), and for the FFmpeg libraries one of micromamba, mamba or
conda. The pinned flash-attn and vLLM wheels target CUDA 12.

```bash
streamreplace env list                       # what is defined and built, and which models use each environment
streamreplace env build all --dry-run        # print the commands
streamreplace env build ffmpeg core modern   # or: streamreplace env build all
streamreplace env check all                  # interpreter, packages vs the lock, patches, imports
```

For each environment, `env build` runs

1. `uv venv --python <envs/<name>/python-version> --seed <envs root>/<name>` (only when it does not exist yet, or
   with `--recreate`);
2. `uv pip sync --python <envs root>/<name>/bin/python envs/<name>/requirements.lock`: exactly the packages of the
   paper's runs, including the prebuilt flash-attn wheels (2.8.3 in `modern`, 2.7.4.post1 in `livestar`, 2.6.3 in
   `mmduet2`) and, for `judge`, the CUDA 12.9 builds of vLLM and torch from the PyTorch index;
3. the environment's patches from `envs/patches/manifest.yaml` (see below);
4. for `core` only, this repository in editable mode (`uv pip install --no-deps -e .`), which provides the
   `streamreplace` command;
5. a record `<envs root>/<name>/streamreplace-env.json` with the lock's sha256 and the applied patches.

`env build ffmpeg` creates a conda-forge prefix with FFmpeg 6.1.2 (`ffmpeg=6.1.2=gpl_hdfc89ed_706`, the build of the
paper's runs). Rebuilding an existing environment re-syncs it to the lock.

FlashAttention-2 is part of the paper's numerics where it was used (`modern`, `livestar`, `mmduet2`): an adapter falls
back to PyTorch's sdpa attention when flash-attn is missing and records `info.attention`, but such rows are not
comparable bit for bit. The `llava` environment has no flash-attn: the paper's LLaVA-OneVision and MMDuet runs used
sdpa.

The `judge` environment holds more than vLLM because the judges run the benchmarks' official code: HarmBench's
evaluation utilities import spaCy (3.8.16, with the `en_core_web_sm` 3.8.0 model) and datasketch (1.6.5) when they are
imported, VideoReasonBench's evaluation utilities import `google-genai`, and the RJScore judge loads its checkpoint
with transformers and accelerate (`device_map="auto"`). `env check judge` probes all of these imports.

## The AURA patch

AURA's official evaluation requires a two-line change to the installed transformers 4.57.1 (its README, section
"Required Transformers Patch"): its streaming client sends 1-second video chunks of 2 frames, which the stock Qwen3-VL
video processor rejects (fewer frames than the temporal patch size) and would sample to at least 4 frames.
`envs/patches/aura-transformers-4.57.1-qwen3vl-video.patch` is that change as a unified diff of
`transformers/models/qwen3_vl/video_processing_qwen3_vl.py`. `env build aura` (or `env patch aura` for an environment
installed by hand) applies it only if the file has the expected sha256 of the unmodified 4.57.1 file, and checks the
sha256 of the result; `env check aura` reports whether it is in place. Patches are applied without fuzz: a different
transformers version is refused rather than patched approximately.

The other runtime patch of the paper's runs does not touch installed files: the Qwen2.5-VL temporal-position fix is
applied to the loaded model by `src/streamreplace/models/patches/qwen25_rope.py` and is version-checked there.

## Checking

`streamreplace env check NAME|all` reports, per environment:

- whether the interpreter runs and its Python version (a different minor version is a problem, a different patch
  version a warning);
- packages that are missing, have a different version, or are not in the lock (the comparison reads the installed
  distributions' metadata, so it needs no pip);
- the status of each patch (`applied`, `unpatched`, `unexpected`, `missing`);
- whether the lock changed since the environment was built;
- import probes of the modules the environment's code needs (skip them with `--no-probe`, for example on a machine
  without the FFmpeg libraries).

The exit code is 1 when any environment has a problem.

## GPU support

Every environment supports NVIDIA GPUs of compute capability 8.0-9.0 (Ampere, Ada Lovelace, Hopper). `modern` and
`aura` also support Blackwell GPUs. The pinned builds of the `llava`, `livestar` and `mmduet2` stacks and
ThinkStream's FlashInfer kernels do not support Blackwell. The RJScore judge
(Qwen2.5-72B-Instruct in bf16) needs at least two large GPUs with working peer-to-peer access (about 145 GB; `score`
uses GPUs 0 and 1 unless `--judge-gpus` says otherwise); the judge runs a self-test before scoring.

## CPU-only use

Nothing in the offline adapters' input construction needs a GPU. With an environment built and the processor and
tokenizer files of a checkpoint downloaded, an adapter constructed with `load=False` builds the exact model inputs on a
CPU (`build_inputs`), which is how the inputs of the paper's runs were checked (docs/models/offline-adapters.md).
Set `CUDA_VISIBLE_DEVICES=` and `HF_HUB_OFFLINE=1` for such checks.

## Re-locking

`requirements.in` lists each environment's direct requirements. To move an environment to other versions (which can
change results), edit it and re-lock, for example `uv pip compile --python-version 3.12 envs/modern/requirements.in
-o envs/modern/requirements.lock`, then rebuild. The patch manifest pins the exact files it patches, so a different
transformers in `aura` also needs a re-made patch.
