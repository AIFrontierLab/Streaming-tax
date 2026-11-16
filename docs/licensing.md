# Licensing

## This repository

streamreplace's own code, configuration, manifests, documentation and shipped artefacts are licensed under the
Apache License 2.0 (LICENSE; "Copyright 2026 The streamreplace Authors"). The shipped artefacts are small numeric or
identifier files derived by the authors: the plugin's steering vectors and thresholds (`plugin_artefacts/`), the
frozen item and id lists, seeds and expected numbers under `paper/`, and the per-video frame metadata under
`manifests/video_meta/` (video paths relative to each dataset's folder, frame counts, frame rates).

The repository ships **no** third-party code, model weights, benchmark text, images or videos, and no model
outputs or judge caches. Frozen lists (for example the robustness items, the plugin's calibration items or the
fine-tuning subsets) hold ids and indices only; the texts are rebuilt from your own copies of the data.

## Third-party code: fetched, never vendored

The models' streaming runtimes and the benchmarks' official scorers are other people's code. `streamreplace
fetch-upstream` downloads them from the authors' own repositories at pinned commits (manifests/upstream.yaml; see
docs/data.md); apart from the short fragments that NOTICE section 1 lists, none of it is copied into this repository.
That is required for the seven repositories that have no licence file at the pinned commit, and done the same way for
all of them:

| Licence at the pinned commit | Repositories |
|---|---|
| MIT | HarmBench, MMDuet, ThinkStream, PromptBench, OVO-Bench, StreamingBench |
| Apache-2.0 | VMDT, LLaVA-NeXT, MMDuet2, Self-in-Space (OpenUAV-QA), VLMEvalKit |
| Wayve Dataset Licence Agreement for Non-Commercial Use | LingoQA (the licence covers its evaluation code as part of the dataset: personal, non-commercial, non-sublicensable) |
| no licence file | LiveStar, TOMATO, TimeBlind, Video-MME's evaluation script, VideoReasonBench, Video-SafetyBench, MCV_Jailbreak |

Code without a licence file is not open source: its authors keep all rights. streamreplace reads it from the
authors' published repository on your machine, the same way their own instructions do, and does not redistribute it.
The few words and lines of such code that this repository does contain are listed in NOTICE section 1, where the
paper's numbers depend on them byte for byte (TimeBlind's answer-format suffixes, the Video-SafetyBench prompt prefix,
the AURA system prompt, TOMATO's prompt as lmms-eval carries it) or a local edit restates them (the `patches`
fragments). If you redistribute a workspace (for example a container image with `upstream/` filled in), check each
project's terms first.

Three fetched files are edited locally after download (manifests/upstream.yaml, `patches`); the manifest stores only
the few lines involved, and NOTICE lists them.

## Text quoted in this repository

A few short texts are quoted verbatim because the paper's numbers depend on them byte for byte: the AURA live-stream
system prompt, the PromptBench StressTest sentence, prompt templates of lmms-eval (MIT; among them TOMATO's prompt and
system prompt), VLMEvalKit (Apache-2.0) and VisuLogic-Eval (Apache-2.0), qwen-vl-utils' frame-selection helpers
(Apache-2.0), MMDuet's frame geometry (MIT), the Video-SafetyBench prompt prefix, TimeBlind's answer-format suffixes,
VMDT's JSON-repair prompt and refusal phrases (Apache-2.0), StreamingBench's answer cue (MIT) and EyePCR's
conversation turns (CC BY 4.0, from the dataset). NOTICE gives each one's location, source, commit and licence. Everything else an official scorer contributes
(parsers, judge prompts, scenario policies, attack prompts, the RJScore computation) is read from the fetched upstream
file at run time, not copied; the tests check those texts through sha256 digests and a few identifying words.

## Model checkpoints

Downloaded from the Hugging Face Hub at pinned revisions (manifests/checkpoints.yaml), under the licence on each
model card: Apache-2.0 (LLaVA-OneVision-7B, SigLIP, Qwen3-VL-8B, AURA, both LiveStar models, both MOSS-VL models,
the extraction judge), MIT (MMDuet, the two HarmBench classifiers), the Qwen Research licence (Qwen2.5-VL-3B,
MMDuet2), the Qwen licence (Qwen2.5-72B-Instruct, the RJScore judge), "other" (Lingo-Judge: see its card), and none
declared (ThinkStream-3B). Several are research-only or carry use restrictions; read the card before use.

## Datasets

Downloaded from their publishers (manifests/datasets.yaml) and used under their own terms:

| Licence or terms (as declared by the publisher) | Datasets |
|---|---|
| Apache-2.0 | DynaMath, MMMU, VisuLogic, MCV-SafetyBench, OpenUAV-QA |
| CC BY-SA 4.0 | MathVista, TOMATO, OVO-Bench |
| CC BY 4.0 | EyePCR, XSTest (harm-score probes) |
| CC BY-NC-SA 4.0, gated | Video-SafetyBench |
| CC BY-NC 4.0 (the repository's current statement; its DATA_LICENSE file held the ODC-By text when the data file last changed) | Stanford Alpaca (plugin calibration, harm-score probes) |
| MIT | HarmBench data, AdvBench (plugin calibration, harm-score probes) |
| Wayve non-commercial licence | LingoQA |
| not declared on the dataset card (see the project) | MME, MMStar, ScienceQA (lmms-lab mirror; original ScienceQA terms), Video-MME, TimeBlind, VideoReasonBench, VMDT V2T, StreamingBench (its code repository is MIT) |
| not stated with the file | MMBench (OpenCompass mirror of the official TSV) |

VMDT's fairness videos come from third-party face-video datasets that permit evaluation use only. Video-SafetyBench,
MCV-SafetyBench, HarmBench, AdvBench and XSTest contain harmful requests; they are used to measure refusals, and model
replies to them are not released. LingoQA's licence forbids redistributing the data or anything derived from its
text.

`streamreplace download` prints each dataset's licence, refuses LingoQA until `--accept-terms lingoqa` is given, and
prints the access step for Video-SafetyBench (docs/data.md).

## What you may publish from your runs

Generations and judge outputs contain benchmark text and, for the safety benchmarks, harmful completions; whether
you may publish them depends on each dataset's terms. Scores, ids and numeric per-row records (correct / safe, parse
status, the plugin's projection) are what this project's own analysis needs, and avoid the problem in most cases.
