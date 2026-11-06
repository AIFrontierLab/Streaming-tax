# Documentation

Start with the [README](../README.md) (what streamreplace is, installation, quickstart). The pages below go into
detail; every page names the code and configuration it describes.

## Getting set up

| Page | Contents |
|---|---|
| [data.md](data.md) | fetching third-party code, checkpoints and benchmark data at pinned versions (`fetch-upstream`, `download`), manual access steps, where files go |
| [environments.md](environments.md) | the per-model Python environments (`streamreplace env`), the AURA transformers patch, GPU support, CPU-only use |
| [licensing.md](licensing.md) | what this repository licenses, what it fetches, and the terms of every model and dataset (with [NOTICE](../NOTICE)) |

## The evaluation protocol

| Page | Contents |
|---|---|
| [protocol.md](protocol.md) | P, C and S precisely: matched frames, prompts, chat template and decoding offline; the native streams with the question at the end; silence and forced answers; how the gaps are reported |
| [frames.md](frames.md) | which video frames each model sees: the per-pair offline rules, benchmark overrides, frame digests, the `video_meta` manifests, decoder caveats |
| [models/offline-adapters.md](models/offline-adapters.md) | the four offline adapters (conditions P and C): inputs, templates, decoding, CPU input checks, LoRA loading |
| [models/stream-adapters.md](models/stream-adapters.md) | the six children's native streams (condition S): frames, memory, question injection, answer readout, silence, deviations from the official evaluations, plugin hook points |

## Benchmarks

One page per benchmark: data source and subset, prompt, parsing and scoring, judge use, metrics and statistics units,
deviations from the official protocol.

| Group | Benchmarks | Conditions |
|---|---|---|
| Image understanding | [DynaMath](benchmarks/dynamath.md), [MathVista](benchmarks/mathvista.md), [MMBench](benchmarks/mmbench.md), [MME](benchmarks/mme.md), [MMMU](benchmarks/mmmu.md), [MMStar](benchmarks/mmstar.md), [ScienceQA](benchmarks/scienceqa.md), [VisuLogic](benchmarks/visulogic.md) (supplement) | P, C |
| Video understanding | [Video-MME](benchmarks/videomme.md), [TOMATO](benchmarks/tomato.md), [TimeBlind](benchmarks/timeblind.md), [VideoReasonBench](benchmarks/videoreasonbench.md) (supplement) | P, C, S |
| Safety | [HarmBench](benchmarks/harmbench.md) | P, C |
| Safety | [Video-SafetyBench](benchmarks/video_safetybench.md), [MCV-SafetyBench](benchmarks/mcv_safetybench.md) | P, C, S |
| Fairness and privacy | [VMDT fairness](benchmarks/vmdt_fairness.md), [VMDT privacy](benchmarks/vmdt_privacy.md) | P, C, S |

The fine-tuning datasets (LingoQA, EyePCR, OpenUAV-QA) are described in
[generalization/finetune.md](generalization/finetune.md).

## Scoring and statistics

| Page | Contents |
|---|---|
| [judges.md](judges.md) | the scoring pipeline (`score`), the extraction judge, HarmBench classifiers, RJScore and Lingo-Judge, self-tests, judge caches |
| [statistics.md](statistics.md) | counting rules, per-job metrics (`report`), analysis units, the bootstrap engines, Holm families, margins and verdicts, flags, generation-cap reruns, tables (`stats`, `tables`) |

## The plugin and generalization

| Page | Contents |
|---|---|
| [plugin.md](plugin.md) | S_fix and S+: Recall, Respond, the hidden-state gate and parent-restoration steering; per-child arming points; thresholds; reproducing the plugin figure; calibrating a new child |
| [generalization/robustness.md](generalization/robustness.md) | prompt robustness: items, perturbations, frozen edits, analysis |
| [generalization/finetune.md](generalization/finetune.md) | LoRA fine-tuning: datasets, recipe, training units, evaluation, analysis, controls, seed replication |

## Reproducing and extending

| Page | Contents |
|---|---|
| [reproducing-the-paper.md](reproducing-the-paper.md) | which command produces which table and figure; the CPU path from scored outputs and the GPU path; nondeterminism; known differences from the paper |
| [paper/README.md](../paper/README.md) | the frozen item lists, expected numbers and audit labels shipped in `paper/`, and why no benchmark text or model output is |
| [adding-a-model.md](adding-a-model.md) | a new pair or streaming child: adapters, environment, configuration, CPU checks, analysis roster |
| [adding-a-benchmark.md](adding-a-benchmark.md) | a new benchmark: module, registration, configuration, data, fingerprints, statistics, tests |

## Reference

The configuration files document their own fields in comments: `configs/defaults.yaml`, `models.yaml`, `pairs.yaml`,
`benchmarks.yaml`, `judges.yaml`, `analysis.yaml`, `plugin.yaml` and `configs/generalization/*.yaml`;
`configs/examples/cpu-trial.yaml` is an extra `--config` file for the CPU trial in the README. The manifests
(`manifests/upstream.yaml`, `checkpoints.yaml`, `datasets.yaml`, `prepared.yaml`) start with a description of their
format, and [envs/README.md](../envs/README.md) lists the environments. `streamreplace <command> --help` describes
every command and option.
