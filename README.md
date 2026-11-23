# Streaming Tax: Can Streaming VLMs Replace Their Offline Parents?

We evaluates six streaming video-language models against the
offline models they were adapted from, diagnoses where they fall short, and provides **Stream++**, a training-free
plugin that narrows the gap.

## Overview

Each streaming *child* is compared with its offline *parent* under three conditions, on the same items with the same
scorer:

| | Model | Inference |
|---|---|---|
| **P** | parent | offline: the whole video, then the question |
| **C** | child | the same offline pipeline (matched frames, prompt and decoding) |
| **S** | child | its native streaming interface; the question arrives at the end of the video |

```
S - P           =   (C - P)          +   (S - C)
streaming tax       adaptation tax       execution tax
```

**Stream++** acts on a child's memory and answer turn, without training:

- **Recall** keeps earlier frames within the native memory budget (the newest chunks plus a uniform sample of older ones);
- **Respond** makes sure an answer is delivered (no silence at the answer start; option readout for closed questions);
- **Refuse** restores the parent's refusals (parent-restoration steering and a calibrated refusal gate).

| Pair | Offline parent | Streaming child |
|---|---|---|
| p1 | LLaVA-OneVision-7B | MMDuet |
| p2 | Qwen2.5-VL-3B | ThinkStream, MMDuet2 |
| p3 | Qwen3-VL-8B | AURA |
| p4 | LiveStar-InternVideo-8B | LiveStar |
| p5 | MOSS-VL-Instruct | MOSS-VL-Realtime |

| Dimension | Benchmarks | Conditions |
|---|---|---|
| Image understanding | DynaMath, MathVista, MMBench, MME, MMMU, MMStar, ScienceQA, VisuLogic | P, C |
| Video understanding | Video-MME, TOMATO, TimeBlind, VideoReasonBench | P, C, S |
| Safety | HarmBench (P, C), Video-SafetyBench, MCV-SafetyBench | P, C, S |
| Fairness and privacy | VMDT fairness, VMDT privacy | P, C, S |
| Generalization | prompt robustness (Video-MME, MMBench), LoRA fine-tuning (LingoQA, EyePCR) | P, C |
| Native streaming | OVO-Bench, StreamingBench | S, Stream++ |

Every benchmark is scored with its official prompt and scorer; differences come with paired bootstrap intervals and
Holm-corrected tests ([docs/protocol.md](docs/protocol.md), [docs/statistics.md](docs/statistics.md)).

## Installation

```bash
git clone <this repository> streamreplace && cd streamreplace
pip install -e ".[test,download,prepare]"
streamreplace doctor                                  # resolved paths and what is present
```

The command needs Python 3.10+ and no GPU: planning, scoring from cached judge answers, statistics and tables all run
on a CPU. Generating outputs needs Linux with NVIDIA GPUs, [uv](https://docs.astral.sh/uv/) and conda (for FFmpeg);
each model family runs in its own pinned environment.

```bash
streamreplace env build all                           # model environments from the lock files in envs/
streamreplace fetch-upstream all                      # official model and benchmark code at pinned commits
streamreplace download models all                     # the 11 models at pinned revisions
streamreplace download judges all
streamreplace download datasets paper-core paper-supp
streamreplace prepare paper-core paper-supp           # checked against the fingerprints in manifests/
```

Every download accepts `--dry-run`. Everything lands in `workspace/` (or `$STREAMREPLACE_HOME`). Sizes, gated
datasets and manual steps: [docs/data.md](docs/data.md), [docs/environments.md](docs/environments.md).

## Quick start

```bash
# P, C and S for one pair on one benchmark (resumable; add --dry-run to see the plan)
streamreplace run --pairs p2 --protocols P,C,S --benchmarks videomme --gpus 0,1
streamreplace score --judge --judge-gpus 0,1
streamreplace report                                  # metrics.json, RESULTS.md
streamreplace stats --no-overlay                      # paired C-P, S-C and S-P contrasts (GAPS.md)

# Stream++ on the paper's evaluation items
streamreplace --outputs plugin_runs plugin run --items full --gpus 0,1
streamreplace --outputs plugin_runs plugin assemble --items full
streamreplace --outputs plugin_runs score --judge --judge-gpus 0,1
streamreplace --outputs plugin_runs plugin report --items full --formal <outputs>

# a diagnostic experiment (streamreplace experiment list shows all of them)
streamreplace experiment run window --gpus 0,1
streamreplace --outputs <outputs>/experiments/window score
streamreplace experiment report window
```

Global options (`--outputs`, `--config`, `--paths`, `--home`) go before the command; `<outputs>` defaults to
`workspace/outputs`.

**No GPU?** A mock parent-child pair takes one benchmark through the whole pipeline in seconds:

```bash
streamreplace fetch-upstream timeblind && streamreplace download datasets timeblind && streamreplace prepare timeblind
trial() { streamreplace --config configs/examples/cpu-trial.yaml --outputs workspace/cpu-trial "$@"; }
trial run --pairs pm --benchmarks timeblind --limit 40 && trial score && trial report && trial stats --no-overlay
```

## Reproducing the paper

| Paper | Commands |
|---|---|
| Sec. 4: Fig. 3, Table 2, absolute scores | `run`, `score`, `report`, `stats`, `tables` |
| Sec. 4.3: prompt robustness, fine-tuning | `robustness`, `finetune` |
| Sec. 4.2, 5: response forms, harmless requests, history, response and refusal controls | `experiment run / label / report <name>` |
| Sec. 6: Stream++ | `plugin run / assemble / report` |
| Supplement: native streaming benchmarks | `experiment run streaming_benchmarks` |

[docs/reproducing-the-paper.md](docs/reproducing-the-paper.md) maps every table and figure to its command, and lists
what was checked against the paper, the known differences and what is not included.

- **From scored outputs (CPU, exact).** The analysis commands rebuild the paper's statistics and tables;
  `streamreplace tables --check` compares every printed cell with `paper/expected/`.
- **From scratch (GPU, approximate).** Items, prompts, frames, decoding and scoring are the paper's; generations are
  not bit-identical across hardware.

Model outputs are not distributed: they contain benchmark text and harmful completions.

## Documentation

| | |
|---|---|
| [protocol](docs/protocol.md), [frames](docs/frames.md) | P, C and S precisely; which frames each model sees |
| [offline adapters](docs/models/offline-adapters.md), [stream adapters](docs/models/stream-adapters.md) | how each model is run |
| [benchmarks](docs/index.md#benchmarks), [judges](docs/judges.md) | data, prompts, parsing and scoring per benchmark |
| [statistics](docs/statistics.md) | intervals, tests, Holm families, tables |
| [experiments](docs/experiments.md) | the diagnostic and supplementary experiments |
| [plugin](docs/plugin.md) | Stream++: components, thresholds, calibrating a new child |
| [robustness](docs/generalization/robustness.md), [fine-tuning](docs/generalization/finetune.md) | the generalization experiments |
| [adding a model](docs/adding-a-model.md), [adding a benchmark](docs/adding-a-benchmark.md) | extending the code |

## Repository layout

```
configs/            models, pairs, benchmarks, judges, analysis profile, experiments, Stream++
manifests/          pinned third-party code, checkpoints and datasets; fingerprints of prepared data
envs/               one locked Python environment per model family
plugin_artefacts/   Stream++ vectors and thresholds per child
paper/              frozen item lists and expected numbers (no benchmark text)
src/streamreplace/  cli, models (offline and stream adapters), benchmarks, judges, scoring, run, analysis,
                    plugin, experiments, generalization, provisioning
tests/              python -m pytest -q   (CPU; no data, checkpoints or GPU needed)
docs/               start at docs/index.md
```

## License and citation

Apache-2.0 ([LICENSE](LICENSE)). Third-party code, model weights and datasets are fetched from their publishers and
keep their own terms ([NOTICE](NOTICE), [docs/licensing.md](docs/licensing.md)).

```bibtex
@misc{anonymous2026streamingtax,
  title  = {Streaming Tax: Can Streaming VLMs Replace Their Offline Parents?},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```
