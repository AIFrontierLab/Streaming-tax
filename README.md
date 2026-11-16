# streamreplace

streamreplace is the code of the paper *Streaming Tax: Can Streaming VLMs Replace Their Offline Parents?* (under
anonymous review). It compares six streaming video-language models (the *children*) with the offline models they
were post-trained from (their *parents*) on understanding, trustworthiness and generalization benchmarks, under a
protocol that separates what streaming adaptation changes in the weights from what the streaming interface changes.
It contains everything the paper reports:

- **Streaming Tax** (Sec. 3-4): the P / C / S evaluation on 15 benchmarks, its statistics and tables;
- **the diagnosis** (Sec. 5): controls for the visual evidence a stream keeps, for its response policy, and for the
  gap between recognizing a harmful request and refusing it;
- **Stream++** (Sec. 6): the training-free plugin (Recall, Respond, Refuse), its calibration and its evaluation,
  including native streaming benchmarks.

[docs/reproducing-the-paper.md](docs/reproducing-the-paper.md) maps every table and figure of the paper to a command.

## What it measures

For each parent-child pair and benchmark, three conditions are scored by the same scorer on the same items:

| Condition | Model | How it is run |
|---|---|---|
| **P** | parent | offline: the whole image set or sampled video at once, then the question |
| **C** | child | the same offline pipeline, with matched frames, prompt, chat template and decoding |
| **S** | child | its own native streaming interface: the video is replayed in order, the child keeps its own memory and response policy, and the question arrives at the end of the video |

```
S - P           =  (C - P)         +  (S - C)
streaming tax      adaptation tax     execution tax
```

C - P is the difference that appears when the weights change under a fixed protocol; S - C the difference that
appears when the same checkpoint runs as a stream. Every difference is reported with paired cluster-bootstrap
intervals, Holm-corrected tests, equivalence and non-inferiority marks and the response coverage of both arms. A
stream that answers none of a set's requests gets no attack-success or privacy score there (silence alone would give
the best value). Image benchmarks and HarmBench have no streaming input and get P and C only. Details:
[docs/protocol.md](docs/protocol.md), [docs/statistics.md](docs/statistics.md).

**Stream++** is training-free and acts on the stream's memory and on its answer turn
([docs/plugin.md](docs/plugin.md)):

- **Recall** keeps earlier frames without enlarging the memory: once the child's native visual budget is full, the
  newest 10 chunks stay and the other slots hold a uniform sample of everything older, updated as each chunk arrives;
- **Respond** makes sure an answer is delivered: the answer turn is requested, it may not open with the child's
  silence marker, and a multiple-choice or yes/no reply without a valid option is read from the option logits at
  the answer start;
- **Refuse** restores the parent's refusal of harmful requests: *parent-restoration steering* (the mean
  parent-minus-child hidden state, scaled, added at one decoder layer while the child answers) and a *refusal gate*
  (a fixed refusal when the answer-position hidden state projects past a calibrated threshold on the child's harm
  direction).

S_fix = Recall + Respond; S_plug = S_fix + Refuse. The per-child vectors and thresholds ship in `plugin_artefacts/`;
calibrating the plugin for a new child is included.

**The diagnostic and supplementary experiments** are declared in one file, `configs/experiments.yaml`: each is a set
of arms (ordinary generation jobs with one protocol component replaced) on a frozen item list, run and read with
`streamreplace experiment` ([docs/experiments.md](docs/experiments.md)): the offline input restricted to the
streaming window and the stream with nothing evicted; memory policies at a fixed budget; MMDuet's forced answer;
question timing; AURA without its streaming system prompt and with its parent's weights; the response-form audit
and harmless how-to requests about the same videos; refusal under the bare request, the video-framing sentence,
black frames and the real video; the models' own risk judgments and their hidden-state harm scores; steering along
the harm direction; OVO-Bench and StreamingBench.

**Generalization**, offline, parent against child:

- *prompt robustness*: typos, paraphrases and distractor sentences in Video-MME and MMBench questions; the child's
  accuracy drop is compared with its parent's ([docs/generalization/robustness.md](docs/generalization/robustness.md));
- *LoRA fine-tuning*: one recipe for every model on LingoQA and EyePCR (OpenUAV-QA optional) at 1K, 2K and 5K
  training examples; the child's gain over its own zero-shot accuracy is compared with its parent's
  ([docs/generalization/finetune.md](docs/generalization/finetune.md)).

## Models

| Pair | Parent (P) | Child (C and S) | The child's native streaming interface (S) |
|---|---|---|---|
| p1 | LLaVA-OneVision-7B | MMDuet-7B | two score heads decide when to reply; they rarely fire at the question, so most questions go unanswered (the control S† forces a reply at the final frame) |
| p2 | Qwen2.5-VL-3B-Instruct | ThinkStream-3B | a short thought on every 2-frame chunk, a reply at the question's chunk; a window of recent chunks |
| p2 | | MMDuet2-3B | after every 2-frame turn, a reply or "NO REPLY"; the full history |
| p3 | Qwen3-VL-8B-Instruct | AURA-8B | the video as one-second segments, with a silence token for "no reply"; the last 30 segments; served by vLLM |
| p4 | LiveStar-InternVideo-8B | LiveStar-8B | narrates when its narration gate fires, then answers from its narration history |
| p5 | MOSS-VL-Instruct-0708 | MOSS-VL-Realtime | a realtime session fed on the wall clock, with a silence token |

Checkpoints are pinned to Hugging Face revisions in `configs/models.yaml`; each pair's single offline video protocol
is in `configs/pairs.yaml`. The adapters are documented in
[docs/models/offline-adapters.md](docs/models/offline-adapters.md) and
[docs/models/stream-adapters.md](docs/models/stream-adapters.md).

## Benchmarks

| Benchmark | Items | Reported metric | Conditions | Scored by |
|---|---|---|---|---|
| DynaMath | 5,010 (501 questions x 10 variants) | average accuracy; worst-case accuracy | P, C | rules; an LLM extraction judge reads replies the rules cannot parse |
| MathVista (testmini) | 1,000 | accuracy | P, C | rules + extraction judge |
| MMBench (English dev) | 4,329 rows (1,164 questions, circular) | circular accuracy | P, C | rules + extraction judge |
| MME | 2,374 | perception; cognition | P, C | rules + extraction judge |
| MMMU (validation) | 900 | accuracy | P, C | rules + extraction judge |
| MMStar | 1,500 | accuracy | P, C | rules + extraction judge |
| ScienceQA (test questions with an image) | 2,017 | accuracy | P, C | rules + extraction judge |
| VisuLogic (supplement) | 1,000 | accuracy | P, C | rules + extraction judge |
| Video-MME (short videos, no subtitles) | 900 (300 videos) | accuracy | P, C, S | the official answer regex |
| TOMATO | 1,484 | accuracy | P, C, S | the official parser and extraction prompt |
| TimeBlind | 2,400 (600 instances x 4) | instance accuracy | P, C, S | the official extractor and prompt |
| VideoReasonBench (supplement) | 1,440 | accuracy | P, C, S | the official LLM judge and state checkers |
| HarmBench (text and image behaviours) | 510 | attack success rate | P, C | the official HarmBench classifiers |
| Video-SafetyBench | 2,264 (harmful and benign queries) | attack success rate per query type | P, C, S | RJScore (Qwen2.5-72B-Instruct) |
| MCV-SafetyBench | 2,920 (explicit and implicit attacks) | attack success rate per attack | P, C, S | RJScore |
| VMDT fairness | 2,000 (a fixed subset) | fairness score (stereotype, decision, factual) | P, C, S | the official rules and JSON-repair prompt |
| VMDT privacy | 600 (200 videos) | privacy score (location leakage) | P, C, S | the official rules |
| OVO-Bench (supplement) | 1,576 rows (clips of at most 120 s) | question accuracy; official total average | S, Stream++ arms | the official scorer |
| StreamingBench, real-time (supplement) | 700 questions (asked within 120 s) | accuracy | S, Stream++ arms | the official counting rule |

One page per benchmark under [docs/benchmarks/](docs/index.md#benchmarks) gives the data source, prompt, parsing,
judge use, metric and every deviation from the official protocol; the judges are described in
[docs/judges.md](docs/judges.md). The five pairs on the first 17 benchmarks make 235 generation jobs.

## Installation

### Requirements

- The `streamreplace` command: Python 3.10 or newer with numpy, PyYAML, Pillow and PyAV (installed with it);
  preparing the benchmarks that ship as parquet files also needs pyarrow (extra `prepare`), and downloading needs
  huggingface_hub and gdown (extra `download`). Scoring from cached judge answers, the statistics and the tables need
  no GPU and no model environment.
- Generating outputs: Linux x86_64 with NVIDIA GPUs of compute capability 8.0 to 9.0 (the `modern` and `aura`
  environments also support Blackwell), CUDA 12 drivers, [uv](https://docs.astral.sh/uv/), and micromamba, mamba or
  conda for the FFmpeg 6 libraries. The RJScore safety judge (Qwen2.5-72B-Instruct, about 145 GB in bf16) needs at
  least two large GPUs with working peer-to-peer copies.
- Disk for the data: the largest benchmarks are Video-MME (101 GB) and VMDT fairness (84 GB); the two optional
  streaming benchmarks of the supplement are larger (OVO-Bench 153 GB, StreamingBench 132 GB).
  [docs/data.md](docs/data.md) lists every size.

### The command

```bash
git clone <this repository> streamreplace && cd streamreplace
python -m pip install -e ".[test,download,prepare]"   # the command, pytest, huggingface_hub + gdown, pyarrow
streamreplace --help
streamreplace doctor                       # resolved paths, configuration, and what is present
```

streamreplace runs **from a source checkout**: `configs/`, `manifests/`, `plugin_artefacts/` and `paper/` live at
the repository root and are not part of an installed package. Install in editable mode (as above) or run
`python -m streamreplace` with `src/` on `PYTHONPATH`; otherwise the command looks for `configs/defaults.yaml` in the
current directory and its parents.

For the exact orchestrator versions of the paper's analysis (Python 3.12.14, numpy 2.5.3, the pinned PyAV and
decord), build the `core` environment below and use its `streamreplace`: it has this checkout installed in editable
mode.

### Workspace

Everything downloaded or generated goes under one workspace, `STREAMREPLACE_HOME` (default: `workspace/` in the
checkout), with the roots `data/`, `checkpoints/`, `envs/`, `outputs/`, `cache/` and `upstream/`. Any root can be
moved with `STREAMREPLACE_DATA`, `STREAMREPLACE_CHECKPOINTS`, `STREAMREPLACE_ENVS`, `STREAMREPLACE_OUTPUTS`,
`STREAMREPLACE_CACHE` or `STREAMREPLACE_UPSTREAM`, with a paths file (`--paths FILE` or `STREAMREPLACE_PATHS`), or with
a `paths:` block in the git-ignored `configs/local.yaml`; a paths file can also point single checkpoints or datasets
at existing copies (`checkpoint_dirs`, `dataset_dirs`). `streamreplace doctor` shows where each root resolves.

### Model environments

The models need incompatible software stacks, so each model family runs in its own Python environment, built from the
lock files under `envs/` (exactly the packages of the paper's runs); the command starts each worker with that
environment's interpreter and this checkout's `src/` on its path.

```bash
streamreplace env list                     # environments, whether built, which models use them
streamreplace env build all --dry-run      # print the uv / conda commands
streamreplace env build all                # or some of them: env build ffmpeg core modern
streamreplace env check all                # packages against the locks, patches, import probes
```

See [docs/environments.md](docs/environments.md) and [envs/README.md](envs/README.md).

### Third-party code

Model runtimes and official scorers are other people's code. They are not copied into this repository (apart from
the short fragments NOTICE lists): `fetch-upstream` downloads the exact files the paper used from 19 repositories at
pinned commits and checks each one against its git blob SHA-1.

```bash
streamreplace fetch-upstream --list        # what is pinned, with licences
streamreplace fetch-upstream all           # into <upstream root>/<name>/
```

### Checkpoints and data

Resolve every pin first (`--dry-run` downloads nothing), then download:

```bash
streamreplace download models all --dry-run
streamreplace download models all          # the 11 models and the SigLIP vision tower, at pinned revisions
streamreplace download judges all --dry-run
streamreplace download judges all          # extraction judge, HarmBench classifiers, RJScore, Lingo-Judge
streamreplace download datasets paper-core paper-supp --dry-run
streamreplace download datasets paper-core paper-supp
streamreplace prepare paper-core paper-supp   # raw -> prepared samples, checked against manifests/
```

A few sources need a manual step: Video-SafetyBench is gated on Hugging Face (accept its conditions and provide a
token), LingoQA requires `--accept-terms lingoqa`, TOMATO's videos come from Google Drive (`gdown`), and OpenUAV-QA's
frames have no immutable revision (`--allow-unpinned`). Details for every dataset and checkpoint:
[docs/data.md](docs/data.md).

### Licences

This repository's code is Apache-2.0. Third-party code, model weights and benchmark data are not redistributed;
each keeps its own licence and terms, several of them non-commercial or evaluation-only, and the safety benchmarks
contain harmful requests. [NOTICE](NOTICE) lists the short texts quoted verbatim, the fetched code, and every model
and dataset with its licence; [docs/licensing.md](docs/licensing.md) explains what that means for you.

## Quickstart

Global options (`--outputs DIR`, `--config FILE`, `--paths FILE`, `--home DIR`) go **before** the command name.

```bash
# 1. Data: resolve the pins, download, prepare and check one benchmark
streamreplace download datasets videomme --dry-run
streamreplace download datasets videomme
streamreplace prepare videomme

# 2. Generate P, C and S for one pair: plan first, then run (resumable; one worker per GPU slot,
#    repeat an id to pack two workers on one card)
streamreplace run --pairs p2 --protocols P,C,S --benchmarks videomme --dry-run
streamreplace run --pairs p2 --protocols P,C,S --benchmarks videomme --gpus 0,0,1,1
streamreplace status --pairs p2 --benchmarks videomme

# 3. Score: benchmark rules, then the judges (requests without a cached answer are sent only with --judge)
streamreplace score --judge --judge-gpus 0,1

# 4. Metrics and the paired C-P, S-C and S-P contrasts
streamreplace report                       # <outputs>/metrics.json, RESULTS.md, pairs.csv
streamreplace stats --no-overlay           # <outputs>/analysis/no-overlay/contrasts.json, contrasts.csv, GAPS.md

# 5. Stream++ on the paper's evaluation items (its own outputs folder), then the recovery tables, which compare
#    S_fix and S_plug with P, C and S of the same items from the outputs folder of the P/C/S runs
streamreplace --outputs plugin_runs plugin run --arms sfix,sfix_prs --items full --gpus 0,1
streamreplace --outputs plugin_runs plugin assemble --items full
streamreplace --outputs plugin_runs score --judge --judge-gpus 0,1
streamreplace --outputs plugin_runs plugin report --items full --formal <outputs>   # <outputs>: the P/C/S runs

# 6. A diagnostic experiment: the offline input restricted to the streaming window, the stream with nothing evicted
streamreplace experiment list
streamreplace experiment run window --gpus 0,1
streamreplace --outputs <outputs>/experiments/window score
streamreplace experiment report window
```

`<outputs>` is `<STREAMREPLACE_HOME>/outputs` unless `--outputs` names another folder. `plugin run` and `plugin
report` take `--children` to restrict the plugin to some children (default: all six).

A quick end-to-end trial on a few items: `streamreplace --outputs runs/trial run --pairs p1 --benchmarks videomme
--limit 3 --gpus 0` (`--limit` is part of the run identity, so it gets its own outputs folder). Without a GPU you can
still run the test suite, every `--dry-run`, `streamreplace plugin verify`, the statistics on scored outputs, the
CPU checks of model inputs described in [docs/adding-a-model.md](docs/adding-a-model.md), and the trial below.

### Try it on a CPU

[configs/examples/cpu-trial.yaml](configs/examples/cpu-trial.yaml) pairs a mock parent and child: deterministic
stand-ins without weights that run in the command's own interpreter. They take a benchmark through planning, the
worker processes, scoring, metrics and the paired statistics in seconds, without a GPU or a checkpoint. TimeBlind
(4 GB of data) suits it: a video benchmark, so P, C and S all run, whose rule-based scorer reads every mock reply.

```bash
streamreplace fetch-upstream timeblind && streamreplace download datasets timeblind && streamreplace prepare timeblind
trial() { streamreplace --config configs/examples/cpu-trial.yaml --outputs workspace/cpu-trial "$@"; }
trial run --pairs pm --benchmarks timeblind --limit 40      # P, C and S on 40 queries (10 instances)
trial status --pairs pm --benchmarks timeblind --limit 40
trial score && trial report                                 # workspace/cpu-trial/RESULTS.md
trial stats --no-overlay                                    # analysis/no-overlay/contrasts.json, GAPS.md
trial tables --contrasts workspace/cpu-trial/analysis/no-overlay/contrasts.json   # tables/
```

The numbers mean nothing (the mock answers right with a fixed probability); the point is that every step runs. A new
model follows the same path with its own entry in a `--config` file ([docs/adding-a-model.md](docs/adding-a-model.md)).

## Reproducing the paper

[docs/reproducing-the-paper.md](docs/reproducing-the-paper.md) maps every table and figure to the commands that
produce it. In short:

- **From scored outputs, on a CPU.** `report`, `stats`, `tables`, `robustness report`, `finetune report`, `plugin
  report` and `experiment report` rebuild the paper's statistics, tables and figure data from generation and score
  rows. The paper's outputs are not distributed with this repository (they contain benchmark text and harmful
  completions); `paper/expected/` holds the numbers they produce, and `tables --check` compares every printed cell
  of the main tables with the paper's ([paper/README.md](paper/README.md)).
- **Regenerating on GPUs, approximately.** The items, prompts, frames, stream schedules, decoding settings, scoring
  rules and statistics are those of the paper. New generations are not bit-identical, though: GPU kernels differ
  across hardware, ThinkStream samples its thoughts, AURA's stream is batched by vLLM, MOSS-VL-Realtime runs on the
  wall clock, and LiveStar's narration varies between runs.

What was checked while preparing this release, and the few analyses of the supplement that are not part of it, are
listed in the same document.

## Extending

- A new pair or streaming child: [docs/adding-a-model.md](docs/adding-a-model.md).
- A new benchmark: [docs/adding-a-benchmark.md](docs/adding-a-benchmark.md).
- Stream++ for a new child: [docs/plugin.md](docs/plugin.md#calibrating-a-new-child).
- A new control experiment: an entry in `configs/experiments.yaml` ([docs/experiments.md](docs/experiments.md)).

New models, pairs, benchmarks and suites can live in an extra config file passed with `--config FILE`, merged over
`configs/`.

## Repository layout

```
configs/              what is evaluated and how: models, pairs, benchmarks, judges, the analysis profile, Stream++
                      (plugin.yaml), the diagnostic and supplementary experiments (experiments.yaml), the
                      generalization experiments (configs/local.yaml: your paths and overrides, git-ignored)
manifests/            pinned third-party code, checkpoints and datasets; fingerprints of the prepared samples and
                      prompts; per-video frame counts and rates (video_meta/)
envs/                 one Python environment per software stack: python-version, requirements.in, requirements.lock;
                      patches/ (the one checked patch of an installed package)
plugin_artefacts/     Stream++'s per-child steering vectors and thresholds (sha256-checked)
paper/                frozen item lists, expected numbers and audit labels of the paper (no benchmark text)
docs/                 documentation; start at docs/index.md
src/streamreplace/
  cli/                the streamreplace command, one module per subcommand
  config/  core/      configuration and settings resolution; data contracts, run identity, file I/O
  data/  media/       raw -> prepared samples and prompts; frame sampling, video metadata, frame digests
  models/             offline adapters (P, C) and native stream adapters (S)
  benchmarks/         one module per benchmark: preparation, prompt, parsing, metrics
  judges/  scoring/   extraction judge, HarmBench classifiers, RJScore, Lingo-Judge, caches; rules -> score rows
  run/                job planning, dispatch to the model environments, the worker, generation-cap reruns
  analysis/           metrics, paired statistics, flags, the paper's tables
  plugin/             Stream++: Recall, Respond, Refuse; runtime hooks, post-hoc assembly, recovery report,
                      calibration
  experiments/        the diagnostic and supplementary experiments: arms and contrasts, response forms, harm-score
                      probes and steering
  generalization/     prompt robustness and LoRA fine-tuning
  provisioning/       environments, third-party code, downloads
tests/                CPU test suite (pytest)
```

An outputs folder (default `<STREAMREPLACE_HOME>/outputs`) holds `<benchmark>/<job>.jsonl` (generation rows, append
only), `<job>.meta.json` (run identity and run segments) and `<job>.scores.jsonl` (score rows) for every job, where a
job is `<model>.offline` or `<model>.stream` (a plugin arm: `<child>@<arm>.stream`); judge caches under `judges/`;
`metrics.json`, `RESULTS.md`, `analysis/` and `tables/` from the analysis commands; `experiments/<name>/` (each an
outputs folder of its own, with its item list and report); and logs under `logs/`.

## Tests

```bash
python -m pytest -q                        # the whole suite, on a CPU
python -m pytest -q tests/core             # or one directory at a time
```

The default suite needs no data, checkpoints, third-party code or GPU. Tests marked `requires_data` or
`requires_upstream` run when the prepared benchmarks or the fetched third-party code are present and are skipped
otherwise.

## Citation

```bibtex
@misc{anonymous2026streamreplace,
  title  = {Streaming Tax: Can Streaming VLMs Replace Their Offline Parents?},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```

## License

Apache License 2.0 ([LICENSE](LICENSE)), Copyright 2026 The streamreplace Authors. The licence covers the code and
files of this repository only; third-party code, model weights and datasets remain under their own terms
([NOTICE](NOTICE), [docs/licensing.md](docs/licensing.md)).
