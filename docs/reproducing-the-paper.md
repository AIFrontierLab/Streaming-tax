# Reproducing the paper

There are two ways to reproduce the paper's numbers, and they promise different things:

| Path | Needs | Result |
|---|---|---|
| **A. From scored outputs (CPU)** | the paper's outputs folders (generation rows and score rows; judge caches to re-score), plus your own prepared copies of the benchmarks | every statistic and table **exactly**, bit for bit, with the documented exceptions below |
| **B. Regenerate (GPU)** | checkpoints, model environments, GPUs | new generations under the same protocol; numbers **close to** the paper's but not bit-identical (see "Nondeterminism") |

The outputs of the paper's runs are **not** distributed with this repository: they contain benchmark text and model
replies, including harmful completions to safety prompts, and several benchmarks do not allow redistribution
([licensing.md](licensing.md)). `paper/expected/` holds the numbers they produce
([paper/README.md](../paper/README.md)), so either path can be checked against the paper without them.

## Which command produces which table and figure

All commands are run from the repository root. `--outputs DIR` (a global option, before the command) selects the
outputs folder; the default is `<STREAMREPLACE_HOME>/outputs`. "Formal" = the 235 P/C/S jobs.

### Main paper

| Paper item | Command (in order) | Writes |
|---|---|---|
| Fig. 3: adaptation tax C-P (image, video) and execution tax S-C (video) | `stats`, `tables` | `tables/gaps.json` (the figure's cells; no plotting code), `gaps_figure.json` (C-P and S-P against the parent) |
| Table 2: trust and generalization (C-P, S-C; robustness; fine-tuning) | `stats`, `robustness report`, `finetune report`, `tables` | `tables/gaps.tex`, `gaps.csv` |
| Fig. 4: response forms on 400 harmful VSB requests | `experiment label response_forms`, `experiment report response_forms` | `experiments/response_forms/REPORT.md`, `report.json` |
| Sec. 4.2: 40 harmless how-to questions about VSB videos (answered offline and under streaming) | `experiment run harmless_requests`, `experiment label harmless_requests`, `experiment report harmless_requests` | `experiments/harmless_requests/` |
| Fig. 5a: window controls on VMME-750 (C_W - S, S_full - C) | `experiment run window`, `score`, `experiment report window` | `experiments/window/` |
| Sec. 5.1: fixed-budget memory control | `experiment run memory`, `score`, `experiment report memory` | `experiments/memory/` |
| Sec. 5.2: MMDuet forced to answer (0.8% -> 66.0%) | `experiment run response`, `score`, `experiment report response` | `experiments/response/` |
| Sec. 5.2: AURA without its streaming system prompt; its parent with it | `experiment run system_prompt`, `score --judge`, `experiment label`, `experiment report` | `experiments/system_prompt/` |
| Sec. 5.3: risk judgments, refusal decomposition | `experiment run harm_probe`, `experiment label harm_probe`, `experiment report harm_probe` | `experiments/harm_probe/` |
| Fig. 5b: harm scores at the request and before generation | the same | `experiments/harm_probe/report.json` (`pairs[].new_help`) |
| Sec. 5.3: the video-framing sentence | `experiment run framing`, `experiment label framing`, `experiment report framing` | `experiments/framing/` |
| Fig. 6: Stream++, means over the children | `plugin run`, `plugin assemble`, `score --judge`, `plugin report --items full` | `plugin/REPORT.md`, `report.json` (`means`), `figure.json` |

### Supplement

| Paper item | Command | Writes |
|---|---|---|
| A: models, protocols, scoring and statistics | - (documented in [protocol.md](protocol.md), [models/](models/stream-adapters.md), [judges.md](judges.md), [statistics.md](statistics.md)) | |
| B.1, B.2: image C-P per benchmark; absolute scores (image, video, safety, VMDT) | `report`, `stats`, `tables` | `tables/eval_image.tex`, `eval_video`, `eval_safety`, `eval_vmdt` (+ .csv), `gaps.json` |
| B.2: MCV-SafetyBench contrasts | `stats`, `tables` | `tables/mcv_gaps.tex` |
| B.2: response forms on VSB benign-query and MCV requests | `experiment report response_forms` | `experiments/response_forms/` |
| B.2: harmless requests (the 40 how-to questions) | `experiment report harmless_requests` | `experiments/harmless_requests/` |
| B.3: prompt robustness (pooled, per benchmark, instruction level) | `robustness report`, `tables` | `robustness/report/`, `tables/eval_robustness.tex`, `eval_instruction.tex` |
| B.3: fine-tuning accuracies, gains, curves, three seeds | `finetune report`, `finetune seeds`, `tables` | `finetune/FT_RESULTS.md`, `ft_tables/`, `finetune/seeds/SEED_RESULTS.md`, `tables/eval_finetuning*.tex` |
| A.4, B.1: generation-cap reruns | `rerun-capped report` | `rerun_capped/TRUNC.md` |
| C.1: execution tax by duration; window controls on VMME-300 | `experiment report duration`, `experiment run/report window_300` | `experiments/duration/`, `experiments/window_300/` |
| C.1: parent controls, AURA rows (P_S); offline-format input | `experiment run/report parent_stream`, `offline_input` | `experiments/parent_stream/`, `experiments/offline_input/` |
| C.2: question timing | `experiment run/report question_timing` | `experiments/question_timing/` |
| C.3: refusal context; refusal decomposition; new harmful help; probe scores; harmless-request controls | `experiment ... framing`, `harm_probe` | `experiments/framing/`, `experiments/harm_probe/` |
| C.3: steering along the harm direction and random directions | `experiment run steering`, `experiment label steering`, `experiment report steering` | `experiments/steering/` |
| D.1: per-child Stream++ figures, MCV figure | `plugin report --items full` | `plugin/report.json` (`cells`), `figure.json` |
| D.2: ablations (S_fix, gate only, PRS only), gate activation, answer readout, MOSS-VL-Realtime's per-task table | `plugin report --items full` | `plugin/report.json` (`loss`, `gate`, `prs`, `fire` and each arm's score per cell; rows' `info.plugin.readout`) |
| D.2: memory budget and retention; gains by duration; Recall settings | `experiment run/report recall_budget`, `recall_settings` | `experiments/recall_budget/`, `experiments/recall_settings/` |
| D.4: native streaming performance (OVO-Bench, StreamingBench) | `experiment run streaming_benchmarks`, `score`, `experiment report streaming_benchmarks` | `experiments/streaming_benchmarks/` |
| Stream++ vectors and thresholds | shipped; `plugin verify` checks them; `plugin calibrate` rebuilds them for a child | `plugin_artefacts/<child>/` |

Command details: [statistics.md](statistics.md) (`report`, `stats`, `tables`, `rerun-capped`),
[experiments.md](experiments.md), [plugin.md](plugin.md),
[generalization/robustness.md](generalization/robustness.md), [generalization/finetune.md](generalization/finetune.md).

## Path A: from scored outputs (CPU)

You need, under one outputs folder `OUT` (the layout every command writes):

```
OUT/<benchmark>/<job>.jsonl, .meta.json, .scores.jsonl    the 235 formal P/C/S jobs
OUT/judges/<judge>.<config hash>.jsonl                    judge caches (only to re-score)
OUT/rerun_capped/<pair>/<benchmark>/<model>.offline.*     generation-cap reruns, pairs p2-p5 (for `stats`)
OUT/robustness/<condition>/...                            prompt-robustness condition folders
OUT/finetune/...                                          fine-tuning evaluations, blank/, images/, seeds/
```

and, for Stream++, a separate folder `PLUGIN` holding the `<child>@sfix` and `<child>@sfix_prs` stream jobs and
`fixed_refusal.offline` on the evaluation items, and one folder per experiment under `OUT/experiments/`. The
analysis also reads each item's grouping keys (its video, question group, category, scenario) from the prepared
benchmarks, and several benchmarks read their official prompts, parsers and annotation files from the fetched
third-party code, so fetch the code and download and prepare the data first (`fetch-upstream all`, `download
datasets`, `prepare`; [data.md](data.md)). No GPU and no model environment is needed.

```bash
streamreplace --outputs OUT score --cache-only    # optional: rebuild every score row from the judge caches
streamreplace --outputs OUT report                # metrics.json, RESULTS.md, pairs.csv
streamreplace --outputs OUT rerun-capped report   # rerun_capped/TRUNC.md, trunc.json
streamreplace --outputs OUT stats --no-overlay    # analysis/no-overlay/contrasts.json (formal rows)
streamreplace --outputs OUT stats                 # analysis/contrasts.json (generation-cap reruns merged)
streamreplace --outputs OUT robustness report     # robustness/report/
streamreplace --outputs OUT finetune report       # finetune/FT_RESULTS.md, READOUT.md, ft_tables/
streamreplace --outputs OUT finetune seeds        # finetune/seeds/SEED_RESULTS.md
streamreplace --outputs OUT tables --check        # tables/*.tex (+ .csv), gaps.json, gaps_figure.json; every cell
                                                  # compared with paper/expected/tables.json
streamreplace --outputs PLUGIN plugin report --items full --formal OUT   # plugin/report.json, figure.json
streamreplace --outputs OUT experiment report response                   # ... and every other experiment
```

`tables --check` lists every printed cell that differs from the paper's; on the paper's rows none of the 935
compared cells does.

`score --cache-only` fails, writing nothing, if any judge answer is missing from the caches, so a successful run
proves that every score came from the cached judge answers. Each experiment folder (a robustness condition, a
fine-tuning control) is an outputs folder of its own and is scored the same way (`--outputs <folder>`).

To compare with the paper automatically, point the test suite at the outputs folder: the tests in
`tests/analysis/test_analysis_golden.py` recompute every contrast (with and without the reruns) and every job's
metrics and compare them field by field with `paper/expected/` when the outputs folder holds the 235 scored jobs:

```bash
STREAMREPLACE_OUTPUTS=OUT python -m pytest -q tests/analysis/test_analysis_golden.py
```

The statistics are bit-exact with numpy 2.2 to 2.5 (the `core` environment pins numpy 2.5.3, the version of the
paper's analysis); numpy does not promise identical random streams across versions, see "The two bootstrap engines"
in [statistics.md](statistics.md).

## Path B: regenerate (GPU)

```bash
# once: third-party code, environments, checkpoints, data (docs/data.md, docs/environments.md)
streamreplace fetch-upstream all
streamreplace env build all
streamreplace download models all --dry-run && streamreplace download models all
streamreplace download judges all --dry-run && streamreplace download judges all
streamreplace download datasets paper-core paper-supp --dry-run
streamreplace download datasets paper-core paper-supp
streamreplace prepare paper-core paper-supp       # checks counts, samples and prompt fingerprints

# the 235 P/C/S jobs (resumable: run the same command again after an interruption)
streamreplace run --pairs all --protocols P,C,S --benchmarks paper-core,paper-supp --dry-run
streamreplace run --pairs all --protocols P,C,S --benchmarks paper-core,paper-supp --gpus 0,1,2,3
streamreplace score --judge --judge-gpus 0,1      # rules, then the judges (RJScore self-tests first)

# generation-cap reruns at 8,192 tokens (pairs p2-p5 on MathVista, MMMU, MMStar)
streamreplace rerun-capped select                 # your capped ids -> <outputs>/rerun_capped/
streamreplace rerun-capped run --gpus 0,1
streamreplace --outputs <outputs>/rerun_capped/p3 score --judge --judge-gpus 0,1   # each of p2, p3, p4, p5

# statistics and tables
streamreplace report && streamreplace stats && streamreplace tables
```

`--gpus` lists GPU slots, one worker per slot; repeating an id packs several workers on one card (`--gpus 0,0,1,1`).
`--launcher print` writes each worker's payload and prints its command instead of running it, for a batch scheduler.
The prompt-robustness, fine-tuning, Stream++ and diagnostic experiments have their own sequences:
[generalization/robustness.md](generalization/robustness.md), [generalization/finetune.md](generalization/finetune.md),
[plugin.md](plugin.md), [experiments.md](experiments.md). Judge GPU needs: the extraction judge one GPU per replica, the HarmBench classifiers one
GPU, the RJScore judge (Qwen2.5-72B-Instruct, about 145 GB in bf16) at least two large GPUs with working
peer-to-peer copies ([judges.md](judges.md)).

### Nondeterminism

Regenerated outputs will not equal the paper's bit for bit, so their statistics will differ slightly. The sources:

- **GPU kernels.** Greedy decoding is deterministic only for a fixed GPU type, driver and library build; across
  GPU types a near-tie between two tokens can flip and the reply diverges from there. This affects every offline
  run and the greedy streams (MMDuet, MMDuet2, AURA, LiveStar's answer). FlashAttention-2 is part of the numerics
  where the paper used it; an adapter that falls back to sdpa records `info.attention`.
- **Sampled decoding.** ThinkStream samples its thoughts (seeded per question); its streams repeat exactly only on
  the same GPU type and FlashInfer version.
- **Batching.** AURA's stream is served by vLLM with 8 requests in flight; vLLM's batching is not batch-invariant.
- **Wall clock.** MOSS-VL-Realtime runs on the wall clock: how much it generates between frames, and whether the
  question shares a processing cycle with the last frame, depend on GPU speed.
- **Narration.** LiveStar's narration gate decides when to write a caption from perplexities averaged over three
  passes; the number of narrations varies between runs. The LiveStar pair also showed about 1% run-to-run change
  offline (non-deterministic kernels in its vision encoder), whereas a repeated offline evaluation in the paper's
  own runs (the prompt-robustness items) reproduced the other nine models' replies exactly. With the plugin,
  LiveStar shares one streaming session per video among that video's questions, unlike its native stream.
- **Judges.** New judge requests are answered by the judge models at temperature 0 (extraction judge, HarmBench
  classifiers) or by a forward pass (RJScore); the RJScore self-test refuses to judge when its reference scores move
  on your hardware by more than 0.15 (bf16 scores differ slightly between GPU models; see [judges.md](judges.md)).
- **Re-built inputs.** Paraphrases for the robustness experiment depend on the paraphrasing model and its serving
  stack; re-calibrated plugin vectors depend on GPU kernels and are not bit-identical to the shipped ones (use the
  shipped artefacts to reproduce the paper); fine-tuned adapters are bit-reproducible only with deterministic
  kernels on the same GPU type.

What does not depend on the hardware: the items, the prompt bytes, the sampled frames and stream schedules (with
the pinned decoding libraries of the environments), the decoding settings, the scoring rules, how judge requests and
their cache keys are built, and the statistics computed from a given set of rows.

### What has been checked against the paper's original runs

Every check below ran on CPUs against the original runs' outputs. No model or judge was re-run on a GPU to prepare
this release.

| Stage | Check |
|---|---|
| Plan and settings | the 235 planned jobs are the paper's; their resolved settings equal the original runs' settings |
| Prompts | the 457,392 stored prompts of the 235 jobs equal the prompts rendered from the prepared samples; the OVO-Bench (1,576) and StreamingBench (700) prompts equal the stored ones |
| Frames | the recomputed frame digest equals the recorded one on every offline video row (154,088 rows), and digests agree within every pair; [frames.md](frames.md) |
| Model inputs | processor inputs (token counts, frames, frame sizes, sampling rates) of sampled rows of every offline job equal the recorded ones; [models/offline-adapters.md](models/offline-adapters.md) |
| Stream schedules | `schedule()` reproduces the recorded structure of every row of the 48 stream jobs (84,040 rows) |
| Scoring | re-scoring the original generations with the original judge caches gives identical scores on every row, with no missing judge answer; OVO-Bench and StreamingBench: the official per-row score of every stored row |
| Statistics | every C-P, S-C and S-P cell behind Fig. 3, Table 2 and the MCV table (delta, 95% interval, Holm significance: 139 + 108 cells) and the Holm families of the paper (42 / 18 image, 18 per video gap, safety 51 / 21, VMDT 89 / 41) |
| Tables | `tables --check`: the 935 printed cells of Table 2, the MCV table and the eight absolute-score, robustness and fine-tuning tables |
| Stream++ | thresholds recomputed from their three quantiles; the 54 recovery cells (430 of 432 contrasts identical to the paper's snapshot, see below) and the means of Fig. 6 (Video-MME +12.5, VSB harmful +7.7, fairness +4.0, privacy +5.6) |
| Recall | the segments and chunks each row of the paper's Stream++ runs recorded as kept equal the ones this code's rule keeps: AURA 20,280 rows (3,562 over budget), ThinkStream 3,804 rows with a recorded retention (3,726 over budget) |
| Experiments | `response` (0.8 / 66.0 / 65.6), `response_forms` (every share and ASR of Fig. 4), `harmless_requests` (answered offline -> streaming: 16 -> 1, 18 -> 5, 12 -> 4, 30 -> 20, 15 -> 9, 28 -> 34), `question_timing` (every cell of the table), `window` (C_W and P_W on VMME-750 and their contrasts with S, C and P; the S_full arm was not checked), `recall_budget`, `recall_settings`, the duration split, `streaming_benchmarks`, and the harm scores of `harm_probe` (AURA pair: the probe-score and harmless-control rows) |
| Generalization | robustness item lists and edit files byte-identical (paraphrases: given the paper's cache); fine-tuning test subsets, training orders and selection files byte-identical; the planned fine-tuning evaluations (268 jobs, the three-seed replication of every checkpoint included) are the paper's; both reports and the seed report equal the paper's tables |
| Third-party code | `fetch-upstream` reproduces the third-party files of the paper's runs byte for byte |

The control settings of the stream adapters (question timing, the reservoir eviction of Recall inside ThinkStream's
KV engine, AURA's offline-format input) and the experiments' generation paths are ports of the code that produced
the paper's rows. In this layout they were exercised on stand-in models only (the unit tests), not on GPUs.

## Known differences from the published paper

- **MMDuet's native stream.** The paper's MMDuet rows were generated with a reply forced at the last frame and
  rescored as silence where the heads had not fired. Here the adapter returns that silence itself
  (`stream.force_answer` gives the forced control), which leads to the same scores. On the two RJScore benchmarks
  the judge then scores the empty reply; the paper used the judge's cached score of an empty reply to the same
  request where one existed (233 rows, all safe) and counted the other empty replies as safe.
- **Holm-adjusted p-values.** The paper's snapshot keeps some adjusted p-values from before the unreported cells
  left their families; the values here follow the stated families. No significance mark differs.
- **Two gate-only cells of Stream++** (MMDuet2 on TOMATO, MOSS-VL-Realtime on VMDT fairness) follow the final
  threshold rule here; the paper's snapshot holds their values under an earlier one. The complete plugin's cells
  are identical.
- **Bootstrap intervals of three analyses** (the streaming-benchmark table, the long-minus-short contrast of the
  accuracy-by-duration table, and AURA's window control on VMME-750) come from the analysis engine here (10,000
  draws) and differ from the paper's in the last digit of some bounds (at most 0.2 points). The point estimates are
  identical.
- **Sec. 5.1's fixed-budget numbers on VMME-750** were produced with an earlier implementation of the same memory
  rule (other hash seeds for the sample); `experiment run memory` reruns the control with Stream++'s Recall. The
  same contrast on the 666 evaluation questions (`recall_budget`) is reproduced exactly.
- **Captions** are regenerated from the data.
- **Paraphrases** regenerate exactly only with the paper's paraphrase cache, which contains benchmark-derived text
  and is not shipped; another model or serving stack produces other paraphrases (and other skipped units).
- **OpenUAV-QA** (optional fine-tuning set) keeps its frames on ModelScope, which has no immutable revision and
  serves some files empty; a new download may not give the paper's 3,893 test questions. The paper's selected ids
  ship.
- **Figures** are delivered as data (`gaps_figure.json`, `plugin/figure.json`, the experiments' `report.json`,
  `ft_tables/`), not as plots.

## Not part of this release

These analyses of the supplement are not scripted here; the paper states their numbers:

- A.4: the two blinded raters' check of the response-form labels, the second answer-extraction judge, the
  TimeBlind null models as a table, the DynaMath truncation bound, and the repeated streaming runs (the three
  ThinkStream seeds can be run with `--set stream.seed=k`);
- B.2: the neutral scoring prompt on narration replies, and the harmless requests on 96 benign clips;
- C.1: parent controls other than AURA's (the parent's weights inside MMDuet's and MMDuet2's streams, ThinkStream's
  input proxies, MOSS-VL-Realtime), the 49-token-per-frame control on TimeBlind, AURA's parent on the VMDT rows,
  the interval of the interaction column of the parent-control table (its point estimate is the difference of the
  two contrasts `parent_stream` reports), and the concept-readability probes;
- C.2: LiveStar's criterion shift on MME and its language-model swap, MMDuet2's compact prompt and its
  system-prompt removal on VMDT;
- C.3: the visual-content recognition checks (matching against mismatched video, MCV scene swap);
- D.3: the harmless-request evaluation on 142 benign clips and the gate-selectivity table that uses it. The
  requests of the first 96 clips ship (`paper/frozen/plugin/harmless_requests.json`), and
  `experiments/forms.py` labels replies to harmless requests.
