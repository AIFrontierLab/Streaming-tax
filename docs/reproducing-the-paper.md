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
outputs folder; the default is `<STREAMREPLACE_HOME>/outputs`.

| Paper item | Command (in order) | Writes | Reads | Expected values |
|---|---|---|---|---|
| Table 2 (offline C-P on the image headlines, HarmBench, prompt robustness) | `stats`, `robustness report`, `tables --paper` | `tables/gaps.tex`, `gaps.csv`, `gaps.json` | scored formal jobs, `rerun_capped/` rows, `robustness/report/contrasts.csv` | `paper/expected/tables.json` (`gaps`), `contrasts.json` |
| Gap figure (C-P and S-P on the video, safety and VMDT columns) | `stats`, `tables --paper` | `tables/gaps_figure.json`, `.csv` (figure data; no plotting code) | `analysis/contrasts.json` | `tables.json` (`figure`, 108 cells) |
| Supp. master tables (every C-P and every S-P contrast) | `stats`, `tables --paper` | `tables/master_cp.*`, `master_sp.*` | `analysis/contrasts.json` | `tables.json` (`master_cp`, `master_sp`) |
| Supp. absolute scores (image, video, safety, VMDT) | `report`, `tables --paper` | `tables/eval_image.tex`, `eval_video`, `eval_safety`, `eval_vmdt` | `metrics.json` (formal rows only) | `metrics.json`, `tables.json` |
| Supp. common-response refusal table | `tables --paper` | `tables/eval_common_response.tex` | `paper/flags/common_response.csv` (audit labels) | `tables.json` |
| Every P/C/S contrast, the counts quoted in the text, the Video-MME S-C duration split | `stats --no-overlay` (formal rows) and `stats` (generation-cap reruns merged) | `analysis/[no-overlay/]contrasts.json`, `.csv`, `GAPS.md` | scored formal jobs (+ `rerun_capped/`) | `paper/expected/contrasts_no_overlay.json`, `contrasts.json` (354 rows each, every field) |
| Generation-cap rerun sensitivity (Supp. B) | `rerun-capped report` | `rerun_capped/TRUNC.md`, `trunc.json` | formal and rerun rows | - |
| Prompt robustness: Sec. 4.3, Supp. `eval_robustness`, `eval_instruction` | `robustness report`, then `tables --paper` | `robustness/report/` (`ROBUSTNESS.md`, `results.json`, csv tables); `tables/eval_robustness.tex`, `eval_instruction.tex` | the scored condition folders | `tables.json` |
| Fine-tuning: the LingoQA figure of Sec. 4.3, Supp. `eval_finetuning`, `eval_finetuning_gains`, the fine-tuning curves | `finetune report`, `finetune seeds`, then `tables --paper` | `finetune/FT_RESULTS.md`, `READOUT.md`, `ft_tables/` (`results.json`: the tables' data, `cells.csv`: the figure's); `finetune/seeds/SEED_RESULTS.md`; `tables/eval_finetuning.tex`, `eval_finetuning_gains.tex` | the scored fine-tuning folders | `tables.json` (`eval_finetuning`, `eval_finetuning_gains`) |
| Plugin figure (Sec. 6) and recovery tables | `plugin run`, `plugin assemble`, `score --judge`, `plugin report --items pilot2` | `plugin/figure.json`, `report.json`, `REPORT.md` | the plugin item-set folder and the formal P/C/S jobs | `paper/expected/plugin_recovery.json` |
| Plugin vectors and thresholds | shipped; `plugin verify` checks them; `plugin calibrate` rebuilds them for a child | `plugin_artefacts/<child>/` | - | the shipped manifests |

Command details: [statistics.md](statistics.md) (`report`, `stats`, `tables`, `rerun-capped`),
[generalization/robustness.md](generalization/robustness.md), [generalization/finetune.md](generalization/finetune.md),
[plugin.md](plugin.md). The paper's description of the native streaming protocols (Supp. A.2) is hand-written;
[models/stream-adapters.md](models/stream-adapters.md) documents the same protocols in full.

## Path A: from scored outputs (CPU)

You need, under one outputs folder `OUT` (the layout every command writes):

```
OUT/<benchmark>/<job>.jsonl, .meta.json, .scores.jsonl    the 235 formal P/C/S jobs
OUT/judges/<judge>.<config hash>.jsonl                    judge caches (only to re-score)
OUT/rerun_capped/<pair>/<benchmark>/<model>.offline.*     generation-cap reruns, pairs p2-p5 (for `stats`)
OUT/robustness/<condition>/...                            prompt-robustness condition folders
OUT/finetune/...                                          fine-tuning evaluations, blank/, images/, seeds/
```

and, for the plugin, a separate folder `PLUGIN` holding the `<child>@sfix` and `<child>@sfix_prs` stream jobs and
`fixed_refusal.offline` on the second pilot's items. The analysis also reads each item's grouping keys (its video,
question group, category, scenario) from the prepared benchmarks, and several benchmarks read their official prompts,
parsers and annotation files from the fetched third-party code, so fetch the code and download and prepare the data
first (`fetch-upstream all`, `download datasets`, `prepare`; [data.md](data.md)). No GPU and no model environment is
needed.

```bash
streamreplace --outputs OUT score --cache-only    # optional: rebuild every score row from the judge caches
streamreplace --outputs OUT report                # metrics.json, RESULTS.md, pairs.csv
streamreplace --outputs OUT rerun-capped report   # rerun_capped/TRUNC.md, trunc.json
streamreplace --outputs OUT stats --no-overlay    # analysis/no-overlay/contrasts.json (formal rows)
streamreplace --outputs OUT stats                 # analysis/contrasts.json (generation-cap reruns merged)
streamreplace --outputs OUT robustness report     # robustness/report/ (Table 2 robustness columns, Supp. E)
streamreplace --outputs OUT finetune report       # finetune/FT_RESULTS.md, READOUT.md, ft_tables/
streamreplace --outputs OUT finetune seeds        # finetune/seeds/SEED_RESULTS.md
streamreplace --outputs OUT tables --paper --check   # tables/*.tex (+ .csv), gaps.json, gaps_figure.json; every cell
                                                    # compared with paper/expected/tables.json
streamreplace --outputs PLUGIN plugin report --items pilot2 --formal OUT   # plugin/report.json, figure.json
```

`tables --paper --check` lists every printed cell that differs from the paper's; on the paper's rows these are
exactly the four ThinkStream master-table cells under "Known differences" below.

`score --cache-only` fails, writing nothing, if any judge answer is missing from the caches, so a successful run
proves that every score came from the cached judge answers. Each experiment folder (a robustness condition, a
fine-tuning control) is an outputs folder of its own and is scored the same way (`--outputs <folder>`).

To compare with the paper automatically, point the test suite at the outputs folder: the tests in
`tests/analysis/test_analysis_golden.py` recompute all 354 contrasts (with and without the reruns) and every job's
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
streamreplace report && streamreplace stats && streamreplace tables --paper
```

`--gpus` lists GPU slots, one worker per slot; repeating an id packs several workers on one card (`--gpus 0,0,1,1`).
`--launcher print` writes each worker's payload and prints its command instead of running it, for a batch scheduler.
The prompt-robustness, fine-tuning and plugin experiments have their own sequences:
[generalization/robustness.md](generalization/robustness.md), [generalization/finetune.md](generalization/finetune.md),
[plugin.md](plugin.md). Judge GPU needs: the extraction judge one GPU per replica, the HarmBench classifiers one
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
| Prompts | the 457,392 stored prompts of the 235 jobs equal the prompts rendered from the prepared samples |
| Frames | the recomputed frame digest equals the recorded one on every offline video row (154,088 rows), and digests agree within every pair; [frames.md](frames.md) |
| Model inputs | processor inputs (token counts, frames, frame sizes, sampling rates) of sampled rows of every offline job equal the recorded ones; [models/offline-adapters.md](models/offline-adapters.md) |
| Stream schedules | `schedule()` reproduces the recorded structure of every row of the 48 stream jobs (84,040 rows) |
| Scoring | re-scoring the original generations with the original judge caches gives identical scores on every row, with no missing judge answer |
| Statistics and tables | the 354 contrasts, with and without the reruns, bit for bit; RESULTS.md; every table cell, with the exceptions below |
| Plugin | thresholds recomputed exactly; the 135 recovery cells and the figure cells exact |
| Generalization | robustness item lists and edit files byte-identical (paraphrases: given the paper's cache); fine-tuning test subsets, training orders and selection files byte-identical; both reports equal the paper's tables |
| Third-party code | `fetch-upstream` reproduces the third-party files of the paper's runs byte for byte |

## Known differences from the published paper

- **Four ThinkStream VMDT cells of the supplement's master tables** were rendered from an unreleased variant of
  ThinkStream's answer cleaning. The released scoring, which every other number uses (including Table 2 and the gap
  figure), gives C-P privacy +5.1 and state +15.0 (bold) instead of +4.8 and +12.5, and S-P stereotype -1.0 and
  decision +1.2 instead of -1.1 and +1.4.
- **Gap-figure flags.** The S-P flag sets of the gap figure were assembled by hand; only the printed marks (dagger
  for a forced stream, double dagger for capped rows) are reproduced, and they agree.
- **"21 of 42".** A supplement sentence counts 21 of 42 image contrasts as equivalent within +-3 points; that count
  predates the generation-cap reruns. With the reruns, as in Table 2, it is 19 of 42.
- **Captions** are regenerated from the data (for example, the video table's caption names the incomplete entries).
- **Plugin over-refusal guardrail.** The paper quotes S+ against P on 46 held-out harmless clips. That needs refusal
  labels from an LLM judge with a harmlessness rubric that is not part of this package; the requests ship
  (`paper/frozen/plugin/harmless_requests.json`, role `guardrail`), the labels and the comparison do not.
- **MOSS-VL-Realtime plugin results are not reported in the paper.** Its plugin artefacts and thresholds ship and
  `plugin run` supports it, but the paper reports S_fix and S+ for the other five children only; the paper's plugin
  figure is the second pilot (908 items). The `full` item set (10,140 items) is defined but its results are not in
  the paper.
- **Robustness sentences.** Two statements of the supplement's robustness reanalysis (a count of unrelated model
  pairs separated by accuracy versus by probability, and MOSS-VL-Realtime's share of additionally flipped answers)
  are not computed by `robustness report`; every other robustness number is.
- **Paraphrases** regenerate exactly only with the paper's paraphrase cache, which contains benchmark-derived text and
  is not shipped; another model or serving stack produces other paraphrases (and other skipped units).
- **OpenUAV-QA** (optional fine-tuning set) keeps its frames on ModelScope, which has no immutable revision and serves
  some files empty; a new download may not give the paper's 3,893 test questions. The paper's selected ids ship.
- **Figures** are delivered as data (`gaps_figure.json`, `plugin/figure.json`, `ft_tables/`), not as plots.

## Not part of this release

- The diagnostic analyses of Sec. 5 and their supplementary tables and figures (history restrictions, the parent in
  the child's streaming protocol, internal-representation probes), the exploratory intervention screening of the
  supplement, and the hand-assembled evidence table.
- The LLM-labelled response-form audit behind the response-form figures. Its per-model shares and verdicts ship as
  numbers in `paper/flags/` (they qualify cells of the analysis); the labelling is not included.
- The second-judge sensitivity analysis of the supplement (re-asking the extraction requests to another judge). The
  extraction judge is configurable in `configs/judges.yaml`, but that comparison is not scripted.
