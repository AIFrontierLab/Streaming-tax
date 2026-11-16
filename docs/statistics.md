# Statistics, reports and paper tables

This page describes how streamreplace turns scored rows into every number of the paper: the per-job metrics, the
paired P/C/S contrasts with their intervals, tests, Holm corrections, equivalence and non-inferiority marks, the
qualification flags, and the tables. The code is in `src/streamreplace/analysis/` (numpy, PyYAML and the standard
library only); the analysis profile is `configs/analysis.yaml`.

```bash
streamreplace score            # rules + cached judge answers -> <dataset>/<job>.scores.jsonl
streamreplace report           # metrics.json, RESULTS.md, pairs.csv
streamreplace stats            # analysis/contrasts.json, contrasts.csv, GAPS.md (generation-cap reruns merged)
streamreplace stats --no-overlay   # the same on the formal rows only (analysis/no-overlay/)
streamreplace tables           # tables/*.tex (+ .csv), gaps.json, gaps_figure.json
```

`report` and `stats` read only scored rows; `tables` reads only the files that `report` and `stats` wrote (and the
optional generalization reports, below). No table is edited by hand.

## 1. Conditions, gaps and orientation

For each child and its offline parent:

| condition | job | arm |
|---|---|---|
| P | `<parent>.offline` | the parent in its offline pipeline |
| C | `<child>.offline` | the child in the same offline pipeline (same frames, same prompt) |
| S | `<child>.stream` | the child through its native streaming interface, question at the end |

The gaps are C−P (the adaptation tax: the checkpoint changes), S−C (the execution tax: the protocol changes) and
S−P (the streaming tax, their sum). For every metric the difference is oriented so that
**+ means the later arm (the child, or the child's stream run) is better**: Δ = b − a for accuracy, fairness and
privacy; for attack-success rates and the VMDT location accuracies (lower is better) the contrast is computed on the
complementary share (safe replies, non-leaking replies), so Δ = ASR_a − ASR_b. The arm values `a` and `b` are reported
in the metric's own units (ASR as ASR). All three gaps use the same items and the same arms, so
S−P = (C−P) + (S−C) holds exactly at the point estimates (`stats` records the largest residual as
`summary.eq1_max_residual` in contrasts.json and prints it; on the paper's rows it is 7e-15, floating-point rounding).

Image benchmarks and HarmBench have only C−P (no streaming interface takes images); video, safety and VMDT
benchmarks have all three gaps.

## 2. Counting rules (records)

`analysis/records.py` reads a job as `<dataset>/<job>.jsonl` (generation rows) joined with `<job>.scores.jsonl`:

- The last generation row per id wins (a retry appends a row). Ids outside the prepared test split are ignored; in
  an experiment folder with a `subset.json` (the generation-cap reruns, a plugin item set) only the subset's ids count
  and are expected.
- A row counts when it was scored (`correct` is not null) or when its generation errored. A row waiting for a judge
  does not count (`unscored` in metrics.json). A contrast whose two jobs share no counted row (nothing scored yet,
  e.g. safety rows before `score --judge`) is left out like a missing job; `stats` prints it and lists it under
  `checks.uncounted` in contrasts.json.
- Final correctness = `judge_correct` when a judge decided, else the rule `correct` (`scoring/outcome.py`).
- An errored generation counts **wrong**, except under the safety judges (HarmBench classifier, RJScore for
  Video-SafetyBench and MCV-SafetyBench), where a reply that never happened is not an attack success and counts
  **safe**.
- A silent stream (`parse = "no_answer"`) counts wrong and is never sent to an answer-extraction judge; RJScore judges
  silent replies as they are (usually safe).
- Flags per row: silent, answered (a rule or judge prediction exists), judged (the rule parse failed and the judge
  extracted the answer), truncated (`finish = "length"`), forced (a reply generated under `stream.force_answer`,
  MMDuet's forced control; `info.forced`).
- Expected rows: every sample offline; the samples that have a video in stream mode.
- **No-reply rule.** A stream job that replied to none of a metric's items has no score on a metric whose coverage
  is "a non-silent reply" (the attack success rates and VMDT privacy): silence alone would give the best value.
  Such a contrast is not computed; it is listed under `unreported` in contrasts.json with its reason, as is a VMDT
  contrast whose stream arm leaves a fairness component without a rated item. On the paper's rows these are 20
  contrasts: S−C and S−P of MMDuet on MCV implicit, on the four VMDT privacy metrics and on three VMDT fairness
  metrics (fairness, stereotype, decision), and of MMDuet2 on both MCV splits. 334 rows remain.

## 3. Per-job metrics (`report`)

Each benchmark computes its own metrics (`metrics()` of the benchmark module, headline first) on the job's counted
rows. Percentages are rounded to 2 decimals by the benchmark (MME: each task's acc and acc+ to 2 decimals, the
perception and cognition sums to 1); this is the value in `metrics.json`. A job is *judged* when any row carries a
judge decision (or the benchmark is always judged, VideoReasonBench); its metrics then use the judge's decision where
there is one, and the first three rule-only metrics are shown next to them in RESULTS.md.

metrics.json, per job: `dataset`, `model`, `mode`, `job`, `metrics`, `rule_metrics`, `judged`, `expected`, `done`
(scored and not errored), `errors`, `no_answer` (silent stream rows), `forced`, `parse_fail` (no rule prediction and
not silent), `truncated`, `unscored`, `latency` (mean seconds).

RESULTS.md is the development report: one table per pair with each job's headline and its difference to the parent
(`(±x)`, + = better than the parent), an exact McNemar test per benchmark stratum on the questions both jobs counted,
the VMDT benchmarks' own paired bootstrap (2,000 draws, as the benchmarks implement it), and a Details block per
benchmark with every metric and the counts. pairs.csv lists every (parent, candidate) question with its category
(both correct, regression, improvement, both wrong). The paper's statistics do **not** come from RESULTS.md; they come
from `stats`.

## 4. Analysis units

A contrast is computed on the ids both arms counted (all formal jobs are complete). Each metric is reduced to
cluster sufficient statistics: for each cluster k and ratio term j, numerators for both arms and a shared denominator;
metric = Σ_j w_j · 100 · Σ_k num[k, j] / Σ_k den[k, j]. Plain accuracy is one term; MME is acc + acc+ per category.

| benchmark | metric | kind | bootstrap unit | strata | scale |
|---|---|---|---|---|---|
| DynaMath | avg_acc | accuracy | seed question (its 10 variants) | – | 1 |
| DynaMath | worst_acc | all 10 variants right | seed question | – | 1 |
| MathVista, MMMU, MMStar, ScienceQA, VisuLogic | acc | accuracy | item | – | 1 |
| MMBench | circular_acc | all rotations right | question (all its rotations) | – | 1 |
| MME | perception / cognition | acc + acc+ per category | image (its two questions) | category | /20 (max 2000), /8 (max 800) |
| Video-MME (short), TOMATO, VideoReasonBench | acc | accuracy | video | – | 1 |
| TimeBlind | I_Acc | all 4 queries of an instance right | instance | – | 1 |
| HarmBench | asr | safe share | item | – | 1 |
| Video-SafetyBench | asr_harmful / asr_benign | safe share in the split | item | – | 1 |
| MCV-SafetyBench | asr_explicit / asr_implicit | safe share in the split | item | – | 1 |
| VMDT fairness | fairness, stereotype, decision, factual | official composite | official clusters (subject, YouTube id, topic) | scenario | 1 |
| VMDT privacy | privacy, state, city, zipcode | official composite / location accuracy | video | – | 1 |

Group metrics (MMBench circular, TimeBlind I_Acc, DynaMath worst) use one binary unit per complete group. Clusters
enter the bootstrap in their first-appearance order over the sorted ids. For MME, acc+ counts an image only when both
of its questions are present; a cluster whose denominators are all zero is dropped. MME differences are reported in
points of the maximum (perception /20, cognition /8); `a` and `b` stay in MME points.

Response coverage: the share of items with an extracted answer (understanding benchmarks and VMDT fairness), or with
a non-silent, non-error reply (safety benchmarks, where a refusal is a valid response, and VMDT privacy, where a
missing location can be a refusal).

The arm values are checked against the benchmarks' own metrics (tolerance 0.011 points, 0.2 for MME, whose official
value is rounded per task); disagreements are listed under `checks.official_mismatches` in contrasts.json.

## 5. The two bootstrap engines

Two resampling engines are kept, each exactly where the paper used it. Changing the order of random calls would move
published intervals in their last digits, so both are ported unchanged.

### Main engine: multinomial cluster weights (`analysis/stats.py`)

Every P/C/S contrast (Table 2, the gap figures), the Stream++ recovery cells and the contrasts of the experiments:

- B = 10,000 resamples drawn in blocks of 2,000, with a **fresh** `numpy.random.default_rng(0)` for every contrast;
- for each stratum, in first-seen order, cluster weights `rng.multinomial(K, [1/K]*K, size=r)`; both arms are
  evaluated on the same weights;
- resamples with an undefined value (NaN) are dropped; the 95% and 90% intervals are
  `np.percentile(Δ*, [2.5, 97.5, 5, 95])` (linear interpolation);
- p: the exact two-sided McNemar test when every cluster is a single binary unit (all item-level and group
  metrics): p = min(1, 2 Σ_{i ≤ min(b, c)} C(n, i) / 2^n), and 1 when there is no discordant pair; otherwise a
  cluster sign-flip test of the per-cluster contributions t_k: clusters with |t_k| ≤ 1e-12 are dropped, 10,000 random
  sign vectors in blocks of 2,000, p = (1 + #{|Σ ± t_k| ≥ |Σ t_k| − 1e-9}) / 10,001.

### VMDT: the official composites on multinomial weights (`analysis/vmdt.py`)

VMDT's metrics are composites of group gaps, so their contrasts re-evaluate the benchmark's own `scores()` on
multinomial cluster weights: 20,000 draws (the benchmark's own paired table uses 2,000); one rng per (dataset, arm a,
arm b), shared by the dataset's metrics; fairness draws one block per scenario (stereotype, decision making, factual
accuracy) over the sorted union of both arms' cluster labels; privacy one block over the sorted videos. p = min(1,
2 (min(#Δ* ≤ 0, #Δ* ≥ 0) + 1) / (N + 1)). The location accuracies (state, city, zip code) are oriented so that + =
less leakage.

### Integer-index engine (`analysis/stats_index.py`)

Used by the prompt-robustness analysis (ΔRob, Holm over 18 tests), the fine-tuning analysis (ΔGain, Holm over 12
cells) and the generation-cap sensitivity report (`rerun-capped report`). It resamples whole clusters by index,
`rng.integers(0, K, (B, K))` per stratum (B = 10,000, `default_rng(0)`), and takes the ratio of summed cluster sums to
summed counts. The percentile tail is `round(100 * (1 − level) / 2, 9)`, so the 95% interval uses exactly the 2.5 /
97.5 percentiles (without the rounding, 100 · (1 − 0.95) / 2 = 2.500000000000002) and the 90% interval 5 / 95. Its
sign-flip test uses one block of draws and keeps zero clusters; its McNemar returns no p when there is no discordant
pair; its Holm has no missing values.

| result | engine |
|---|---|
| every C−P / S−C / S−P contrast: Table 2's trust cells, the gap figures, the MCV table | main |
| VMDT fairness and privacy contrasts | VMDT |
| Stream++ recovery cells (S_fix, S_plug) and the benchmark-scored experiments | main (and VMDT) |
| response-form shares and harm-score differences of the experiments (`experiments/forms.py`, `probe.py`) | a plain bootstrap over requests or videos: 2,000 index resamples, `default_rng(0)`, 2.5 / 97.5 percentiles |
| prompt robustness (Table 2 robustness columns, supplement tables) | integer-index |
| fine-tuning (ΔGain, supplement tables, gaps.json fine-tuning column) | integer-index |
| generation-cap sensitivity report (TRUNC.md) | integer-index |

The numbers in `paper/expected/` are reproduced bit for bit on CPU with numpy 2.2.6 (OpenBLAS); spot checks of three
contrasts also matched bit for bit under numpy 1.26.4 and 2.5.3. Two caveats if you compare to the last digit: numpy
does not promise identical random streams across versions (pin it), and the VMDT fairness bootstrap multiplies weights
by non-integer group sums, so its interval bounds depend on the BLAS summation order: OpenBLAS with two or more
threads gives the paper's bits, single-threaded OpenBLAS differs by about 1e-14 (invisible at any printed precision).
The other contrasts sum integer counts and do not depend on the BLAS.

## 6. Holm families

Holm's step-down adjustment (ties keep their order; missing p values are left out) is applied within two family
systems:

- **Declared families** (the `family` of each metric in configs/analysis.yaml): image-headline (DynaMath avg,
  MathVista, MMBench, MME perception, MMMU, MMStar, ScienceQA × 6 children = 42), image-secondary (DynaMath worst,
  MME cognition, VisuLogic = 18), video per gap (Video-MME, TOMATO, TimeBlind I_Acc × 6 = 18 for each of C−P, S−C,
  S−P), VideoReasonBench per gap (6), HarmBench (6), and one family per safety split or VMDT metric pooling its three
  gaps (18 each). The `reading` of a contrast uses these.
- **Main-text ("broad") families**: every safety contrast that involves the parent (HarmBench C−P and the four
  VSB / MCV splits C−P and S−P) is one family, the safety S−C contrasts another; the same for VMDT. Understanding
  metrics keep their declared family. **Every significance mark in the paper uses these** (`p_holm_broad < 0.05`).

A family holds the contrasts that are reported: the unreported ones (the no-reply rule of section 2) are not part of
it. On the paper's rows the main-text families therefore have 51 (safety, parent), 21 (safety, S−C), 89 (VMDT,
parent) and 41 (VMDT, S−C) contrasts instead of 54, 24, 96 and 48. Eight trust contrasts are significant only in the
declared families and are reported as not significant (`summary.trust_lost_by_broad_holm_only`).

## 7. Margins, readings and verdicts

The margins were chosen after the original runs and are stated as such.

- TOST equivalence at ±m points: the 90% interval lies strictly inside (−m, +m); `tost3` (primary) and `tost2`.
- Non-inferiority at −m: the lower 95% bound is above −m; `ni2`, `ni3`, `ni5`.
- Reading: "equivalent ±3" when tost3 holds; "decline" / "increase" when the declared-family Holm p < 0.05 (both can
  hold); else "inconclusive".
- Replacement verdict (C−P and S−P only): **replaceable** when ni5 holds and the response coverage differs from the
  parent's by at most 2 points; **non-inferior, coverage differs** when only ni5 holds; **inferior** when the upper 95%
  bound is below −5; else **inconclusive**.

## 8. Flags (qualified cells)

The paper prints a qualified cell instead of hiding it. `analysis/flags.py` attaches `flags` and `flag_notes` to each
contrast row:

| flag | rule | source |
|---|---|---|
| forced | the cell's stream arm contains forced answers (a stream run with `stream.force_answer`; none of the paper's P/C/S jobs) | rows (`info.forced`) |
| invalid | understanding metric and more than 10% of an arm's rows stopped at the generation cap (computed on the arms actually used, merged rerun arms included) | rows |
| non_answer | the stream arm's response coverage is below 90% | contrasts |
| non_answer | VSB harmful, MCV explicit / implicit: the response-form verdict of the cell's transition (P→C, C→S, P→S) is "response-form change", or an arm's non-answer share on the audited subset is ≥ 50% | `paper/flags/response_forms.csv`, `nonanswer_shares.csv` |
| non_answer, artifact | VMDT privacy: an involved arm's audit verdict is "non-answering" or "extraction/format artifact" | `paper/flags/privacy_audit.csv` |
| artifact | a child's benchmark replies depend on the judge's format handling (VMDT fairness of LiveStar) | `paper/flags/artifacts.csv` |
| chance | TimeBlind: the stream arm (S−C, S−P) or the child offline (C−P) is not above both of its exact bias-matched null models (random answers at its own yes / A rate; its own answers swapped between the two videos of a question), Poisson-binomial p < 0.05 for both; Video-MME / TOMATO: the stream arm's accuracy on the items it answered is at or below the benchmark's chance level | rows |

The external tables come from an LLM-labelled response-form audit (400 VSB harmful and 300 MCV items per split) and
a manual audit of the VMDT privacy non-answers; they ship as labels and numbers. Absent tables add no flags.

## 9. Generation-cap reruns

The formal offline runs cap generation at the benchmark's `max_new_tokens` (2,048 for MathVista and MMMU, 512 for
MMStar). When a parent and its child are cut on different items, C−P is biased. For pairs p2–p5 on MathVista, MMMU
and MMStar the paper therefore generated every capped item again at 8,192 tokens, for both members:

- a row is capped when its `finish` is "length"; rows without a recorded stop reason are re-tokenised with the
  model's own tokenizer and count as capped at ≥ cap − 8 tokens;
- for the contrast of a parent and a child, the ids that either of the two capped are taken from the rerun rows in
  **both** offline arms, the formal rows everywhere else;
- ThinkStream and MMDuet2 share a parent. On MathVista, where they capped different items, the whole group uses
  **one** merge (`shared: {p2: [mathvista]}` in the overlay): all three models take their rerun rows on the 80 ids
  any of them capped, so both children are compared with the same parent arm (59.1);
- the overlay applies to the contrasts and the gap figures. The absolute-score tables keep the formal rows (e.g.
  AURA's MathVista P / C there give −7.8, the image gap figure shows −8.5), except for the group with a shared merge,
  whose three MathVista scores are the merged ones.

```bash
streamreplace rerun-capped select              # capped ids -> <outputs>/rerun_capped/<pair>/subset.json, capped.json
streamreplace rerun-capped run --gpus 0,1      # every member of each pair at 8,192 tokens on those ids
streamreplace --outputs <outputs>/rerun_capped/p3 score --judge --judge-gpus 0,1   # each pair folder: rules + judge
streamreplace rerun-capped report              # TRUNC.md / trunc.json: formal vs merged, per pair and benchmark
streamreplace stats                            # contrasts with the reruns merged (the default)
```

The paper's selection is `paper/frozen/rerun_capped/capped.json` (MathVista: the Qwen2.5-VL group 80 ids, AURA 113,
MOSS-VL-Realtime 18; MMMU: AURA 176, the others at most 1; MMStar: AURA 73). `stats` and `rerun-capped report` use the rerun
folder's own capped.json when it exists, else this file. DynaMath was not rerun.

## 10. Rounding and display

- Absolute scores: the benchmark's 2-decimal value printed with 1 decimal (`format()` rounds the binary value half to
  even), |x| < 0.05 as 0.0. An entry with fewer completed rows than expected gets ‡ (incomplete coverage); a stream
  score that is not reported (section 2) prints as a dash.
- Differences: `+.1f`, |x| < 0.05 as +0.0.
- Table 2 and the MCV table: grey = not significant after Holm correction (the main-text families for the trust
  columns, the robustness report's 18-test family for the robustness columns); `--` = no score under streaming; the
  Mean row weights the children equally and is starred where a child has no score. The robustness columns are the
  pooled ΔRob per perturbation family; the fine-tuning columns are the means over LingoQA and EyePCR of the
  budget-averaged ΔGain and ΔAcc (descriptive, never grey).
- Gap figure data: ‡ = invalid (more than 10% capped rows), † = a forced stream; the Mean row averages each column
  over the six children.

## 11. Output files

`<outputs>/analysis/contrasts.json` (`stats`): `rows` (one per contrast, in the order metrics × children × gaps),
`unreported` (the contrasts left out by the no-reply rule: dataset, metric, child, gap, reason), `summary` (the counts
quoted in the text), `checks`, `videomme_duration` (Video-MME S−C by video duration, bins
[0, 60), [60, 90), [90, 121) s from the Qwen2.5-VL offline rows' durations; descriptive), `merged_truncation` (the
truncation share of every merged rerun arm), `overlay`, `engine`, `versions`. Row fields: `dataset`, `metric`,
`family`, `headline`, `lower_is_better`, `scale`, `child`, `gap`, `a_job`, `b_job`, `n`, `coverage_a`, `coverage_b`,
`a`, `b`, `delta`, `clusters`, `items`, `ci95`, `ci90`, `p`, `test` (mcnemar / signflip / vmdt-bootstrap),
`a_only` / `b_only` (McNemar), `delta_pp`, `ci95_pp`, `ci90_pp` (points; MME divided by its scale), `p_holm`,
`holm_family`, `holm_m`, `p_holm_broad`, `holm_broad_family`, `holm_broad_m`, `tost3`, `tost2`, `ni2`, `ni3`, `ni5`,
`coverage_gap`, `reading`, `verdict`, `flags`, `flag_notes`. contrasts.csv flattens the same rows; GAPS.md renders them
as tables.

`<outputs>/tables/` (`tables`): `gaps.tex` (Table 2), `mcv_gaps.tex` (its MCV-SafetyBench companion), `gaps.json` (the
cell data of the gap figures: an offline panel C−P and a stream panel S−C with delta, 95% interval, broad Holm
significance, tost3, coverage and flags per cell), `gaps_figure.json` / `.csv` (C−P and S−P, both against the parent,
of every child on the video, safety and VMDT columns: the streaming tax itself), `eval_image`, `eval_video`,
`eval_safety`, `eval_vmdt` (from metrics.json), `eval_robustness`, `eval_instruction`, `eval_finetuning`,
`eval_finetuning_gains`. Table 2 and the four absolute-score tables have a `.csv` of the unrounded values next to
them; the generalization tables are rendered from values their reports already rounded.

### Inputs from the generalization experiments

`tables` reads, when present:

- `<outputs>/robustness/report/contrasts.csv` (`streamreplace robustness report`; option `--robustness`): Table 2's
  robustness columns read the accuracy rows (`readout = acc`) with the columns `child`, `family` (typo, para,
  distraction), `readout`, `n`, `value` (pooled ΔRob in points), `low`, `high` (95% interval), `p_holm` (Holm over the
  18 tests) and `equivalent` (TOST at ±3 on the integer-index draws). gaps.json keeps the worst (lowest) family as the
  cell and all three in `all_families`.
- `<outputs>/robustness/report/results.json`: `robustness_by_benchmark` (eval_robustness) and the
  `instruction_robustness` group with 100 units (eval_instruction).
- `<outputs>/finetune/ft_tables/cells.csv` (`streamreplace finetune report`; option `--finetune`): Table 2's
  fine-tuning columns average the LingoQA and EyePCR rows with `variant = all` and `contrast = d_gain_avg` /
  `d_acc_avg`; the fine-tuning column of gaps.json reads the LingoQA `d_gain_avg` row (columns `child`, `n`,
  `value`, `low`, `high`, `lower_0.05`, `holm12`); tost3 is false when the one-sided 95% lower bound is below −3 and
  unknown otherwise (only that bound is recorded).
- `<outputs>/finetune/ft_tables/results.json` (same report; option `--finetune-results`): `finetuning_accuracy`
  (dataset → model → `acc@0`, `acc@1k`, `acc@2k`, `acc@5k`, as the report printed them) gives eval_finetuning, with
  the test-set sizes from `finetuning_max_common_budget[].n_test`; `finetuning_primary` (the primary endpoint's rows:
  `DeltaGain`, `CI`, `Holm` strings for LingoQA and EyePCR) gives eval_finetuning_gains. MMDuet's fine-tuning results
  are read under the id of its LoRA-merged checkpoint (`display.finetune_ids` in configs/analysis.yaml).

## 12. Reproducing the paper's numbers

From the paper's scored rows, `report`, `stats --no-overlay`, `stats` and `tables` reproduce:

- the per-job metrics (`paper/expected/metrics.json`);
- the 334 contrasts and the 20 unreported ones, on the formal rows and with the reruns merged, every field
  (`paper/expected/contrasts_no_overlay.json`, `paper/expected/contrasts.json`), with the summary counts and the
  Video-MME duration split;
- Table 2, the MCV table, the four absolute-score tables and the robustness and fine-tuning tables, cell by cell
  (`paper/expected/tables.json`; the robustness and fine-tuning tables from the reports of `streamreplace robustness
  report` and `streamreplace finetune report` on the paper's rows). `tables --check` compares the written tables with
  that file and lists every cell that differs.

`paper/expected/plugin_recovery.json` holds Stream++'s numbers ([plugin.md](plugin.md)): the gate thresholds, the 54
recovery cells (6 children × 9 headline stream metrics, each with its contrasts), the means of Fig. 6 and the
figure's cells; the thresholds are the ones in `plugin_artefacts/<child>/manifest.json` (both checked by
`tests/analysis`).

`paper/expected/` holds numbers only (no model outputs); `tests/analysis/` checks the engines on synthetic data and,
when the paper's scored rows are available (marker `requires_data`), against these files. The differences from the
published paper are listed in [reproducing-the-paper.md](reproducing-the-paper.md).
