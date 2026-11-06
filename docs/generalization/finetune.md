# LoRA fine-tuning

Does streaming adaptation make a model harder to adapt to a new domain? Every model of the five pairs gets the same
LoRA fine-tuning recipe on the same training samples of a new domain, and each streaming child's improvement is
compared with its offline parent's. The experiment feeds Supp. Tables `eval_finetuning` and `eval_finetuning_gains`,
the fine-tuning figures of the supplement and the fine-tuning column of Table 2 (LingoQA).

Everything runs **offline** through the main evaluation's pipeline (parent and child see the same frames and differ
only in their weights). Code: `src/streamreplace/generalization/finetune/` (selection, LoRA, training, reports),
`src/streamreplace/benchmarks/finetune/` (LingoQA, EyePCR, OpenUAV-QA) and `src/streamreplace/judges/lingo.py`
(Lingo-Judge); specification: `configs/generalization/finetune.yaml`; the paper's id lists:
`paper/frozen/finetune/`; command: `streamreplace finetune`.

## Design

**Models.** The 11 checkpoints of the five pairs, MMDuet as its LoRA-merged checkpoint (`mmduet_7b_merged`: MMDuet's
LoRA folded into LLaVA-OneVision's weights and its projector swapped in; the main evaluation keeps that LoRA
unmerged). Within a pair the settings are identical; zero-shot and fine-tuned evaluations of a model differ only in
the adapter. Model-side settings that differ from the main evaluation (zero-shot and fine-tuned alike):

| Pair | Setting | Why |
|---|---|---|
| p1 (LLaVA-OneVision, MMDuet) | bf16 weights, vision tower included (main evaluation: fp16) | LoRA gradients underflow on fp16 base weights |
| p5 (MOSS-VL, MOSS-VL-Realtime) | `video.max_pixels` 602,112 per frame (the p3 budget) | near-native frames of a 16-frame 1080p clip need ~32K tokens: not trainable in memory |

**Datasets.**

| | LingoQA (driving) | EyePCR (ophthalmic surgery) | OpenUAV-QA (drone flight; optional) |
|---|---|---|---|
| Source | wayveai/LingoQA (Google Drive) | HF EvergreenTree/EyePCR | HF choucsan/OpenUAV-QA + ModelScope frames |
| Test | the official evaluation set: 500 questions x 2 reference answers, 100 segments | 3,492 multiple-choice questions of the official test cases (153 cases) | 3,893 questions (seen + unseen maps) |
| Train pool | Action + Scenery, 413,829 (segment, question) pairs | 39,504 questions (1,375 cases) | 11,443 questions |
| Frames | the 5 frames of the 4 s segment (1 fps), as a frame list | 16 uniform frames, `np.linspace(0, total - 1, 16, dtype=int)` | the official selection: sorted files, at most 32 by linspace, one video at 2 fps, pixel floor 200,704 |
| Prompt | the question verbatim (no official template) | the dataset's instruction + question + `CHOICE: [options]`, options reordered (below) | the official template (`UAV_PROMPT_TEMPLATES["default"]`) |
| Decoding | greedy, 256 new tokens | greedy, 512 | greedy, 128 |
| Scoring | the official Lingo-Judge: correct iff its max logit over the references is > 0 | the reply matched to an option's text (no official scorer; below) | the official `extract_answer` |
| Groups (clusters, leakage) | segment | surgery case | trajectory |
| Strata of the test subset | none (all 500 used) | the MCQ category | the map |

LingoQA is the only dataset whose fine-tuning gains clearly need the video (blank-frame control below); OpenUAV-QA's
tables are reported but the dataset is outside the primary endpoint (below).

**Test subsets and training orders** (`finetune prepare`, written once per dataset, identical for every model and
budget; seed 0):

- Test: all questions up to 2,000; above that a stratified sample over the strata keys, seats by largest remainder,
  within a stratum the questions in `sha256("0:<id>")` order. LingoQA 500, EyePCR 2,000 of 3,492, OpenUAV-QA 2,000 of
  3,893.
- Training pool: the training split minus every sample that shares a group (segment, case, trajectory) with a test
  question (none did), and for LingoQA minus samples with a frame whose 32x20 grayscale thumbnail correlates >= 0.95
  with a test frame (3 dropped; random training frames reach at most 0.93, frames of one test segment 0.93-0.98).
- Training order: the pool sorted by `sha256("0:<id>")`; LingoQA alternates Action and Scenery 1:1. The 1K, 2K and 5K
  budgets are the first 1,000 / 2,000 / 5,000 ids of that order (nested), each trained on its own.
- `selection.json` records the counts, what was dropped and the sha256 of both lists; every trained adapter's
  DONE.json repeats the two hashes.

**LoRA recipe** (identical for all 11 models; `lora.py`, `train.py`):

- LoRA r 16, alpha 32, dropout 0, no bias, on every linear layer of the language-model decoder (attention and MLP
  projections; a per-family pattern and an exact module count are checked, so a vision-tower layer with the same
  name is never adapted). LoRA weights are kept in fp32. The training seed is set right before injection, so parent
  and child of a pair start from the same A matrices (B = 0).
- Vision tower, projector, embeddings and LM head are frozen and stay in eval mode; gradient checkpointing.
- AdamW (0.9, 0.999, eps 1e-8), learning rate 1e-4, weight decay 0, linear warmup over 3% of the steps then cosine to
  0, gradient clipping 1.0, micro-batch 1 with 8 accumulation steps, 1 epoch.
- Loss: per sample the mean token cross-entropy over the answer and the end-of-turn token (fp32), averaged over the 8
  samples of a step. The training input of a sample is built by the model adapter's `encode()`, the same code that
  builds its evaluation input; a sample whose answer would change the tokenization of the prompt is refused.
- Determinism: training and the evaluation of these ids run with torch's deterministic algorithms,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `FLASH_ATTENTION_DETERMINISTIC=1` (LiveStar's direct flash-attn calls get
  `deterministic=True`). A checkpoint (LoRA + AdamW state + step) is written every 25 steps or 10 minutes; a resumed
  unit is bit-identical to an uninterrupted one.

**Evaluation ids.** For each model and dataset: `<model>@<dataset>-0` (zero-shot) and `<model>@<dataset>-1k|2k|5k`
(the trained adapters). They are ordinary model entries (`streamreplace finetune models` writes them as a models file):
the member's checkpoint and offline adapter, its pair's video protocol and the overrides above in the model's own
settings layer, and the options `deterministic: true` and, for a trained id, `lora: <adapter folder>`. The run
identity holds the sha256 of the adapter's weights (never the path), and the run worker attaches the adapter unmerged
on the freshly loaded weights (the offline adapter's `load_lora()`), so it computes exactly the trained function.

## Statistics (`finetune report`)

Accuracy in %; Δ = child − parent, so + means the streaming child is better.

- acc(b) for b = 0 (zero-shot), 1K, 2K, 5K; Gain(b) = acc(b) − acc(0); normalized gain at 5K =
  Gain / (100 − acc(0)).
- ΔAcc(b) = acc_C(b) − acc_P(b); ΔGain(b) = Gain_C(b) − Gain_P(b) = ΔAcc(b) − ΔAcc(0).
- **Primary endpoint**, per child and dataset: ΔGain averaged over the three budgets, per question
  ((C_b − C_0) − (P_b − P_0)) and then over questions. 95% interval: stratified cluster bootstrap (clusters = the
  dataset's groups, strata = its strata keys, 10,000 resamples, seed 0; the integer-index engine of
  `analysis/stats_index.py`); p: cluster sign-flip test (10,000 draws); Holm over the 12 cells of LingoQA and EyePCR.
  Exact McNemar tests are added to per-budget ΔAcc only where no two questions share a group.
- Errors count as wrong; a LingoQA answer counts once the Lingo-Judge has scored it.
- Multiple-choice sets also show the first-token letter accuracy (argmax over the option letters' logits).
- EyePCR's first-round parser gets its own accuracy table (sensitivity, below).

Paper values (budget-averaged ΔGain): LingoQA MOSS-VL-Realtime −7.13 [−12.00, −2.13] (Holm p 0.038; the only
significant LingoQA cell); EyePCR MMDuet +14.63, ThinkStream +11.22, AURA +4.10, MOSS-VL-Realtime +4.83 (Holm
significant), MMDuet2 and LiveStar not significant.

**Post-fine-tuning reading** (`READOUT.md`, `ft_tables/cells.csv`, `pooled.csv`). The same rows read through what
fine-tuning reaches: per cell ΔAcc(0), ΔAcc_avg (mean over the budgets), ΔAcc(5K) and ΔGain_avg = ΔAcc_avg − ΔAcc(0)
(the primary endpoint, exactly), each with its 95% interval, sign-flip p, one-sided 95% lower bound and
Bonferroni-12 simultaneous lower bound; Holm over the 12 primary cells and over all 18 cells. The worst simultaneous
lower bound of ΔAcc_avg over the 12 primary cells is −3.5 points (every child's budget-averaged fine-tuned accuracy is
at most 3.5 points below its parent's): the parity statement of the paper. A ΔGain cell whose ΔAcc_avg interval
includes 0 is read as a zero-shot difference rather than a difference in what fine-tuning reaches; this reading was
chosen after the results were seen (it follows from the identity above). Also reported: the equal-weight
cross-dataset estimate (each dataset bootstrapped with its own seed) and LingoQA without its fixed-format question
(20% of the evaluation questions ask for the current action and its justification in a fixed answer format, which the
training data never contains; a sensitivity view). Table 2's fine-tuning column is cells.csv's LingoQA ΔGain_avg (`analysis/render/gaps_json.py`).

**Controls** (each in its own folder under the experiment root, scored there):

- Blank frames (`blank`): the frames replaced by black frames of the same number and size, for the zero-shot and the
  5K ids, on 500 questions per dataset (the test-subset rule applied to the test subset). The 2x2 per model:
  VD(b) = acc_real(b) − acc_blank(b); Gain = Gain_blank + ΔVD exactly, with Gain_blank the gain kept without frames
  (text, format and answer priors) and ΔVD = VD(5K) − VD(0) the gain that needs the frames; the share of answered
  questions per arm; and per pair (exploratory, Holm within the table) ΔVis = ΔVD_child − ΔVD_parent.
  Findings: OpenUAV-QA is answered as well without frames (61-83% zero-shot, 98-100% after fine-tuning);
  EyePCR's gains come from text priors and the answer format (ΔVD ≤ 0); LingoQA depends on the frames (22-33
  points), already zero-shot.
- 1K for 3 epochs (`extra_units`, tag `e3`): the three p2 models on OpenUAV-QA (an under-fitting check).
- Image list (`images`): Qwen2.5-VL-3B zero-shot on OpenUAV-QA with the frames as separate images (the benchmark's
  official multi-image path) instead of one video.

**Training-seed replication** (`finetune seeds`, folder `seeds/`). The formal units are seed 0. Seeds 1 and 2 repeat
the LingoQA units of p3 and p5 with another LoRA initialisation and another order of the same training samples
(epoch 0 reshuffled by `sha256("<seed>:0:<id>")`); test questions, settings and the zero-shot rows are the formal ones.
Over the seeds: mean, SD and range of ΔAcc per budget and averaged over budgets, and a seed-inclusive 95% interval
mean ± 1.96·sqrt(SE_test² + SD²/k). Decision rules fixed before the replication was run: SD(ΔAcc avg) ≤ 1.5 points
with the interval inside ±3.6 → parity with total uncertainty; seeds more than 3 points apart → unresolved within seed
noise; the 2K and 5K units only if a pair's 1K SD exceeds 1.5 points (both did). Paper: AURA ΔAcc avg +1.2 (SD 0.8,
[−0.6, +2.9], parity); MOSS-VL-Realtime +0.3 (SD 1.5, [−2.2, +2.7], range 3.1: unresolved within seed noise).

## Running it

```bash
streamreplace prepare lingoqa eyepcr openuav_qa             # raw data -> prepared/ (docs/data.md)
streamreplace finetune prepare openuav_qa lingoqa eyepcr    # subset.json, train_order.json, selection.json (once)
streamreplace finetune check                                # equal to the paper's lists in paper/frozen/finetune?
streamreplace finetune eval --budgets 0 --gpus 0,1,2,3      # zero-shot evaluations (all models and datasets)
streamreplace finetune train --model aura_8b --dataset lingoqa --budget 5000 --gpus 0   # one unit per process
streamreplace finetune eval --gpus 0,1,2,3                  # every trained id (untrained ones are listed, skipped)
streamreplace finetune eval --control blank --gpus 0,1      # the controls, each in its folder
streamreplace --outputs <root> score --judge --judge-gpus 0  # and the same for <root>/blank, <root>/images
streamreplace finetune report                               # FT_RESULTS.md, READOUT.md, ft_tables/
streamreplace finetune eval --seeds --gpus 0                # after training the seed units: <root>/seeds
streamreplace --outputs <root>/seeds score --judge --judge-gpus 0
streamreplace finetune seeds                                # SEED_RESULTS.md, seed_tables/
```

`<root>` is `<outputs root>/finetune` (or `--root`). `streamreplace finetune commands --gpus 0,1,2,3` prints every
step of the whole experiment (training units longest first), one independent command per line, for a shell loop or a
batch scheduler; each line repeats the global options it was called with (`--config`, `--home`, `--paths`,
`--config-dir`), `--spec` and the experiment root as `--root`, so a trial root stays separate from the formal one.
`finetune units` shows each unit's state. A training unit runs in its model's environment on one GPU
(`train --launcher print` prints the command instead); it resumes from its checkpoint when restarted, and a second
process on the same unit exits at once. The evaluations are ordinary runs of the run pipeline: equivalently,
`streamreplace finetune models --out ft_models.yaml` and then
`streamreplace --config ft_models.yaml --outputs <root> run --models <ids> --protocols P,C --benchmarks lingoqa`.

`finetune prepare` needs the prepared training splits, and for LingoQA the frames of the near-duplicate check;
`finetune prepare --frozen` copies the paper's lists instead. OpenUAV-QA's preparation needs only the ModelScope file
listing; its frames are fetched afterwards for the selected trajectories only
(`streamreplace download datasets openuav_qa --allow-unpinned --frames-for <root>`, docs/data.md). LingoQA scoring needs the `lingo` environment
(`envs/lingo`, LingoQA's own pins) with the fetched LingoQA code (`streamreplace fetch-upstream lingoqa`) and the
Lingo-Judge checkpoint (`streamreplace download judges`).

Cost of the paper's run: about 17 (LingoQA), 24 (EyePCR) and 59 (OpenUAV-QA) GPU-hours of training plus 19 for the
seed replication, on 48 GB GPUs (peak 8-23 GB per unit); the adapters take 16 GB.

## Files

In the experiment root:

| Path | Content |
|---|---|
| `subset.json`, `train_order.json`, `selection.json` | the test subsets, training orders and selection records (write-once) |
| `adapters/<dataset>/<model>/<label>/` | `unit.json` (what the unit is; a resume must match), `loss.jsonl` (one line per step), `ckpt.pt` (while training), `adapter/` (PEFT safetensors + config), `DONE.json` (adapter sha256, data hashes, loss summary, versions), `payload.json`, `train.log` |
| `<dataset>/<model>@<dataset>-<label>.offline.jsonl` (+ `.meta.json`, `.scores.jsonl`) | evaluations, as for any outputs folder |
| `judges/` | the Lingo-Judge cache (`lingo.<config hash>.jsonl`: {key, score}) |
| `FT_RESULTS.md`, `ft_tables/{accuracy,pairs,blank_2x2_<dataset>}.csv` | the report |
| `ft_tables/results.json` | the data of the paper's fine-tuning tables (Supp. Tables eval_finetuning and eval_finetuning_gains and the per-pair summary), read back from FT_RESULTS.md, accuracy.csv and pairs.csv by the rules that produced the paper's numbers |
| `READOUT.md`, `ft_tables/{cells,pooled}.csv` | the post-fine-tuning reading |
| `blank/`, `images/` | the controls (own subset.json, evaluations, scores) |
| `seeds/` | the seed replication (adapters, evaluations, `SEED_RESULTS.md`, `seed_tables/{seeds,summary}.csv`) |

`paper/frozen/finetune/` holds the paper's `subset.json`, `train_order.json`, `selection.json` and the control
questions (`controls.json`): sample ids, counts and hashes only, no benchmark text. `finetune prepare` regenerates the
three lists byte for byte from the prepared data, and the controls' questions follow from the stated rule.

## Deviations and decisions made after seeing results

- **EyePCR has no official scorer.** A reply counts as an option when, after `normalize`, it equals the option's text
  or names exactly one option as a whole phrase (an option contained in another named option does not count). Two
  rules were added after the first zero-shot results: the option a marked answer starts with ("The correct answer is
  'X'. ..." followed by text naming other options, ThinkStream's habit) and a lone letter ("C", "The answer is C.")
  read as the position in the listed order (MOSS and ThinkStream answered with letters that were right far above
  chance). The first-round parser is kept as the `SENSITIVITY` table; under it ThinkStream's primary endpoint is +19.1
  and MOSS-VL-Realtime's +0.7 (not significant). The options of every question are reordered by
  `sha256("<id>:<option>")`: in the released order the answer is the third option for 40% of the questions, a
  position prior that fine-tuning would pass off as adaptation. Questions whose answer is no option are dropped (16
  test, 228 train; an answer equal to an option after normalisation is respelled as that option) and so are questions
  with two equal options. Token budget 512 (128 cut ThinkStream's reasoning).
- **OpenUAV-QA left the primary endpoint** after the results: every model reaches 98-100% after fine-tuning, also
  with blank frames, and choosing "the option most similar to the other three" without any model gives 94.0% (the
  distractors are one-step edits of the correct path). Its tables are kept. Its frames come from ModelScope, which has
  no revision pin and serves some listed files empty (such trajectories are skipped, as the official loader does).
- **LingoQA** is scored with the official classifier but not through evaluate.py (its input path fails on the
  released data); references come from the official parquet. No official prompt template or token budget exists.
- **MMDuet merged, p1 in bf16, MOSS pixel cap**: see Models. These make the zero-shot numbers differ from the main
  tables by design.
- The post-fine-tuning reading rule (above) was chosen after the results.
- One training seed for the formal units; the seed replication covers LingoQA for p3 and p5 only.

## Licences

LingoQA's data and code are licensed for personal, non-commercial use without sublicensing: nothing of LingoQA is
shipped (ids only), the code is fetched from its repository and the data downloaded by the user. EyePCR is CC-BY-4.0.
OpenUAV-QA's code (Self-in-Space) is Apache-2.0. See docs/licensing.md.
