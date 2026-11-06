# Prompt robustness

Does streaming adaptation make a model more sensitive to how a question is worded? This experiment perturbs the
wording of questions (and, separately, of the answer instruction) with the perturbation types of PromptRobust (Zhu et
al., 2023) and compares each streaming child's accuracy loss with its offline parent's on the same perturbed prompts.
It feeds the three prompt-robustness columns of Table 2, Supp. Tables `eval_robustness` and `eval_instruction`, and
the robustness paragraph of Sec. 4.3.

Everything runs **offline** (parent and child share the input pipeline and differ only in their weights), with the
main evaluation's frames, decoding and token budgets. Code: `src/streamreplace/generalization/robustness/`;
specification: `configs/generalization/robustness.yaml`; frozen ids and digests: `paper/frozen/robustness/`.

## Design

**Items.** 300 Video-MME short questions and 300 MMBench DEV-EN questions, the same for every model and condition
(seed 0):

- Video-MME: one question per video, the one whose `sha256("0:<sample id>")` is smallest (300 videos, 300 units).
- MMBench: 300 questions by largest-remainder allocation over the `l2` category (attribute reasoning 51, coarse
  perception 76, fine-grained cross-instance 37, fine-grained instance-level 76, logic reasoning 30, relation
  reasoning 30); within a category in `sha256("0:<id of the question's first rotation>")` order. All circular
  rotations of a chosen question are kept (1,118 rows).

A **unit** is one item of the analysis: a Video-MME question, or an MMBench question that counts as right only if
every rotation is answered right (the official circular rule). Clusters for the bootstrap are Video-MME videos and
MMBench questions.

**What is perturbed.** Only the question stem: Video-MME's `question`, MMBench's raw question. The official
instruction, the options, the MMBench hint and the answer format stay byte-identical, so the benchmarks' own parsers
and scoring apply unchanged, and the judge requests still read the clean question. All rotations of an MMBench
question share one perturbed stem.

| Condition | PromptRobust level | Edit |
|---|---|---|
| `clean` | – | none (a rerun of the main evaluation on these items) |
| `typo` | character (DeepWordBug) | DeepWordBug's four character operations (swap adjacent letters, substitute, delete, insert) on `round(0.2 × eligible words)` randomly drawn words, at least 1 and at most 4. The first letter is never touched; substituted and inserted letters are QWERTY neighbours; case is preserved. Eligible words: at least 3 ASCII letters, not all capitals (acronyms, option letters), not inside quotes, `<tags>` or tokens with digits |
| `para` | semantic | back-translation through Chinese by a language model, verified (below); a unit whose paraphrase fails is skipped for every model |
| `stress` | sentence (StressTest) | PromptBench's StressTest text `" and true is true " * 5`, stripped and appended after the stem's last word |
| `irrelevant` | sentence (natural distractor) | one of 50 everyday personal sentences (`irrelevant.txt`: no numbers, colours, people descriptions, animals, vehicles, clothing, spatial or time-order words): the first one, in `sha256("0:<sentence>")` order, that shares no word of 4+ letters with the stem and the option lines, appended after the stem |

Draws are seeded with the string `"<seed>|<dataset>|<unit>|<condition>"` through Python's `random.Random(str)`
(SHA-512 seeding), so they are identical on every machine.

**Paraphrases.** The stem is translated into Chinese and back into English by a chat model (temperature 0, thinking
disabled). The candidate must pass rule checks (word-level edit distance at least 20% of the longer text, the same
numbers, length within 0.5-2x, and it must not newly mention the correct option's text) and the model must answer
"yes" to "Do the two versions ask exactly the same thing, so that the correct option is the same for both?". If the
back-translation fails either check, one direct "reword this question" request goes through the same checks. In the
paper 579 of 600 units were accepted (319 back-translations, 260 direct rewordings); 6 Video-MME and 15 MMBench units
were skipped, leaving 294 and 285.

**Instruction level** (a side experiment, MMBench only). The same edits are applied to MMBench's official instruction
"Answer with the option's letter from the given choices directly." instead of the stem (stem, options and hint
unchanged), on a nested, stratified 100-question subset (373 rows) and on all 300 questions:

| Condition | Stem twin | Edit |
|---|---|---|
| `ins_typo`, `ins300_typo` | `typo` | the typo operations on the instruction |
| `ins_para`, `ins300_para` | `para` | one of 8 hand-written rewrites that keep the demand (the letter, directly), picked by `int(sha256(seed), 16) % 8`; every rewrite passes the paraphrase rule checks |
| `ins_irrelevant`, `ins300_irrelevant` | `irrelevant` | an irrelevant sentence sharing no 4+ letter word with the stem, options or instruction, appended to the instruction |

The 300-question conditions draw their edits with the 100-question conditions' seeds (`seed_as`), so the 100 shared
questions get identical edits and their replies replicate the earlier run. Video-MME states its answer format twice,
so it has no single instruction sentence.

## Running it

Each condition is an ordinary outputs folder under the experiment root (`<outputs root>/robustness` by default)
holding `subset.json` and, except `clean`, `prompt_overrides.jsonl`: one row per sample,
`{"dataset", "id", "unit", "condition", "edits": [[old, new]], "skip"}`. Every run on that folder applies the edits
to the rendered prompt through the runtime's prompt-override hook (`data/prompts.py`): each `old` must occur exactly
once, so an edit can only touch the span it names, and the edits enter the run identity. The runner knows nothing
about perturbations.

```bash
streamreplace prepare videomme mmbench     # raw data as for the main evaluation (docs/benchmarks/)
streamreplace robustness subset            # subset.json in every condition folder (written once)
streamreplace robustness paraphrase --endpoint http://127.0.0.1:8000/v1 --model judge   # see below
streamreplace robustness edits             # prompt_overrides.jsonl (written once)
streamreplace robustness check --frozen    # edits only touch the stem; ids and edit files equal the paper's
streamreplace robustness audit             # optional: audit/<condition>.md for a manual read
streamreplace robustness runs              # prints, per condition folder:
streamreplace --outputs <root>/typo run --pairs p1,p2,p3,p4,p5 --protocols P,C --benchmarks videomme,mmbench
streamreplace --outputs <root>/typo score --judge
streamreplace robustness report            # <root>/report/
```

Written files are never replaced by different content: delete a folder's outputs before changing its edits.
`check --frozen` compares the item lists with `paper/frozen/robustness/items.json` and the sha256 of every override
file with `paper/frozen/robustness/edits.json`. With the prepared data, the typo, stress, irrelevant and every
instruction-level file regenerate **byte for byte**.

**Paraphrases are not exactly regenerable.** The paper's paraphrases came from the same Qwen3.5-family 27B FP8
instruct checkpoint as the extraction judge (`configs/judges.yaml`), served by vLLM; any OpenAI-compatible endpoint
can run the procedure (`paraphrase` settings in the specification, or `robustness: {paraphrase: {endpoint: ...}}` in
`configs/local.yaml`), for example `vllm serve <checkpoint> --served-model-name judge`. Another model, revision,
serving stack or GPU produces other paraphrases and other skips. Results are cached in
`<root>/paraphrase_cache.jsonl`, keyed by a hash of (dataset, unit, stem, pivot language); with the paper's cache in
place, `edits` reproduces the paper's `para` file exactly (sha256 in `edits.json`). The benchmark text itself is not
shipped, so the repository holds only ids, skip reasons and digests.

The offline adapters record the first generated token's logits of the option letters (`info.letter_logits`) for
every multiple-choice request; the probability readout below needs them. Cost in the paper: about 6.6 hours on 8
48-GB GPUs for the five question-stem conditions (about 53 GPU-hours); the clean rerun reproduced the main
evaluation's replies exactly for the nine deterministic models, while the LiveStar pair showed about 1% run-to-run
change (non-deterministic GPU kernels in its vision encoder).

## Analysis (pre-registered)

- acc(m, c) on units; errors and unparsed replies count as wrong (rule-parse failures on MMBench go to the extraction
  judge as in the main evaluation). Drop(m, c) = acc(m, clean) − acc(m, c); PDR(m, c) = 1 − acc(m, c) / acc(m, clean).
- **ΔRob** = Drop_parent − Drop_child, computed per unit as (P_clean − P_c) − (C_clean − C_c): **+ = the child loses
  less, − = the child loses more**. A family's value is the per-unit mean over its conditions (typo; para;
  distraction = stress and irrelevant). Units a condition skipped leave it for both models.
- **Primary endpoint:** per child × family, pooled over Video-MME and MMBench (bootstrap strata): the mean, a 95%
  stratified cluster bootstrap interval (10,000 draws, numpy seed 0, whole clusters resampled by index within each
  stratum), a cluster sign-flip p (10,000 flips) and Holm over the 18 tests (6 children × 3 families). This is the
  integer-index engine of `analysis/stats_index.py`, the engine these numbers were reported with (docs on the two
  engines: `docs/statistics.md`). All 18 Holm-adjusted p values are 1.
- **Bounds and readouts:** the one-sided 95% lower bound and the Bonferroni-18 simultaneous lower bound of the same
  bootstrap, TOST at ±3 points (90% interval inside), and the same contrast on the **gold-option probability**: the
  softmax of the first token's logits restricted to the item's option letters, the gold letter's share averaged over
  an MMBench question's rotations (Holm-18 within this readout). ΔRob_PDR = PDR_parent − PDR_child on the same draws.
- Per benchmark: the same contrasts on one benchmark (uncorrected p); accuracy, Drop and PDR per model and condition;
  diagnostics: C→W and W→C against clean, answer change, answered, bare-letter replies and mean reply tokens.
- **Instruction level**, per group of conditions run on the same items (the 100- and the 300-question runs, each with
  its own Holm family over 6 children × 3 conditions): items present in clean, the condition and its stem twin;
  per model the level effect Drop_instruction − Drop_stem (+ = the instruction edit hurts more); ΔRob for the
  instruction condition and for its twin on the same items, and their difference (DiD); for the 300-question run, the
  share of identical raw replies on the 100 earlier questions and ΔRob on the earlier vs the new questions
  (descriptive). Both readouts with bounds for every instruction condition, Holm over all instruction probability
  tests.

## Outputs of `robustness report`

| File | Content |
|---|---|
| `ROBUSTNESS.md` | all tables of the analysis above |
| `results.json` | `primary` (per child × family: ΔRob, both readouts with bounds, PDR contrast), `robustness_by_benchmark` (Supp. Table `eval_robustness`: per child × benchmark, clean accuracies and each family's ΔRob, `rob_tax = −ΔRob`), `per_benchmark` (accuracy, per-condition ΔRob, diagnostics), `instruction_robustness` (Supp. Table `eval_instruction`: per group and child, each condition's ΔRob keyed by its stem twin, with matched clean and perturbed accuracies), `instruction_contrasts` |
| `items.csv` | unit outcomes: condition, dataset, model, unit, cluster, correct, the parsed answers of the unit's samples |
| `primary.csv` | the pooled ΔRob table: pair, parent, child, family, `d_rob_pp`, low, high, p, `p_holm`, items |
| `contrasts.csv` | pair, parent, child, family, readout (`acc` / `prob`), n, value, low, high, p, `lower_0.05`, `lower_0.00277778`, margin, `tost_p`, equivalent, low90, high90, `p_holm`. Table 2 shows, per child, the family with the lowest accuracy ΔRob (`analysis/render/gaps_json.py`) |
| `instruction.csv` | pair, child, condition, readout, n, value, low, high, p, `lower_0.05`, `lower_0.00277778`, `p_holm` |

All values are in percentage points. The report reproduces the paper's files exactly from the original runs' rows:
`items.csv` and `primary.csv` byte for byte, the supplement tables' values, and the Table 2 cells.

## Results in the paper

Pooled ΔRob (points, 95% interval; + = the child loses less); every Holm-adjusted p is 1:

| Pair | Child | Typo | Paraphrase | Distraction |
|---|---|---|---|---|
| LLaVA-OneVision-7B | MMDuet | +0.7 [−1.0, +2.3] | +0.7 [−1.2, +2.8] | +0.1 [−1.7, +1.8] |
| Qwen2.5-VL-3B | ThinkStream | +0.5 [−1.2, +2.2] | +1.9 [−0.2, +4.0] | +1.5 [−0.2, +3.2] |
| Qwen2.5-VL-3B | MMDuet2 | −0.2 [−1.8, +1.5] | +1.2 [−1.0, +3.5] | −0.2 [−1.9, +1.4] |
| Qwen3-VL-8B | AURA | +0.2 [−1.3, +1.7] | −0.5 [−2.1, +1.0] | +0.4 [−1.2, +2.1] |
| LiveStar-InternVideo-8B | LiveStar | +0.3 [−1.8, +2.5] | −1.6 [−3.6, +0.3] | −0.8 [−2.5, +0.8] |
| MOSS-VL | MOSS-VL-Realtime | +1.7 [−0.2, +3.5] | −0.9 [−2.8, +1.0] | −0.8 [−2.8, +1.1] |

On the gold-option probability readout, the only Holm survivor is ThinkStream under distraction (−0.95 points);
under instruction edits ThinkStream's probability ΔRob reaches −2.2 points (100-question `ins_irrelevant`) while no
accuracy contrast is Holm-significant at either instruction-level size.

## Differences from PromptRobust

1. Non-adaptive perturbations: every model sees the same perturbed text instead of a per-model adversarial search,
   so parent and child form a paired comparison. The character operations are therefore implemented here (the
   official recipes rank words by querying each model and depend on TextAttack).
2. The question stem, which the user writes in visual question answering, is perturbed rather than the task
   instruction (the instruction level is a separate side experiment).
3. Paraphrases come from a local open-weight model by back-translation through Chinese, not from GPT-4 and Google
   Translate.
4. No word-level attacks (TextFooler, BertAttack): synonym substitution often changes the meaning of a visual question.
