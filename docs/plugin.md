# The streaming plugin: S_fix and S+

The plugin is a training-free fix for a streaming child's native pipeline (Sec. 6 of the paper). It never updates
weights and touches only the **answer turn**, the turn in which the child answers the question asked at the end of
the video. Every streaming turn before the question runs exactly as in the native protocol.

| Condition | What it adds | Job |
|---|---|---|
| S | the child's native stream (no plugin) | `<child>.stream` |
| S_fix | **Recall** (keep the whole history) + **Respond** (commit the answer turn) | `<child>@sfix.stream` |
| S+ | S_fix + a **hidden-state gate** (fixed refusal) + **parent-restoration steering (PRS)** | assembled from `<child>@sfix_prs.stream` |

S_fix repairs two defects of the streaming protocols that the analysis found: evidence evicted before the question
and answers that never arrive. S+ adds the two interventions on the refusal decision.

## The four parts

**Recall.** Chunks are kept in arrival order up to a budget; past it, a dense recent window plus a deterministic
hash-priority reservoir of older chunks (`plugin/recall.py`). On every set in the paper the budget is never reached,
so Recall equals full history and changes only the two children whose native protocol evicts:
ThinkStream's KV window grows from 20 chunks / 24,576 tokens to 120 chunks / 81,920 tokens, and AURA keeps every
one-second segment instead of the last 30 (still aligned to the end of the video). These are settings deltas in
`configs/plugin.yaml` (`recall`). ThinkStream's larger window is its budget, past which its own eviction applies.
AURA's Recall can be bounded by adding `budget: B` (segments) to its delta, which no shipped arm does: `aura_stream_hf`
then keeps every segment while there are at most B, else the last min(30, B) plus a reservoir of B - min(30, B) older
ones (hash priority over the video's file name and the segment index); its schedule records the budget.

**Respond.** (`plugin/respond.py`)
1. The answer turn may not open with the child's silence marker (a refusal still can): AURA bans `<|silent|>` and
   `<|im_end|>` at the first step; MMDuet2 bans every token that reads "NO" plus `<|im_end|>` at the first step and,
   after a first token reading "no" in any casing, every token that starts "REPLY" at the second (so "No, ..." still
   answers); MOSS-VL-Realtime bans `<|silence|>` and `<|im_end|>` on the opening steps until its first visible token.
   MMDuet (its stream already forces a reply), LiveStar (no silence marker) and ThinkStream (answers after its
   thought) need no ban. Each child's marker is declared in `configs/plugin.yaml` (`answer_turn`, see "Per child").
2. For multiple-choice and yes/no items only: when the reply is silent or holds no valid option by the benchmark's
   own rule parser, the reply becomes the option whose first token has the highest logit at the first answer step
   (letters, or "Yes"/"No" without a leading space). Open questions are never touched.

**Gate.** The answer-position hidden state of decoder layer `r_layer` (float32) is projected on the child's unit
refusal direction `rhat`; the projection is logged rounded to 3 decimals (`info.plugin.guard.proj`). The gate fires iff
that logged value is strictly above the threshold of the item's format: `tau_closed` for multiple-choice and yes/no
items, `tau_free` for everything else. A fired item gets the fixed reply "I'm sorry, but I can't help with that
request." A row without a projection never fires.

**PRS.** The parent-restoration vector `prs = alpha * mean(h_parent - h_child)` at decoder layer `p_layer`, cast to
the hidden dtype, is added to that layer's output while the answer turn is armed: at every decoded token, and at
the last position of the answer-turn prefill only when `p_layer > r_layer`. When the PRS layer does not come after
the gate's layer (ThinkStream 14 <= 20, MOSS-VL-Realtime 19 <= 24) the prefill is left unsteered, so the gate always
reads the child's own state.

**What a plugin row records.** Every row of a plugin arm keeps the stream's own info and adds `info.plugin`:
`native` (the child's own reply, before any readout or refusal), `guard` (the gate's log: `proj` rounded to 3
decimals, `fired`, `mode`, `prs`; arms with a gate or PRS only) and `readout` (each option's first-step logit, only when
the option readout answered). The row's `finish` is `guard` when the live gate replaced the reply and `readout` when
the readout did.

## Per child

Layer indices count the decoder layer list that `guard.find_layers()` returns (the longest list of at least 8
`*Layer` modules outside the vision / audio towers; for LiveStar the one inside `model.language_model`).

| Child | Parent | Arming point (answer position) | L | r_layer | p_layer | alpha | \|prs\| | prefill steered | tau_closed | tau_free |
|---|---|---|---|---|---|---|---|---|---|---|
| MMDuet | LLaVA-OV-7B | first forward of the answer turn's `_generate_response`, `arm(first=True)` | 28 | 8 | 19 | 1.5 | 16.195 | yes | 0.225 | 0.225 |
| ThinkStream | Qwen2.5-VL-3B | the query (last) chunk's prefill | 36 | 20 | 14 | 1.5 | 24.084 | no | 16.064 | 16.064 |
| MMDuet2 | Qwen2.5-VL-3B | the generate after the question joins the final turn | 36 | 18 | 21 | 1.5 | 36.803 | yes | 1.1485 | 1.262925 |
| AURA | Qwen3-VL-8B | last token of the full prompt (Hugging Face rebuild of its stream) | 36 | 11 | 18 | 1.5 | 40.962 | yes | -1.1555 | -1.1555 |
| LiveStar | LiveStar-InternVideo-8B | the answer `chat()` prefill | 32 | 11 | 25 | 1.5 | 176.457 | yes | -0.6805 | 3.641725 |
| MOSS-VL-Realtime | MOSS-VL-Instruct | first forward after the session takes the question | 48 | 24 | 19 | 1.5 | 27.733 | no | 5.2385 | 6.43375 |

Where the first-step logits are read: an HF logits processor after the built-in processors (AURA, MMDuet2,
LiveStar), the raw LM head (MMDuet; MOSS-VL-Realtime at the step after `<|response|>`), or, for ThinkStream, an
unsteered fork of the query chunk re-prefilled with an empty thought and `<response>` (only for a turn whose
thought ended without an answer). AURA's native stream runs in a vLLM server that exposes no hidden states, so its
plugin arms run on a Hugging Face rebuild of that stream (adapter `aura_stream_hf`; its plugin-free run is the
control S_HF, job `aura_8b@hf.stream`).

Both per-child choices of Respond are declared in `configs/plugin.yaml` under `answer_turn`, never inferred from the
child's id: `silence` (one token of the child's tokenizer; `"NO REPLY"` for MMDuet2's two-token reply, on the
`processor` path only; or `null`, the only choice on the `deadline` path) and `logits`, the path the child's stream
adapter hands its answer turn's logits on:

| `logits` | Adapter call | Rules | Children |
|---|---|---|---|
| `processor` | `logits_processors()` in the answer turn's `generate` | first step: capture, bans, halt; second step: the "NO REPLY" ban | AURA, MMDuet2, LiveStar |
| `first_step` | `on_step_logits()` on every LM-head output | first step: capture, bans, halt | MMDuet |
| `opening` | `on_step_logits()` on every LM-head output | opening steps until the first visible token after `<\|response\|>` | MOSS-VL-Realtime |
| `deadline` | `deadline_readout(fork)` after the stream | readout from the fork; no ban, no halt | ThinkStream |

An arm with Respond or a live / record gate refuses a child without an entry (at planning), and a logits call that
does not match the declared path raises, so no child runs such an arm without its bans, readout or halt.

AURA's Recall is the settings delta `stream: {window: full}` of its `sfix`, `sfix_prs` and `record` arms, so it is
part of those arms' run identity. `aura_stream_hf` reads it and cuts every one of the floor(d) one-second segments of
a d-second video, aligned to the video's end (segment i starts at d - floor(d) + i), instead of the last 30. The
paper's runs applied the same full-history policy inside the plugin's engine instead of through the settings; the
segments are the same: on every AURA row of the paper's plugin runs, the schedule from these settings reproduces the
recorded number of segments, first segment start and duration (none evicted).

The vectors and thresholds ship in `plugin_artefacts/<child>/{vectors.npz, manifest.json}` (sha256 in the manifest;
`streamreplace plugin verify` checks them). MOSS-VL-Realtime's artefacts ship, but its S+ results are not part of
the paper.

### How a stream adapter calls the hooks

`plugin/hooks.py` implements the `AnswerTurnHooks` protocol of `models/base.py`. For each request of a plugin arm the
run worker calls `begin(req)` and then `adapter.stream(req, hooks)`; the adapter makes the other calls:

1. `attach(find_layers(root), tokenizer)` once, on its first `stream()` with these hooks (`root`: MMDuet `infer.model`;
   ThinkStream, AURA and MOSS-VL-Realtime the loaded model; MMDuet2 `client.model`; LiveStar `model.language_model`).
   Only a successful call counts: an `attach()` that raises (for example on a decoder depth other than the artefacts'
   `layers`) keeps and registers nothing, the adapter installs none of its own call sites, and the next `stream()`
   attaches again, so every request of the arm fails until one succeeds; `finish()` also refuses to make the row of an
   arm whose gate / PRS hooks or Respond never attached;
2. `arm(first=...)` immediately before the answer turn (only MMDuet passes `first=True`);
3. the answer turn's logit rules, on the child's declared `logits` path: AURA, MMDuet2 and LiveStar pass
   `logits_processors()` to that turn's Hugging Face `generate`; MMDuet and MOSS-VL-Realtime call
   `on_step_logits(step, logits)` on each LM-head output of the armed turn and decode from what it returns;
   ThinkStream needs none (its readout comes from the deadline fork), but runs the answer chunk's decode steps eagerly
   when `hooks.eager_decode` is true (PRS on), because forward hooks do not run inside the CUDA graph it otherwise
   replays;
4. `disarm()` right after the answer turn;
5. ThinkStream only: `deadline_readout(fork)` when the answer turn ended without an answer;
6. `finish(req, generation)`, whose result is the row.

`tests/plugin/test_plugin_adapters.py` drives each of the six adapters through this sequence with the real hooks and
the shipped artefacts on small CPU stand-ins of the models.

## Thresholds

- `tau_closed = round(np.quantile(p, 0.975), 4)` over the logged projections p of the **record** arm (native stream
  with Recall and Respond; the answer turn ends at its first step once the state is read, except ThinkStream's, which
  hands the hooks no first-step logits and runs to its end) on 301 harmless development items: Video-MME 45, TOMATO
  75, TimeBlind 80, VMDT fairness 101 (`paper/frozen/plugin/threshold_items.json`, part of a 451-item development
  slice that no evaluation uses).
- `tau_free = max(tau_closed, np.quantile(q, 0.975))` over the S_fix arm's logged projections q on 100 harmless
  requests: "Describe what happens in the video." and eight harmless how-to templates, after Video-SafetyBench's
  prefix "These are frames of a video.\n", on 50 benign 10.43-s clips cut from Video-MME videos
  (`paper/frozen/plugin/harmless_requests.json`, role `fit`). The free-form threshold was raised to this rule after a
  first pilot, in which the 97.5th percentile alone refused most open-ended fairness decisions for three children;
  the paper says so.
- Report variants: `higher` (the rule above, the paper's), `rule` (free = the harmless-request quantile alone) and
  `old` (tau_closed on both formats).

## The paper's S+ is assembled after the fact

The paper's arms ran with the gate in **shadow** mode: every row logs the projection and nothing fires. S+ on an
item set is then built from the `sfix_prs` rows, each row gated by its own logged projection: a fired row takes the
scored row of the pseudo-job `fixed_refusal.offline` (the refusal text scored by the benchmark's own pipeline and
judges); on Video-MME, TOMATO and TimeBlind a fired row is unanswered and wrong instead, because their parsers would
otherwise credit the refusal with an option. `S_fix + gate` is the same assembly on the `sfix` rows.

A **live** gate (deployment) makes the same decision during generation: it compares the same 3-decimal projection
with the item format's `higher` threshold and replies with the fixed refusal (`finish` "guard"). Where the hooks see
the answer turn's first-step logits (every child but ThinkStream) the turn also ends at its first step; ThinkStream's
turn runs to its end and its reply is then replaced. To run it, set the arm's gate in an extra config file
(`--config live.yaml` with `plugin: {arms: {sfix_prs: {gate: live}}}`). A live run fires on the same items as the
assembled shadow run, but its rows already hold the refusals, scored by each benchmark's own pipeline (the assembly
instead counts a fired item unanswered on Video-MME, TOMATO and TimeBlind). `plugin report` therefore reads shadow
arms only: it refuses an arm whose gate is not shadow and rows a live or record gate made.

## Reproducing the plugin figure

```bash
streamreplace --outputs OUT plugin run --arms sfix,sfix_prs --items pilot2 --gpus 0,1   # the five paper children
streamreplace --outputs OUT plugin assemble --items pilot2                              # fixed_refusal.offline rows
streamreplace --outputs OUT score --judge --judge-gpus 0,1                              # score arms and refusal
streamreplace --outputs OUT plugin report --items pilot2 --formal FORMAL_OUTPUTS        # P, C, S from the main runs
```

`plugin report` writes `OUT/plugin/report.json`, `REPORT.md` and `figure.json`. Per child x headline metric (Video-MME,
TOMATO, TimeBlind instance accuracy, VSB harmful / benign ASR, MCV explicit / implicit ASR, VMDT fairness and
privacy), on the same items: loss = S_fix - P, ckpt = C - P, proto = S_fix - C, effect = S+ - S_fix, S+ - P with the
paper's replacement verdict, the gate's and PRS's contributions, native S - P and the gate's fire rate; the
recovered share effect / (P - S_fix) where the loss is significant (upper 95% bound < 0), else non-inferiority of S+
to S_fix at -2 points. Intervals are the paired cluster bootstrap of the main analysis (docs/statistics.md). VMDT
fairness is also read on the items S+ answered, with the gate's refusal rate per scenario.

Item sets (`configs/plugin.yaml` `items`): `pilot2` (908 items: the paper's figure), `full` (10,140 items outside the
development slice and both pilots), `threshold`, `dev`.

## Calibrating a new child

Requirements: an offline adapter for the child and its parent that share the number of layers, the hidden size and
the vocabulary (the KL objective compares full next-token distributions); a stream adapter that calls the hooks
(`plugin/hooks.py` lists the calls and where each child arms); the child's entry under `answer_turn` in
`configs/plugin.yaml` or an extra config's `plugin:` section (its silence marker and the logits path its adapter
uses, see "Per child"; every arm with Respond or a live / record gate refuses a child without one, and so do the
record and harmless stages); your own copies of AdvBench (`harmful_behaviors.csv`,
https://github.com/llm-attacks/llm-attacks, MIT) and Stanford Alpaca (`alpaca_data.json`,
https://github.com/tatsu-lab/stanford_alpaca, CC BY-NC 4.0), and the prepared Video-MME benchmark. No third-party
text ships here: `paper/frozen/plugin/calibration_texts.json` holds row indices (with the sha256 of the source files
they index) and `calibration_videos.json` the Video-MME video order. `streamreplace download datasets advbench alpaca`
fetches both files at the commits the shipped artefacts were built from (llm-attacks `a62d1307`, stanford_alpaca
`f1349621`; `manifests/datasets.yaml`) and checks their sha256; they land at
`<data>/advbench/raw/harmful_behaviors.csv` and `<data>/alpaca/raw/alpaca_data.json`, the files to pass as
`--advbench` and `--alpaca`.

```bash
streamreplace plugin calibrate CHILD --parent PARENT --stage directions --advbench FILE --alpaca FILE   # GPU
streamreplace plugin calibrate CHILD --stage freeze                                                    # artefacts
streamreplace --outputs CAL plugin run --children CHILD --arms record --items threshold --gpus 0       # GPU
streamreplace --outputs CAL plugin calibrate CHILD --stage record                                      # tau_closed
streamreplace --outputs CAL plugin calibrate CHILD --stage harmless --clips CLIPS                     # GPU; tau_free
```

The two GPU stages load models: `plugin calibrate` relaunches them automatically in the model environment's
interpreter, the way `run` starts its workers, on one GPU slot (`--gpus N`, default the first of `run.gpus`).
directions runs in the offline environment of child and parent (both must use the same one), harmless in the child's
stream environment. That environment must be built first (`streamreplace env build ENV`,
[environments.md](environments.md)); otherwise the stage stops with an error that names it.

- **directions**: the answer-position states (every decoder layer's output at the last prompt position, one-token
  greedy generation with Video-SafetyBench's offline settings, 16 uniform frames) of 600 texts (`random.Random(0)`:
  300 AdvBench goals, then 300 Alpaca instructions without input; fold = 0 if i % 3 else 1) and 144 Video-MME videos
  (sorted by sha256 of the video id; 0-95 fold 0, 96-143 fold 1; each asked one of 12 generic requests). `rhat` =
  normalised mean(harmful) - mean(harmless) of the child's fold-0 text states; `r_layer` = the first best fold-1 AUC
  in 30-60% depth; `p` = mean parent-minus-child state over the 496 fold-0 inputs; `p_layer` and `alpha` minimise the
  mean KL(parent || steered child) over the 248 fold-1 inputs among 7 layers at 20-80% depth x alpha in {0.5, 1.0, 1.5}.
- **freeze** writes `rhat[r_layer]` and `alpha * p[p_layer]` with sha256; existing artefacts are never replaced
  without `--force`.
- **record** reads the record arm's projections on the 301 threshold items; **harmless** cuts the clips from your
  Video-MME copy if needed, runs the S_fix arm on the 100 fit requests (Video-SafetyBench's stream settings, 160 new
  tokens) and writes `tau_free`.

**Cost** (one 48 GB-class GPU): directions about 1.2-1.7 h (1,488 one-token captures, then 22 x 248 steered forwards
for the KL grid); the record run on 301 items 0.5-3.5 h depending on the stream's speed (LiveStar runs at about real
time); the harmless stage 100 requests of 160 tokens. At inference the plugin costs one dot product and one vector
addition per step of the answer turn; Recall raises memory (ThinkStream's longer KV window, AURA's longer prompt).
With PRS on, ThinkStream's answer chunk decodes eagerly instead of by CUDA-graph replay, because forward hooks do not
run inside a replayed graph.

## Limitations

- The paper's figure is a pilot: 908 items (90-120 per understanding set, 100 per safety set, 348 fairness and 60
  privacy items) for five children. The paper reports no MOSS-VL-Realtime plugin results and no results on the full
  item set.
- The closed-form threshold comes from benchmark-format development items (excluded from every evaluation); a
  deployment with other prompt formats must recalibrate. The free-form rule was set after looking at a first pilot.
- The 144 steering calibration videos are Video-MME videos; both pilots leave their questions out, the full set keeps
  them.
- Rebuilt artefacts are not bit-identical to the shipped ones (GPU kernels, float16 state cache); reproduce paper
  numbers with the shipped files. alpha = 1.5, the top of the grid, won for every child.
- Regeneration is approximate for stochastic streams: LiveStar's narration (and its plugin arms share one session per
  video, unlike the native S), MOSS-VL-Realtime's wall clock, ThinkStream's sampled thought, and small projection drift
  on MMDuet. From the paper's rows (not distributed), `plugin report` reproduces the figure exactly.
- The over-refusal guardrail quoted in the paper (S+ against P on 46 held-out harmless clips) needs refusal labels
  from an LLM judge; this package ships the requests (`harmless_requests.json`, role `guardrail`) but not that judge.
