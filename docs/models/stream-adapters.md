# Native streaming adapters (condition S)

Condition **S** runs each streaming child through its own streaming interface: the video is replayed on the media
clock and the benchmark question is asked **at the end of the video**. S - C then isolates the change of protocol
(the same checkpoint in the offline pipeline is condition C). This page describes, for each of the six children, what
its stream is fed, how the question enters, how the reply is read, what counts as silence, and where the study's
procedure departs from the child's official evaluation.

One adapter module per child lives in `src/streamreplace/models/`:

| Child (pair) | Adapter | Environment | Upstream code (fetched at the pinned commit) |
|---|---|---|---|
| MMDuet-7B (p1) | `mmduet_stream` | `llava` | MMDuet `demo/liveinfer.py`, `test/inference.py`, `models/*`; LLaVA-NeXT |
| ThinkStream-3B (p2) | `thinkstream_stream` | `modern` | ThinkStream `thinkstream/model/inference.py`, `thinkstream/data/*` |
| MMDuet2 (p2) | `mmduet2_stream` | `mmduet2` | MMDuet2 `demo/api_server.py`, `demo/frontend.py`, `demo/inference.py`, `proactive_eval/qwen2_5_vl/*` |
| AURA-8B (p3) | `aura_stream` (vLLM), `aura_stream_hf` (transformers) | `aura` | none (AURA's request is re-implemented; its system prompt is quoted) |
| LiveStar-8B (p4) | `livestar_stream` | `livestar` | LiveStar `inference/*` (remote model code, `streaming_infer.py`) |
| MOSS-VL-Realtime (p5) | `mossrt_stream` | `modern` | the checkpoint's own remote code (`create_realtime_session`) |

`streamreplace fetch-upstream <name>` downloads the upstream files; see `manifests/upstream.yaml` for repositories,
commits and file hashes.

## Rules shared by every child

- **Question time.** The question is asked when the whole video has been streamed, with the benchmark's prompt
  (byte-identical to the offline prompt). The dataset's system prompt (only TOMATO has one) is **not** used: each
  child keeps its own streaming system prompt.
- **Budget.** The reply budget is the benchmark's `generation.max_new_tokens`, as offline.
- **Silence.** When the child stays silent (its own silence marker, an empty reply, or no reply by the deadline),
  the row carries `info.no_answer = true`. Such rows are scored **wrong** and are never parsed or sent to a judge
  (a parser could otherwise read MMDuet2's "NO REPLY" as "no"); the RJScore safety judge still scores the raw text.
- **No forcing.** No child is made to reply: silence stays silence. (MMDuet's forced-answer control is a setting,
  below.)
- **Frames.** MMDuet2, LiveStar and MOSS-VL-Realtime read the framework's fixed-step frames at `stream.fps = 1`
  (frame index `min(total - 1, int(i * fps_source))`, the same rule as the offline frames of pairs p4 and p5, so
  for those pairs offline and stream see the same frames). MMDuet, ThinkStream and AURA read frames with their own
  official loaders, reproduced exactly by `schedule()`.

## MMDuet

MMDuet has no official "question at time t" evaluation; the adapter follows its benchmark loop (the question is
encoded, then the next frame; the model replies when its two score heads fire) inside the official live-inference
class, with the MAGQA settings (`informative + relevance > 0.5`).

- **Frames.** The demo's `load_video(output_fps=2)` (OpenCV): one frame each time the accumulated frame time reaches
  the next multiple of 0.5 s, the long side resized to 384 and padded to a black 384 x 384 square, 49 tokens per
  frame. The accumulation sums `1 / fps` frame by frame in floating point, which moves some frames by one (at 30 fps
  the 0.5 s frame is frame 16, not 15); `load_video_indices()` reproduces this. Videos longer than 200 s (400 frames)
  are refused, because `load_video` would cut them silently.
- **Memory.** Every frame stays in the KV cache; replies are not kept in context.
- **Question.** Encoded after all frames but the last; then the last frame arrives.
- **Answer and silence.** If the heads fire on that last frame, the model's own reply is the answer; otherwise
  MMDuet stays silent (`info.no_answer`). The heads fire on few questions (14 of 900 on Video-MME), so its native S
  mostly goes unanswered.
- **Forced-answer control (S† in the paper).** With `stream.force_answer: true`, one reply is generated at the last
  frame with MMDuet's own `_generate_response` when the heads have not fired, and the row is marked `info.forced`.
  This replaces the response policy by "always answer": the control of Sec. 5.2 (experiment `response`) and what
  Stream++'s Respond does for MMDuet.
- **Decoding.** MMDuet's `fast_greedy_generate` (argmax, no repetition penalty).
- **Row info.** `frames`, `forced`, `last_frame_scores`, `replies_before_question`, `events` (last five), `no_answer`.

## ThinkStream

ThinkStream publishes a streaming evaluation (`eval_common.py`, OVO-Bench and StreamingBench-RT); the adapter runs the
same upstream engine and generator (`StreamingWindowInferenceEngine`, `streaming_video_chat`) with its settings.

- **Frames.** Chunks of `max(1, ceil(duration / 120))` seconds (at most 120 chunks), two frames per chunk, read
  uniformly over the whole video by qwen-vl-utils with torchcodec, at 200,704-401,408 pixels per frame (the
  official evaluation's values; the checkpoint's own processor config would resize again). Clips with too few frames
  get fewer, longer chunks (upstream's rule).
- **Memory.** A 24,576-token KV cache keeping the last 20 video chunks; older chunks are evicted.
- **Question.** At the torchcodec duration of the video, which places it in the last chunk, as
  `"\n" + prompt` after that chunk's frames.
- **Decoding.** Each chunk thinks for at most 20 tokens, sampled (temperature 1.0, top-k 50, top-p 0.95, through
  FlashInfer) after `torch.manual_seed(int(sha256(id)[:8], 16))`, set once per question. Chunks before the question
  are forced to `</think><silent>`. The question chunk is the deadline: after `</think>` the model must answer.
  Multiple-choice and yes/no questions are answered by `<response>` followed by the argmax over the option tokens
  (the last token of each letter, or of "No"/"Yes"), the official restricted readout. Open questions have no
  official procedure: `<response>` is forced after `</think>` and the answer then decodes freely (still sampled)
  up to the budget.
- **Answer and silence.** The text between `<response>` and `<|im_end|>`. A thought that ends with `<|im_end|>`
  before `</think>` makes upstream raise "Deadline exceeded": the row is unanswered (`info.failure`). A
  `<response>` sampled inside an observation chunk's thought makes upstream end the stream early with that turn as
  the reply (1 of 900 Video-MME rows in the paper's runs); this is upstream behaviour and is kept.
- **Deviations from the official evaluation.** The video is not extended by `slack_time` past the question; the
  budget is the benchmark's (official: 30 tokens); the question is the harness prompt (official: options followed
  by "Please select the correct answer.").
- **Row info.** `chunks` (streamed chunks: one less after a missed deadline), `window_chunks`, `kept_chunks`,
  `duration`, `seed`, `restricted`, `raw_turn`, `no_answer`, `failure`.
- Settings `stream.seed: k` gives a replicate run its own seeds (`"<id>#seed<k>"`); `stream.window` (chunks, or
  `full`) and `stream.max_len` change the KV window, and `stream.reservoir: k` replaces its eviction by Recall's
  (see "Settings of the controls").

## MMDuet2

MMDuet2 has no official "question at time t" evaluation. The adapter drives the official in-process proactive client
(`ProactiveInferenceAPIClient` from the demo server, loaded without its web routes by
`models/patches/mmduet2_client.py`, which also replaces the hard-coded 512-token reply budget by the benchmark's;
both upstream files are checked against their pinned git blob ids first).

- **Frames.** 1 fps fixed-step frames, each downscaled by the demo front end's `resize_frame` (longest side at most
  640, `cv2.INTER_AREA`) and then by the client's qwen-vl-utils resize.
- **Memory.** Two frames per user turn (the last turn may hold one); the client replies after every turn; the full
  KV cache is reused and every reply, including "NO REPLY", stays in the history.
- **Question.** Added as text to the final turn, after its images (as in MMDuet2's training data and demo).
- **Decoding.** Greedy with the checkpoint's repetition penalty (1.05).
- **Answer and silence.** The reply to the final turn. Silence is a reply equal to "NO REPLY" in any case (trailing
  dots allowed) or an empty reply - the exact `NO_REPLY_STR` of MMDuet2's own evaluation; the demo's
  `startswith("NO")` test is not used because it would also swallow a genuine "No." to a yes/no question.
- **No forcing.** Forcing variants were tried (MMDuet2's own "I must reply" prefill, banning "NO", an extra user
  turn, restricting the first token to the options): the forced answers were near chance and biased towards "A" or
  "No", so they only replace silence by guessing. Silence is kept and scored wrong.
- **Row info.** `frames`, `turns`, `duration`, `replies_before_question`, `no_answer`.

## AURA

AURA publishes benchmark adapters that ask a question at time t against a vLLM server; `aura_stream` re-implements
that request in AURA's official evaluation environment (vLLM 0.11.0 with the transformers patch from AURA's README,
applied when the `aura` environment is built).

- **Server.** `vllm serve <checkpoint> --max-model-len 32768 --gpu-memory-utilization 0.9 --media-io-kwargs
  '{"video": {"num_frames": -1}}' --limit-mm-per-prompt '{"video": 50}' --served-model-name aura`, started by the
  adapter on a free local port (and stopped again when it does not come up).
- **Frames and memory.** The last `min(30, floor(d))` one-second segments, aligned to the end of the video
  (`d` = decodable frames / fps; a video shorter than one second is one segment). Each segment is cut with the FFmpeg
  of `imageio-ffmpeg` (pinned in the `aura` environment; no other FFmpeg build is used) into an mp4 holding exactly two
  frames (`-vf fps=2`, lossless x264, no audio): the 2-fps chunks AURA was trained on.
- **Question.** Each segment is one user turn; every earlier segment is followed by a prefilled assistant
  `<|silent|>` turn; the last segment's turn carries the prompt. AURA's live-stream system prompt.
- **Decoding.** One request, temperature 0, `mm_processor_kwargs {fps: 1.0, size: {shortest_edge: 4096,
  longest_edge: 602112}}`. With `fps 1.0` each two-frame segment is stamped "<0.5 seconds>"; kept as in AURA's request.
- **Answer and silence.** The reply with `<|silent|>` removed; empty means silence. No forcing.
- **Deviations from the official evaluation.** Official segments are re-encoded at the source fps with audio and
  their length comes from ffprobe; the official reply budget is 128 tokens.
- **Row info.** `segments`, `window_start`, `duration`, `prompt_tokens`, `no_answer`.

**`aura_stream_hf`.** vLLM exposes no hidden states and accepts at most 50 videos per request, so the plugin and
the memory controls run on a rebuild of the same procedure in transformers (same segments, cuts, system prompt and
turns; frames decoded with OpenCV as vLLM does; per-segment video metadata `{fps 1.0, frames_indices range(n),
total_num_frames n, duration n}` so that the prompt carries the same "<0.5 seconds>" stamps; greedy). On smoke items
its prompt token counts equal vLLM's and most replies are identical. Run without the plugin it is the **S_HF**
control (job variant `hf`), which separates the engine change from the plugin's effect.

## LiveStar

LiveStar has no official "question at time t" evaluation. The adapter runs its `StreamingSession` (the SVeD
narration gate) with the settings of LiveStar's accuracy evaluation, then asks the question once, as in its
fixed-schedule evaluation.

- **Frames.** 1 fps fixed-step frames, one 448 x 448 tile per frame (16 tokens after token merging).
- **Memory.** No compression and no retrieval: every frame and every narration stays in context.
- **Narration.** A new caption (greedy, repetition penalty 1.05, at most 384 tokens, upstream's narration prompt)
  is written when a frame's perplexity exceeds the reference times `alpha = 1.04`; the perplexity is averaged over
  `num_runs = 3` passes, as upstream (a single pass changed the narration count on 6 of 8 test videos, so it is not
  equivalent).
- **Question.** The frames that arrived after the last narration's first frame move into the question turn
  (`"Frame-<t>: <image>\n"` each, 0-based), followed by the prompt; one `chat()` call on the narration history, greedy,
  repetition penalty 1.05.
- **Answer.** LiveStar has no silence marker and always answers (`info.no_answer` is never set).
- **Row info.** `frames`, `context_frames`, `frames_in_question`, `narrations`, `alpha`, `num_runs`.

## MOSS-VL-Realtime

MOSS-VL-Realtime has no official benchmark script; the adapter drives the checkpoint's realtime session as its
README asks (playback speed 1).

- **Frames.** 1 fps fixed-step frames near native resolution. All frames but the last are pushed on the wall clock
  at their media timestamps (a 256-frame queue, nothing dropped); the adapter then waits until every frame is
  consumed and the wall clock has reached the video's duration.
- **Question.** Pushed together with the last frame (`push_prompt_frame`), with the session's default system
  prompt.
- **Decoding.** The session's greedy decoding, paced at 12 tokens per one-second turn, with a total budget of
  `max(4096, 12 * frames + budget)`.
- **Answer and silence.** The output after the question's `<|round_start|>`, read by the official client's display
  rule (`readouts.read_reply`): control tokens hidden, the first visible text up to `<|silence|>` / `<|im_end|>`;
  `<|response|>` is not required. Reading stops when the reply is complete, after 3 s without output, after 180 s,
  or when the chunks with visible text reach the budget. No visible text is silence.
- **Why `<|response|>` is not required.** `push_prompt_frame` puts the frame and the question into two queues one
  after the other. When the session's thread wakes up in between, the last frame is processed alone and the question
  arrives in the next cycle without a frame; no `<|silence|>` is injected then, and the model writes the answer
  directly, without `<|response|>`. The official client hides `<|response|>` anyway, so the user would see the same
  answer. `info.prompt_with_frame` tells which case occurred.
- **Provenance of the paper's rows.** On Video-MME, TOMATO, TimeBlind and VideoReasonBench the paper's MOSS-VL-Realtime
  rows were generated with an earlier reading that took only the text after `<|response|>` and counted a reply without
  it as silence. The rows it left silent were then re-read, without new inference, by this same `read_reply` applied
  to the saved output tail (`info.tail`, from its last `<|round_start|>`); those rows carry `info.reread` with their
  earlier state. 226 of the 230 silent Video-MME rows held an answer and are scored with it; the 12 TOMATO, 13
  TimeBlind and 16 VideoReasonBench silent rows stayed silent. The Video-SafetyBench, MCV-SafetyBench and VMDT rows were
  read live by `read_reply`. Regenerated rows are all read live; for the reply shapes seen in those runs
  (`<|round_start|>A<|silence|>`, `A.<|im_end|>`) the live reading gives the same text as the re-read.
- **Row info.** `frames`, `events`, `no_answer`, `response_tag`, `prompt_with_frame`, `tail`.

## Settings of the controls

None of these is part of condition S. The diagnostic experiments ([../experiments.md](../experiments.md)) and
Stream++ ([../plugin.md](../plugin.md)) set them through derived model ids or an arm's settings:

| Setting | Children | Effect |
|---|---|---|
| `stream.force_answer: true` | MMDuet | one reply is generated at the last frame when the heads have not fired (`info.forced`) |
| `stream.question_at: start \| middle` | MMDuet, ThinkStream, MMDuet2, AURA, MOSS-VL-Realtime | the question is posed once at the first step or at step n // 2, the stream runs on under the child's own response policy and the last reply after the question is scored; when no reply came by the end, an answer is requested there (`info.fallback`). ThinkStream's query interface returns one answer and ends the run, so its first answer is scored. LiveStar's narration stream takes no question before its final chat turn |
| `stream.window: N \| full` | AURA (segments), ThinkStream (chunks, with `stream.max_len` for the size of its KV cache) | the recent-only memory of N steps, or nothing evicted (S_full: AURA on `aura_stream_hf`; ThinkStream with `max_len: 81920`) |
| `stream.budget: B`, `stream.recent: k` with `stream.window: full` | AURA (`aura_stream_hf`) | Recall: every segment while floor(d) <= B, else the newest k plus a reservoir of B - k older ones (`info.kept_starts`) |
| `stream.reservoir: k` | ThinkStream | Recall inside its KV window: the newest k chunks stay and the older chunk the reservoir would not keep is evicted instead of the oldest (`info.retained`) |
| `stream.memory_seed: s` | AURA, ThinkStream | another reservoir sample |
| `stream.system: none` | AURA | the live-stream system turn is dropped |
| `stream.input: offline` | AURA (`aura_stream_hf`) | the video is shown as the offline pipeline shows it: one video from 0 s in one user turn, every frame at the offline pixel budget and stamped with its time |

Another checkpoint can be loaded into a child's stream adapter (a derived model id with `checkpoint:`): the parent
control P_S of Supp. C.1 serves Qwen3-VL-8B through AURA's stream.

## Schedules (CPU only)

Every adapter has `schedule(req)`, and its module a function `schedule(req, meta=None)`: what the stream is fed for a
request - frame indices and their digest, chunks, segments or turns, the question time - computed without a GPU or a
model (`meta` takes the values of a `manifests/video_meta` row, so no video is decoded either). An adapter built
with `load=False` loads no model and serves `schedule()` only. Stream rows record the schedule's digest in
`info.schedule`.

On the paper's formal stream runs, `schedule()` reproduces the structural fields of every row (84,040 rows of the 48
stream jobs: ThinkStream chunks and seeds, AURA segments, window start and duration, MMDuet2 frames and turns,
MMDuet, LiveStar and MOSS-VL-Realtime frame counts). Eight MMDuet TimeBlind rows are errors in those runs: OpenCV
decodes no frame of those four videos.

## Plugin hook points

`stream(req, hooks=None)` is exactly the native path described above. With `hooks` (the plugin's
`AnswerTurnHooks`), each adapter calls the plugin around its **answer turn** - the part of the stream that answers
the question - and never touches the turns before it:

| Child | Answer turn (`hooks.arm()` right before, `hooks.disarm()` right after) | Where the first-step logits are |
|---|---|---|
| AURA (`aura_stream_hf`) | the single `generate` call on the full prompt | `hooks.logits_processors()` in that call |
| ThinkStream | the query (deadline) chunk's `engine.generate` | upstream's restricted readout; a missed deadline is read out by `hooks.deadline_readout(fork)` |
| MMDuet2 | the client's `generate` for the final turn | `hooks.logits_processors()` in that call |
| LiveStar | the final `chat()` call | `hooks.logits_processors()` in its generation config |
| MMDuet | the first `_generate_response` after the question (`arm(first=True)`) | `hooks.on_step_logits(step, logits)` on every LM-head forward of the turn |
| MOSS-VL-Realtime | from the session update that drains the question to the end of the stream | `hooks.on_step_logits(step, logits)` on every LM-head forward after arming |

- `hooks.attach(decoder_layers, tokenizer)` is called with the language model's decoder layers, once per hooks object
  until it succeeds (a call that raises is retried on the next `stream()`), before the adapter installs its own call
  sites.
- `on_step_logits` receives the LM head's output (shape `[1, positions, vocab]`); a returned tensor replaces it.
- ThinkStream decodes by CUDA-graph replay, where forward hooks do not run: when `hooks.eager_decode` is true
  (parent-restoration steering), the query chunk's decode steps run the same forward eagerly. Its `fork` callable
  re-prefills the query chunk from its state before the prefill (cache length, next rotary position and CUDA RNG
  restored) with the question followed by `<think></think><response>`, unsteered, and returns the logits at
  `<response>`; on a clip of a single chunk, a missed deadline is the same unanswered turn.
- LiveStar with the plugin shares one session per video among that video's consecutive questions, as in the paper's
  plugin runs; native S streams the video again for each question.
- `hooks.finish(req, generation)` makes the final row (the plugin's refusal or option readout; the native reply
  is kept in the row's info).

AURA's vLLM adapter refuses hooks: the plugin's AURA arms run on `aura_stream_hf`.

## Reproducibility

What the CPU can check is exact (the schedules reproduce the structure of the paper's stream rows, see Schedules);
regenerating streams on GPUs is approximate:

- MMDuet, MMDuet2, AURA (`aura_stream_hf`) and LiveStar's answer are greedy, but GPU kernels can flip near-tied
  tokens across GPU types.
- ThinkStream samples its thoughts: identical only on the same GPU type and FlashInfer version.
- AURA's vLLM server answers 8 requests at a time (`stream.aura_concurrency`); vLLM batching is not batch-invariant.
- LiveStar's narration count varies between runs (its perplexity gate).
- MOSS-VL-Realtime runs on the wall clock: how much it generates between frames, and whether the question shares a
  cycle with its frame, depend on GPU speed.
