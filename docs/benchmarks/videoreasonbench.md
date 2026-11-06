# VideoReasonBench

Vision-centric video reasoning: the model must track a sequence of operations on a latent state (cards, chips,
cups, a file system, a grid, a sliding-block puzzle) and answer free-form questions. Supplementary video benchmark:
evaluated under all three conditions, P (parent offline), C (child in the same offline pipeline) and S (child
through its native streaming interface, question at the end of the video); reported in the supplement's video
table and its contrasts (a secondary metric family). Module: `src/streamreplace/benchmarks/video/videoreasonbench.py`.

## Data

- Source: Hugging Face dataset `lyx97/reasoning_videos` at revision `ebab429914370bc5fd83f9bf6e21a57d707fc90e`:
  `<task>/test-*.parquet` for the six tasks (card, chip, cup, file_sys, grid, hrd) and `videos.zip` (extracted into
  `media/`).
- Subset: all **1,440** questions = 240 videos x 6. 1,200 have a reference answer text; the 240 operation questions
  have an empty answer and are checked against the video's final state.
- Sample id = the dataset id. The states, coordinates and moves of each video are kept in the metadata for the
  judge's state checkers; they are private metadata that models never see.

## Prompt

The question as released (it already ends by asking for a summary after "Final Answer:"). Generation: greedy, at
most 4,096 new tokens. No system prompt. Frames: the pair's offline video protocol for P and C, each child's native
stream for S (docs/protocol.md).

## Parsing and scoring

The official `extract_final_answer` of `utils/eval_utils.py` (VideoReasonBench repository, fetched at the pinned
commit) on the cleaned reply: the text from the FIRST "Final Answer" (case-sensitive, colon optional and kept)
through the end of the reply. A reply without "Final Answer" has no answer and is wrong.

## Judge

Always: every extracted answer is judged with the official procedure, routed to a local open model (temperature 0,
thinking off):

- questions with a reference answer: the official `get_judge_prompt` / `llm_judge_api` (max_tokens 16; correct when
  the judge replies "Correct");
- operation questions: the official state checkers (`eval_op_*`), which ask the judge to extract the moves from the
  answer (max_tokens 512) and replay them on the video's initial or final state. For the file-system task the
  released parquet keeps only the file lists of each state, which are passed in the checker's format.

A rule-only score (exact match after normalisation) is reported beside the judged metrics but is not the paper's
metric.

## Metrics and statistics

- `acc` (headline): judged accuracy over all 1,440 questions; replies without a final answer, errored and silent
  rows count as wrong. `len/<=80s`, `len/>80s`, `task/*` and `dim/*` breakdowns.
- Paired statistics resample videos (the 6 questions of a video form one cluster).

## Deviations from the official protocol

- The judge is a local open model instead of the official API model.
- The judge's replies are compared after stripping surrounding whitespace, and its move lists are parsed with
  `ast.literal_eval` (a malformed list fails the check) instead of `eval`, a small patch applied to the fetched file
  (manifests/upstream.yaml).
