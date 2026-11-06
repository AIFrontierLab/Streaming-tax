# TimeBlind

Temporal understanding with minimal pairs: each instance has two videos and two questions whose answers flip
between the videos, so a model that ignores time cannot get all four right. Video benchmark: evaluated under all
three conditions, P (parent offline), C (child in the same offline pipeline) and S (child through its native
streaming interface, question at the end of the video); reported in the video results table and the C-P, S-C and
S-P contrasts. Module: `src/streamreplace/benchmarks/video/timeblind.py`.

## Data

- Source: Hugging Face dataset `BaiqiL/TimeBlind` at revision `bafceda3a2bd0e5128c390ec25fdc54e470a743d`:
  `data.jsonl` and `videos/` (read in place from the raw folder); `tags.json` from the TimeBlind repository at the
  pinned commit (metadata only, see manifests/datasets.yaml).
- Subset: all **2,400** questions = 600 instances x 4; 1,200 yes/no and 1,200 A/B questions (options inline in the
  question as `Option A: ... Option B: ...`).
- Sample id = the dataset index; instance = index // 4; role = index % 4 (0 = question 0 on video 0, 1 = q0/v1,
  2 = q1/v0, 3 = q1/v1). The role determines the reference answer, so it is private metadata that models never see.

## Prompt

The official `add_question_suffix`; the options stay inside the question text:

- yes/no: `{question} Please output Yes or No.`
- A/B: `{question} Please output A or B.`

Generation: greedy, at most 512 new tokens. No system prompt. Frames: the pair's offline video protocol for P and
C, each child's native stream for S (docs/protocol.md).

## Parsing and scoring

The official `utils.extract_answer` of the TimeBlind repository (fetched at the pinned commit) on the cleaned reply:
1 = yes / A, 0 = no / B, -1 = no answer (status `official_invalid`). It reads letters only: a reply that just
repeats an option's text is not an answer for the rules (the judge may extract it).

## Judge

Fallback with the official prompt: for a reply where `extract_answer` returns -1, the judge receives the system
message and `create_extraction_prompt` of the official `llm_judge.py` (raw reply; max_tokens 10, temperature 0),
and its reply goes through `extract_answer` again; -1 stays wrong. Silent stream rows are never judged.

## Metrics and statistics

- `I_Acc` (headline): share of complete instances whose four questions are all correct. Errored and silent rows
  count as wrong.
- `Q_Acc` (question pairs: roles 0 and 1, 2 and 3), `V_Acc` (video pairs: roles 0 and 2, 1 and 3), `Acc` (all
  questions of complete instances), `instances`, `type/*` (yes/no, multiple choice), following the official
  `get_scores`.
- Paired statistics: one binary unit per instance (all four correct), group size 4.

## Deviations from the official protocol

- The official `llm_judge.py` script re-extracts every reply with a GPT model; following the repository's use of it
  for outputs the regex misses, only replies without a regex answer are sent to the judge, a local open model.
