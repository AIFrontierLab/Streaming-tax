# MMBench

Multiple-choice perception and reasoning with circular evaluation. Image benchmark: evaluated for the parent (P) and
the child in the same offline pipeline (C); reported in the image results table and the C-P contrasts.
Module: `src/streamreplace/benchmarks/image/mmbench.py`.

## Data

- Source: `MMBench_DEV_EN_legacy.tsv` from the OpenCompass mirror of MMBench (md5
  `6be6358b6c8d6f633e7cd3c602a49c6b`; see manifests/datasets.yaml).
- Subset: the English dev split, **4,329** rows = 1,164 questions, each asked once per circular rotation of its
  options (926 questions with 4 rotations, 149 with 3, 89 with 2).
- Row index = rotation x 1,000,000 + question number; the sample id is the row index. A row's image field may hold
  another row's index instead of the image; it is followed to the image. One image per question:
  `media/<question>.png`.
- Options A-D, empty or `nan` options dropped; a hint of `''` or `nan` means no hint.

## Prompt

As lmms-eval (tasks/mmbench: `mmbench_doc_to_text`, `create_options_prompt`, the system prompt of mmbench.yaml):

```
{hint} {question} There are several options:
A. ...
B. ...

Answer with the option's letter from the given choices directly.
```

(`{hint} ` is omitted when there is no hint). Generation: greedy, at most 512 new tokens. No system prompt.

## Parsing and scoring

The common option rules on the cleaned reply: a bare letter, `(B)`, `B.`, the exact or normalised option text,
`B. text`; an answer naming several options is a parse failure. A row is correct when the parsed letter is the
reference letter of that rotation.

## Judge

Fallback: a reply the rules cannot read goes to the generic LLM extraction judge (question with its hint and the
options; temperature 0). The extraction is read again by the rules as `Final answer: <x>`.

## Metrics and statistics

- `circular_acc` (headline): share of questions, among those with all their rotations counted, whose every rotation
  is correct. Errored generations count as wrong.
- `vanilla_acc` (rotation 0 rows only), `questions`, `l2/*` (circular accuracy per level-2 category).
- Paired statistics: one binary unit per question (all rotations correct), group size = the question's number of
  rotations.

## Deviations from the official protocol

- The official evaluation (VLMEvalKit) matches replies to options with exact rules and then a ChatGPT matcher, and
  uses its own prompt. Here the prompt is lmms-eval's, the rules are this repository's common option rules, and
  replies they cannot read go to the local extraction judge.
