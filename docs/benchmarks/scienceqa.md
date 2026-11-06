# ScienceQA

Science multiple-choice questions with an image and an optional textual context. Image benchmark: evaluated for the
parent (P) and the child in the same offline pipeline (C); reported in the image results table and the C-P
contrasts. Module: `src/streamreplace/benchmarks/image/scienceqa.py`.

## Data

- Source: Hugging Face dataset `lmms-lab/ScienceQA` at revision `69dd4d6b67373d38f96a5badd5d24d0eb5bcdc50`, files
  `ScienceQA-IMG/test-*.parquet`.
- Subset: the test questions that come with an image, **2,017** questions with 2-5 options.
- Sample id: `<shard>-<row>`, where the row number counts every row of the shard, including the skipped rows without
  an image (e.g. `00000-0`). Images `media/<id>.png`. The reference is the letter of the answer index.
- The hint becomes a context line: question = `Context: {hint}\n{question}` (no context line for an empty hint).

## Prompt

As LLaVA / lmms-eval (tasks/scienceqa), rendered by the common prompt:

```
Context: {hint}
{question}
A. ...
B. ...
Answer with the option's letter from the given choices directly.
```

Generation: greedy, at most 512 new tokens. No system prompt.

## Parsing and scoring

The common option rules on the cleaned reply: a bare letter, `(B)`, `B.`, the exact or normalised option text,
`B. text`; an answer naming several options is a parse failure.

## Judge

Fallback: a reply the rules cannot read goes to the generic LLM extraction judge (temperature 0); the extraction is
read again as `Final answer: <x>`.

## Metrics and statistics

- `acc` (headline): accuracy over all 2,017 questions; errored generations count as wrong. `subject/*` breakdowns.
- Paired statistics resample items.

## Deviations from the official protocol

- Replies are read with this repository's common option rules plus the local extraction judge.
