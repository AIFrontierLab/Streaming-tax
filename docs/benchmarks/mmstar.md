# MMStar

Vision-indispensable multiple-choice questions over 6 core capabilities. Image benchmark: evaluated for the parent
(P) and the child in the same offline pipeline (C); reported in the image results table and the C-P contrasts.
Module: `src/streamreplace/benchmarks/image/mmstar.py`.

## Data

- Source: Hugging Face dataset `Lin-Chen/MMStar` at revision `bc98d668301da7b14f648724866e57302778ab27`, file
  `mmstar.parquet`.
- Subset: all **1,500** questions (6 capabilities x 250). Sample id = the dataset's index; images `media/<index>.png`.
- The options are inline in the released question, either as `Options: A: x, B: y, ...` or in the MathVista style
  `Hint: ...\nQuestion: ...\nChoices:\n(A) x\n(B) y`. They are split out for scoring only; the prompt keeps the
  released question.

## Prompt

As lmms-eval (tasks/mmstar): the released question, stripped, followed by the task's default post-prompt, which has
no final period:

```
{released question}
Answer with the option's letter from the given choices directly
```

Generation: greedy, at most 512 new tokens. No system prompt.

## Parsing and scoring

The common option rules on the cleaned reply: a bare letter, `(B)`, `B.`, the exact or normalised option text,
`B. text`; an answer naming several options is a parse failure.

## Judge

Fallback: a reply the rules cannot read goes to the generic LLM extraction judge (temperature 0); the extraction is
read again as `Final answer: <x>`.

## Metrics and statistics

- `acc` (headline): accuracy over all 1,500 questions; errored generations count as wrong. `cat/*` breakdowns.
- Paired statistics resample items.
- The paper's gap tables replace the rows that stopped at the 512-token cap with 8,192-token reruns of both pair
  members (docs/statistics.md); the absolute-score table uses the formal rows.

## Deviations from the official protocol

- Replies are read with this repository's common option rules plus the local extraction judge instead of the
  official matching.
