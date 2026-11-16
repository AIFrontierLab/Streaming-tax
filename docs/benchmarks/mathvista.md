# MathVista

Mathematical reasoning in visual contexts. Image benchmark: evaluated for the parent (P) and the child in the same
offline pipeline (C); reported in the image results table and the C-P contrasts.
Module: `src/streamreplace/benchmarks/image/mathvista.py`.

## Data

- Source: Hugging Face dataset `AI4Math/MathVista` at revision `2b6ad69445fbb5695c9b165475e8decdbeb97747`, files
  `data/testmini-*.parquet`.
- Subset: testmini, **1,000** questions: 540 multiple choice, 458 integer or float answers, 2 list answers (scored
  as text). Sample id = the dataset's `pid`; images are saved as `media/<pid>.png`.
- Multiple-choice answers are stored as the letter whose option text equals the reference answer.
- Numeric tolerance from the dataset: a float answer with `precision` p is correct within 0.5 x 10^-p (+1e-9), an
  integer answer must match exactly; list answers are compared element-wise within 1e-3.

## Prompt

The dataset's own zero-shot `query` field, verbatim, for example:

```
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: ...
Choices:
(A) ...
```

Generation: greedy, at most 2,048 new tokens. No system prompt.

## Parsing and scoring

The common rules on the cleaned reply (`<think>...</think>` blocks and special tokens removed):

- multiple choice: a bare letter, `(B)`, `B.`, the exact or normalised option text, `B. text`; an answer naming
  several options is a parse failure;
- numbers: the last `\boxed{}`, else the text after the last answer marker ("final answer", "the answer is",
  "answer:"), else a single-line reply; fractions, `\frac{}{}`, pi, percentages, degrees and thousands separators
  are understood; a multi-line reply without a marker is left to the judge.

## Judge

Fallback: a reply the rules cannot read goes to the generic LLM extraction judge (with the row's question, not the
full query; temperature 0). The extraction is read again by the rules as `Final answer: <x>`.

## Metrics and statistics

- `acc` (headline): accuracy over all 1,000 questions; errored generations count as wrong. `task/*` breakdowns.
- Paired statistics resample items.
- The paper's gap figures replace the rows that stopped at the 2,048-token cap with 8,192-token reruns of both pair
  members (docs/statistics.md); the absolute-score table uses the formal rows, except for Qwen2.5-VL-3B, ThinkStream
  and MMDuet2, which share one merge on the 80 items any of the three capped and are shown with it.

## Deviations from the official protocol

- The official evaluation extracts answers with a GPT model and normalises them by answer type and precision. Here
  the rules above decide, with the local extraction judge only for replies they cannot read, and numbers are
  compared with the precision-derived tolerance instead of rounding.
