# VisuLogic

Visual logic puzzles whose options are drawn inside the image. Supplementary image benchmark: evaluated for the
parent (P) and the child in the same offline pipeline (C); reported in the supplement's image table and its C-P
contrasts (a secondary metric). Module: `src/streamreplace/benchmarks/image/visulogic.py`.

## Data

- Source: Hugging Face dataset `VisuLogic/VisuLogic` at revision `3e483f8cca0bc766ecc5f70299f2b8a34b5035ca`: `data.jsonl`
  and `images.zip` (extracted into `media/`).
- Subset: all **1,000** questions. Sample id = the dataset's id; the text options are just A-D (the real options are
  in the image); the reference is the upper-cased label; `category` = the dataset's tag.

## Prompt

The official default of VisuLogic-Eval (models/prompts.py, `SFT_PROMPT`), appended to the question without a
separator exactly as upstream does (`text + self.user_prompt`):

```
{question}Just Answer the question following this format: Answer:\boxed{$LETTER}
```

(the question already ends with its `A: A ... D: D` lines). Generation: greedy, at most 2,048 new tokens. No system
prompt.

## Parsing and scoring

The common option rules on the cleaned reply; the answer span is the last `\boxed{}` (one level of nested braces),
else the text after the last answer marker, else a single-line reply.

## Judge

Fallback: a reply the rules cannot read goes to the generic LLM extraction judge (temperature 0); the extraction is
read again as `Final answer: <x>`.

## Metrics and statistics

- `acc` (headline): accuracy over all 1,000 questions; errored generations count as wrong. `cat/*` breakdowns.
- Paired statistics resample items. Long reasoning can reach the 2,048-token cap; the analysis flags a cell as
  invalid when more than 10% of an arm's rows stopped at the cap (docs/statistics.md).

## Deviations from the official protocol

- The official evaluation's answer extraction is not used: the common option rules read the reply (the last boxed
  letter first) and the local extraction judge reads what they cannot.
