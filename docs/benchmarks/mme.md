# MME

Yes/no perception and cognition questions, two per image. Image benchmark: evaluated for the parent (P) and the
child in the same offline pipeline (C); reported in the image results table and the C-P contrasts.
Module: `src/streamreplace/benchmarks/image/mme.py`.

## Data

- Source: Hugging Face dataset `lmms-lab/MME` at revision `d6c9023f017b564f7b3ccccf5348166bce8fdbcd`, files
  `data/test-*.parquet`.
- Subset: all **2,374** questions = 1,187 images x 2 questions, 14 subtasks (10 perception, 4 cognition:
  commonsense_reasoning, numerical_calculation, text_translation, code_reasoning).
- Sample id: `<question_id>#<k>`, k = 1, 2 for the image's first and second question (the dataset's question_id
  names the image). Images: `media/<sha256(question_id)[:16]>.png`. Answers are lower-cased.

## Prompt

The question verbatim; every question already ends with "Please answer yes or no." (the common prompt adds
"Answer with yes or no." only to questions that do not mention "yes or no"). Generation: greedy, at most 16 new
tokens. No system prompt.

## Parsing and scoring

The common yes/no rule on the cleaned reply: the first word when it is yes or no (and the answer is long, or names
only one of them), else the single yes/no word in the answer; anything else is a parse failure.

## Judge

Fallback: a reply the rule cannot read goes to the generic LLM extraction judge (temperature 0); the extraction is
read again as `Final answer: <x>`.

## Metrics and statistics

- Official arithmetic: subtask score = accuracy + accuracy+ (the share of images with both questions right), at most
  200, each rounded to 2 decimals. `perception` (headline, at most 2,000) and `cognition` (at most 800) are the sums
  of their subtasks, rounded to 1 decimal. `task/*` lists the subtask scores. Errored generations count as wrong.
- Paired statistics: strata = subtask category, cluster = image (both questions); accuracy and accuracy+ are the two
  terms of each category. Deltas are reported in percentage points of the maximum (perception / 20, cognition / 8).

## Deviations from the official protocol

- The official script looks for yes or no at the start of the reply and counts anything else as wrong. Here the
  common yes/no rule reads the reply and the local extraction judge reads what the rule cannot.
