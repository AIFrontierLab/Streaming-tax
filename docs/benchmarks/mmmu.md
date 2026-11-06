# MMMU

College-level multi-discipline questions with up to 7 images. Image benchmark: evaluated for the parent (P) and the
child in the same offline pipeline (C); reported in the image results table and the C-P contrasts.
Module: `src/streamreplace/benchmarks/image/mmmu.py`.

## Data

- Source: Hugging Face dataset `MMMU/MMMU` at revision `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`, files
  `<Subject>/validation-*.parquet` (30 subjects).
- Subset: the validation split, **900** questions (847 multiple choice, 53 open). Sample id = the dataset's id.
- Options come from the `options` field (a Python list literal). Images `image_1` ... `image_7` are saved as
  `media/<Subject>/<id>_<i>.png` and passed to the model before the text; the `<image n>` tags stay in the question
  as plain text.

## Prompt

The official MMMU prompt (MMMU repository: configs/llava1.5.yaml and construct_prompt):

```
{question}
(A) ...
(B) ...

Answer with the option's letter from the given choices directly.
```

Open questions: `{question}\nAnswer the question using a single word or phrase.` Generation: greedy, at most 2,048
new tokens. No system prompt.

## Parsing and scoring

- Multiple choice: the common option rules on the cleaned reply; correct when the parsed letter equals the
  reference letter.
- Open: the common answer span (last `\boxed{}`, else the text after the last answer marker, else a single-line
  reply). A reference that starts with `[` lists several accepted forms (e.g. `['$MgS$', 'MgS']`); dollar signs are
  ignored and numbers are correct within a 1% relative tolerance (at least 1e-3).

## Judge

Fallback: a reply the rules cannot read goes to the generic LLM extraction judge (temperature 0); the extraction is
read again as `Final answer: <x>`.

## Metrics and statistics

- `acc` (headline): accuracy over all 900 questions; errored generations count as wrong. `subject/*` breakdowns.
- Paired statistics resample items.
- The paper's gap tables replace the rows that stopped at the 2,048-token cap with 8,192-token reruns of both pair
  members (docs/statistics.md); the absolute-score table uses the formal rows.

## Deviations from the official protocol

- The official evaluation picks a random option when it finds none and uses its own open-answer matching. Here the
  common rules decide, replies they cannot read go to the local extraction judge, and no random answer is drawn.
