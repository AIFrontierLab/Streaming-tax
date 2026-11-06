# DynaMath

Visual math reasoning under programmatic variation. Image benchmark: evaluated for the parent (P) and the child in
the same offline pipeline (C); reported in the image results table and the C-P contrasts.
Module: `src/streamreplace/benchmarks/image/dynamath.py`.

## Data

- Source: Hugging Face dataset `DynaMath/DynaMath_Sample` at revision `bc7fcd1ae58caa445d61f6dc0c241b0bb8a7f4c6`,
  files `data/sample{1..10}-*.parquet` (one file set per variant).
- Subset: the full sample split, 501 seed questions x 10 generated variants = **5,010** questions
  (2,960 numeric, 1,740 multiple choice, 310 free-form text).
- Sample ids: `sha256("dynamath" + "\x1f" + "sample_variant<v>/default" + "\x1f" + "<seed>:<v>")[:24]`, e.g. seed 6,
  variant 1 -> `4632251fe0dcaf354fd88f5b`. Images are saved as `media/variant<v>/<seed>.png`.
- Multiple-choice options are split out of the question text (`(A) ... (B) ...` markers, or `A.` line starts; the
  labels must run A, B, C, ...); the raw question is kept in `metadata.raw_question` for the prompt.

## Prompt

The official DynaMath prompt of VLMEvalKit (`vlmeval/dataset/dynamath.py`), built from the raw question:

```
## Question
 {raw question}
## Answer Instruction Please provide an answer to the question outlined above. Your response should adhere to the
following JSON format, which includes two keys: 'solution' and 'short answer'. The 'solution' key can contain
detailed steps needed to solve the question, and the 'short answer' key should provide a concise response. {INST}

Example of expected JSON response format:

{
    "solution": "[Detailed step-by-step explanation]",
    "short answer": "[Concise Answer]"
}
```

(one line in the actual text; shown wrapped here). `{INST}` depends on the answer type, upstream wording including
its typo:

| kind | instruction |
|---|---|
| multiple choice | Provide the corresponing choice option in the 'short answer' key, such as 'A', 'B', 'C', or 'D'. |
| numeric | Format the answer as a three-digit floating-point number and provide it in the 'short answer' key. |
| text | Float numbers in the answer should be formatted as three-digit floating-point numbers. |

Generation: greedy, at most 2,048 new tokens. No system prompt.

## Parsing and scoring

1. The reply is cleaned (`<think>...</think>` blocks and special tokens removed).
2. The JSON short answer is read as VLMEvalKit preprocesses it: the text between the first `{` and the last `}`,
   every backslash removed, `json.loads(..., strict=False)["short answer"]`. If that fails (for example JSON broken
   by LaTeX), the last `"short answer": ...` field is read with a regular expression.
3. The short answer is read with the common rules for its kind (option letter or option text; a number; a short
   phrase). Status strings are prefixed with `json_`. Without a usable short answer the common rules read the whole
   reply (last `\boxed{}`, else the text after the last "answer" marker, else a single-line reply).
4. Correctness follows the official rules: numbers within an absolute 1e-3; options by letter; free-form answers
   are correct when the stripped, lower-cased prediction is a substring of the lower-cased reference.

## Judge

Fallback: a reply the rules cannot read goes to the generic LLM extraction judge (the raw reply, the question stem
and the options; temperature 0). The judge's extraction is read again by the rules above as `Final answer: <x>`.

## Metrics and statistics

- `avg_acc` (headline): accuracy over all 5,010 questions; errored generations count as wrong.
- `worst_acc`: share of seeds whose 10 variants are all correct (complete seeds only); `complete_seeds`;
  `kind/*` and `subject/*` breakdowns.
- Paired statistics: `avg_acc` resamples seeds (the 10 variants of a seed form one cluster); `worst_acc` is one
  binary unit per seed (group size 10).

## Deviations from the official protocol

- Answers the rules cannot read are extracted by this repository's local extraction judge instead of the official
  evaluation's own answer-extraction step.
- DynaMath is not part of the generation-cap reruns used in the paper's gap tables (docs/statistics.md).
