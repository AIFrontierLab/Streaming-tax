# TOMATO

Temporal reasoning over visual dynamics (counting, direction, rotation, shape and trend, velocity and frequency,
visual cues), multiple choice. Video benchmark: evaluated under all three conditions, P (parent offline), C (child
in the same offline pipeline) and S (child through its native streaming interface, question at the end of the
video); reported in the video results table and the C-P, S-C and S-P contrasts.
Module: `src/streamreplace/benchmarks/video/tomato.py`.

## Data

- Source: Hugging Face dataset `yale-nlp/TOMATO` at revision `aa4827bbd64ce5a93588b09e5c6da6eb58dd09c9`, files
  `data/<reasoning>-*.parquet`; the videos come from the authors' `videos.zip` (linked from the TOMATO repository,
  see manifests/datasets.yaml), extracted into `media/`.
- Subset: all **1,484** questions (direction 403, count 292, rotation 286, shape_trend 223, velocity_frequency 210,
  visual_cues 70), 3-7 options each (9 questions have a G option), on 1,397 distinct video files.
- Sample id: `<reasoning>:<key>`, with `:<row>` appended when the id repeats. The ids seed the random fallback
  below. Video: `media/videos/<demonstration type>/<key>.mp4`. The question is kept verbatim (not stripped).

## Prompt

As lmms-eval (tasks/tomato, from the official `construct_prompt`), trailing spaces included; `{index2ans}` is the
Python dict representation of the options, e.g. `{'A': '2', 'B': '1'}`:

```
You will be provided with frames uniformly sampled from a video, the frames are provided in chronological order of the video. Analyze these frames and provide the answer to the question about the video content. Answer the multiple-choice question about the video content.

You must use these frames to answer the multiple-choice question; do not rely on any externel knowledge or commonsense. 

<question> 
{question} 
</question>

<options> 
{index2ans} 
</options>

Even if the information in these separate frames is not enough to answer the question, PLEASE TRY YOUR BEST TO GUESS AN ANSWER WHICH YOU THINK WOULD BE THE MOST POSSIBLE ONE BASED ON THE QUESTION. 

DO NOT GENERATE ANSWER SUCH AS 'NOT POSSIBLE TO DETERMINE.' 
```

System prompt (offline adapters only; stream adapters send no dataset system prompt): "You are an expert in
understanding dynamics of objects." Generation: greedy, at most 1,024 new tokens. Frames: the pair's offline video
protocol for P and C, each child's native stream for S (docs/protocol.md).

## Parsing and scoring

The official `src/parse_result.py` of the TOMATO repository (fetched at the pinned commit, manifests/upstream.yaml):

1. `pre_parser` on the cleaned reply: a single letter; `X: option` when the upper-cased reply equals it (so only
   for options without lower-case letters); or `Answer is:(X)` style replies for A-F.
2. Otherwise the judge receives `gpt_parser`'s prompt verbatim (the raw reply, the list of letters and the options)
   with max_tokens 16 at temperature 0.
3. If the judge's reply is not one of the letters, an option is drawn at random, seeded per question:
   `random.Random(int(sha256("tomato-random:<id>")[:8], 16)).choice(letters)`.

## Judge

Fallback with the official prompt (`chat` request, max_tokens 16). Silent stream rows and empty replies are never
judged and get no random option: they are wrong.

## Metrics and statistics

- `acc` (headline): accuracy over all 1,484 questions; errored and silent rows count as wrong. `reasoning/*` and
  `demo/*` (human, object, simulated) breakdowns.
- Paired statistics resample video files (questions on the same clip form one cluster: 1,397 clusters).

## Deviations from the official protocol

- The judge is a local open model instead of gpt-4o-mini, called once at temperature 0 (upstream retries up to 5
  times; at temperature 0 the reply does not change); the judge's reply is compared after stripping surrounding
  whitespace.
- The random fallback is seeded per question (upstream uses the unseeded global random generator), so scores are
  reproducible.
