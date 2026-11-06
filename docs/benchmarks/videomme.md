# Video-MME (short videos, no subtitles)

General video understanding, multiple choice. Video benchmark: evaluated under all three conditions, P (parent
offline), C (child in the same offline pipeline) and S (child through its native streaming interface, question at
the end of the video); reported in the video results table and the C-P, S-C and S-P contrasts.
Module: `src/streamreplace/benchmarks/video/videomme.py`.

## Data

- Source: Hugging Face dataset `lmms-lab/Video-MME` at revision `ead1408f75b618502df9a1d8e0950166bf0a2a0b`:
  `videomme/test-00000-of-00001.parquet`, the video archives `videos_chunked_01.zip` ... `videos_chunked_20.zip` and
  `subtitle.zip` (all extracted into `media/`).
- Subset: the **short** videos only (under 2 minutes): **900** questions on 300 videos (the full set has 2,700
  questions over short, medium and long videos). Sample id = the dataset's question_id; video
  `media/data/<videoID>.mp4`.
- Options are parsed from the `A. ...` strings of the release. `metadata.subtitle` records the subtitle file
  (`media/subtitle/<videoID>.srt`, relative to the dataset folder); no prompt uses subtitles.

## Prompt

As lmms-eval (tasks/videomme, default model kwargs; its wording mentions subtitles even without them):

```
Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.
{question}
A. ...
B. ...
C. ...
D. ...

Answer with the option's letter from the given choices directly.
```

Generation: greedy, at most 512 new tokens. No system prompt. Frames: the pair's offline video protocol for P and
C, each child's native stream for S (docs/protocol.md); P, C and S send the same prompt text.

## Parsing and scoring

The official scorer, `eval_your_results.py` of the Video-MME evaluation repository (fetched at the pinned commit,
manifests/upstream.yaml): `extract_characters_regex` on the cleaned reply. It removes a list of answer prefixes and
returns the first capital A-D anywhere in the rest; a reply of more than 10 words without any A-D letter gives no
answer. The prediction is correct when it equals the reference letter.

## Judge

None. The official evaluation states that it introduces no third-party models; a reply without a letter is wrong.

## Metrics and statistics

- `acc` (headline): correct / all 900 questions; unanswered, errored and silent stream rows count as wrong.
- `acc_official`: correct / answered questions (upstream's ratio); `answered`: the answered share.
- `duration/short`, `len/<=80s` and `len/>80s` (video length split at 80 s: the LLaVA-OneVision pair samples at most
  160 frames, i.e. 80 s at 2 fps), `task/*`, `domain/*`.
- Paired statistics resample videos (the 3 questions of a video form one cluster).
- Stream rows where the model stayed silent (no reply, "NO REPLY", no visible text) are wrong and never parsed.

## Deviations from the official protocol

- Only the short split, without subtitles.
- The headline divides by all questions, not by the questions a letter could be extracted from (`acc_official`
  keeps upstream's ratio).
