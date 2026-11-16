# StreamingBench (Real-Time Visual Understanding)

Four-option questions asked at a time stamp t during a video, in 10 tasks. Only the Real-Time Visual Understanding
part is run (the part streaming-model papers usually report); Omni-Source (audio) and Contextual Understanding are
not. Streaming benchmark: it is **not** one of the P/C/S benchmarks. The paper uses it (with OVO-Bench) to check that
Stream++ keeps a child's native streaming performance (Supp. D.4, experiment `streaming_benchmarks`,
[../experiments.md](../experiments.md)); only stream conditions are run.
Module: `src/streamreplace/benchmarks/streaming/streamingbench.py`.

## Data

- Source: Hugging Face dataset `mjuicem/StreamingBench` at revision `48872fa707124474ce4c5172ddc58efb8bc88058`:
  the ten `Real-Time Visual Understanding_*.zip` archives (about 132 GB). The questions are the official
  `src/data/questions_real.json` of the benchmark's code, fetched at the pinned commit (`streamreplace
  fetch-upstream streamingbench`; manifests/upstream.yaml). The dataset's CSV is another revision with edited texts
  and is not used.
- Rows: 2,500 questions on 499 videos. Subset: the questions asked within the first 120 s, **700** questions.
- Clips: for each question the video from 0 s to t, cut as the official `split_video` does (`ffmpeg -ss 0 -t t`,
  libx264 + aac, re-encoded) with the FFmpeg of the `ffmpeg` environment (`STREAMREPLACE_FFMPEG` overrides it;
  `STREAMREPLACE_CLIP_WORKERS` sets the number of parallel ffmpeg processes). Re-encoding costs about 3 CPU-seconds
  per second of 1080p video.
- Sample id: `<sample>_<k>`, k = the question's position among that video's questions in the file.

## Prompt

The official `PROMPT_TEMPLATE` followed by "\n\nThe best option is:", verbatim (options are lettered when the
annotation does not letter them). Generation: at most 128 new tokens. The clip plays through the child's native
stream and the question arrives at its end (the official protocol with `context_time = -1`).

## Parsing and scoring

The official `src/data/count.py` logic, no judge: a reply is right iff its first character is the answer letter.
The reply is first cleaned of think blocks and control tokens.

## Metrics and statistics

- `acc` (the paper's accuracy): correct / all questions; unanswered questions are wrong.
- `acc_official`: correct / answered questions (the official accuracy skips empty replies); `answered`; `task/*`.
- Paired statistics resample source videos.

## Deviations from the official protocol

- Only the questions asked within the first 120 s.
- The headline divides by all questions, not by the answered ones (`acc_official` keeps the official ratio).
