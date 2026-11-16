# OVO-Bench

Online video understanding: questions asked at a time t during a video, in three modes, Backward Tracing (EPM, ASI,
HLD), Real-Time Visual Perception (STU, OJR, ATR, ACR, OCR, FPD) and Forward Active Responding (REC, SSR, CRR).
Streaming benchmark: it is **not** one of the P/C/S benchmarks. The paper uses it (with StreamingBench) to check
that Stream++ keeps a child's native streaming performance (Supp. D.4, experiment `streaming_benchmarks`,
[../experiments.md](../experiments.md)); only stream conditions are run.
Module: `src/streamreplace/benchmarks/streaming/ovo_bench.py`.

## Data

- Source: Hugging Face dataset `JoeLeelyf/OVO-Bench` at revision `fec29e3385747b5642d995370143ba92d2819bd2`: the
  official pre-chunked clips `chunked_videos.tar.partaa` ... `partao` (16 files with the README, about 153 GB).
  Annotations (`data/ovo_bench_new.json`), prompts and the scorer come with the benchmark's code, fetched at the
  pinned commit (`streamreplace fetch-upstream ovo_bench`; manifests/upstream.yaml).
- Rows: one per backward / real-time query (clip `chunked_videos/<id>.mp4`, the video from 0 s to the query time)
  and one per probe of a forward query (clip `chunked_videos/<id>_<i>.mp4`): 3,035 rows in all (1,468 queries and
  1,567 probes).
- Subset: the rows whose clip lasts at most 120 s, **1,576** rows (several children's native streams cannot run
  much longer clips). `prepare` reads the split tar as one stream and extracts only these clips.
- Sample id: `<id>` or `<id>_<probe>`.

## Prompt

The official templates (`constant.py`) through the official `OVOBenchOffline.build_prompt`, verbatim: a
multiple-choice prompt for backward and real-time queries; for REC the count of an activity so far; for SSR and
CRR a yes/no question about a step or an event. Generation: at most 128 new tokens.

The clip plays through the child's native stream and the question arrives at its end, the protocol the streaming
models' own OVO-Bench evaluations follow.

## Parsing and scoring

The official scorer (`utils/OVOBenchScore.py`), verbatim, no judge: a backward or real-time reply is right iff the
option letter occurs in it; REC iff the reply's digits joined equal the count; SSR / CRR iff the reply is exactly
"Y" / "N" for the probe's type or contains "Yes" / "No". The reply is first cleaned of think blocks and control
tokens; an empty reply is the official `None` (wrong).

## Metrics and statistics

- `acc_rows` (the paper's row accuracy): correct / all rows; unanswered rows are wrong.
- `overall`: the official Total Avg, the mean of the three modes' averages, each the mean of its tasks' accuracies
  (a forward task's accuracy is over its probes); `mode/*`, `task/*`.
- Paired statistics resample source videos.

## Deviations from the official protocol

- Only the rows asked within the first 120 s.
- Forward probes are scored one by one at their probe time, as the official offline scorer does; a child is not
  asked to choose its own response time.
