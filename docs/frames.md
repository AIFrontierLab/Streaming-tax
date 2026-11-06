# Frames: which video frames each model sees

The P/C/S comparison is only meaningful if the frames are controlled. In the offline pipeline (conditions **P** and
**C**) the parent and its children must see *exactly the same frames*, so that C - P measures the weights and nothing
else. In the native stream (condition **S**) each child reads the video the way its own release does. This page gives
the rules, the numbers they depend on, how they are recorded and checked, and the decoder details that can move a
frame. The code is in `src/streamreplace/media/`:

| Module | Contents |
|---|---|
| `frames.py` | `frame_indices`, `sample_frames`, `offline_frames`, `offline_indices`: the framework sampler |
| `video_meta.py` | decodable frame count (cached), reader fps, frame-list videos, the `video_meta` manifests |
| `qwen_sampling.py` | a CPU mirror of the frame selection of qwen-vl-utils 0.0.14 (the Qwen pairs' sampler), for checks |
| `digest.py` | `frame_digest`, the per-row fingerprint of the sampled frames |

## Offline frames (conditions P and C)

Each pair has **one** `video` block in `configs/pairs.yaml`, applied verbatim to every member of the pair. The frames
follow the child's streaming input as closely as the parent allows:

| Pair | Models | Sampler | Frames | Pixels per frame |
|---|---|---|---|---|
| p1 | LLaVA-OneVision-7B, MMDuet-7B | fixed step | 2 fps from frame 0; above 160 frames (videos longer than about 80 s), 160 uniform frames over the whole video (LLaVA-OneVision's 32k context) | long side scaled to 384, then black borders to 384 x 384, like MMDuet's own loader |
| p2 | Qwen2.5-VL-3B, ThinkStream-3B, MMDuet2-3B | qwen-vl-utils | about 2 fps: an even number of uniform frames, at least 4 and at most 768 | 200,704 to 401,408 (ThinkStream's evaluation; MMDuet2's 640-pixel frames fall inside) |
| p3 | Qwen3-VL-8B, AURA-8B | qwen-vl-utils | as p2 | at most 602,112 (AURA's stream) |
| p4 | LiveStar-InternVideo-8B, LiveStar-8B | fixed step | 1 fps from frame 0 | one 448 x 448 tile (the architecture's) |
| p5 | MOSS-VL-Instruct, MOSS-VL-Realtime | fixed step | 1 fps from frame 0 | MOSS-VL-Realtime's budget: at most 16,777,216 per frame and 65,536,000 per video |

The pixel keys are applied by the model adapters (`docs/models/offline-adapters.md`); the media code only chooses
*which* frames.

### Fixed-step and uniform frames (`frames.frame_indices`)

With `T` the decodable frame count and `r` the source frame rate of the video (next section):

- **Fixed step** at `fps`: step `s = r / fps`; `n = max(1, int(T / s))` frames; frame `i` is `min(T - 1, int(i * s))`.
  One frame every `1 / fps` seconds starting at frame 0; a trailing part shorter than `1 / fps` is not sampled.
- **Cap**: if that gives more than `max_frames` frames, `max_frames` uniform frames over the whole video are used
  instead (first and last frame included).
- **Uniform** `n` frames (`nframes`, or the cap): frame `i` is `round(i * (T - 1) / (n - 1))` with Python's `round`,
  which rounds halves to even (2.5 -> 2). Kept exactly: another rounding moves frames.
- **`sampler: linspace_int`**: the same positions truncated, `int(i * (T - 1) / (n - 1))`; on every video of the
  benchmarks that use it this equals `np.linspace(0, T - 1, n, dtype=int)`, the official samplers' expression.
- **`sampler: all`**: every frame (frame lists whose frames `prepare` already chose by the dataset's own rule).
- `fps` wins over `nframes` when both are set (the `defaults` block carries `nframes: 64` as a fallback only).

Example: a 12-frame video at 4 fps gives `[0, 4, 8]` at 1 fps and `[0, 4, 7, 11]` with `nframes: 4`.

### Qwen's sampler (`sampler: qwen_vl_utils`, pairs p2 and p3)

The Qwen adapters call qwen-vl-utils 0.0.14 `fetch_video` (`return_video_sample_fps=True,
return_video_metadata=True`) as in the Qwen2.5-VL and Qwen3-VL model cards. It reads the video with **torchcodec**
(`VideoDecoder`, exact seek mode), whose frame count `T'` and average fps `r'` it uses, and selects:

- `n = T' / r' * 2`, clamped to `[4, max_frames]` with `max_frames = min(768, T')` rounded down to an even number,
  at most `T'`, then rounded down to an even number;
- frame `i` = `torch.linspace(0, T' - 1, n)[i].round()`: positions in **float32**, rounded half to even;
- `sample_fps = n / T' * r'`, passed to Qwen2.5-VL as `fps`; Qwen3-VL receives the metadata instead.

On the paper's videos `n` is never 768 (the longest video, 185 s, gets 370 frames), and the 4-frame floor applies to
74 clips shorter than 2 seconds (70 in TimeBlind, 4 in TOMATO), whose sample fps is then above 2 (up to 9.2). Because
`n` is even, these frames are never padded.

`media/qwen_sampling.py` recomputes this selection from `T'` and `r'` alone (no torch, no decoding). The float32
detail matters: for long videos at many frames a position can fall on the other side of a .5 boundary than in exact
arithmetic (for a 12,342-frame video at 768 frames, position 528 is frame 8,496, not 8,495), so `linspace_f32`
reproduces torch's CPU result bit for bit: each value is one fused multiply-add rounded once to float32, which is
what torch 2.8 gives on an x86 AVX-512 CPU (checked). `fma=False` gives the variant with the product and the sum
rounded separately; both variants select the same frames on every video of the paper's benchmarks.

### Benchmark overrides (`offline_frames` in `configs/benchmarks.yaml`)

A benchmark can replace the frame-sampling keys of the pair (`fps`, `nframes`, `sampler`, `max_frames`,
`min_frames`, `last_seconds`) for the offline conditions only; pixel budgets and square padding stay the pair's.

| Benchmark | Offline frames | Why |
|---|---|---|
| Video-SafetyBench | 16 uniform frames (`linspace_int`) | the benchmark's official 16 frames |
| MCV-SafetyBench | 33 uniform frames (`linspace_int`) | the official 4 fps over the 8.25 s clips |
| EyePCR | 16 uniform frames (`linspace_int`) | fine-tuning experiments (`docs/generalization/finetune.md`) |
| LingoQA, OpenUAV-QA | `sampler: all` on frame lists | the datasets ship frame sequences; `prepare` picks the official frames |

With an override the Qwen pairs read the framework's frames too. Qwen's vision encoder takes frames in pairs, so the
Qwen adapters repeat the last frame of an odd count (MCV-SafetyBench: 33 frames sampled, 34 given to the model).

`blank: true` in a model's `video` settings (a control of the fine-tuning experiments) keeps the frame count and size
but makes every frame black.

## Frame count, frame rate and duration

The fixed-step and uniform rules need two numbers per video, measured as in the paper's runs:

- **`frames`, the decodable frame count**: the number of frames PyAV (FFmpeg) actually decodes, counted once by
  decoding the whole video and cached in `<cache root>/frame_counts/` (`STREAMREPLACE_CACHE`, see `paths.py`). The
  cache key is `sha256("<resolved path>|<size>|<mtime_ns>")[:24]` and an entry is `{"path", "frames"}`.
- **`fps`**: decord's `VideoReader.get_avg_fps()` (decord 0.6.0 in every model environment of the paper's runs).
  Without decord the PyAV header rate is used (`average_rate`, or 30).
- `duration = frames / fps`.

Why not the container's frame count: a clip cut without re-encoding keeps the frames from the previous keyframe up to
the cut, and the MP4 edit list hides them. FFmpeg-based readers (PyAV, and torchcodec in exact mode) skip the hidden
frames; decord's `len()` counts them and pads indices past the real end with the last frame. 172 TOMATO videos and
58 TimeBlind videos are such cuts, with up to 364 hidden frames. The decodable count is the paper's rule for every
model.

A video can also be a **frame list**, `<id>.frames.json` = `{"fps": f, "frames": [image paths relative to the
file]}`, which `prepare` writes for datasets released as image sequences; its frames are read as stored, `frames` is
the list's length and `fps` its recorded rate.

## The frame digest

Every offline video row records `info.frame_indices = "<number of frames>:<first 12 hex digits of
sha256(comma-joined frame indices)>"`, e.g. `"4:aa99d842319d"` for frames `[0, 4, 7, 11]`. For a frame list the hash
covers the stored frame paths instead of the indices. The digest is taken on the sampled frames, *before* any
model-specific padding, so it is the same for every member of a pair. `streamreplace score` refuses to score a
benchmark whose offline digests differ within a pair (`--no-frame-check` skips this), and two runs with the same
digest on a row saw the same frames. Stream rows record their own `info.schedule` digest
(`docs/models/stream-adapters.md`).

## Stream frames (condition S)

In the stream condition each child reads the video with its own release's input path; the question arrives with the
last frame. Details, including the deliberate deviations from the official evaluations, are in
`docs/models/stream-adapters.md`.

| Child | Stream frames | Compared with its pair's offline frames |
|---|---|---|
| MMDuet (p1) | the official loader (OpenCV) at 2 fps, square-padded, at most 400 frames | the same rate, but no 160-frame cap: videos above 80 s see more frames in the stream |
| ThinkStream (p2) | qwen-vl-utils chunks of 2 frames, chunk length `max(1, ceil(duration / 120))` s | about 2 fps for videos up to 120 s, fewer above |
| MMDuet2 (p2) | the fixed-step frames at 1 fps, two per turn | half the offline rate |
| AURA (p3) | the last `min(30, floor(duration))` one-second segments, 2 frames each | only the end of the video |
| LiveStar (p4) | the fixed-step frames at 1 fps | the same frames as offline |
| MOSS-VL-Realtime (p5) | the fixed-step frames at 1 fps, pushed on the wall clock | the same frames as offline |

For p4 and p5 the stream uses the same `frame_indices` function at the same rate, so offline and stream see
identical frame indices.

## The video_meta manifests

`manifests/video_meta/<benchmark>.jsonl` records, for every video of the eight video benchmarks, the values the
paper's runs used. One row per video, sorted, keyed by the video path relative to the benchmark's dataset folder (the
`video` field of `prepared/samples.jsonl`); no absolute paths:

```json
{"video": "media/data/-ApL8d6tX5U.mp4", "frames": 2233, "fps": 23.976023976023978, "header_frames": 2233,
 "torchcodec": {"frames": 2233, "fps": 23.97602406183525, "duration": 93.134708}}
```

`frames` is the decodable count, `fps` decord's rate, `header_frames` decord's `len()` (larger than `frames` for
edit-list videos), and `torchcodec` the exact-mode metadata that Qwen's sampler reads. With a manifest, every frame
index of the paper's offline rows and every stream schedule can be computed on a CPU without the videos (the stream
adapters' `schedule(req, meta)` take a manifest row).

| Benchmark | Videos | Edit-list videos | Above 160 frames at 2 fps (p1 cap) | Shorter than 2 s (Qwen 4-frame floor) |
|---|---:|---:|---:|---:|
| Video-MME (short) | 300 | 0 | 164 | 0 |
| TOMATO | 1,397 | 172 | 0 | 4 |
| TimeBlind | 1,200 | 58 | 0 | 70 |
| VideoReasonBench | 240 | 0 | 54 | 0 |
| Video-SafetyBench | 1,132 | 0 | (16 frames) | (16 frames) |
| MCV-SafetyBench | 2,920 | 0 | (33 frames) | (33 frames) |
| VMDT fairness | 2,000 | 0 | 104 | 0 |
| VMDT privacy | 200 | 0 | 0 | 0 |

Commands (run them with a model environment's Python: decord and torchcodec live there, not in the core install):

```bash
# check a local copy of the data against the shipped manifest: the same videos, and the same values when measured
# with the local libraries (all videos, or a seeded random fraction; --recount decodes again instead of using the cache)
streamreplace video-meta verify videomme tomato --sample 0.1 --recount
# prepare also compares the prepared video list with the manifest; --video-meta measures the videos as well
streamreplace prepare videomme --verify --video-meta
# write a manifest for another benchmark (needs decord; --allow-pyav-fps records PyAV's rate instead)
streamreplace video-meta build mybench --manifests manifests/video_meta
```

`verify` skips, and reports, the fields whose library is not installed, and exits with code 1 on any difference.

## Decoder caveats

- **Library versions.** The paper's runs used decord 0.6.0 for the frame rate, PyAV 13.1.0 for the decodable count,
  and torchcodec 0.7.0 with the FFmpeg 6 libraries for the Qwen pairs. A different FFmpeg build could decode a
  different number of frames from a damaged file; `video-meta verify` shows whether a local setup reproduces the
  manifest values.
- **decord versus PyAV frame rate.** On every video of the paper's benchmarks PyAV's header rate equals decord's, so
  the frame indices would not change without decord. This is not guaranteed for other videos, which is why
  `video-meta build` requires decord unless `--allow-pyav-fps` is given.
- **torchcodec frame rate.** torchcodec's exact-mode average fps differs from decord's in the last digits on many
  videos (1,135 of the 1,397 TOMATO videos). Only Qwen's sampler reads it, and the manifests record both. Its frame
  count equals the PyAV decodable count on every paper video.
- **Silent fallback.** qwen-vl-utils falls back to torchvision without an error when torchcodec cannot load (for
  example when the FFmpeg libraries are missing), which gives other frames; the Qwen adapters refuse that case.
- **OpenCV.** MMDuet's stream loader and the transformers rebuild of AURA's stream decode with OpenCV. Four
  TimeBlind videos cannot be decoded by OpenCV, so MMDuet's stream rows on them are errors (counted wrong).

## How this was checked against the paper's runs

- Unit tests (`tests/media/`): the rules above on generated videos (including an edit-list cut), the qwen-vl-utils
  selection and torch's float32 values recorded from the real libraries, and the paper's recorded digest, frame
  count, duration and sample fps for a few videos of every benchmark and pair, recomputed from the shipped manifests.
- On the paper's outputs, for every offline video row of the 88 offline jobs (eleven models on eight video
  benchmarks), the frames chosen from the shipped configuration and manifests, without decoding, give the recorded
  digest, frame count, duration and sample fps, and the digests agree within every pair: 154,088 rows on the
  evaluated samples, plus 10,318 rows that the original VMDT fairness runs produced for samples outside its fixed
  subset (two failed generations recorded no frames).
- The manifests' frame counts equal the original runs' frame-count cache for every video, and rebuilding the VMDT
  privacy manifest with `video-meta build` from those counts gives the shipped file byte for byte. Re-decoding a
  random 2% of the videos (188) with PyAV, decord and torchcodec reproduces every manifest value, and PyAV's header
  frame rate equals decord's on all 9,389 videos.
- `qwen_sampling` equals the real `fetch_video` (frame indices, frame count, sample fps, metadata) in 196 calls on
  66 videos of the eight benchmarks, with the p2 and p3 settings and five other selection settings, all read with
  torchcodec.
