# Adding a benchmark

A benchmark is one Python module that turns the raw release into samples and defines the prompt, the answer
parsing, the correctness rule and the metrics, plus a few lines of configuration. Once registered, every pair runs it
under the conditions its media allow (image: P and C; video: P, C and S), and `score`, `report` and `stats` treat it
like the paper's benchmarks.

| Step | File |
|---|---|
| 1. the module | `src/streamreplace/benchmarks/<group>/<name>.py` |
| 2. register it | `src/streamreplace/benchmarks/registry.py` |
| 3. configure it | `configs/benchmarks.yaml`, or an extra config file passed with `--config` |
| 4. get and prepare the data | `<data>/<name>/raw/`; optionally a pinned source in `manifests/datasets.yaml` |
| 5. record what the paper-style checks compare | `manifests/prepared.yaml`, `manifests/video_meta/<name>.jsonl` |
| 6. add it to the statistics | `metrics` of the analysis profile (`configs/analysis.yaml` or `--config`) |
| 7. tests | `tests/<your dir>/` |

## 1. The module

The interface is the `Benchmark` protocol in `src/streamreplace/benchmarks/base.py`. Only `MEDIA` and `prepare()` are
required; everything else falls back to the shared implementations in `src/streamreplace/benchmarks/common/`, which
most of the paper's benchmarks use. A short real example is `benchmarks/image/mmstar.py`; the test fixture
`tests/fixtures/demo_benchmark.py` is a complete synthetic video benchmark in 50 lines.

```python
"""MyBench: what it measures, how many items, which split; how it is prompted, parsed and scored (docs page)."""
from __future__ import annotations

from pathlib import Path

from .. import common
from ..base import JudgePolicy, MetricSpec
from ...core.schema import Sample

NAME = "mybench"
MEDIA = "video"                       # "image" (P and C) or "video" (P, C and S)
JUDGE = JudgePolicy.FALLBACK          # the extraction judge reads replies the rules cannot parse
EXPECTED_COUNT = 1200                 # prepare refuses a release with another number of test samples
METRICS = [MetricSpec("acc", "acc", cluster="video")]     # first = headline; the unit structure of the statistics
PRIVATE_METADATA = frozenset()        # metadata keys the model must not see (ground truth, references)


def prepare(raw: Path, media: Path) -> list[Sample]:
    """<data>/mybench/raw -> samples, in a fixed order with stable ids; media written to media/ (or read in place
    under raw/)."""
    samples = []
    for row in common.read_parquet(raw / "test.parquet"):
        samples.append(Sample(id=str(row["id"]), question=row["question"], answer=row["answer"], kind="mcq",
                              choices=dict(zip("ABCD", row["options"])), video=str(media / row["video"]),
                              metadata={"category": row["category"]}))
    return samples
```

`Sample.kind` is one of `mcq`, `yesno`, `numeric`, `text` and `open`; it selects the shared prompt suffix ("Answer with
the option's letter from the given choices directly." and its siblings), the shared parser and the shared
correctness rule. Optional hooks, each overriding the shared default:

| Hook | Default | Override when |
|---|---|---|
| `prompt(view) -> str` | question + options + the kind's answer instruction (`common/prompts.py`) | the benchmark has an official prompt template |
| `parse(view, text) -> (pred, status)` | letter / yes-no / number / final-answer rules (`common/parsing.py`) | the benchmark has an official answer extractor |
| `correct(sample, pred) -> bool` | normalised equality, numeric tolerance from `metadata` | scoring is not plain equality |
| `metrics(rows) -> dict` | `acc` plus per-`category` accuracy, in percent, 2 decimals | the official metric differs (group accuracy, composites) |
| `stratum(sample)` | none | a metric is reported per split (e.g. harmful / benign) |
| `smoke(samples)` | about 50 ids in a fixed hash order, one item at a time | `--limit` should keep groups together, e.g. `return common.smoke(samples, group="video")` |
| `judge_messages(sample, raw)`, `judge_result(sample, reply)` | the generic extraction request | the benchmark has an official LLM extraction prompt |

Rules that the paper's benchmarks follow and a new one should too:

- `prompt()` and `parse()` receive a `SampleView`: no answer, and no `PRIVATE_METADATA`. A parser must never use the
  answer.
- The prompt is the only text the model receives, identical for P, C and S. Keep it deterministic: the run refuses
  to continue an output file whose stored prompts differ from the rendered ones.
- Keep official strings, regexes and rules verbatim and say where they come from. Code you would rather not copy
  (an official scorer, a judge prompt with an unclear licence) can be fetched at a pinned commit: add an entry to
  `manifests/upstream.yaml` (every file with its git blob SHA-1), run `streamreplace fetch-upstream <name>`, and read
  it at run time with the helpers of `benchmarks/common/io.py` (`common.upstream_path()`,
  `common.upstream_functions()`, `common.upstream_assignment()`, ...), as TOMATO's module reads the official parser.
- Media files must exist under the dataset folder when `prepare()` returns; the paths may be absolute, and `prepare`
  stores them relative to the dataset folder.
- Judge policies: `none` (rules only), `fallback` (the extraction judge reads rule failures), `always` (a judge
  decides every parsed answer), `classifier` and `rjscore` (safety judges; an errored row then counts safe), `lingo`
  (LingoQA). See [judges.md](judges.md). A silent stream row is never parsed or sent to an extraction judge.

## 2. Register it

Add one line to `BENCHMARKS` in `src/streamreplace/benchmarks/registry.py`:

```python
    "mybench": f"{_PACKAGE}.video.mybench",
```

The configuration alone is not enough: a benchmark that is configured but not registered is reported as "Unknown
benchmark".

## 3. Configure it

```yaml
# configs/benchmarks.yaml, or an extra config file (streamreplace --config mybench.yaml ...)
benchmarks:
  mybench: {media: video, suite: video, settings: {generation: {max_new_tokens: 512}}}
suites:
  mine: [mybench, videomme]
```

- `media` must equal the module's `MEDIA`.
- `settings` is the benchmark's settings layer: the token budget (`generation.max_new_tokens`; default 512) and a
  dataset system prompt (`system`), which only the offline adapters use.
- `offline_frames` replaces the pair's frame selection in the offline conditions only (for example
  `{nframes: 16, sampler: linspace_int}` for a benchmark whose official evaluation uses 16 uniform frames); pixel
  budgets and padding stay the pair's. Do not set it just to save compute: the pair protocol is what keeps P, C and
  S comparable ([protocol.md](protocol.md)).

## 4. Data

Put the raw release in `<data>/mybench/raw/` (`<data>` is `$STREAMREPLACE_DATA`, by default
`$STREAMREPLACE_HOME/data`), then:

```bash
streamreplace prepare mybench         # raw -> <data>/mybench/prepared/samples.jsonl (+ manifest.json, media/)
```

`prepare` prints one JSON line with the sample count, the sha256 of `samples.jsonl` and the fingerprint of every
rendered prompt (`prompts_sha256`). To make the download reproducible for others, add the source to
`manifests/datasets.yaml` pinned to a full revision or commit, with sizes and checksums (the file's header documents
the source types); `streamreplace download datasets mybench --dry-run` then resolves it and `download datasets
mybench` fetches and checks it. Unpinned sources are refused.

## 5. Record the fingerprints

So that others can confirm they evaluate exactly your items and prompts:

- add `mybench: {count, samples_sha256, prompts_sha256}` with the values `prepare` printed to
  `manifests/prepared.yaml`; `streamreplace prepare mybench --verify` then checks a local preparation against them
  (until the row exists, `prepare` reports the benchmark with `"verified": false` and a note);
- for a video benchmark, write the per-video frame counts and rates with `streamreplace video-meta build mybench`
  (run it with a model environment's Python: it needs decord, and torchcodec for the Qwen pairs' metadata). The
  manifest lets every frame selection and stream schedule be recomputed on a CPU, and `prepare --verify` compares
  the prepared video list with it ([frames.md](frames.md)).

## 6. Statistics

`report` works at once: it uses the module's own `metrics()`. The paired contrasts of `stats` are defined by the
analysis profile: add one entry per reported metric to its `metrics` list, with the unit structure the bootstrap
should resample ([statistics.md](statistics.md), "Analysis units"):

```yaml
# an extra config file (lists replace: repeat the profile's metrics list, then add yours), or configs/analysis.yaml
analysis:
  metrics:
    # ... the existing entries of configs/analysis.yaml ...
    - {dataset: mybench, name: acc, kind: acc, cluster: video, family: mybench}
```

`kind` is `acc`, `group` (all members of a group correct), `mme`, `safe` (reported as an attack success rate) or
`vmdt`; `cluster` names the resampling unit (`video`, a metadata key, or the item when absent); `family` is the Holm
family; `modes: [offline]` restricts an image benchmark to C-P. Keep the module's `METRICS` consistent with these
entries.

## 7. Tests

Put tests under your own `tests/<dir>/`: the parsing rules on fixed replies (no data needed), and checks on the
prepared data marked `@pytest.mark.requires_data("mybench")` (skipped where the data is absent), or
`@pytest.mark.requires_upstream("<name>")` when they need fetched upstream code. `python -m pytest -q tests/<dir>`
runs them on a CPU.

## 8. Run it

```bash
streamreplace run --pairs all --protocols P,C,S --benchmarks mybench --dry-run
streamreplace --outputs runs/mybench-smoke run --pairs p2 --benchmarks mybench --limit 3 --gpus 0
streamreplace run --pairs all --protocols P,C,S --benchmarks mybench --gpus 0,1,2,3
streamreplace score --benchmarks mybench --judge --judge-gpus 0,1
streamreplace report --benchmarks mybench
```

Write a page `docs/benchmarks/<name>.md` (data source and subset, prompt, parsing, judge use, metric, deviations from
the official protocol), following the existing pages.
