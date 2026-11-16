# Adding a model: a new pair or a new streaming child

A new comparison needs an offline **parent**, a streaming **child** initialised from it, and the three conditions of
[protocol.md](protocol.md): P and C in one shared offline pipeline, S through the child's native streaming interface.
This page walks through what to write, what to configure and what to check before spending GPU hours.

| You need | Where |
|---|---|
| an offline adapter that runs parent and child identically (P, C) | `src/streamreplace/models/<name>.py`, or an existing adapter |
| a stream adapter for the child (S) | `src/streamreplace/models/<name>.py` |
| the adapters registered | `src/streamreplace/models/registry.py` |
| the model entries and the pair, with its one video protocol | an extra config file (`--config FILE`) or `configs/models.yaml` / `configs/pairs.yaml` |
| a Python environment per software stack | `envs/<env>/` |
| third-party code at a pinned commit, if the adapters import any | `manifests/upstream.yaml` |
| the child in the analysis roster | `analysis:` in an extra config file, or `configs/analysis.yaml` |
| optionally, the plugin for the child | [plugin.md](plugin.md#calibrating-a-new-child) |

## 1. Choose or write the offline adapter (P and C)

The parent and the child must go through the **same** offline code path: same frames, same prompt, same chat
template, same decoding. Usually the child keeps its parent's architecture, so one adapter serves both. The existing
offline adapters:

| Adapter | Serves | Environment |
|---|---|---|
| `qwen_offline` | Qwen2.5-VL and Qwen3-VL architectures (chosen by the checkpoint's `model_type`) | `modern` |
| `llava_offline` | LLaVA-OneVision (and a LoRA on it, kept unmerged) | `llava` |
| `internvl_offline` | LiveStar / InternVideo2.5 (remote model code from the fetched LiveStar repository) | `livestar` |
| `moss_offline` | MOSS-VL | `modern` |

A checkpoint of one of these architectures with the same processor can reuse the adapter; check it on a CPU first
(step 5). Otherwise write an adapter. The contract is in `src/streamreplace/models/base.py`; the smallest complete
example is the test model `src/streamreplace/models/mock.py`, and
[models/offline-adapters.md](models/offline-adapters.md) describes the four real ones. In short:

```python
"""Offline adapter of MyFamily (conditions P and C): ... (what it does and which pairs it serves)."""
from __future__ import annotations

from ..core.schema import Generation, Request
from ..media.digest import frame_digest
from ..media.frames import offline_frames
from .base import BaseAdapter, PreparedInputs
from .decoding import greedy


class MyOffline(BaseAdapter):
    PROTOCOL_VERSION = 1      # bump whenever the generated text can change: it is part of the run identity
    LOAD_KEYS = ()            # dotted settings keys read at load time (jobs that differ on them get separate workers)

    def __init__(self, spec, settings, upstream, load: bool = True):
        super().__init__(spec, settings, upstream)
        ...                   # processor and tokenizer always; the weights only when load is True

    def build_inputs(self, req: Request) -> PreparedInputs:
        """CPU only: the exact processor inputs and the row info, without a forward pass."""
        info = {}
        if req.video:
            frames, meta = offline_frames(req.video, req.settings["video"])   # the pair's frame protocol
            info.update(frames=len(frames), duration=meta["duration"], frame_indices=frame_digest(meta))
        ...
        return PreparedInputs(inputs, info)

    def generate(self, req: Request) -> Generation:
        prepared = self.build_inputs(req)
        ...                   # greedy(req.settings) + the configured repetition penalty
        return Generation(text, tokens=n, finish="length" if n >= budget else "eos", info=prepared.info)
```

Rules that keep the comparison fair and reproducible:

- Frames come from the pair's `video` settings through `media/frames.py` (or, for a qwen-vl-utils pair, from
  qwen-vl-utils as `qwen_offline` does), and every video row records `info.frame_indices` from
  `media/digest.frame_digest` *before* any model-specific padding. `streamreplace score` compares these digests within
  the pair and refuses to score when they differ.
- The prompt is `req.prompt`, unchanged; the request's sample view has no answer. Use the benchmark's system prompt
  (`req.settings["system"]`, only TOMATO has one) in place of the template default, and otherwise the template's own
  default.
- Decoding is `models/decoding.greedy(req.settings)` plus the model's `decoding.repetition_penalty`; call
  `decoding.check_repetition_penalty()` against the checkpoint's `generation_config.json` so a different checkpoint
  cannot silently change it.
- `finish` is `"length"` when the reply used the whole budget (the generation-cap analysis reads it), else `"eos"`.
- Third-party model code is imported inside `with self.upstream.use("<name>"):` from a repository fetched at a pinned
  commit (add it to `manifests/upstream.yaml` with every file's git blob SHA-1, then `streamreplace fetch-upstream
  <name>`); nothing third-party is copied into this repository.
- Optional capabilities used only by the generalization experiments: `load_lora(path)`, `encode()` /
  `train_forward()` and first-token letter logits (`models/base.py`).

## 2. Write the stream adapter (S)

The stream adapter runs the child the way its own release streams video, with the benchmark question asked when the
whole video has been streamed. Read [models/stream-adapters.md](models/stream-adapters.md) first: it shows, for six
children, what each decision looks like (frames, memory, how the question enters, how the answer is read, what counts
as silence) and records every deliberate difference from the official evaluations. The contract
(`src/streamreplace/models/base.py`):

- `schedule(req) -> dict`, **CPU only**: what `stream()` will feed (frames with their source indices, chunks,
  segments or turns), the question time on the media clock, and `"digest"`. A module-level `schedule(req, meta=None)`
  computes the same from a `manifests/video_meta` row, without decoding the video. `stream()` writes the digest to
  `info.schedule`.
- `stream(req, hooks=None) -> Generation`: with `hooks=None` exactly the child's native path. Silence (its silence
  marker, an empty reply, a missed deadline) is reported as `info["no_answer"] = True`; never replace silence by a
  guessed answer in the native path. A control that does so is a setting and marks its rows (MMDuet's
  `stream.force_answer` sets `info["forced"]`).
- The constructor takes `load: bool = True`; `load=False` builds a schedule-only adapter without weights.
- `concurrency`: requests in flight (1 unless the backend batches safely); `LOAD_KEYS` as offline.
- For the plugin: the run worker calls `hooks.begin(req)` before each `stream(req, hooks)`; the adapter calls the
  hooks only around its answer turn: `attach(decoder layers, tokenizer)` on its first `stream()` with these hooks,
  `arm()` right before the answer-turn prefill, `logits_processors()` or `on_step_logits()` while armed, `disarm()`
  after it, and `finish(req, generation)` to make the row. An adapter that cannot expose hidden states raises when
  hooks are passed. [plugin.md](plugin.md) lists where each child arms.

Register both adapters in `src/streamreplace/models/registry.py`: add `"my_offline":
"streamreplace.models.my_offline:MyOffline"` to `ADAPTERS` and `"my_offline": frozenset({"offline"})` to `MODES`
(`{"stream"}` for a stream adapter); the configuration is validated against both. Workers import an adapter module
only inside the model's own environment, but the test suite and CPU checks import it too: keep heavy imports (torch,
transformers, upstream code) inside the functions that need them, as the existing adapters do.

## 3. Environment

Reuse an environment whose packages fit (`streamreplace env list` shows which models use which), or define a new one:

```
envs/<env>/python-version       e.g. 3.12.14
envs/<env>/requirements.in      the direct requirements
envs/<env>/requirements.lock    the exact packages (uv pip compile ... -o requirements.lock)
```

```bash
streamreplace env build <env>          # uv venv + uv pip sync of the lock (+ checked patches, envs/patches/)
streamreplace env check <env>          # interpreter, packages against the lock, import probes
```

Set `ffmpeg: true` on a model whose decoding needs the FFmpeg 6 shared libraries (torchcodec, qwen-vl-utils); `env
build ffmpeg` provides them. See [environments.md](environments.md).

## 4. Configure the models and the pair

Put new models in an extra config file and pass it with the global `--config` option (it is merged over
`configs/`; new pairs, models, benchmarks and suites can all be added this way):

```yaml
# my_pair.yaml
models:
  my_parent:
    role: parent
    checkpoint: my-org/my-offline-parent@<40-hex commit>
    offline: {adapter: qwen_offline, env: modern, ffmpeg: true}
    decoding: {repetition_penalty: 1.05}          # must equal the checkpoint's generation_config.json
  my_child:
    role: child
    checkpoint: my-org/my-streaming-child@<40-hex commit>
    offline: {adapter: qwen_offline, env: modern, ffmpeg: true}
    stream: {adapter: my_stream, env: modern, ffmpeg: true}
    decoding: {repetition_penalty: 1.05}
pairs:
  p6:
    parent: my_parent
    children: [my_child]
    video: {sampler: qwen_vl_utils, fps: 2, min_pixels: 200704, max_pixels: 401408}
```

- Checkpoints are written `<Hugging Face repo>@<full commit>`; `streamreplace download` refuses anything else.
  `base` (a LoRA's base model) and `vision` (a separate vision tower) are optional.
- The pair's `video` block is applied verbatim to every member offline. Choose it to follow the **child's native
  streaming input** as closely as the parent's architecture allows (frame rate, frame count cap, pixel budget,
  padding), as the five paper pairs do (`configs/pairs.yaml` explains each). Benchmarks with an `offline_frames`
  override (Video-SafetyBench, MCV-SafetyBench) replace only the frame selection.
- `settings` (an extra settings layer) and `options` (adapter options) are optional; both enter the run identity.
- Parents are offline only. Each model belongs to one pair.

## 5. Check on a CPU before using a GPU

```bash
streamreplace --config my_pair.yaml download models my_parent my_child --dry-run          # resolve the pins
streamreplace --config my_pair.yaml download models my_parent my_child --processors-only  # no weights
streamreplace --config my_pair.yaml run --pairs p6 --benchmarks paper-core --dry-run      # the planned jobs
```

Then, in the model's environment (`<envs root>/<env>/bin/python` with `PYTHONPATH=src`, `CUDA_VISIBLE_DEVICES=` and
the prepared benchmark), build the inputs of a few requests with `load=False` and compare the members of the pair:

```python
from streamreplace.benchmarks import registry as benchmarks
from streamreplace.config.load import load_config
from streamreplace.core.schema import Request
from streamreplace.data import load_samples, render_prompts
from streamreplace.models import registry as models

config = load_config(extra=["my_pair.yaml"])
bench = benchmarks.get("videomme")
samples = load_samples("videomme", limit=3)
prompts = render_prompts("videomme", samples)
for model in ("my_parent", "my_child"):
    spec = config.model(model)
    settings = config.settings("videomme", model, "offline")
    adapter = models.create(spec.adapter("offline").adapter, spec, settings, load=False)
    for s in samples:
        req = Request(id=s.id, dataset="videomme", prompt=prompts[s.id], images=list(s.images), video=s.video,
                      settings=settings, kind=s.kind, choices=dict(s.choices), sample=bench.view(s))
        info = adapter.build_inputs(req).info
        print(model, s.id, info["frame_indices"], info.get("input_tokens"))
```

Parent and child must print the same `frame_indices` for every video. Do the same with the stream adapter's
`schedule(req)` (settings of mode `"stream"`) to see what the child will be fed and when the question arrives.

## 6. Run

```bash
streamreplace --config my_pair.yaml --outputs runs/p6-smoke run --pairs p6 --benchmarks videomme --limit 3 --gpus 0
streamreplace --config my_pair.yaml run --pairs p6 --protocols P,C,S --benchmarks paper-core --gpus 0,1
streamreplace --config my_pair.yaml score --judge --judge-gpus 0,1
streamreplace --config my_pair.yaml report
```

`--limit` is part of the run identity, so a smoke run belongs in its own `--outputs` folder. Pass the same `--config`
file to every command: the run identities, the frame check of `score` and the analysis read the pair from it.

## 7. Add the child to the analysis

`stats` computes the contrasts of the children listed in the analysis profile's roster, each against the parent of
its pair. Lists replace rather than extend, so give the whole list (order = table order):

```yaml
# my_analysis.yaml (pass it with --config next to my_pair.yaml)
analysis:
  roster:
    children: [mmduet_7b, thinkstream_3b, mmduet2_3b, aura_8b, livestar_8b, moss_vl_rt, my_child]
    models: [llava_ov_7b, mmduet_7b, qwen25vl_3b, thinkstream_3b, mmduet2_3b, qwen3vl_8b, aura_8b,
             livestar_iv_8b, livestar_8b, moss_vl, moss_vl_rt, my_parent, my_child]
  display:
    names: {my_parent: MyParent, my_child: MyChild}
```

```bash
streamreplace --config my_pair.yaml --config my_analysis.yaml stats --no-overlay
```

Holm families grow with the roster, so adjusted p values of the paper's children change when a child is added; the
paper's numbers come from the paper's roster ([statistics.md](statistics.md)).

## 8. The plugin for the new child

The plugin needs the child's stream adapter to call the hooks around its answer turn (step 2), offline adapters of
child and parent with the same number of decoder layers, hidden size and vocabulary (the steering calibration
compares their next-token distributions), and a calibration of the refusal direction, the steering vector and the
thresholds: [plugin.md](plugin.md#calibrating-a-new-child). In `configs/plugin.yaml` (or an extra config's
`plugin:` section), add the child to `children`, and declare its answer turn under `answer_turn` as
`{silence: ..., logits: ...}`: its silence marker (one token of its tokenizer, `"NO REPLY"`, or `null`) and the path
on which its stream adapter hands the hooks the answer turn's logits (`processor`, `first_step`, `opening` or
`deadline`; [plugin.md](plugin.md#per-child) lists the choices). Arms with Respond or a live / record gate, and the
calibration's record and harmless stages, refuse a child without that entry. If its stream evicts history, add a
Recall settings delta under `recall` as well.
