"""Model adapter interfaces (conditions P and C use an OfflineAdapter, condition S a StreamAdapter).

An adapter module lives in `streamreplace/models/<name>.py`, is listed in models/registry.py, and is imported only
inside the model's own Python environment (torch, transformers and the upstream code are imported lazily there).

Construction. Every model adapter is constructed as `Adapter(spec: ModelSpec, settings: dict, upstream: Upstream,
load: bool = True)`: the model's config entry, the resolved settings of the first job it serves (load-time keys are
listed in LOAD_KEYS), and the resolver for fetched third-party code (models/upstream.py). `load=False` builds a
CPU-only adapter without model weights: build_inputs() (offline) and schedule() (stream) work on it, generate() and
stream() do not. `registry.create(name, spec, settings, upstream, load=False)` passes the keyword through (the
test-only mock takes no `load`).

OfflineAdapter (P and C): the whole media at once, framework-sampled frames for video, greedy decoding
(models/decoding.py).

StreamAdapter (S): the model's native streaming interface replays the video on the media clock and the question is
asked at the end of the video. `hooks=None` must be exactly the native path; the plugin passes AnswerTurnHooks. The
contract every stream adapter implements:
- schedule(req) is CPU-only and returns a JSON-serialisable dict of what stream() feeds the model: "adapter", the
  frames / segments / chunks / turns (with their source frame indices), "question_time" (seconds on the media clock),
  adapter-specific fields (e.g. ThinkStream's chunks and seed, AURA's segments and window start), and "digest" = the
  fingerprint (core.io.fingerprint) of the dict without "digest". stream() writes that digest to the row as
  info["schedule"]. Each adapter module also exposes a module-level `schedule(req, meta=None)` that takes a
  manifests/video_meta row instead of decoding the video.
- stream(req, hooks=None) returns a core.schema.Generation; silence is reported as info["no_answer"] = True.
- concurrency: requests in flight at once; 1 except AURA's vLLM client. With hooks, the run worker sends requests
  one at a time (the hooks hold per-request state). An adapter that cannot host the plugin (AURA's vLLM client)
  raises when hooks are passed; the plugin runs AURA on its transformers rebuild (aura_stream_hf).
- With hooks, each adapter calls them only around its own answer turn and documents its arming point in its module
  docstring (docs/models/stream-adapters.md and docs/plugin.md list them per child):
      hooks.attach(find_layers(root), tokenizer[, eos=...])  on the first stream() with these hooks, before the
                                                             adapter installs its own call sites; a call that
                                                             raises is retried on the next stream()
      hooks.arm(first=...)                                   immediately before the answer-turn prefill
      logits_processors() or on_step_logits(step, logits)    while armed, depending on the decoding path
      hooks.disarm()                                         right after the answer turn
      hooks.deadline_readout(fork)                           ThinkStream only, when the turn ended unanswered
      hooks.finish(req, generation)                          makes the final row
  The run worker calls hooks.begin(req) before each stream(req, hooks), so on a run's first request begin() comes
  before attach() (AnswerTurnHooks lists the order). The plugin never reaches into private adapter state; adapters
  hand it model handles (decoder layers, tokenizer, logits) only.

Optional adapter capabilities:
- load_lora(path) -> list[str]: every offline adapter has it. It attaches a fine-tuned id's LoRA adapter (the folder
  `path` with adapter_config.json and adapter_model.safetensors) unmerged, with float32 LoRA weights, and returns the
  names of the wrapped modules. It raises on any mismatch (saved keys the model lacks, injected modules the file does
  not cover, another placement, trainable non-LoRA parameters), on an adapter built with load=False, and (LLaVA) on
  a LoRA for the unmerged MMDuet. The run worker calls it right after construction for a model whose config sets
  `options.lora`, and refuses adapters without it, so a fine-tuned id never silently evaluates its base model. The
  class attribute LORA_LAYERS = (dotted path of the decoder-layer list inside the adapter's `model`, number of
  LoRA-wrapped linear layers per decoder layer) says where the LoRA must sit; models/lora.py holds the shared attach
  and check code.
- Trainable: encode() / train_forward() for LoRA fine-tuning (generalization/finetune); the training inputs are built
  by the same code that builds evaluation inputs.
- first-token option-letter logits: an adapter that sets CAPTURES_LETTER_LOGITS = True records, for every mcq
  request, Generation.info["letter_logits"] = {letter: round(logit, 4)} for the single-token option letters, read
  from the first generated token's logits (models/lm_head.py: head_wrapper() with capture + letter_logits()). The
  prompt-robustness analysis reads them.
These are used by the generalization experiments only, never by the paper's P/C/S runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

from ..core.schema import Generation, Request

if TYPE_CHECKING:
    from ..config.schema import ModelSpec
    from .upstream import Upstream

EOT = "<|im_end|>"      # every family's assistant turn ends with it (Qwen, LLaVA qwen_1_5, InternVL internvl2_5, MOSS)


@dataclass
class PreparedInputs:
    """CPU-only result of OfflineAdapter.build_inputs(): the processor inputs plus the info the row will carry
    (frames, frame_size, sample_fps, input_tokens, frame_indices digest). Used to verify requests without a GPU."""
    inputs: Any
    info: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class OfflineAdapter(Protocol):
    PROTOCOL_VERSION: int                  # bump when the generated text can change; part of the run identity
    LOAD_KEYS: tuple[str, ...]             # dotted settings keys read at load time: jobs that differ on them are
                                           # never served by one process (a process loads with its first job's settings)

    def build_inputs(self, req: Request) -> PreparedInputs:
        """CPU-only: the processor inputs and the row info for this request (no forward pass)."""

    def generate(self, req: Request) -> Generation: ...

    def close(self) -> None: ...


@runtime_checkable
class StreamAdapter(Protocol):
    PROTOCOL_VERSION: int
    LOAD_KEYS: tuple[str, ...]
    concurrency: int                       # requests in flight; 1 except AURA's vLLM client

    def schedule(self, req: Request) -> dict:
        """CPU-only: frames / segments / chunks / turns, question time and the digest of what stream() feeds (see
        the module docstring for the fields)."""

    def stream(self, req: Request, hooks: AnswerTurnHooks | None = None) -> Generation:
        """The native stream; with hooks, the plugin's hooks are called around the answer turn (module docstring)."""

    def close(self) -> None: ...


@runtime_checkable
class AnswerTurnHooks(Protocol):
    """The plugin's view of a stream adapter's answer turn (plugin/hooks.py implements it). Each stream adapter
    documents its arming point and owns the upstream-specific code; the plugin never reaches into adapter state.

    Call order, per request: begin (the run worker, before stream()) -> attach (inside stream(); only the first
    successful call on a hooks object acts, so it does its work on the run's first request, after that request's
    begin; a call that raises keeps nothing and is retried on the next stream()) -> arm ->
    logits_processors / on_step_logits -> disarm -> [deadline_readout] -> finish. plugin/hooks.py also accepts attach
    before the first begin. deadline_readout is called (ThinkStream) whenever the answer turn ended unanswered; the
    hooks call the fork only for a closed-choice item with Respond on.

    Attribute `eager_decode: bool` (optional, default False): True when the hooks act on decode-step forwards (parent-
    restoration steering). ThinkStream then runs the answer turn's decode steps eagerly instead of by CUDA-graph replay,
    where forward hooks would not run."""

    def settings(self, settings: dict) -> dict:
        """Recall: the stream settings with the plugin's settings delta merged in. The run planner merges the same
        delta into the job's settings (so it enters the run identity) and the worker does not call settings(); it is
        kept for callers that build an arm's settings outside the planner."""

    def attach(self, decoder_layers, tokenizer, *, eos: int | None = None) -> None:
        """After load: forward hooks are registered on the decoder layers (a ModuleList, or a module searched with
        plugin.guard.find_layers), inert until armed. tokenizer: the child's. eos: the token id that ends the answer
        turn, for a child whose turn does not end with <|im_end|> (default: <|im_end|>). Only the first successful
        call acts: a call that raises keeps and registers nothing, and the adapter calls attach() again on its next
        stream(). So an adapter calls it on its first stream() with these hooks, before installing its own call
        sites."""

    def begin(self, req: Request) -> None:
        """Per request, before streaming (called by the run worker): resets per-request state and leaves the hooks
        disarmed until the answer turn's arm(); the live gate picks its threshold from the request's answer format
        (closed-choice or free-form)."""

    def arm(self, first: bool = False) -> None:
        """Immediately before the answer-turn prefill. first=True (MMDuet): the gate reads the first forward after
        arming instead of the prefill's last position."""

    def logits_processors(self) -> list:
        """HF generate path (AURA's transformers rebuild, MMDuet2, LiveStar): a list of LogitsProcessors (first-step
        logit capture, silence bans, halt); the adapter wraps it in a LogitsProcessorList for the answer turn's call."""

    def on_step_logits(self, step: int, logits) -> Any:
        """LM-head path (MMDuet's greedy loop, MOSS-VL-Realtime's session): called with the LM head's output tensor
        [1, positions, vocab] of every forward while armed (`step` counts from 0). Returns the logits to decode from:
        a non-None return value replaces the head's output."""

    def deadline_readout(self, fork: Callable[[], Any]) -> None:
        """ThinkStream only, after an answer turn that ended unanswered: fork() re-prefills the query chunk unsteered
        with an empty thought and returns float32 CPU logits at <response>. The hooks call it only for a closed-choice
        item with Respond on."""

    def disarm(self) -> None:
        """Right after the answer turn."""

    def finish(self, req: Request, gen: Generation) -> Generation:
        """The final row: the fixed refusal when the live gate fired, else the option readout for a closed-choice
        reply that is silent or unparseable. Writes info["plugin"] = {native, guard: {proj, fired, ...}, readout}."""


@runtime_checkable
class Trainable(Protocol):
    """Optional offline-adapter capability (generalization/finetune)."""

    def encode(self, req: Request, answer: str | None = None) -> dict:
        """{"inputs": model kwargs on CPU, "n": supervised tokens (answer + EOT), "info"}, built by the same code
        as generate()'s inputs; answer=None encodes the evaluation prompt only."""

    def train_forward(self, enc: dict) -> Any:
        """Logits of the last n + 1 positions of encode()'s inputs (the LM head projects only those)."""


class BaseAdapter:
    """Convenience base for adapters: stores the constructor arguments and provides the defaults. Subclasses accept
    the `load` keyword themselves (see the module docstring)."""
    PROTOCOL_VERSION = 1
    LOAD_KEYS: tuple[str, ...] = ()
    CAPTURES_LETTER_LOGITS = False
    concurrency = 1

    def __init__(self, spec: ModelSpec, settings: dict, upstream: Upstream):
        self.spec, self.settings, self.upstream = spec, settings, upstream

    def close(self) -> None:
        pass


def is_trainable(adapter) -> bool:
    return callable(getattr(adapter, "encode", None)) and callable(getattr(adapter, "train_forward", None))


def captures_letter_logits(adapter) -> bool:
    return bool(getattr(adapter, "CAPTURES_LETTER_LOGITS", False))


def supervised_length(tokenize: Callable[[str], list[int]], prompt: str, answer: str) -> int:
    """Trailing tokens of prompt + answer + EOT that a training step supervises (answer + EOT). Refuses a sample whose
    answer changes the prompt's tokenization (a merge across the boundary), so the trained prompt is exactly the one
    evaluated; tokenize(text) -> token ids."""
    head, full = tokenize(prompt), tokenize(prompt + answer + EOT)
    if full[:len(head)] != head or len(full) - len(head) < 2 or full[-1] != tokenize(EOT)[-1]:
        raise ValueError(f"answer {answer[:40]!r} does not tokenize as a clean continuation of the prompt")
    return len(full) - len(head)
