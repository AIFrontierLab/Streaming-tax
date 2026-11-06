"""Deterministic stand-in model for tests of the pipeline (planning, workers, resume, sharding, the plugin wiring).
It produces no paper number.

It answers "The answer is X.", where X is the reference answer with probability `accuracy` and a wrong answer
otherwise. Whether a sample is answered correctly is decided by a hash of the seed and the sample id, so every run,
shard and thread gives the same reply. Sample ids listed in `fail` raise, to exercise error rows. The mock serves
both modes: offline (conditions P and C) and stream (condition S, where it also reports the stream rate it was given).

Options (the `options:` block of a model whose adapter is `mock`; like every option they enter the run identity):
    accuracy     probability of answering correctly (default 1.0)
    seed         hash seed (default 0)
    fail         sample ids whose generation raises
    concurrency  requests in flight at once, as AURA's vLLM client does (default 1)

The mock is the only adapter that receives the reference answer (Request.reference): the run worker fills it in for
this adapter and for no other.

Load-time behaviour, for tests of the worker: the mock declares one load-time settings key, `stream.mock_load` (not
used by any real model), and reports the value it was loaded with in `info["load"]`; a fine-tuned id's LoRA path
handed over by the worker (load_lora) is reported in `info["lora"]`."""
from __future__ import annotations

import hashlib

from ..core.schema import Generation, Request
from .base import BaseAdapter, PreparedInputs


class MockModel(BaseAdapter):
    PROTOCOL_VERSION = 1
    # A load-time key that only the mock reads: jobs whose values differ must be served by separate processes.
    LOAD_KEYS: tuple[str, ...] = ("stream.mock_load",)

    def __init__(self, spec, settings: dict, upstream):
        super().__init__(spec, settings, upstream)
        self.options = dict(getattr(spec, "options", None) or {})
        self.loaded_with = (settings.get("stream") or {}).get("mock_load")
        self.lora: str | None = None

    @property
    def concurrency(self) -> int:              # samples in flight, as AURA's vLLM client
        return int(self.options.get("concurrency", 1))

    def load_lora(self, path: str) -> None:
        """Called by the worker right after loading, for a fine-tuned model id (options.lora)."""
        self.lora = str(path)

    def _answer(self, req: Request) -> Generation:
        if req.id in self.options.get("fail", []):
            raise RuntimeError("configured mock failure")
        roll = int(hashlib.sha256(f"{self.options.get('seed', 0)}:{req.id}".encode()).hexdigest(), 16) % 1000
        good = roll < 1000 * float(self.options.get("accuracy", 1.0))
        reference = req.reference or ""
        wrong = {"A": "B", "B": "A", "yes": "no", "no": "yes"}.get(reference, reference + "0")
        info = {"frames": len(req.images) or (1 if req.video else 0)}
        if self.loaded_with is not None:
            info["load"] = self.loaded_with
        if self.lora is not None:
            info["lora"] = self.lora
        return Generation(f"The answer is {reference if good else wrong}.", 5, "eos", info)

    def build_inputs(self, req: Request) -> PreparedInputs:
        return PreparedInputs(None, {"frames": len(req.images) or (1 if req.video else 0)})

    def generate(self, req: Request) -> Generation:
        return self._answer(req)

    def schedule(self, req: Request) -> dict:
        return {"fps": req.settings["stream"]["fps"]}

    def stream(self, req: Request, hooks=None) -> Generation:
        if hooks is not None:
            hooks.arm()
        generation = self._answer(req)
        generation.info["stream_fps"] = req.settings["stream"]["fps"]
        if hooks is not None:
            hooks.disarm()
            generation = hooks.finish(req, generation)
        return generation
