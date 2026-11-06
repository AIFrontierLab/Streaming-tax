"""The prompt text a model is sent for each sample, and per-folder prompt overrides.

The prompt comes from the benchmark module's prompt() applied to the sample's model-safe view (no answer, no private
metadata). P, C and S send the same text, and it must be byte-identical to the text the paper's runs recorded in each
row's `prompt` field: the run worker refuses to append to an output file whose stored prompts differ from the
rendered ones (use `run --fresh` to archive such outputs).

Prompt overrides (the prompt-robustness experiment). An outputs folder may hold `prompt_overrides.jsonl`, one row per
sample: {"dataset": <benchmark>, "id": <sample id>, "edits": [[old, new], ...]}. Every run on that folder applies each
row's edits, in order, to the rendered prompt; each `old` must occur exactly once in the text at the time it is
applied, so an edit can only touch the span it names. A folder with overrides for a benchmark needs a row for each of
its samples, and the overrides enter the run identity. The main outputs folder of the paper has none.

prompt_map_sha256() is the per-benchmark fingerprint of all prompts, recorded per benchmark in
manifests/prepared.yaml (`prompts_sha256`) and checked by `streamreplace prepare` (data/prepare.py check_prepared):
sha256 of the UTF-8 bytes of json.dumps({id: prompt}, sort_keys=True, ensure_ascii=False)."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

from ..core.identity import canonical_json
from ..core.io import read_jsonl

OVERRIDES_FILE = "prompt_overrides.jsonl"


def read_overrides(outputs: str | Path) -> dict[str, dict[str, list]] | None:
    """The outputs folder's prompt overrides as {benchmark: {sample id: edits}}, or None when the folder has none."""
    path = Path(outputs) / OVERRIDES_FILE
    if not path.is_file():
        return None
    edits: dict[str, dict[str, list]] = {}
    for row in read_jsonl(path):
        edits.setdefault(row["dataset"], {})[row["id"]] = row["edits"]
    return edits


def apply_edits(text: str, edits: list) -> str:
    """Each [old, new] replaces `old`, which must occur exactly once (so an edit can only touch the span it names)."""
    for old, new in edits:
        if text.count(old) != 1:
            raise ValueError(f"prompt edit target occurs {text.count(old)} times, not once: {old[:80]!r}")
        text = text.replace(old, new)
    return text


def render_prompts(name: str, samples, edits: Mapping[str, list] | None = None, bench=None) -> dict[str, str]:
    """id -> the text the model is sent: the benchmark's prompt of the sample's view, then the folder's edits for
    this benchmark (`edits`: {sample id: [[old, new], ...]}, None for no overrides)."""
    if bench is None:
        from ..benchmarks import registry
        bench = registry.get(name)
    prompts = {}
    for sample in samples:
        prompts[sample.id] = bench.prompt(bench.view(sample))
        if edits is not None:
            if sample.id not in edits:
                raise ValueError(f"{name}: the folder's {OVERRIDES_FILE} has no row for {sample.id}")
            prompts[sample.id] = apply_edits(prompts[sample.id], edits[sample.id])
    return prompts


def prompt_map_sha256(prompts: Mapping[str, str]) -> str:
    """sha256 (hex) of the canonical JSON of the {id: prompt} map (sorted keys, unescaped unicode, UTF-8): independent
    of the order of the samples. `prompts` = {sample id: bench.prompt(bench.view(sample))} over the prepared set."""
    return hashlib.sha256(canonical_json(dict(prompts)).encode("utf-8")).hexdigest()


prompt_map_digest = prompt_map_sha256     # earlier name, kept for callers that use it
