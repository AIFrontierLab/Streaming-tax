"""Benchmark data as every condition (P, C, S) and every later stage reads it.

- prepare.py  raw data -> `<data>/<benchmark>/prepared/samples.jsonl` + manifest; checks against the paper's counts,
              samples sha256 and prompt fingerprint (manifests/prepared.yaml) and the video_meta manifests
- samples.py  load_samples(): prepared order, absolute media paths, an experiment folder's subset, `--limit` in the
              benchmark's smoke order
- prompts.py  the prompt each sample is sent, per-folder prompt overrides (exact single-occurrence edits), and the
              prompt fingerprint

Typical use:
    from streamreplace.data import load_samples, render_prompts
    samples = load_samples("videomme")
    prompts = render_prompts("videomme", samples)"""
from .prompts import (OVERRIDES_FILE, apply_edits, prompt_map_digest, prompt_map_sha256, read_overrides,
                      render_prompts)
from .samples import SUBSET_FILE, dataset_dir, load_samples, prepared_dir, read_subset, samples_file, smoke_order

__all__ = ["OVERRIDES_FILE", "SUBSET_FILE", "apply_edits", "dataset_dir", "load_samples", "prepared_dir",
           "prompt_map_digest", "prompt_map_sha256", "read_overrides", "read_subset", "render_prompts", "samples_file",
           "smoke_order"]
