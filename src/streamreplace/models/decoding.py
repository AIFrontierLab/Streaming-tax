"""Decoding settings of the offline conditions (P and C) and the per-question seed of sampled streams.

Offline decoding is greedy with the benchmark's token budget. greedy() deliberately leaves repetition_penalty out:
the original runs inherited it from each checkpoint's generation_config.json (1.05 for LLaVA-OneVision, MMDuet,
Qwen2.5-VL-3B, ThinkStream-3B and MMDuet2; 1.0 for Qwen3-VL-8B and AURA). configs/models.yaml now states the value
per model (`decoding.repetition_penalty`), and adapters call check_repetition_penalty() against the checkpoint so a
different checkpoint cannot silently change it. A checkpoint folder that does not exist is reported as missing
(missing_checkpoint(), with the download command), never read as a checkpoint without a generation config. LiveStar's
adapter passes 1.05 itself and MOSS's offline API passes 1.0 explicitly, as in the original runs."""
from __future__ import annotations

import json
from pathlib import Path

from ..core.schema import seed_of


def greedy(settings: dict) -> dict:
    """Generation kwargs of every offline adapter that calls HF generate()."""
    return {"do_sample": False, "temperature": None, "top_p": None, "top_k": None, "num_beams": 1,
            "max_new_tokens": int(settings["generation"]["max_new_tokens"])}


def sample_seed(sample_id: str) -> int:
    """Per-question seed for adapters whose official decoding samples (reproducible, independent of order):
    int(sha256(id).hexdigest()[:8], 16). ThinkStream's replicate k uses the id f"{id}#seed{k}"."""
    return seed_of(sample_id)


def missing_checkpoint(model_id: str, folder: Path, what: str = "checkpoint", detail: str = "") -> FileNotFoundError:
    """The error for a checkpoint folder that is absent or incomplete: names the folder and the download command
    (`streamreplace download models <id>` fetches every checkpoint a model uses, a LoRA's base included)."""
    who = f"{model_id}: " if model_id else ""
    return FileNotFoundError(f"{who}{what} not found at {folder}{f' ({detail})' if detail else ''}; run "
                             f"`streamreplace download models {model_id or '<model id>'}`")


def checkpoint_repetition_penalty(checkpoint_dir: Path, model_id: str = "") -> float:
    """repetition_penalty of a checkpoint's generation_config.json (1.0 when the folder has none, the transformers
    default). A folder that does not exist raises FileNotFoundError (missing_checkpoint) instead of reading as 1.0."""
    folder = Path(checkpoint_dir)
    if not folder.is_dir():
        raise missing_checkpoint(model_id, folder, detail="no such folder")
    path = folder / "generation_config.json"
    if not path.exists():
        return 1.0
    return float(json.loads(path.read_text(encoding="utf-8")).get("repetition_penalty", 1.0))


def check_repetition_penalty(expected: float | None, checkpoint_dir: Path, model_id: str = "") -> None:
    """Raises ValueError when configs/models.yaml's decoding.repetition_penalty disagrees with the checkpoint's
    generation_config.json (the value the original runs actually used), FileNotFoundError when the checkpoint folder
    does not exist. expected=None skips the check (adapters that set the value themselves)."""
    if expected is None:
        return
    found = checkpoint_repetition_penalty(checkpoint_dir, model_id)
    if abs(found - float(expected)) > 1e-9:
        if (Path(checkpoint_dir) / "generation_config.json").exists():
            source = f"the checkpoint's generation_config.json says {found}"
        else:       # an incomplete download is the likelier cause than a wrong configuration
            source = (f"the checkpoint folder {checkpoint_dir} has no generation_config.json (transformers' default "
                      f"{found}); if its download is incomplete, run `streamreplace download models "
                      f"{model_id or '<model id>'}`")
        raise ValueError(f"{model_id or checkpoint_dir}: configs say repetition_penalty={expected}, {source}")
