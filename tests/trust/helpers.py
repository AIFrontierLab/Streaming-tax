"""Helpers of the safety/trust benchmark tests: prepared data and fetched upstream code (tests that need them are
skipped when they are absent), and the official function definitions read from upstream source with ast."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from streamreplace import paths
from streamreplace.core.io import read_jsonl
from streamreplace.core.schema import Sample


def prepared_samples(name: str) -> list[Sample]:
    """The benchmark's prepared samples (as stored), or skip the test when it is not prepared."""
    path = paths.current().dataset_dir(name) / "prepared" / "samples.jsonl"
    if not path.is_file():
        pytest.skip(f"{name} is not prepared ({path} is missing)")
    return [Sample.from_dict(row) for row in read_jsonl(path)]


def raw_dir(name: str, probe: str) -> Path:
    """The benchmark's raw data folder, or skip the test when `probe` (a file inside it) is missing."""
    raw = paths.current().dataset_dir(name) / "raw"
    if not (raw / probe).exists():
        pytest.skip(f"{name} raw data is not downloaded ({raw / probe} is missing)")
    return raw


def require_upstream(name: str) -> Path:
    """The fetched upstream repository `name`, or skip the test (streamreplace fetch-upstream)."""
    from streamreplace.models.upstream import available, upstream_path
    if not available(name):
        pytest.skip(f"upstream code {name!r} is not fetched (streamreplace fetch-upstream {name})")
    return upstream_path(name)


def function_def(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def class_constants(path: Path, class_name: str) -> dict:
    """The literal class attributes of an upstream class (e.g. RJScore_Judge's prompt templates)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return {t.id: ast.literal_eval(n.value) for n in node.body if isinstance(n, ast.Assign) for t in n.targets}


def metric_row(sample_id: str, correct: bool, metadata: dict, **extra) -> dict:
    return {"id": sample_id, "correct": correct, "metadata": metadata, **extra}
