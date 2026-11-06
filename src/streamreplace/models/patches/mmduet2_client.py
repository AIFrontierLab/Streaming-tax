"""Runtime patch: MMDuet2's official proactive streaming client, usable in-process (paper condition S for MMDuet2).

MMDuet2 ships its streaming client inside a Flask demo server (demo/api_server.py, ProactiveInferenceAPIClient). The
MMDuet2 stream adapter needs the client without the server, and with the reply budget of the benchmark instead of
the hard-coded 512 tokens. Rather than copying upstream code, this module loads the fetched file itself and edits it
in memory:
- the one occurrence of `max_new_tokens=512,` becomes `max_new_tokens=getattr(self, 'max_new_tokens', 512),`, so the
  adapter sets `client.max_new_tokens` per request (unset, the client behaves as upstream);
- only the module's imports and classes are kept, without the Flask import (no routes, no argument parsing, no
  `__main__` block), and executed as the module `_streamreplace_mmduet2_api`;
- `resize_frame` (the demo front end's frame downscaling, max side 640 with cv2.INTER_AREA) is taken from
  demo/frontend.py the same way.
Both files are version-checked first: their git blob SHA-1 must equal the pinned upstream commit's (the values
manifests/upstream.yaml lists for MMDuet2 @7bd91d5), so the edit is never applied to a file it was not written for.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys
from types import ModuleType
from typing import Callable

API_SERVER = "mmduet2/demo/api_server.py"
FRONTEND = "mmduet2/demo/frontend.py"
# git blob SHA-1 of the two files at MMDuet2 commit 7bd91d5239adcb273df0977bc6c23faff950912c
API_SERVER_BLOB = "b7509463adefc7a341305511062b82d7f5b7a8d1"
FRONTEND_BLOB = "ccbf41bf6b42be3bfd9cf86d3a7385f563b17b22"
BUDGET = "max_new_tokens=512,"
BUDGET_PATCHED = "max_new_tokens=getattr(self, 'max_new_tokens', 512),"
MODULE = "_streamreplace_mmduet2_api"


class PatchError(RuntimeError):
    """The upstream file is not the version this patch was written for."""


def git_blob_sha1(data: bytes) -> str:
    """The blob id git gives a file's bytes (`git hash-object FILE`)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def checked_source(path: Path, blob: str) -> str:
    """The file's text after checking its blob id."""
    data = Path(path).read_bytes()
    found = git_blob_sha1(data)
    if found != blob:
        raise PatchError(f"{path}: git blob {found}, expected {blob} (MMDuet2 @7bd91d5); "
                         f"run `streamreplace fetch-upstream mmduet2`")
    return data.decode("utf-8")


def client_tree(code: str) -> ast.Module:
    """api_server.py with the budget edit, reduced to its imports (Flask's dropped) and classes."""
    if code.count(BUDGET) != 1:
        raise PatchError(f"expected exactly one {BUDGET!r} in the client source, found {code.count(BUDGET)}")
    tree = ast.parse(code.replace(BUDGET, BUDGET_PATCHED))
    tree.body = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef))
                 and not (isinstance(node, ast.ImportFrom) and node.module == "flask")]
    return tree


def load_client_module(upstream) -> ModuleType:
    """The patched client module (ProactiveInferenceAPIClient, ProactiveTestAPIArguments). Its own imports
    (`from inference import ...`, qwen_vl_utils, transformers) resolve against sys.path, so the caller puts
    mmduet2/demo and mmduet2/proactive_eval on it first (Upstream.use)."""
    path = upstream.path(API_SERVER)
    tree = client_tree(checked_source(path, API_SERVER_BLOB))
    module = ModuleType(MODULE)
    module.__file__ = str(path)
    sys.modules[MODULE] = module          # dataclasses look their module up while the class body is executed
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def load_resize_frame(upstream) -> Callable:
    """demo/frontend.py's resize_frame(frame, max_size=640), alone (the rest of the front end needs a camera)."""
    import cv2
    path = upstream.path(FRONTEND)
    tree = ast.parse(checked_source(path, FRONTEND_BLOB))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "resize_frame")
    scope: dict = {"cv2": cv2}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope["resize_frame"]
