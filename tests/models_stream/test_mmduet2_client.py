"""The MMDuet2 client patch (models/patches/mmduet2_client.py): version check, budget edit, kept definitions."""
from __future__ import annotations

import ast

import pytest

from streamreplace.models.patches import mmduet2_client as patch
from streamreplace.models.upstream import Upstream

SOURCE = '''import copy
from flask import Flask, request
from inference import ProactiveInferenceClient


class ProactiveInferenceAPIClient(ProactiveInferenceClient):
    def add_image(self, image):
        return self.model.generate(max_new_tokens=512, do_sample=False)


def get_args():
    return None


if __name__ == "__main__":
    get_args()
'''


def test_git_blob_ids():
    assert patch.git_blob_sha1(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"      # git's empty blob
    assert patch.git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_client_tree_keeps_imports_and_classes_and_patches_the_budget():
    tree = patch.client_tree(SOURCE)
    kinds = [type(node).__name__ for node in tree.body]
    assert kinds == ["Import", "ImportFrom", "ClassDef"]                       # flask, get_args and __main__ dropped
    code = ast.unparse(tree)
    assert "getattr(self, 'max_new_tokens', 512)" in code and "max_new_tokens=512" not in code


def test_client_tree_refuses_an_unexpected_source():
    with pytest.raises(patch.PatchError, match="exactly one"):
        patch.client_tree(SOURCE.replace("max_new_tokens=512,", "max_new_tokens=256,"))
    with pytest.raises(patch.PatchError, match="exactly one"):
        patch.client_tree(SOURCE + "\nx = dict(max_new_tokens=512, y=1)\n")


def test_checked_source_refuses_another_version(tmp_path):
    path = tmp_path / "api_server.py"
    path.write_text(SOURCE)
    with pytest.raises(patch.PatchError, match="git blob"):
        patch.checked_source(path, patch.API_SERVER_BLOB)
    assert patch.checked_source(path, patch.git_blob_sha1(path.read_bytes())) == SOURCE


@pytest.mark.requires_upstream("mmduet2")
def test_pinned_upstream_files_pass_the_check():
    upstream = Upstream()
    if not upstream.available("mmduet2"):
        pytest.skip("MMDuet2 upstream code not fetched (streamreplace fetch-upstream mmduet2)")
    code = patch.checked_source(upstream.path(patch.API_SERVER), patch.API_SERVER_BLOB)
    names = [node.name for node in patch.client_tree(code).body if isinstance(node, ast.ClassDef)]
    assert names == ["ProactiveTestAPIArguments", "ProactiveInferenceAPIClient"]
    frontend = ast.parse(patch.checked_source(upstream.path(patch.FRONTEND), patch.FRONTEND_BLOB))
    assert any(isinstance(n, ast.FunctionDef) and n.name == "resize_frame" for n in frontend.body)
