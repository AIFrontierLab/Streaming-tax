"""manifests/upstream.yaml and the fetcher behind `streamreplace fetch-upstream` (offline: downloads are faked)."""
from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile

import pytest

from streamreplace import paths
from streamreplace.provisioning import upstream
from streamreplace.provisioning.upstream import Edit, Entry, FetchError, FilePatch, ManifestError, git_blob_sha1

PRISTINE = b"def f(x):\n    if x < 1:\n        return 0\n    return 1\n"
PATCHED = b"def f(x):\n    if x <= 1:\n        return 0\n    return 1\n"
README = b"hello\n"


def toy_entry(**changes) -> Entry:
    patch = FilePatch("pkg/mod.py", git_blob_sha1(PRISTINE), (Edit("if x < 1:", "if x <= 1:"),), "toy")
    fields = dict(name="toy", repo="https://github.com/someone/toy", commit="a" * 40, license="MIT", group="core",
                  used_for="test", files={"README.md": git_blob_sha1(README), "pkg/mod.py": git_blob_sha1(PATCHED)},
                  patches={"pkg/mod.py": patch})
    fields.update(changes)
    return Entry(**fields)


def remote(entry: Entry, files: dict[str, bytes]) -> dict[str, bytes]:
    return {entry.raw_url(path): data for path, data in files.items()}


def fake_get(table: dict[str, bytes], calls: list | None = None):
    def get(url: str) -> bytes:
        if calls is not None:
            calls.append(url)
        if url not in table:
            raise FetchError(f"GET {url}: HTTP Error 404")
        return table[url]
    return get


def tarball(entry: Entry, files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    prefix = f"{entry.owner_repo[1]}-{entry.commit}"
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path, data in {**files, "extra/unlisted.txt": b"x"}.items():
            info = tarfile.TarInfo(f"{prefix}/{path}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


# ------------------------------------------------------------------------------------------------ hashing, patches

def test_git_blob_sha1_matches_git_hash_object():
    assert git_blob_sha1(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    assert git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_file_blob_sha1_streams_the_same_value(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"\x00\x01" * 1_000_000)
    assert upstream.file_blob_sha1(path) == git_blob_sha1(path.read_bytes())


def test_patch_applies_only_to_the_pinned_upstream_file():
    patch = toy_entry().patches["pkg/mod.py"]
    assert patch.apply(PRISTINE) == PATCHED
    with pytest.raises(FetchError, match="upstream blob"):
        patch.apply(PATCHED)                                      # already edited: not the file it was written for
    twice = FilePatch("p", git_blob_sha1(b"a a"), (Edit("a", "b"),))
    with pytest.raises(FetchError, match="expects 1 occurrence"):
        twice.apply(b"a a")
    assert FilePatch("p", git_blob_sha1(b"a a"), (Edit("a", "b", 2),)).apply(b"a a") == b"b b"


# ------------------------------------------------------------------------------------------------ manifest parsing

def minimal(**changes) -> dict:
    entry = {"repo": "https://github.com/o/r", "commit": "b" * 40, "license": "none", "group": "core",
             "files": {"a.py": "c" * 40}}
    entry.update(changes)
    return {"version": 1, "upstream": {"r": entry}}


@pytest.mark.parametrize("changes, message", [
    ({"commit": "main"}, "full 40-hex commit"),
    ({"commit": "abc1234"}, "full 40-hex commit"),
    ({"files": {"a.py": "xyz"}}, "git blob"),
    ({"files": {"../a.py": "c" * 40}}, "clean relative path"),
    ({"files": {"/abs/a.py": "c" * 40}}, "clean relative path"),
    ({"group": "misc"}, "group"),
    ({"repo": "https://gitlab.com/o/r"}, "github.com"),
    ({"patches": {"b.py": {"upstream_blob": "d" * 40, "edits": [{"old": "x", "new": "y"}]}}}, "not in files"),
    ({"patches": {"a.py": {"upstream_blob": "d" * 40, "edits": []}}}, "no edits"),
    ({"group": "reference"}, "never fetched"),
    ({"surprise": 1}, "unknown keys"),
])
def test_manifest_validation_refuses_unpinned_or_malformed_entries(changes, message):
    with pytest.raises(ManifestError, match=message):
        upstream.parse_manifest(minimal(**changes))


def test_select_resolves_all_groups_and_refuses_reference_entries():
    manifest = upstream.parse_manifest({"version": 1, "upstream": {
        "a": minimal()["upstream"]["r"],
        "b": {**minimal()["upstream"]["r"], "group": "generalization"},
        "c": {**minimal()["upstream"]["r"], "group": "reference", "fetch": False}}})
    assert [e.name for e in upstream.select(manifest, None)] == ["a", "b"]
    assert [e.name for e in upstream.select(manifest, ["all"])] == ["a", "b"]
    assert [e.name for e in upstream.select(manifest, ["generalization", "a"])] == ["b", "a"]
    with pytest.raises(ManifestError, match="provenance only"):
        upstream.select(manifest, ["c"])
    with pytest.raises(ManifestError, match="unknown upstream"):
        upstream.select(manifest, ["zzz"])


# ------------------------------------------------------------------------------------------------ fetching

def test_raw_fetch_applies_the_patch_verifies_and_is_idempotent(tmp_path):
    entry = toy_entry()
    calls: list[str] = []
    get = fake_get(remote(entry, {"README.md": README, "pkg/mod.py": PRISTINE}), calls)
    status = upstream.fetch(entry, tmp_path, method="raw", get=get, log=lambda m: None)
    assert status.ok and status.files == 2 and status.action == "fetched (raw)"
    assert (tmp_path / "toy" / "pkg" / "mod.py").read_bytes() == PATCHED
    marker = json.loads((tmp_path / "toy" / upstream.MARKER).read_text())
    assert marker["commit"] == entry.commit and marker["patched"] == ["pkg/mod.py"]
    assert not (tmp_path / upstream.STAGING).exists()
    assert len(calls) == 2
    assert upstream.fetch(entry, tmp_path, get=get).action == "kept"           # verified tree: no download
    assert len(calls) == 2
    upstream.fetch(entry, tmp_path, get=get, force=True)
    assert len(calls) == 4


def test_verify_reports_missing_and_modified_files_and_fetch_repairs(tmp_path):
    entry = toy_entry()
    get = fake_get(remote(entry, {"README.md": README, "pkg/mod.py": PRISTINE}))
    assert upstream.verify(entry, tmp_path).state == "missing"
    upstream.fetch(entry, tmp_path, get=get)
    (tmp_path / "toy" / "README.md").write_bytes(b"changed\n")
    status = upstream.verify(entry, tmp_path)
    assert status.state == "modified" and status.mismatched[0][0] == "README.md"
    (tmp_path / "toy" / "README.md").unlink()
    assert upstream.verify(entry, tmp_path).state == "incomplete"
    assert upstream.fetch(entry, tmp_path, get=get).ok


def test_wrong_upstream_content_is_refused_and_the_existing_tree_is_kept(tmp_path):
    entry = toy_entry()
    upstream.fetch(entry, tmp_path, get=fake_get(remote(entry, {"README.md": README, "pkg/mod.py": PRISTINE})))
    bad = fake_get(remote(entry, {"README.md": b"tampered\n", "pkg/mod.py": PRISTINE}))
    with pytest.raises(FetchError, match="pinned"):
        upstream.fetch(entry, tmp_path, method="raw", get=bad, force=True)
    assert upstream.verify(entry, tmp_path).ok                                # the old verified tree is untouched


def test_tarball_method_and_auto_fallback(tmp_path):
    entry = toy_entry()
    archive = tarball(entry, {"README.md": README, "pkg/mod.py": PRISTINE})

    def download(url: str, target: Path) -> Path:
        assert url == entry.tarball_url()
        target.write_bytes(archive)
        return target

    status = upstream.fetch(entry, tmp_path / "a", method="tarball", download=download)
    assert status.ok and not (tmp_path / "a" / "toy" / "extra").exists()     # only the listed files are taken
    messages: list[str] = []
    status = upstream.fetch(entry, tmp_path / "b", method="auto", get=fake_get({}), download=download,
                            log=messages.append)
    assert status.ok and status.action == "fetched (tarball)" and "tarball" in messages[0]


def test_tarball_missing_a_listed_file_fails(tmp_path):
    entry = toy_entry()
    archive = tarball(entry, {"README.md": README})
    with pytest.raises(FetchError, match="lacks 1 listed file"):
        upstream.fetch(entry, tmp_path, method="tarball",
                       download=lambda url, target: (target.write_bytes(archive), target)[1])


def test_local_method_accepts_pristine_and_already_patched_copies(tmp_path):
    entry = toy_entry()
    for name, content in (("pristine", PRISTINE), ("patched", PATCHED)):
        source = tmp_path / name / "toy"
        (source / "pkg").mkdir(parents=True)
        (source / "README.md").write_bytes(README)
        (source / "pkg" / "mod.py").write_bytes(content)
        status = upstream.fetch(entry, tmp_path / f"out-{name}", method="local", source=tmp_path / name)
        assert status.ok and (tmp_path / f"out-{name}" / "toy" / "pkg" / "mod.py").read_bytes() == PATCHED


def test_reference_entries_cannot_be_fetched(tmp_path):
    with pytest.raises(FetchError, match="reference-only"):
        upstream.fetch(toy_entry(group="reference", fetch=False), tmp_path)


def test_fetch_all_reports_failures_and_continues(tmp_path):
    good, bad = toy_entry(), toy_entry(name="other", repo="https://github.com/someone/other")
    table = remote(good, {"README.md": README, "pkg/mod.py": PRISTINE})
    results = upstream.fetch_all([bad, good], tmp_path, method="raw", get=fake_get(table), log=lambda m: None)
    assert results["toy"].ok
    assert results["other"].state == "failed" and results["other"].action.startswith("failed")


# ------------------------------------------------------------------------------------------------ the real manifest

@pytest.fixture(scope="module")
def manifest():
    return upstream.load_manifest()


def test_shipped_manifest_pins_every_repository(manifest):
    assert len(manifest) == 21
    assert sorted(n for n, e in manifest.items() if not e.fetch) == ["aura", "moss"]
    assert len(upstream.fetchable(manifest)) == 19
    for entry in manifest.values():
        assert upstream.HEX40.fullmatch(entry.commit)
        assert entry.repo.startswith("https://github.com/")
        for path, patch in entry.patches.items():
            assert patch.upstream_blob != entry.files[path]                  # an edit always changes the blob
    assert sorted((n, p) for n, e in manifest.items() for p in e.patches) == [
        ("llava", "llava/model/multimodal_encoder/builder.py"), ("thinkstream", "thinkstream/model/inference.py"),
        ("videoreasonbench", "utils/eval_utils.py")]


# Every "<upstream name>/<path>" the package reads must be pinned (files), or be a folder of pinned files.
USED_FILES = [
    "videomme/eval_your_results.py", "timeblind/utils.py", "timeblind/llm_judge.py", "tomato/src/parse_result.py",
    "videoreasonbench/utils/eval_utils.py", "openuav_qa/eval/src/uav/vqa/uav_dataset.py",
    "openuav_qa/eval/src/uav/vqa/run_uav.py", "mmduet2/demo/api_server.py", "mmduet2/demo/frontend.py",
    "video_safetybench/model_inference/llava_video.py", "video_safetybench/eval/evaluate.py",
    "video_safetybench/eval/evaluator.py", "video_safetybench/eval/scenario.json",
    "video_safetybench/model_results/LLaVA-Video-72B-Qwen2/video_safetybench_benign/video_safetybench_benign.json",
    "vmdt/VMDT/fairness/video_text/factual_accuracy/factual_accuracy.csv", "vmdt/VMDT/models/v2t_models/qwen2vl.py",
    "promptbench/promptbench/prompt_attack/transformations.py", "mcv_safetybench/Evaluation/policy.txt",
    "mcv_safetybench/Attack/Explicit/Qwen2.5VL_7B_exp.py", "mcv_safetybench/Attack/Implicit/Qwen2.5VL_7B_imp.py",
    "livestar/inference/streaming_infer.py", "lingoqa/benchmark/judge.py", "harmbench/eval_utils.py",
    "thinkstream/thinkstream/model/inference.py", "mmduet/demo/liveinfer.py",
    "llava/llava/model/multimodal_encoder/builder.py",
]
USED_FOLDERS = ["vmdt/VMDT/fairness/video_text", "vmdt/VMDT/privacy/video_text", "video_safetybench/eval",
                "livestar/inference", "mmduet2/demo", "mmduet2/proactive_eval", "mcv_safetybench/Attack/Explicit/Chagned_question"]


@pytest.mark.parametrize("relative", USED_FILES)
def test_files_the_package_reads_are_pinned(manifest, relative):
    name, path = relative.split("/", 1)
    assert manifest[name].fetch and path in manifest[name].files


@pytest.mark.parametrize("relative", USED_FOLDERS)
def test_folders_the_package_reads_hold_pinned_files(manifest, relative):
    name, folder = relative.split("/", 1)
    assert any(p.startswith(folder + "/") for p in manifest[name].files)


def test_mmduet2_client_patch_pins_the_same_blobs(manifest):
    from streamreplace.models.patches import mmduet2_client
    files = manifest["mmduet2"].files
    assert files["demo/api_server.py"] == mmduet2_client.API_SERVER_BLOB
    assert files["demo/frontend.py"] == mmduet2_client.FRONTEND_BLOB


@pytest.mark.requires_upstream
def test_fetched_upstream_trees_verify(manifest):
    root = paths.current().upstream
    present = [n for n in upstream.fetchable(manifest) if (root / n).is_dir()]
    if not present:
        pytest.skip(f"no fetched upstream code under {root} (streamreplace fetch-upstream all)")
    for name in present:
        status = upstream.verify(manifest[name], root)
        assert status.ok, f"{name}: {status.summary()}"
