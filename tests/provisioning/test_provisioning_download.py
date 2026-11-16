"""manifests/{checkpoints,datasets}.yaml and the pinned downloader (offline: every request is faked)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from streamreplace.config.load import load_config
from streamreplace.paths import default_config_dir, resolve
from streamreplace.provisioning import download
from streamreplace.provisioning.download import DownloadError, ManifestError, Refused, Response, Source
from streamreplace.provisioning.upstream import git_blob_sha1


@pytest.fixture(scope="module")
def config():
    return load_config(default_config_dir())


@pytest.fixture(scope="module")
def checkpoints():
    return download.load_checkpoints()


@pytest.fixture(scope="module")
def datasets():
    return download.load_datasets()


def workspace(tmp_path):
    return resolve(home=tmp_path / "home", environ={})


# ------------------------------------------------------------------------------------------------ checkpoints

def test_checkpoint_manifest_agrees_with_the_configuration(config, checkpoints):
    """Every checkpoint models.yaml / judges.yaml reference is listed at the same revision, with the same users,
    and the manifest lists nothing else."""
    referenced: dict[str, dict] = {}
    for kind in ("models", "judges"):
        for ident, fields in download.config_checkpoints(config, kind).items():
            for field, ref in fields.items():
                entry = referenced.setdefault(ref.repo, {"revision": ref.revision, "used_by": {}})
                assert entry["revision"] == ref.revision
                entry["used_by"][ident] = field
    assert sorted(referenced) == sorted(checkpoints)
    for repo, entry in referenced.items():
        assert checkpoints[repo].revision == entry["revision"], repo
        assert dict(checkpoints[repo].used_by) == entry["used_by"], repo
        assert checkpoints[repo].dir == repo
    assert len([c for c in checkpoints.values() if c.role != "judge"]) == 12
    assert len([c for c in checkpoints.values() if c.role == "judge"]) == 5


def test_plan_checkpoints_dedupes_and_targets_the_checkpoints_root(tmp_path, config, checkpoints):
    paths = workspace(tmp_path)
    plans = download.plan_checkpoints(config, "models", ["all"], paths, checkpoints)
    assert len(plans) == 12
    llava = next(p for p in plans if p.repo == "lmms-lab/llava-onevision-qwen2-7b-ov")
    assert llava.users == ["llava_ov_7b.checkpoint", "mmduet_7b.base"]
    assert llava.target == paths.checkpoints / "lmms-lab" / "llava-onevision-qwen2-7b-ov"
    judges = download.plan_checkpoints(config, "judges", None, paths, checkpoints)
    assert sorted(p.repo for p in judges) == sorted(r for r, c in checkpoints.items() if c.role == "judge")
    one = download.plan_checkpoints(config, "models", ["mmduet_7b"], paths, checkpoints, processors_only=True)
    assert len(one) == 3 and all(p.processors_only for p in one)          # checkpoint + base + vision tower
    by_repo = download.plan_checkpoints(config, "judges", ["Qwen/Qwen2.5-72B-Instruct"], paths, checkpoints)
    assert [p.users for p in by_repo] == [["rjscore.checkpoint"]]
    with pytest.raises(ManifestError, match="unknown model"):
        download.plan_checkpoints(config, "models", ["gpt5"], paths, checkpoints)


class FakeModel:
    def __init__(self, refs):
        self._refs = refs

    def checkpoints(self):
        return self._refs


class FakeConfig:
    def __init__(self, refs):
        self.models = {"m": FakeModel(refs)}

    def section(self, name):
        return {}


def test_unpinned_or_conflicting_checkpoints_are_refused(tmp_path, checkpoints):
    from streamreplace.config.schema import CheckpointRef
    paths = workspace(tmp_path)
    unpinned = FakeConfig({"checkpoint": CheckpointRef.parse("org/model")})
    with pytest.raises(Refused, match="no pinned revision"):
        download.plan_checkpoints(unpinned, "models", None, paths, checkpoints)
    short = FakeConfig({"checkpoint": CheckpointRef.parse("org/model@main")})
    with pytest.raises(Refused, match="not a full commit"):
        download.plan_checkpoints(short, "models", None, paths, checkpoints)
    conflict = FakeConfig({"checkpoint": CheckpointRef.parse("Qwen/Qwen3-VL-8B-Instruct@" + "1" * 40)})
    with pytest.raises(Refused, match="manifests/checkpoints.yaml pins"):
        download.plan_checkpoints(conflict, "models", None, paths, checkpoints)
    new = FakeConfig({"checkpoint": CheckpointRef.parse("org/new-model@" + "2" * 40)})   # not in the manifest: allowed
    assert download.plan_checkpoints(new, "models", None, paths, checkpoints)[0].revision == "2" * 40


def api(answers: dict[str, Response]):
    calls = []

    def request(url, method="GET", max_bytes=None):
        calls.append((method, url))
        return answers.get(url, Response(404, {}, b""))
    request.calls = calls
    return request


def test_resolve_checkpoint_checks_the_revision_on_the_hub(tmp_path, config, checkpoints):
    plan = download.plan_checkpoints(config, "models", ["qwen3vl_8b"], workspace(tmp_path), checkpoints)[0]
    url = f"{download.hf_endpoint()}/api/models/{plan.repo}/revision/{plan.revision}"
    siblings = [{"rfilename": n} for n in ("config.json", "model-00001-of-00002.safetensors", "tokenizer.json")]
    good = api({url: Response(200, {}, json.dumps({"sha": plan.revision, "siblings": siblings}).encode())})
    assert download.resolve_checkpoint(plan, good).ok
    plan.processors_only = True
    assert "2 of 3 files" in download.resolve_checkpoint(plan, good).detail
    moved = api({url: Response(200, {}, json.dumps({"sha": "f" * 40}).encode())})
    assert not download.resolve_checkpoint(plan, moved).ok
    assert not download.resolve_checkpoint(plan, api({})).ok


# ------------------------------------------------------------------------------------------------ datasets

def test_dataset_manifest_covers_every_benchmark_and_suite(config, datasets):
    assert set(config.benchmarks) <= set(datasets)
    assert set(datasets) - set(config.benchmarks) == {"advbench", "alpaca", "xstest"}  # calibration and probe texts
    for suite, members in config.suites.items():
        assert [d.name for d in download.select_datasets(datasets, [suite], config.suites)] == list(members)
    assert [d.name for d in download.select_datasets(datasets, ["calibration"])] == ["advbench", "alpaca", "xstest"]
    assert len(download.select_datasets(datasets, None)) == len(datasets)
    with pytest.raises(ManifestError, match="unknown dataset"):
        download.select_datasets(datasets, ["imagenet"])


def test_every_dataset_source_is_pinned(datasets):
    for ds in datasets.values():
        for source in ds.sources:
            s = source.spec
            if source.type == "hf":
                assert download.HEX40.fullmatch(s["revision"]) and s["files"] >= 1
            elif source.type == "github":
                assert download.HEX40.fullmatch(s["commit"])
            elif source.type in ("url", "gdrive"):
                assert download.HEX64.fullmatch(s["sha256"]) and s["size"] > 0
            else:
                assert s["pinned"] is False and ds.name == "openuav_qa"


def test_paper_revisions_and_checksums(datasets):
    """Spot checks of pins the paper's numbers depend on (prepared samples hash these files)."""
    hf = {d.name: d.sources[0].spec for d in datasets.values() if d.sources[0].type == "hf"}
    assert hf["videomme"]["revision"] == "ead1408f75b618502df9a1d8e0950166bf0a2a0b"
    assert hf["videomme"]["files"] == 23                          # README, 1 parquet, subtitle.zip, 20 video archives
    assert hf["timeblind"]["folders"]["videos/"]["files"] == 1200
    tags = datasets["timeblind"].sources[1].spec
    assert tags["commit"] == "03129551918c0830f3ee727485eb9cfb437e21e5"
    assert tags["files"]["tags.json"]["sha256"] == "5257e1a82621976f9277f05d0ce77f8684f5821a4a83b63704fca21eefa68cfe"
    mmbench = datasets["mmbench"].sources[0].spec
    assert mmbench["md5"] == "6be6358b6c8d6f633e7cd3c602a49c6b" and mmbench["to"] == "MMBench_DEV_EN_legacy.tsv"
    harm = datasets["harmbench"].sources[0].spec
    assert harm["commit"] == "8e1604d1171fe8a48d8febecd22f600e462bdcdd" and harm["tree_files"] == 227
    assert datasets["video_safetybench"].sources[0].spec["revision"] == "b04daeb5f6c185df47e2aacb4d30b87912c51114"


def test_terms_and_unpinned_sources_need_explicit_flags(datasets):
    lingoqa, openuav = datasets["lingoqa"], datasets["openuav_qa"]
    with pytest.raises(Refused, match="--accept-terms lingoqa"):
        download.check_allowed([lingoqa])
    download.check_allowed([lingoqa], accepted=["lingoqa"])
    with pytest.raises(Refused, match="--allow-unpinned"):
        download.check_allowed([openuav])
    download.check_allowed([openuav], allow_unpinned=True)
    download.check_allowed([datasets["videomme"], datasets["harmbench"]])


@pytest.mark.parametrize("source, message", [
    ({"type": "hf", "repo": "o/r", "revision": "main"}, "full commit"),
    ({"type": "github", "repo": "o/r", "commit": "a" * 40, "files": {"f": {"size": 1, "blob": "b" * 40}}}, "sha256"),
    ({"type": "github", "repo": "o/r", "commit": "a" * 40}, "either `files` or `tree`"),
    ({"type": "url", "url": "https://x/y", "to": "y", "size": 3}, "missing"),
    ({"type": "gdrive", "id": "x", "to": "../y", "size": 1, "sha256": "c" * 64}, "relative"),
    ({"type": "modelscope", "repo": "o/r", "root": "d", "pinned": True}, "pinned: false"),
    ({"type": "ftp", "url": "ftp://x"}, "type must be"),
])
def test_dataset_source_validation(tmp_path, source, message):
    path = tmp_path / "datasets.yaml"
    path.write_text(json.dumps({"version": 1, "datasets": {"x": {"group": "image", "sources": [source]}}}))
    with pytest.raises(ManifestError, match=message):
        download.load_datasets(path)


def test_resolve_sources_with_faked_endpoints(datasets):
    ds = datasets["timeblind"]
    hf, gh = ds.sources
    names = ["README.md", "data.jsonl", ".gitattributes"] + [f"videos/{i}.mp4" for i in range(1200)]
    hf_url = f"{download.hf_endpoint()}/api/datasets/{hf.spec['repo']}/revision/{hf.spec['revision']}"
    body = json.dumps({"sha": hf.spec["revision"], "siblings": [{"rfilename": n} for n in names]}).encode()
    raw_url = f"https://raw.githubusercontent.com/{gh.spec['repo']}/{gh.spec['commit']}/tags.json"
    request = api({hf_url: Response(200, {}, body), raw_url: Response(200, {"Content-Length": "46131"})})
    assert download.resolve_source(ds, hf, request).ok
    assert download.resolve_source(ds, gh, request).ok
    short = api({hf_url: Response(200, {}, json.dumps({"sha": hf.spec["revision"], "siblings": [
        {"rfilename": n} for n in names[:-1]]}).encode()), raw_url: Response(200, {"Content-Length": "5"})})
    assert "1201 files match" in download.resolve_source(ds, hf, short).detail
    assert not download.resolve_source(ds, gh, short).ok
    harm = datasets["harmbench"].sources[0]
    tree = api({f"https://api.github.com/repos/centerforaisafety/HarmBench/git/trees/{harm.spec['commit']}?recursive=1":
                Response(200, {}, json.dumps({"tree": [{"path": "data/a.csv", "type": "blob", "sha": "0" * 40}]}).encode())})
    assert "digest" in download.resolve_source(datasets["harmbench"], harm, tree).detail


def test_tree_digest_is_order_independent():
    a = download.tree_digest([("data/b", "2" * 40), ("data/a", "1" * 40)])
    assert a == download.tree_digest([("data/a", "1" * 40), ("data/b", "2" * 40)])
    assert a == hashlib.sha256(f"data/a {'1' * 40}\ndata/b {'2' * 40}\n".encode()).hexdigest()


# ------------------------------------------------------------------------------------------------ files on disk

def synthetic_dataset(tmp_path) -> tuple[download.Dataset, Path, dict]:
    raw = tmp_path / "raw"
    content = {"README.md": b"readme\n", "data/q.parquet": b"PAR1" * 100, "videos/a.mp4": b"v" * 10,
               "videos/b.mp4": b"w" * 20, "file.tsv": b"a\tb\n", "data/x/1.csv": b"1\n", "data/x/2.csv": b"2\n"}
    for rel, data in content.items():
        (raw / rel).parent.mkdir(parents=True, exist_ok=True)
        (raw / rel).write_bytes(data)
    sha = lambda b: hashlib.sha256(b).hexdigest()
    hf = Source("hf", {"repo": "o/r", "revision": "a" * 40, "allow": ["README.md", "data/q.parquet", "videos/*"],
                       "files": 4, "checksums": {"README.md": {"size": 7, "blob": git_blob_sha1(content["README.md"])},
                                                  "data/q.parquet": {"size": 400, "sha256": sha(content["data/q.parquet"])}},
                       "folders": {"videos/": {"files": 2, "bytes": 30}}})
    url = Source("url", {"url": "https://example.org/file.tsv", "to": "file.tsv", "size": 4,
                         "sha256": sha(content["file.tsv"]), "md5": hashlib.md5(content["file.tsv"]).hexdigest()})
    tree = Source("github", {"repo": "o/r", "commit": "b" * 40, "tree": "data/x/", "tree_files": 2,
                             "tree_sha256": download.tree_digest([("data/x/1.csv", git_blob_sha1(b"1\n")),
                                                                  ("data/x/2.csv", git_blob_sha1(b"2\n"))])})
    return download.Dataset("toy", "image", "MIT", (hf, url, tree)), raw, content


def test_verify_dataset_checks_sums_folders_and_trees(tmp_path):
    ds, raw, _ = synthetic_dataset(tmp_path)
    assert download.verify_dataset(ds, raw) == []
    (raw / "videos" / "b.mp4").unlink()
    (raw / "file.tsv").write_bytes(b"A\tb\n")
    (raw / "data" / "x" / "2.csv").write_bytes(b"3\n")
    problems = download.verify_dataset(ds, raw)
    assert any(p.startswith("videos/: 1 files") for p in problems)
    assert any(p.startswith("file.tsv: sha256 differs") for p in problems)
    assert any(p.startswith("data/x/") for p in problems)
    assert download.check_file(raw / "file.tsv", {"size": 4, "sha256": "0" * 64}, size_only_above=1) is None


def test_download_url_and_github_files_are_checked_before_they_land(tmp_path, monkeypatch):
    ds, _, content = synthetic_dataset(tmp_path)
    url_source = ds.sources[1]
    served = {"https://example.org/file.tsv": content["file.tsv"]}

    def fake_download(url, target, attempts=4, timeout=300.0):
        target.write_bytes(served[url])
        return target
    monkeypatch.setattr(download, "http_download", fake_download)
    raw = tmp_path / "new"
    toy = download.Dataset("toy", "image", "MIT", (url_source,))
    assert download.download_dataset(toy, raw, log=lambda m: None) == []
    assert (raw / download.SUMS_FILE).read_text() == f"{url_source.spec['sha256']}  file.tsv\n"
    record = json.loads((raw / download.RECORD_FILE).read_text())
    assert record["sources"][0]["url"] == "https://example.org/file.tsv"
    served["https://example.org/file.tsv"] = b"evil"
    with pytest.raises(DownloadError, match="sha256 differs"):
        download.download_dataset(toy, tmp_path / "other", force=True, log=lambda m: None)
    assert not (tmp_path / "other" / "file.tsv").exists()


def test_unzip_unpacks_next_to_the_archive_and_refuses_escapes(tmp_path):
    archive = tmp_path / "images.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("images/val/seg/0.jpg", b"jpeg")
    download._unzip(archive, log=lambda m: None)
    assert (tmp_path / "images" / "val" / "seg" / "0.jpg").read_bytes() == b"jpeg"
    assert (tmp_path / ".unzipped-images.zip").exists()
    evil = tmp_path / "evil" / "x.zip"
    evil.parent.mkdir()
    with zipfile.ZipFile(evil, "w") as handle:
        handle.writestr("../../escape.txt", b"no")
    with pytest.raises(DownloadError, match="outside"):
        download._unzip(evil, log=lambda m: None)


def test_checkpoint_verification_reads_the_revision_marker(tmp_path, config, checkpoints):
    plan = download.plan_checkpoints(config, "models", ["qwen25vl_3b"], workspace(tmp_path), checkpoints)[0]
    assert download.verify_checkpoint(plan) == ["not downloaded"]
    plan.target.mkdir(parents=True)
    (plan.target / ".revision").write_text(plan.revision + "\n")
    assert download.verify_checkpoint(plan) == []
    (plan.target / download.RECORD_FILE).write_text(json.dumps({"processors_only": False}))
    assert plan.files == 14
    assert download.verify_checkpoint(plan) == ["0 of 14 files present (interrupted download?)"]
    for i in range(14):
        (plan.target / f"f{i}.json").write_text("{}")
    assert download.verify_checkpoint(plan) == []
    meta = plan.target / ".cache" / "huggingface" / "download" / "config.json.metadata"
    meta.parent.mkdir(parents=True)
    meta.write_text("f" * 40 + "\netag\n0\n")
    assert "other revisions" in download.verify_checkpoint(plan)[0]


def test_http_request_reads_a_whole_chunked_response(monkeypatch):
    """GitHub's API answers with chunked transfer encoding; the whole body must be read (a regression test)."""
    import http.server
    import threading
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    body = bytes(range(256)) * 400                                    # 102,400 bytes in four chunks

    class Chunked(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for start in range(0, len(body), 30000):
                piece = body[start:start + 30000]
                self.wfile.write(b"%x\r\n" % len(piece) + piece + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Chunked)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/tree"
        assert download.http_request(url).body == body
        assert download.http_request(url, max_bytes=10).body == body[:10]
    finally:
        server.shutdown()
        server.server_close()
