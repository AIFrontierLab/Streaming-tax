"""Pinned downloads of checkpoints and benchmark data: `streamreplace download models|judges|datasets`.

Checkpoints. configs/models.yaml and configs/judges.yaml name every checkpoint as "<repo>@<revision>";
manifests/checkpoints.yaml lists the paper's checkpoints with their licences and must agree with them. A checkpoint
goes to paths.checkpoint_dir(ref) (<checkpoints>/<repo>, or where a paths file maps it) through
huggingface_hub.snapshot_download at exactly that revision; `.revision` in the folder records it (`doctor` reads it).
`processors_only` skips the weight files: enough for the CPU checks that only need tokenizers and processors.

Datasets. manifests/datasets.yaml lists, per benchmark, its sources (Hugging Face at a commit, GitHub files at a
commit, fixed URLs and Google Drive files with sha256, and ModelScope for OpenUAV-QA frames, which has no immutable
revision). Files land in <data>/<benchmark>/raw/, the folder `streamreplace prepare` reads, and are checked against
the manifest's sizes and checksums. `raw/SHA256SUMS` lists the files checked by sha256 and `raw/.download.json`
records the sources and revisions.

Every source must be pinned (a full commit, or a checksum for a file served from a fixed URL); anything else is
refused, except ModelScope with --allow-unpinned. Datasets whose licence needs an explicit agreement are refused
until --accept-terms NAME is given; gated Hugging Face repositories print their manual access step.

A dry run resolves every pin without downloading data: the Hugging Face API at the pinned revision (file names and
counts), GitHub file sizes at the pinned commit (HEAD requests) and folder digests, the size of fixed-URL files,
whether a Google Drive id answers, and the ModelScope listing endpoint. huggingface_hub (models, judges and
Hugging Face datasets) and gdown (Google Drive files) are needed only for real downloads."""
from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import time
from typing import Callable, Iterable, Mapping
import urllib.error
import urllib.parse
import urllib.request

from .upstream import FetchError, file_blob_sha1, git_blob_sha1, http_download, http_get

CHECKPOINTS = Path("manifests") / "checkpoints.yaml"
DATASETS = Path("manifests") / "datasets.yaml"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
HEX32 = re.compile(r"[0-9a-f]{32}")
ROLES = ("model", "vision", "judge")
SOURCE_TYPES = ("hf", "github", "url", "gdrive", "modelscope")
GROUPS = ("image", "video", "safety", "trust", "finetune", "calibration")
# Weight files skipped by --processors-only (tokenizer, processor, config and index files are kept).
WEIGHTS = ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt", "*.gguf", "*.h5", "*.msgpack", "*.onnx", "*.tflite"]
REVISION_FILE = ".revision"
RECORD_FILE = ".download.json"
SUMS_FILE = "SHA256SUMS"
USER_AGENT = "streamreplace-download"
GDRIVE_URL = "https://drive.google.com/uc?export=download&id={id}"
MODELSCOPE_API = "https://www.modelscope.cn/api/v1/datasets/{repo}"


class ManifestError(ValueError):
    """manifests/checkpoints.yaml or manifests/datasets.yaml is malformed or unpinned."""


class DownloadError(RuntimeError):
    """A download failed or a file does not match its pin."""


class Refused(DownloadError):
    """The request is refused: an unpinned source, a revision conflict, or terms that were not accepted."""


def _project() -> Path:
    from ..paths import project_root
    return project_root()


def _yaml(path: Path) -> Mapping:
    import yaml
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"manifest not found: {path}") from exc
    if not isinstance(data, Mapping) or data.get("version", 1) != 1:
        raise ManifestError(f"{path}: expected a version 1 mapping")
    return data


def hf_endpoint() -> str:
    return (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")


# ---------------------------------------------------------------------------------------------------- checkpoints

@dataclass(frozen=True)
class Checkpoint:
    repo: str
    revision: str
    role: str
    used_by: Mapping[str, str]
    dir: str
    license: str
    files: int | None = None
    notes: str = ""


def load_checkpoints(path: Path | str | None = None) -> dict[str, Checkpoint]:
    path = Path(path) if path is not None else _project() / CHECKPOINTS
    data = _yaml(path)
    out = {}
    for repo, raw in (data.get("checkpoints") or {}).items():
        where = f"{path}: {repo}"
        if not isinstance(raw, Mapping) or "/" not in str(repo):
            raise ManifestError(f"{where}: expected '<org>/<name>: {{...}}'")
        unknown = sorted(set(raw) - {"revision", "role", "used_by", "dir", "license", "files", "notes"})
        if unknown:
            raise ManifestError(f"{where}: unknown keys {unknown}")
        revision = str(raw.get("revision", ""))
        if not HEX40.fullmatch(revision):
            raise ManifestError(f"{where}: revision {revision!r} is not a full commit (unpinned checkpoints are refused)")
        if raw.get("role") not in ROLES:
            raise ManifestError(f"{where}: role must be one of {ROLES}")
        used_by = raw.get("used_by") or {}
        if not isinstance(used_by, Mapping) or not used_by:
            raise ManifestError(f"{where}: used_by must map model or judge ids to the referencing field")
        out[str(repo)] = Checkpoint(str(repo), revision, str(raw["role"]), {str(k): str(v) for k, v in used_by.items()},
                                    str(raw.get("dir") or repo), str(raw.get("license", "")),
                                    int(raw["files"]) if raw.get("files") is not None else None,
                                    " ".join(str(raw.get("notes", "")).split()))
    return out


def config_checkpoints(config, kind: str) -> dict[str, dict[str, object]]:
    """{model or judge id: {field: CheckpointRef}} as the configuration references them. kind: models | judges."""
    from ..config.schema import CheckpointRef
    if kind == "models":
        return {mid: dict(model.checkpoints()) for mid, model in config.models.items()}
    if kind != "judges":
        raise ValueError(kind)
    out: dict[str, dict[str, object]] = {}
    for jid, block in (config.section("judges") or {}).items():
        if not isinstance(block, Mapping):
            continue
        refs: dict[str, object] = {}
        if block.get("checkpoint"):
            refs["checkpoint"] = CheckpointRef.parse(block["checkpoint"], f"judges.{jid}.checkpoint")
        for key, value in (block.get("checkpoints") or {}).items():
            refs[f"checkpoints.{key}"] = CheckpointRef.parse(value, f"judges.{jid}.checkpoints.{key}")
        if refs:
            out[str(jid)] = refs
    return out


@dataclass
class CheckpointPlan:
    repo: str
    revision: str
    target: Path
    users: list[str]                                  # "<id>.<field>"
    license: str = ""
    notes: str = ""
    processors_only: bool = False
    files: int | None = None                          # files in the repository at the revision (manifest)

    @property
    def label(self) -> str:
        return f"{self.repo}@{self.revision[:8]}"


def plan_checkpoints(config, kind: str, names: Iterable[str] | None, paths, manifest: Mapping[str, Checkpoint],
                     processors_only: bool = False) -> list[CheckpointPlan]:
    """One plan per distinct repository referenced by the selected models / judges ("all" or nothing = every one).
    Refuses unpinned references and revisions that disagree with manifests/checkpoints.yaml."""
    refs = config_checkpoints(config, kind)
    wanted = [n for n in (names or []) if n] or ["all"]
    chosen: list[str] = []
    for name in wanted:
        if name == "all":
            chosen += list(refs)
        elif name in refs:
            chosen.append(name)
        else:
            by_repo = [i for i, fields in refs.items() for ref in fields.values() if ref.repo == name]
            if not by_repo:
                raise ManifestError(f"unknown {kind[:-1]} {name!r}; choose from: {', '.join(refs)}, all")
            chosen += by_repo
    plans: dict[str, CheckpointPlan] = {}
    for ident in dict.fromkeys(chosen):
        for fieldname, ref in refs[ident].items():
            if not ref.revision:
                raise Refused(f"{ident}.{fieldname}: {ref.repo} has no pinned revision; write it as '<repo>@<commit>'")
            if not HEX40.fullmatch(ref.revision):
                raise Refused(f"{ident}.{fieldname}: revision {ref.revision!r} of {ref.repo} is not a full commit hash")
            known = manifest.get(ref.repo)
            if known is not None and known.revision != ref.revision:
                raise Refused(f"{ident}.{fieldname}: the configuration pins {ref.repo}@{ref.revision}, "
                              f"manifests/checkpoints.yaml pins {known.revision}")
            plan = plans.get(ref.repo)
            if plan is None:
                plan = plans[ref.repo] = CheckpointPlan(ref.repo, ref.revision, Path(paths.checkpoint_dir(ref)), [],
                                                        known.license if known else "", known.notes if known else "",
                                                        processors_only, known.files if known else None)
            elif plan.revision != ref.revision:
                raise Refused(f"{ref.repo} is referenced at two revisions: {plan.revision} and {ref.revision}")
            plan.users.append(f"{ident}.{fieldname}")
    return list(plans.values())


# ---------------------------------------------------------------------------------------------------- datasets

@dataclass(frozen=True)
class Source:
    type: str
    spec: Mapping

    def get(self, key, default=None):
        return self.spec.get(key, default)

    @property
    def label(self) -> str:
        if self.type == "hf":
            return f"hf:{self.spec['repo']}@{self.spec['revision'][:8]}"
        if self.type == "github":
            what = self.spec.get("tree") or ",".join(self.spec.get("files") or {})
            return f"github:{self.spec['repo']}@{self.spec['commit'][:8]}:{what}"
        if self.type == "url":
            return f"url:{self.spec['url']}"
        if self.type == "gdrive":
            return f"gdrive:{self.spec['id']} -> {self.spec['to']}"
        return f"modelscope:{self.spec['repo']}/{self.spec.get('root', '')} (unpinned)"


@dataclass(frozen=True)
class Dataset:
    name: str
    group: str
    license: str
    sources: tuple[Source, ...]
    notes: str = ""
    terms: str = ""
    requires_acceptance: bool = False

    @property
    def unpinned(self) -> list[Source]:
        return [s for s in self.sources if s.type == "modelscope"]


_KEYS = {
    "hf": ({"repo", "revision"}, {"allow", "files", "bytes", "checksums", "folders"}),
    "github": ({"repo", "commit"}, {"files", "tree", "tree_files", "tree_sha256", "bytes", "to"}),
    "url": ({"url", "to", "size", "sha256"}, {"md5"}),
    "gdrive": ({"id", "to", "size", "sha256"}, {"unzip"}),
    "modelscope": ({"repo", "root", "pinned"}, {"frames"}),
}


def _check_sum(where: str, value) -> None:
    if not isinstance(value, Mapping) or not isinstance(value.get("size"), int):
        raise ManifestError(f"{where}: expected {{size: N, sha256: ...}} or {{size: N, blob: ...}}")
    if not (HEX64.fullmatch(str(value.get("sha256", ""))) or HEX40.fullmatch(str(value.get("blob", "")))):
        raise ManifestError(f"{where}: needs a sha256 (64 hex) or a git blob SHA-1 (40 hex)")


def _relative(where: str, path: str) -> str:
    parts = PurePosixPath(path).parts
    if not path or path.startswith("/") or ".." in parts or "\\" in path:
        raise ManifestError(f"{where}: {path!r} is not a clean relative path")
    return path


def _source(where: str, raw: Mapping) -> Source:
    kind = raw.get("type")
    if kind not in SOURCE_TYPES:
        raise ManifestError(f"{where}: type must be one of {SOURCE_TYPES}")
    required, optional = _KEYS[kind]
    spec = {k: v for k, v in raw.items() if k != "type"}
    missing, unknown = sorted(required - set(spec)), sorted(set(spec) - required - optional)
    if missing or unknown:
        raise ManifestError(f"{where}: missing {missing} / unknown {unknown}")
    if kind == "hf" and not HEX40.fullmatch(str(spec["revision"])):
        raise ManifestError(f"{where}: revision {spec['revision']!r} is not a full commit (unpinned sources are refused)")
    if kind == "github":
        if not HEX40.fullmatch(str(spec["commit"])):
            raise ManifestError(f"{where}: commit {spec['commit']!r} is not a full commit (unpinned sources are refused)")
        if bool(spec.get("files")) == bool(spec.get("tree")):
            raise ManifestError(f"{where}: give either `files` or `tree`")
        if spec.get("tree") and not (isinstance(spec.get("tree_files"), int) and HEX64.fullmatch(str(spec.get("tree_sha256", "")))):
            raise ManifestError(f"{where}: a `tree` needs tree_files and tree_sha256")
        for path, value in (spec.get("files") or {}).items():
            _relative(where, str(path))
            _check_sum(f"{where}: {path}", value)
            if not HEX64.fullmatch(str(value.get("sha256", ""))):
                raise ManifestError(f"{where}: {path}: GitHub files are pinned by sha256")
        if spec.get("to") and len(spec.get("files") or {}) != 1:
            raise ManifestError(f"{where}: `to` needs exactly one file")
    if kind in ("url", "gdrive"):
        _relative(where, str(spec["to"]))
        if not isinstance(spec["size"], int) or not HEX64.fullmatch(str(spec["sha256"])):
            raise ManifestError(f"{where}: needs an integer size and a sha256")
        if spec.get("md5") and not HEX32.fullmatch(str(spec["md5"])):
            raise ManifestError(f"{where}: md5 must be 32 hex digits")
    if kind == "hf":
        for path, value in (spec.get("checksums") or {}).items():
            _relative(where, str(path))
            _check_sum(f"{where}: {path}", value)
    if kind == "modelscope" and spec["pinned"] is not False:
        raise ManifestError(f"{where}: ModelScope sources have no immutable revision; mark them `pinned: false`")
    return Source(kind, spec)


def load_datasets(path: Path | str | None = None) -> dict[str, Dataset]:
    path = Path(path) if path is not None else _project() / DATASETS
    data = _yaml(path)
    out = {}
    for name, raw in (data.get("datasets") or {}).items():
        where = f"{path}: {name}"
        if not isinstance(raw, Mapping):
            raise ManifestError(f"{where}: expected a mapping")
        unknown = sorted(set(raw) - {"group", "license", "notes", "terms", "requires_acceptance", "sources"})
        if unknown:
            raise ManifestError(f"{where}: unknown keys {unknown}")
        if raw.get("group") not in GROUPS:
            raise ManifestError(f"{where}: group must be one of {GROUPS}")
        sources = tuple(_source(f"{where} source {i + 1}", s) for i, s in enumerate(raw.get("sources") or []))
        if not sources:
            raise ManifestError(f"{where}: no sources")
        out[str(name)] = Dataset(str(name), str(raw["group"]), str(raw.get("license", "")), sources,
                                 " ".join(str(raw.get("notes", "")).split()), " ".join(str(raw.get("terms", "")).split()),
                                 bool(raw.get("requires_acceptance", False)))
    return out


def select_datasets(manifest: Mapping[str, Dataset], names: Iterable[str] | None,
                    suites: Mapping[str, Iterable[str]] | None = None) -> list[Dataset]:
    """Datasets for NAMEs: benchmark names, suite names (configs/benchmarks.yaml), groups, or "all"."""
    suites = suites or {}
    wanted = [n for n in (names or []) if n] or ["all"]
    chosen: list[str] = []
    for name in wanted:
        if name == "all":
            chosen += list(manifest)
        elif name in manifest:
            chosen.append(name)
        elif name in suites:
            missing = [b for b in suites[name] if b not in manifest]
            if missing:
                raise ManifestError(f"suite {name}: no download entry for {missing}")
            chosen += list(suites[name])
        elif name in GROUPS:
            chosen += [n for n, d in manifest.items() if d.group == name]
        else:
            raise ManifestError(f"unknown dataset {name!r}; choose from: {', '.join(manifest)}, "
                                f"{', '.join(sorted(suites))}, {', '.join(GROUPS)}, all")
    return [manifest[n] for n in dict.fromkeys(chosen)]


def check_allowed(datasets: Iterable[Dataset], accepted: Iterable[str] = (), allow_unpinned: bool = False) -> None:
    """Refuses datasets whose terms were not accepted, and unpinned sources unless allowed."""
    accepted = set(accepted)
    for ds in datasets:
        if ds.requires_acceptance and ds.name not in accepted:
            raise Refused(f"{ds.name}: {ds.terms or 'its licence needs an explicit agreement'} "
                          f"(pass --accept-terms {ds.name})")
        if ds.unpinned and not allow_unpinned:
            raise Refused(f"{ds.name}: {', '.join(s.label for s in ds.unpinned)} cannot be pinned to a revision; "
                          f"pass --allow-unpinned to download it anyway (files are still checked against the "
                          f"recorded listing)")


# ---------------------------------------------------------------------------------------------------- HTTP helpers

def _auth_headers(url: str) -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT}
    host = urllib.parse.urlsplit(url).hostname or ""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token and host == (urllib.parse.urlsplit(hf_endpoint()).hostname or ""):
        headers["Authorization"] = f"Bearer {token}"
    github = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if github and (host.endswith("github.com") or host.endswith("githubusercontent.com")):
        headers["Authorization"] = f"Bearer {github}"
    return headers


@dataclass
class Response:
    status: int
    headers: Mapping[str, str]
    body: bytes = b""


def http_request(url: str, method: str = "GET", max_bytes: int | None = None, attempts: int = 3,
                 timeout: float = 60.0) -> Response:
    """A small request (HEAD, or GET of at most max_bytes) with retries on 429 / 5xx / network errors. HTTP errors
    are returned as responses (status 4xx), network failures raise DownloadError."""
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers=_auth_headers(url), method=method)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if method == "HEAD":
                    body = b""
                elif max_bytes is None:
                    body = response.read()            # read(-1) can stop after one chunk of a chunked response
                else:
                    body = response.read(max_bytes)
                return Response(response.status, dict(response.headers.items()), body)
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt + 1 == attempts:
                return Response(exc.code, dict(exc.headers.items()) if exc.headers else {}, b"")
            error = exc
        except (urllib.error.URLError, OSError) as exc:
            error = exc
        time.sleep(min(20.0, 2.0 ** attempt))
    raise DownloadError(f"{method} {url}: {error}")


Requester = Callable[..., Response]


# ---------------------------------------------------------------------------------------------------- resolving pins

@dataclass
class Resolution:
    what: str
    ok: bool
    detail: str


def _hf_info(kind: str, repo: str, revision: str, request: Requester) -> tuple[Response, dict]:
    url = f"{hf_endpoint()}/api/{kind}/{repo}/revision/{revision}"
    response = request(url)
    try:
        info = json.loads(response.body or b"{}")
    except ValueError:
        info = {}
    return response, info


def resolve_checkpoint(plan: CheckpointPlan, request: Requester = http_request) -> Resolution:
    response, info = _hf_info("models", plan.repo, plan.revision, request)
    if response.status != 200:
        return Resolution(plan.label, False, f"Hugging Face API answered {response.status}")
    if info.get("sha") != plan.revision:
        return Resolution(plan.label, False, f"the API resolves the revision to {info.get('sha')}")
    names = [s.get("rfilename", "") for s in info.get("siblings") or []]
    kept = [n for n in names if not (plan.processors_only and any(fnmatch.fnmatch(n, p) for p in WEIGHTS))]
    gated = f", gated ({info.get('gated')})" if info.get("gated") else ""
    what = f"{len(kept)} of {len(names)} files (processors only)" if plan.processors_only else f"{len(names)} files"
    return Resolution(plan.label, True, f"revision exists, {what}{gated}")


def _hf_matches(names: Iterable[str], allow: Iterable[str] | None) -> list[str]:
    allow = list(allow or [])
    return sorted(n for n in names if not allow or any(fnmatch.fnmatch(n, p) for p in allow))


def resolve_source(ds: Dataset, source: Source, request: Requester = http_request) -> Resolution:
    what = f"{ds.name} {source.label}"
    s = source.spec
    if source.type == "hf":
        response, info = _hf_info("datasets", s["repo"], s["revision"], request)
        if response.status != 200:
            return Resolution(what, False, f"Hugging Face API answered {response.status}")
        if info.get("sha") != s["revision"]:
            return Resolution(what, False, f"the API resolves the revision to {info.get('sha')}")
        names = [x.get("rfilename", "") for x in info.get("siblings") or []]
        matched = _hf_matches(names, s.get("allow"))
        absent = sorted(set(s.get("checksums") or {}) - set(matched))
        problems = []
        if s.get("files") is not None and len(matched) != s["files"]:
            problems.append(f"{len(matched)} files match, the manifest expects {s['files']}")
        if absent:
            problems.append(f"missing at the revision: {', '.join(absent[:3])}")
        for folder, block in (s.get("folders") or {}).items():
            n = sum(1 for m in matched if m.startswith(folder))
            if n != block.get("files"):
                problems.append(f"{folder}: {n} files, expected {block.get('files')}")
        gated = f"; gated ({info.get('gated')})" if info.get("gated") else ""
        if problems:
            return Resolution(what, False, "; ".join(problems))
        return Resolution(what, True, f"revision exists, {len(matched)} files match{gated}")
    if source.type == "github":
        owner_repo, commit = s["repo"], s["commit"]
        if s.get("tree"):
            return _resolve_tree(what, owner_repo, commit, s, request)
        problems, found = [], []
        for path, pin in (s.get("files") or {}).items():
            url = f"https://raw.githubusercontent.com/{owner_repo}/{commit}/{urllib.parse.quote(path, safe='/')}"
            response = request(url, method="HEAD")
            length = response.headers.get("Content-Length") or response.headers.get("content-length")
            if response.status != 200:
                problems.append(f"{path}: HTTP {response.status}")
            elif length is not None and int(length) != pin["size"]:
                problems.append(f"{path}: {length} bytes, pinned {pin['size']}")
            else:
                found.append(path)
        return Resolution(what, not problems, "; ".join(problems) or f"{len(found)} file(s) present at the commit")
    if source.type == "url":
        response = request(s["url"], method="HEAD")
        length = response.headers.get("Content-Length") or response.headers.get("content-length")
        if response.status != 200:
            return Resolution(what, False, f"HTTP {response.status}")
        if length is not None and int(length) != s["size"]:
            return Resolution(what, False, f"{length} bytes, pinned {s['size']}")
        return Resolution(what, True, f"{s['size']} bytes (content checked by sha256 after download)")
    if source.type == "gdrive":
        response = request(GDRIVE_URL.format(id=s["id"]), max_bytes=65536)
        if response.status != 200:
            return Resolution(what, False, f"Google Drive answered {response.status}")
        text = response.body.decode("utf-8", "replace")
        name = Path(s["to"]).name
        disposition = response.headers.get("Content-Disposition") or response.headers.get("content-disposition") or ""
        seen = name in text or name in disposition
        note = f"{name} named in the answer" if seen else "id answers (file name not shown)"
        return Resolution(what, True, f"{note}; {s['size']} bytes, checked by sha256 after download")
    url = MODELSCOPE_API.format(repo=s["repo"]) + (f"/repo/tree?Revision=master&Root={urllib.parse.quote(s['root'])}"
                                                   "&Recursive=false&PageNumber=1&PageSize=1")
    response = request(url, max_bytes=1 << 20)
    ok = response.status == 200
    return Resolution(what, ok, ("listing endpoint answers; no immutable revision (needs --allow-unpinned)" if ok
                                 else f"ModelScope answered {response.status}"))


def tree_digest(entries: Iterable[tuple[str, str]]) -> str:
    """sha256 of the sorted "<path> <git blob SHA-1>" lines (the `tree_sha256` of a GitHub folder source)."""
    return hashlib.sha256("".join(f"{p} {b}\n" for p, b in sorted(entries)).encode("utf-8")).hexdigest()


def _resolve_tree(what: str, owner_repo: str, commit: str, s: Mapping, request: Requester) -> Resolution:
    response = request(f"https://api.github.com/repos/{owner_repo}/git/trees/{commit}?recursive=1")
    if response.status != 200:
        return Resolution(what, False, f"GitHub API answered {response.status} (rate limit? set GITHUB_TOKEN)")
    data = json.loads(response.body or b"{}")
    if data.get("truncated"):
        return Resolution(what, False, "the GitHub tree listing is truncated")
    prefix = s["tree"]
    entries = [(e["path"], e["sha"]) for e in data.get("tree", []) if e.get("type") == "blob" and e["path"].startswith(prefix)]
    if len(entries) != s["tree_files"] or tree_digest(entries) != s["tree_sha256"]:
        return Resolution(what, False, f"{len(entries)} files with digest {tree_digest(entries)[:12]}, pinned "
                                       f"{s['tree_files']} / {s['tree_sha256'][:12]}")
    return Resolution(what, True, f"{len(entries)} files, folder digest matches")


# ---------------------------------------------------------------------------------------------------- checking files

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_file(path: Path, pin: Mapping, size_only_above: int | None = None) -> str | None:
    """None when `path` matches its pin ({size, sha256|blob[, md5]}), else the problem. Files above
    size_only_above bytes are checked by size only (a quick check of very large archives)."""
    if not path.is_file():
        return "missing"
    size = path.stat().st_size
    if size != pin["size"]:
        return f"{size} bytes, pinned {pin['size']}"
    if size_only_above is not None and size > size_only_above:
        return None
    if pin.get("sha256") and sha256_file(path) != pin["sha256"]:
        return "sha256 differs"
    if pin.get("blob"):
        if file_blob_sha1(path) != pin["blob"]:
            return "git blob differs"
    if pin.get("md5") and md5_file(path) != pin["md5"]:
        return "md5 differs"
    return None


def _source_files(source: Source) -> dict[str, dict]:
    """raw-relative path -> pin, for the files of a source that carry a checksum."""
    s = source.spec
    if source.type == "hf":
        return {p: dict(v) for p, v in (s.get("checksums") or {}).items()}
    if source.type == "github" and s.get("files"):
        files = dict(s["files"])
        if s.get("to"):
            (path, pin), = files.items()
            return {s["to"]: dict(pin)}
        return {p: dict(v) for p, v in files.items()}
    if source.type in ("url", "gdrive"):
        pin = {"size": s["size"], "sha256": s["sha256"]}
        if s.get("md5"):
            pin["md5"] = s["md5"]
        return {s["to"]: pin}
    return {}


def verify_dataset(ds: Dataset, raw: Path, size_only_above: int | None = None) -> list[str]:
    """Problems of a downloaded dataset folder (empty list = matches the manifest). Offline."""
    raw = Path(raw)
    problems = []
    if not raw.is_dir():
        return [f"{raw} does not exist"]
    for source in ds.sources:
        for rel, pin in _source_files(source).items():
            problem = check_file(raw / rel, pin, size_only_above)
            if problem:
                problems.append(f"{rel}: {problem}")
        s = source.spec
        if source.type == "hf":
            for folder, block in (s.get("folders") or {}).items():
                base = raw / folder
                found = [p for p in base.rglob("*") if p.is_file()] if base.is_dir() else []
                if len(found) != block["files"] or sum(p.stat().st_size for p in found) != block["bytes"]:
                    problems.append(f"{folder}: {len(found)} files / {sum(p.stat().st_size for p in found)} bytes, "
                                    f"pinned {block['files']} / {block['bytes']}")
        if source.type == "github" and s.get("tree"):
            base = raw / s["tree"]
            entries = []
            if base.is_dir():
                entries = [(str(PurePosixPath(s["tree"]) / p.relative_to(base).as_posix()), file_blob_sha1(p))
                           for p in base.rglob("*") if p.is_file()]
            if len(entries) != s["tree_files"] or tree_digest(entries) != s["tree_sha256"]:
                problems.append(f"{s['tree']}: {len(entries)} files, digest {tree_digest(entries)[:12]}; pinned "
                                f"{s['tree_files']} / {s['tree_sha256'][:12]}")
        if source.type == "modelscope" and not (raw / ".listing_complete").is_file():
            problems.append("ModelScope file listing not downloaded (modelscope_tree.jsonl)")
    return problems


# ---------------------------------------------------------------------------------------------------- downloading

def _import(module: str, why: str):
    try:
        return __import__(module)
    except ImportError as exc:
        raise DownloadError(f"{why} needs the `{module}` package (pip install {module})") from exc


def download_checkpoint(plan: CheckpointPlan, workers: int = 8, force: bool = False, log=print) -> Path:
    """snapshot_download of the pinned revision into plan.target; writes .revision and a download record."""
    _import("huggingface_hub", "downloading checkpoints")
    from huggingface_hub import snapshot_download
    target = plan.target
    target.mkdir(parents=True, exist_ok=True)
    log(f"{plan.label} -> {target}{' (processors only)' if plan.processors_only else ''}")
    snapshot_download(repo_id=plan.repo, revision=plan.revision, local_dir=str(target), max_workers=workers,
                      ignore_patterns=WEIGHTS if plan.processors_only else None, force_download=force)
    commits = _hf_commits(target)
    if commits and commits != {plan.revision}:
        raise DownloadError(f"{target}: files from other revisions are present ({sorted(commits)}); "
                            f"remove the folder and download again")
    (target / REVISION_FILE).write_text(plan.revision + "\n", encoding="utf-8")
    _write_record(target, {"repo": plan.repo, "revision": plan.revision, "processors_only": plan.processors_only,
                           "used_by": plan.users})
    return target


def _hf_commits(folder: Path) -> set[str]:
    """Commits recorded by huggingface_hub's local-dir download metadata (.cache/huggingface/download/*.metadata)."""
    cache = folder / ".cache" / "huggingface" / "download"
    commits = set()
    if cache.is_dir():
        for meta in cache.rglob("*.metadata"):
            try:
                commits.add(meta.read_text(encoding="utf-8").splitlines()[0].strip())
            except (OSError, IndexError):
                pass
    return commits


def verify_checkpoint(plan: CheckpointPlan) -> list[str]:
    target = plan.target
    if not target.is_dir():
        return ["not downloaded"]
    problems = []
    marker = target / REVISION_FILE
    if not marker.is_file():
        problems.append(f"no {REVISION_FILE} file")
    elif marker.read_text(encoding="utf-8").strip() != plan.revision:
        problems.append(f"{REVISION_FILE} says {marker.read_text(encoding='utf-8').strip()}, pinned {plan.revision}")
    commits = _hf_commits(target)
    if commits and commits != {plan.revision}:
        problems.append(f"files from other revisions: {sorted(commits)}")
    record = {}
    if (target / RECORD_FILE).is_file():
        try:
            record = json.loads((target / RECORD_FILE).read_text(encoding="utf-8"))
        except ValueError:
            record = {}
    if plan.files and record and not record.get("processors_only"):      # a complete download has every file
        present = [p for p in target.rglob("*") if p.is_file() and p.name not in (REVISION_FILE, RECORD_FILE)
                   and ".cache" not in p.relative_to(target).parts]
        if len(present) < plan.files:
            problems.append(f"{len(present)} of {plan.files} files present (interrupted download?)")
    return problems


def _write_record(folder: Path, record: dict) -> None:
    path = folder / RECORD_FILE
    old = {}
    if path.is_file():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            old = {}
    old.update(record)
    path.write_text(json.dumps(old, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def _write_sums(raw: Path, sums: Mapping[str, str]) -> None:
    path = raw / SUMS_FILE
    existing = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "  " in line:
                digest, rel = line.split("  ", 1)
                existing[rel] = digest
    existing.update(sums)
    path.write_text("".join(f"{d}  {r}\n" for r, d in sorted(existing.items())), encoding="utf-8")


def _verified_move(part: Path, target: Path, pin: Mapping) -> None:
    problem = check_file(part, pin)
    if problem:
        part.unlink(missing_ok=True)
        raise DownloadError(f"{target.name}: {problem}")
    part.replace(target)


def _unzip(archive: Path, log=print) -> None:
    """Unpacks an archive next to itself once (marker .unzipped-<name>); files already present with the right size
    are kept, so an interrupted unpack resumes."""
    import zipfile
    marker = archive.parent / f".unzipped-{archive.name}"
    if marker.exists():
        return
    with zipfile.ZipFile(archive) as handle:
        members = [m for m in handle.infolist() if not m.is_dir()]
        for number, member in enumerate(members, 1):
            target = (archive.parent / member.filename).resolve()
            if archive.parent.resolve() not in target.parents:
                raise DownloadError(f"{archive}: member {member.filename!r} points outside the folder")
            if target.is_file() and target.stat().st_size == member.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + ".part")
            with handle.open(member) as source, part.open("wb") as out:
                shutil.copyfileobj(source, out, 1 << 22)
            part.replace(target)
            if number % 20000 == 0:
                log(f"  {archive.name}: unpacked {number}/{len(members)}")
    marker.write_text(f"{len(members)} files\n", encoding="utf-8")


def _download_hf(ds: Dataset, source: Source, raw: Path, workers: int, force: bool, log) -> None:
    _import("huggingface_hub", "downloading Hugging Face datasets")
    from huggingface_hub import snapshot_download
    s = source.spec
    log(f"  {source.label} {s.get('allow') or 'all files'}")
    try:
        snapshot_download(repo_id=s["repo"], repo_type="dataset", revision=s["revision"], local_dir=str(raw),
                          allow_patterns=s.get("allow"), max_workers=workers, force_download=force)
    except Exception as exc:                      # noqa: BLE001 - gated repositories: say what to do
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403) or type(exc).__name__ == "GatedRepoError":
            raise Refused(f"{ds.name}: access denied by Hugging Face. {ds.terms}") from exc
        raise DownloadError(f"{ds.name}: {type(exc).__name__}: {exc}") from exc


def _download_github(source: Source, raw: Path, force: bool, log, sums: dict) -> None:
    s = source.spec
    owner_repo, commit = s["repo"], s["commit"]
    if s.get("tree"):
        _download_tree(source, raw, force, log)
        return
    for rel, pin in _source_files(source).items():
        upstream = next(iter(s["files"])) if s.get("to") else rel
        target = raw / rel
        if not force and check_file(target, pin) is None:
            sums[rel] = pin["sha256"]
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/{owner_repo}/{commit}/{urllib.parse.quote(upstream, safe='/')}"
        log(f"  {url}")
        part = target.with_name(target.name + ".part")
        try:
            http_download(url, part)
        except FetchError as exc:
            raise DownloadError(str(exc)) from exc
        _verified_move(part, target, pin)
        sums[rel] = pin["sha256"]


def _download_tree(source: Source, raw: Path, force: bool, log) -> None:
    """A whole folder of a GitHub repository at a commit: one tarball, the folder's files taken from it, checked
    against the pinned file count and folder digest before anything is written."""
    s = source.spec
    prefix = s["tree"]
    base = raw / prefix
    if not force and base.is_dir():
        entries = [(str(PurePosixPath(prefix) / p.relative_to(base).as_posix()), file_blob_sha1(p))
                   for p in base.rglob("*") if p.is_file()]
        if len(entries) == s["tree_files"] and tree_digest(entries) == s["tree_sha256"]:
            return
    url = f"https://codeload.github.com/{s['repo']}/tar.gz/{s['commit']}"
    log(f"  {url} ({prefix})")
    archive = raw / f".{Path(s['repo']).name}-{s['commit'][:12]}.tar.gz"
    raw.mkdir(parents=True, exist_ok=True)
    try:
        http_download(url, archive)
    except FetchError as exc:
        raise DownloadError(str(exc)) from exc
    files: dict[str, bytes] = {}
    try:
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                if member.isfile() and "/" in member.name:
                    rel = member.name.split("/", 1)[1]
                    if rel.startswith(prefix):
                        files[rel] = tar.extractfile(member).read()
    finally:
        archive.unlink(missing_ok=True)
    entries = [(p, git_blob_sha1(d)) for p, d in files.items()]
    if len(entries) != s["tree_files"] or tree_digest(entries) != s["tree_sha256"]:
        raise DownloadError(f"{s['repo']}@{s['commit'][:8]}:{prefix}: {len(entries)} files, digest "
                            f"{tree_digest(entries)[:12]}; pinned {s['tree_files']} / {s['tree_sha256'][:12]}")
    for rel, data in files.items():
        target = raw / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def _download_url(source: Source, raw: Path, force: bool, log, sums: dict) -> None:
    s = source.spec
    target = raw / s["to"]
    pin = _source_files(source)[s["to"]]
    if force or check_file(target, pin) is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        log(f"  {s['url']}")
        part = target.with_name(target.name + ".part")
        try:
            http_download(s["url"], part, timeout=600)
        except FetchError as exc:
            raise DownloadError(str(exc)) from exc
        _verified_move(part, target, pin)
    sums[s["to"]] = s["sha256"]


def _download_gdrive(source: Source, raw: Path, force: bool, log, sums: dict) -> None:
    s = source.spec
    target = raw / s["to"]
    pin = _source_files(source)[s["to"]]
    if force or check_file(target, pin) is not None:
        gdown = _import("gdown", "downloading Google Drive files")
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        log(f"  Google Drive {s['id']} -> {target}")
        if gdown.download(id=s["id"], output=str(part), quiet=False, resume=not force) is None:
            raise DownloadError(f"Google Drive download failed: {s['to']} ({s['id']}); run again to resume")
        _verified_move(part, target, pin)
    sums[s["to"]] = s["sha256"]
    if s.get("unzip"):
        _unzip(target, log)


# ModelScope (OpenUAV-QA frames): as the paper's runs downloaded them. ModelScope allows git/LFS only with an account
# token, so files come one REST request each; the listing records every file's sha256 and each frame is checked
# against it. Files ModelScope lists but serves empty are recorded in raw/modelscope_unavailable.json (prepare skips
# their trajectories, as the official loader could not read them either).

def _ms_json(url: str, tries: int = 5):
    for attempt in range(tries):
        try:
            return json.loads(http_get(url, attempts=1))
        except (FetchError, ValueError):
            if attempt + 1 == tries:
                raise
            time.sleep(2 ** attempt)


def _ms_listing(base: str, root: str) -> list[dict]:
    files, page = [], 1
    while True:
        data = _ms_json(f"{base}/repo/tree?Revision=master&Root={urllib.parse.quote(root)}&Recursive=false"
                        f"&PageNumber={page}&PageSize=1000")
        batch = data["Data"]["Files"]
        files += batch
        if len(batch) < 1000:
            return files
        page += 1


def modelscope_listing(source: Source, raw: Path, workers: int = 16, log=print) -> None:
    """Every folder's file list (name, size, sha256) under the source root -> raw/modelscope_tree.jsonl (appended,
    resumable); raw/.listing_complete when done."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    s = source.spec
    base = MODELSCOPE_API.format(repo=s["repo"])
    tree = raw / "modelscope_tree.jsonl"
    done = {json.loads(line)["folder"] for line in tree.read_text().splitlines()} if tree.exists() else set()
    folders = [f["Name"] for f in _ms_listing(base, s["root"]) if f["Type"] == "tree"]
    todo = [f for f in folders if f not in done]
    log(f"  ModelScope {s['repo']}: {len(folders)} folders, {len(todo)} to list")
    with ThreadPoolExecutor(workers) as pool, tree.open("a") as out:
        futures = {pool.submit(_ms_listing, base, f"{s['root']}/{f}"): f for f in todo}
        for number, future in enumerate(as_completed(futures), 1):
            files = [{"name": x["Name"], "size": x["Size"], "sha256": x["Sha256"]} for x in future.result()
                     if x["Type"] == "blob"]
            out.write(json.dumps({"folder": futures[future], "files": files}) + "\n")
            out.flush()
            if number % 1000 == 0:
                log(f"  listed {number}/{len(todo)}")
    (raw / ".listing_complete").write_text(f"{len(folders)} folders\n", encoding="utf-8")


def openuav_frames(source: Source, raw: Path, selection: Path, workers: int = 16, log=print) -> None:
    """The frames prepare uses (the official <= 32 per trajectory) of the trajectories selected in an outputs folder
    (subset.json test ids + train_order.json training ids from `streamreplace finetune prepare openuav_qa`)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from ..benchmarks.finetune.openuav_qa import select_frames
    s = source.spec
    base = MODELSCOPE_API.format(repo=s["repo"])
    ids = set(json.loads((selection / "subset.json").read_text()).get("openuav_qa", []))
    if (selection / "train_order.json").is_file():
        ids |= set(json.loads((selection / "train_order.json").read_text()).get("openuav_qa", []))
    names = {}
    for split in ("train", "test"):
        for line in (raw / f"TravelUAV_{split}.jsonl").read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                names[item["id"]] = item["video_name"]
    entries = {e["folder"]: e for e in map(json.loads, (raw / "modelscope_tree.jsonl").read_text().splitlines())}
    wanted = []
    for item in sorted(ids):
        files = {f["name"]: f for f in entries[names[item]]["files"]}
        wanted += [(names[item], files[name]) for name in select_frames(list(files))]
    jobs = [(folder, f) for folder, f in wanted
            if not ((raw / "frames" / folder / f["name"]).is_file()
                    and (raw / "frames" / folder / f["name"]).stat().st_size == f["size"])]
    log(f"  frames: {len(wanted)} files for {len(ids)} trajectories, {len(jobs)} to download")

    def fetch(folder: str, file: dict) -> None:
        path = raw / "frames" / folder / file["name"]
        path.parent.mkdir(parents=True, exist_ok=True)
        data = http_get(f"{base}/repo?Revision=master&FilePath={urllib.parse.quote(s['root'] + '/' + folder + '/' + file['name'])}")
        if not data and file["size"]:
            raise FileNotFoundError(f"{folder}/{file['name']}")
        if hashlib.sha256(data).hexdigest() != file["sha256"]:
            raise DownloadError(f"sha256 mismatch: {folder}/{file['name']}")
        part = path.with_name(path.name + ".part")
        part.write_bytes(data)
        part.replace(path)

    failed, empty = 0, []
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(fetch, folder, file) for folder, file in jobs]
        for number, future in enumerate(as_completed(futures), 1):
            error = future.exception()
            if isinstance(error, FileNotFoundError):
                empty.append(str(error))
            elif error:
                failed += 1
                log(f"  ERROR {error}")
            if number % 5000 == 0:
                log(f"  downloaded {number}/{len(jobs)} ({failed} failed, {len(empty)} served empty)")
    listed = raw / "modelscope_unavailable.json"
    known = set(json.loads(listed.read_text())) if listed.exists() else set()
    sizes = {f"{e['folder']}/{f['name']}": f["size"] for e in entries.values() for f in e["files"]}
    known = {k for k in known if not ((raw / "frames" / k).is_file() and (raw / "frames" / k).stat().st_size == sizes.get(k))}
    listed.write_text(json.dumps(sorted(known | set(empty)), indent=0) + "\n")
    if failed:
        raise DownloadError(f"{failed} frames failed; run again to retry")
    log(f"  {len(wanted) - len(empty)} of {len(wanted)} frames present; {len(empty)} served empty")


def download_dataset(ds: Dataset, raw: Path, workers: int = 8, force: bool = False, frames_for: Path | None = None,
                     log=print) -> list[str]:
    """Downloads every source of a dataset into raw/ and verifies it; returns the remaining problems (empty = ok).
    The caller has already checked terms and pins (check_allowed)."""
    raw = Path(raw)
    raw.mkdir(parents=True, exist_ok=True)
    log(f"{ds.name} -> {raw}")
    if ds.terms:
        log(f"  note: {ds.terms}")
    sums: dict[str, str] = {}
    record = {"dataset": ds.name, "sources": []}
    for source in ds.sources:
        if source.type == "hf":
            _download_hf(ds, source, raw, workers, force, log)
            (raw / REVISION_FILE).write_text(source.spec["revision"] + "\n", encoding="utf-8")
        elif source.type == "github":
            _download_github(source, raw, force, log, sums)
        elif source.type == "url":
            _download_url(source, raw, force, log, sums)
        elif source.type == "gdrive":
            _download_gdrive(source, raw, force, log, sums)
        else:
            modelscope_listing(source, raw, workers=max(workers, 16), log=log)
            if frames_for is not None:
                openuav_frames(source, raw, Path(frames_for), workers=max(workers, 16), log=log)
            else:
                log("  frames: pass --frames-for OUTPUTS (a `finetune prepare openuav_qa` folder) to download the "
                    "frames of the selected trajectories")
        record["sources"].append({"type": source.type, **{k: v for k, v in source.spec.items()
                                                          if k in ("repo", "revision", "commit", "url", "id", "to", "tree", "allow")}})
    if sums:
        _write_sums(raw, sums)
    _write_record(raw, record)
    return verify_dataset(ds, raw)
