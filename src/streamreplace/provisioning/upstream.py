"""Third-party code at pinned commits: the manifest (manifests/upstream.yaml), fetching and verification behind
`streamreplace fetch-upstream`.

The streaming models' runtimes and the benchmarks' official scorers are other people's code. This repository does
not redistribute it: each project is downloaded from its own GitHub repository at the commit the paper's runs used,
into `<upstream root>/<name>/<path in the repository>` (upstream root: $STREAMREPLACE_UPSTREAM, default
$STREAMREPLACE_HOME/upstream). models/upstream.py resolves those folders for the adapters and scorers.

Only the files listed in the manifest are fetched, and every one is checked against its git blob SHA-1 (what
`git hash-object FILE` prints), so a fetched tree is byte-for-byte the code the paper's numbers came from. Three files
carry small local edits the paper's runs used (`patches` in the manifest): the downloaded file must have the pristine
upstream blob, each edit's `old` text must occur exactly `count` times, and the edited file must have the blob listed
under `files`. Entries of group `reference` (AURA, MOSS-VL) document provenance only and are never fetched.

Fetching is atomic per repository: files go to a staging folder, are verified there, and only a complete, verified
tree replaces `<root>/<name>`. A tree that already verifies is left alone unless `force` is set. Methods:
  raw      one HTTPS request per file to raw.githubusercontent.com (default; a few hundred small files in all)
  tarball  one GitHub tarball per repository (codeload.github.com), from which the listed files are taken
  auto     raw, falling back to the tarball for a repository whose raw download fails
  local    copy from an existing checkout `<source>/<name>/...` (offline machines); files are verified the same way
A GITHUB_TOKEN (or GH_TOKEN) in the environment is sent with GitHub requests when set; it is never required."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
import time
from typing import Callable, Iterable, Mapping
import urllib.error
import urllib.parse
import urllib.request

MANIFEST = Path("manifests") / "upstream.yaml"
MARKER = ".streamreplace-upstream.json"       # written into every fetched tree: repository, commit, file count
STAGING = ".staging"                          # <root>/.staging/: per-fetch working folders (removed afterwards)
GROUPS = ("core", "generalization", "reference")
METHODS = ("auto", "raw", "tarball", "local")
RAW_URL = "https://raw.githubusercontent.com/{owner}/{repo}/{commit}/{path}"
TARBALL_URL = "https://codeload.github.com/{owner}/{repo}/tar.gz/{commit}"
USER_AGENT = "streamreplace-fetch-upstream"
HEX40 = re.compile(r"[0-9a-f]{40}")
GITHUB = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?")

Getter = Callable[[str], bytes]               # url -> response body (tests pass a fake)


class ManifestError(ValueError):
    """manifests/upstream.yaml is malformed or does not pin something it must."""


class FetchError(RuntimeError):
    """A download failed, or a downloaded file does not have the pinned content."""


# ---------------------------------------------------------------------------------------------------- the manifest

@dataclass(frozen=True)
class Edit:
    old: str
    new: str
    count: int = 1


@dataclass(frozen=True)
class FilePatch:
    """Local edits of one upstream file (the paper's runs used the edited file)."""
    path: str
    upstream_blob: str
    edits: tuple[Edit, ...]
    why: str = ""

    def apply(self, data: bytes) -> bytes:
        """Applies the edits to the pristine upstream file; raises FetchError when the file is not that version or an
        edit does not occur exactly `count` times."""
        if git_blob_sha1(data) != self.upstream_blob:
            raise FetchError(f"{self.path}: blob {git_blob_sha1(data)} is not the upstream blob {self.upstream_blob} "
                             f"the patch was written for")
        text = data
        for number, edit in enumerate(self.edits, 1):
            old, new = edit.old.encode("utf-8"), edit.new.encode("utf-8")
            found = text.count(old)
            if found != edit.count:
                raise FetchError(f"{self.path}: edit {number} expects {edit.count} occurrence(s) of {edit.old[:60]!r}, "
                                 f"found {found}")
            text = text.replace(old, new)
        return text


@dataclass(frozen=True)
class Entry:
    """One upstream repository: where it comes from, what is taken, and the expected blob of every file."""
    name: str
    repo: str
    commit: str
    license: str
    group: str
    used_for: str
    files: Mapping[str, str]                                   # path -> blob as streamreplace uses it (after patches)
    patches: Mapping[str, FilePatch] = field(default_factory=dict)
    fetch: bool = True

    @property
    def owner_repo(self) -> tuple[str, str]:
        match = GITHUB.fullmatch(self.repo)
        if not match:
            raise ManifestError(f"{self.name}: repo {self.repo!r} is not a https://github.com/<owner>/<repo> URL")
        return match.group(1), match.group(2)

    def download_blob(self, path: str) -> str:
        """The blob the file has upstream at the commit (before any local edit)."""
        return self.patches[path].upstream_blob if path in self.patches else self.files[path]

    def raw_url(self, path: str) -> str:
        owner, repo = self.owner_repo
        return RAW_URL.format(owner=owner, repo=repo, commit=self.commit, path=urllib.parse.quote(path, safe="/"))

    def tarball_url(self) -> str:
        owner, repo = self.owner_repo
        return TARBALL_URL.format(owner=owner, repo=repo, commit=self.commit)

    def digest(self) -> str:
        """sha256 of the entry's pinned content (commit, files, patches): recorded in the fetched tree's marker."""
        payload = {"repo": self.repo, "commit": self.commit, "files": dict(sorted(self.files.items())),
                   "patches": {p: {"upstream_blob": f.upstream_blob, "edits": [[e.old, e.new, e.count] for e in f.edits]}
                               for p, f in sorted(self.patches.items())}}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def manifest_path(project: Path | None = None) -> Path:
    if project is None:
        from ..paths import project_root
        project = project_root()
    return Path(project) / MANIFEST


def _safe_relative(path: str, where: str) -> str:
    parts = PurePosixPath(path).parts
    if not path or path.startswith("/") or "\\" in path or ".." in parts or path != str(PurePosixPath(path)):
        raise ManifestError(f"{where}: {path!r} is not a clean relative path inside the repository")
    return path


def _entry(name: str, raw: Mapping) -> Entry:
    where = f"upstream.{name}"
    if not isinstance(raw, Mapping):
        raise ManifestError(f"{where}: expected a mapping")
    unknown = sorted(set(raw) - {"repo", "commit", "license", "group", "used_for", "files", "patches", "fetch"})
    if unknown:
        raise ManifestError(f"{where}: unknown keys {unknown}")
    for key in ("repo", "commit", "license", "group", "files"):
        if key not in raw:
            raise ManifestError(f"{where}: missing {key!r}")
    commit = str(raw["commit"])
    if not HEX40.fullmatch(commit):
        raise ManifestError(f"{where}: commit {commit!r} is not a full 40-hex commit (unpinned sources are refused)")
    group = str(raw["group"])
    if group not in GROUPS:
        raise ManifestError(f"{where}: group {group!r} not in {GROUPS}")
    files = raw["files"]
    if not isinstance(files, Mapping) or not files:
        raise ManifestError(f"{where}: files must map repository paths to git blob SHA-1s")
    clean = {}
    for path, blob in files.items():
        if not HEX40.fullmatch(str(blob)):
            raise ManifestError(f"{where}: {path}: {blob!r} is not a git blob SHA-1")
        clean[_safe_relative(str(path), where)] = str(blob)
    patches = {}
    for path, block in (raw.get("patches") or {}).items():
        path = _safe_relative(str(path), where)
        if path not in clean:
            raise ManifestError(f"{where}: patch for {path!r}, which is not in files")
        if not isinstance(block, Mapping) or not HEX40.fullmatch(str(block.get("upstream_blob", ""))):
            raise ManifestError(f"{where}: patch {path!r} needs upstream_blob (a git blob SHA-1)")
        edits = []
        for number, edit in enumerate(block.get("edits") or [], 1):
            if not isinstance(edit, Mapping) or not isinstance(edit.get("old"), str) or not edit["old"] \
                    or not isinstance(edit.get("new"), str):
                raise ManifestError(f"{where}: patch {path!r} edit {number} needs non-empty `old` and a `new` string")
            count = int(edit.get("count", 1))
            if count < 1:
                raise ManifestError(f"{where}: patch {path!r} edit {number}: count must be >= 1")
            edits.append(Edit(edit["old"], edit["new"], count))
        if not edits:
            raise ManifestError(f"{where}: patch {path!r} has no edits")
        patches[path] = FilePatch(path, str(block["upstream_blob"]), tuple(edits), str(block.get("why", "")).strip())
    fetch = bool(raw.get("fetch", True))
    if group == "reference" and fetch:
        raise ManifestError(f"{where}: reference entries are never fetched (set fetch: false)")
    entry = Entry(name, str(raw["repo"]), commit, str(raw["license"]), group, str(raw.get("used_for", "")).strip(),
                  clean, patches, fetch)
    entry.owner_repo                                            # validates the URL
    return entry


def parse_manifest(data: Mapping, where: str = "upstream manifest") -> dict[str, Entry]:
    if not isinstance(data, Mapping) or not isinstance(data.get("upstream"), Mapping):
        raise ManifestError(f"{where}: expected a top-level `upstream:` mapping")
    if data.get("version", 1) != 1:
        raise ManifestError(f"{where}: unsupported version {data.get('version')!r}")
    return {str(name): _entry(str(name), raw) for name, raw in data["upstream"].items()}


def load_manifest(path: Path | str | None = None) -> dict[str, Entry]:
    """name -> Entry, in manifest order (default file: manifests/upstream.yaml of the source checkout)."""
    import yaml
    path = Path(path) if path is not None else manifest_path()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"upstream manifest not found: {path}") from exc
    return parse_manifest(data, str(path))


def fetchable(manifest: Mapping[str, Entry]) -> list[str]:
    """Names `fetch-upstream all` fetches (every entry except the reference-only ones)."""
    return [name for name, entry in manifest.items() if entry.fetch]


def select(manifest: Mapping[str, Entry], names: Iterable[str] | None) -> list[Entry]:
    """Entries for NAMEs; "all" (or nothing) = every fetchable entry; a group name (core, generalization) = its
    entries. Reference-only entries cannot be selected: they are cited, not fetched."""
    names = [n for n in (names or []) if n] or ["all"]
    chosen: list[str] = []
    for name in names:
        if name == "all":
            chosen += fetchable(manifest)
        elif name in GROUPS and name not in manifest:
            chosen += [n for n in fetchable(manifest) if manifest[n].group == name]
        elif name not in manifest:
            raise ManifestError(f"unknown upstream {name!r}; choose from: {', '.join(fetchable(manifest))}, all, "
                                f"core, generalization")
        elif not manifest[name].fetch:
            raise ManifestError(f"{name} is listed for provenance only ({manifest[name].used_for}); it is never fetched")
        else:
            chosen.append(name)
    return [manifest[n] for n in dict.fromkeys(chosen)]


# ---------------------------------------------------------------------------------------------------- hashing

def git_blob_sha1(data: bytes) -> str:
    """The git blob id of a file's content (`git hash-object`)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def file_blob_sha1(path: Path | str) -> str:
    path = Path(path)
    digest = hashlib.sha1(b"blob %d\0" % path.stat().st_size)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------------------------------- verification

@dataclass
class Status:
    """State of one fetched tree: ok | missing (no tree) | incomplete (files absent) | modified (wrong content)."""
    name: str
    state: str
    files: int = 0
    missing: list[str] = field(default_factory=list)
    mismatched: list[tuple[str, str, str]] = field(default_factory=list)       # (path, found blob, expected blob)
    commit: str | None = None                                                    # from the tree's marker, if any
    action: str = ""                                                             # fetched | kept | verified

    @property
    def ok(self) -> bool:
        return self.state == "ok"

    def as_dict(self) -> dict:
        return {"name": self.name, "state": self.state, "files": self.files, "missing": self.missing,
                "mismatched": [list(m) for m in self.mismatched], "commit": self.commit, "action": self.action}

    def summary(self) -> str:
        if self.state == "ok":
            return f"ok ({self.files} files verified{', ' + self.action if self.action else ''})"
        if self.state == "missing":
            return "missing (not fetched)"
        parts = []
        if self.missing:
            parts.append(f"{len(self.missing)} missing: {', '.join(self.missing[:3])}{' ...' if len(self.missing) > 3 else ''}")
        if self.mismatched:
            parts.append(f"{len(self.mismatched)} modified: " + ", ".join(m[0] for m in self.mismatched[:3])
                         + (" ..." if len(self.mismatched) > 3 else ""))
        return f"{self.state} ({'; '.join(parts)})"


def read_marker(tree: Path) -> dict | None:
    try:
        return json.loads((tree / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def verify(entry: Entry, root: Path | str) -> Status:
    """Checks every listed file of `<root>/<name>` against its pinned blob (offline). Extra files are ignored."""
    tree = Path(root) / entry.name
    if not tree.is_dir():
        return Status(entry.name, "missing")
    status = Status(entry.name, "ok", commit=(read_marker(tree) or {}).get("commit"))
    for path, blob in entry.files.items():
        target = tree / path
        if not target.is_file():
            status.missing.append(path)
            continue
        found = file_blob_sha1(target)
        if found != blob:
            status.mismatched.append((path, found, blob))
    status.files = len(entry.files) - len(status.missing) - len(status.mismatched)
    if status.mismatched:
        status.state = "modified"
    elif status.missing:
        status.state = "incomplete"
    return status


# ---------------------------------------------------------------------------------------------------- downloading

def _headers(url: str) -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    host = urllib.parse.urlsplit(url).hostname or ""
    if token and (host == "github.com" or host.endswith(".github.com") or host.endswith("githubusercontent.com")):
        headers["Authorization"] = f"Bearer {token}"
    return headers


def http_get(url: str, attempts: int = 4, timeout: float = 120.0) -> bytes:
    """GET with retries (back-off on 429, 5xx and network errors). Raises FetchError."""
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=_headers(url)), timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            error = exc
            if exc.code not in (429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, OSError) as exc:
            error = exc
        if attempt + 1 < attempts:
            time.sleep(min(30.0, 2.0 ** attempt))
    raise FetchError(f"GET {url}: {error}")


def http_download(url: str, target: Path, attempts: int = 4, timeout: float = 300.0) -> Path:
    """Streams a (large) response to `target` (written as target.part, then renamed)."""
    error: Exception | None = None
    part = target.with_name(target.name + ".part")
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=_headers(url)), timeout=timeout) as response, \
                    part.open("wb") as handle:
                shutil.copyfileobj(response, handle, 1 << 22)
            part.replace(target)
            return target
        except urllib.error.HTTPError as exc:
            error = exc
            if exc.code not in (429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, OSError) as exc:
            error = exc
        if attempt + 1 < attempts:
            time.sleep(min(30.0, 2.0 ** attempt))
    part.unlink(missing_ok=True)
    raise FetchError(f"GET {url}: {error}")


def _final(entry: Entry, path: str, data: bytes) -> bytes:
    """Checks a downloaded file against its pinned upstream blob and applies the local edits (if any)."""
    expected = entry.download_blob(path)
    found = git_blob_sha1(data)
    if found != expected:
        raise FetchError(f"{entry.name}/{path}: downloaded blob {found}, pinned {expected}")
    if path in entry.patches:
        data = entry.patches[path].apply(data)
        if git_blob_sha1(data) != entry.files[path]:
            raise FetchError(f"{entry.name}/{path}: patched blob {git_blob_sha1(data)}, expected {entry.files[path]}")
    return data


def _from_raw(entry: Entry, get: Getter, jobs: int) -> dict[str, bytes]:
    paths = list(entry.files)
    with ThreadPoolExecutor(max(1, jobs)) as pool:
        bodies = list(pool.map(lambda p: get(entry.raw_url(p)), paths))
    return {path: _final(entry, path, body) for path, body in zip(paths, bodies)}


def _from_tarball(entry: Entry, staging: Path, download: Callable[[str, Path], Path]) -> dict[str, bytes]:
    archive = download(entry.tarball_url(), staging / f"{entry.name}.tar.gz")
    found: dict[str, bytes] = {}
    try:
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                if not member.isfile() or "/" not in member.name:
                    continue
                path = member.name.split("/", 1)[1]              # strip the "<repo>-<commit>/" prefix
                if path in entry.files:
                    found[path] = tar.extractfile(member).read()
    finally:
        archive.unlink(missing_ok=True)
    absent = sorted(set(entry.files) - set(found))
    if absent:
        raise FetchError(f"{entry.name}: the tarball lacks {len(absent)} listed file(s): {', '.join(absent[:5])}")
    return {path: _final(entry, path, found[path]) for path in entry.files}


def _from_local(entry: Entry, source: Path) -> dict[str, bytes]:
    """Files from an existing checkout <source>/<name>; an already-edited copy of a patched file is accepted too."""
    tree = Path(source) / entry.name
    out = {}
    for path in entry.files:
        target = tree / path
        if not target.is_file():
            raise FetchError(f"{entry.name}: {target} is missing")
        data = target.read_bytes()
        if path in entry.patches and git_blob_sha1(data) == entry.files[path]:
            out[path] = data                                     # the copy already carries the local edits
        else:
            out[path] = _final(entry, path, data)
    return out


def _write_tree(entry: Entry, files: Mapping[str, bytes], tree: Path, method: str) -> None:
    for path, data in files.items():
        target = tree / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    marker = {"name": entry.name, "repo": entry.repo, "commit": entry.commit, "files": len(files),
              "patched": sorted(entry.patches), "manifest_digest": entry.digest(), "method": method}
    (tree / MARKER).write_text(json.dumps(marker, indent=1) + "\n", encoding="utf-8")


def fetch(entry: Entry, root: Path | str, method: str = "auto", force: bool = False, jobs: int = 8,
          source: Path | str | None = None, get: Getter | None = None,
          download: Callable[[str, Path], Path] | None = None, log: Callable[[str], None] = print) -> Status:
    """Fetches one repository's pinned files into <root>/<name> (see the module docstring). Returns the verified
    status; raises FetchError when the files cannot be obtained with the pinned content."""
    if not entry.fetch:
        raise FetchError(f"{entry.name} is reference-only and is never fetched")
    if method not in METHODS:
        raise ValueError(f"method {method!r} not in {METHODS}")
    if method == "local" and source is None:
        raise ValueError("method 'local' needs a source folder")
    root = Path(root)
    current = verify(entry, root)
    if current.ok and not force:
        current.action = "kept"
        return current
    get = get or http_get
    download = download or http_download
    root.mkdir(parents=True, exist_ok=True)
    (root / STAGING).mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{entry.name}-", dir=root / STAGING))
    try:
        if method == "local":
            files = _from_local(entry, Path(source))
        elif method == "tarball":
            files = _from_tarball(entry, staging, download)
        else:
            try:
                files = _from_raw(entry, get, jobs)
                method = "raw"
            except FetchError as exc:
                if method == "raw":
                    raise
                log(f"  {entry.name}: per-file download failed ({exc}); trying the repository tarball")
                method = "tarball"
                files = _from_tarball(entry, staging, download)
        new_tree = staging / entry.name
        _write_tree(entry, files, new_tree, method)
        tree = root / entry.name
        old = staging / f"{entry.name}.previous"
        if tree.exists():
            tree.rename(old)
        new_tree.rename(tree)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        try:
            (root / STAGING).rmdir()
        except OSError:
            pass
    status = verify(entry, root)
    status.action = f"fetched ({method})"
    if not status.ok:                                            # cannot happen unless the disk lies; never silent
        raise FetchError(f"{entry.name}: fetched tree does not verify: {status.summary()}")
    return status


def fetch_all(entries: Iterable[Entry], root: Path | str, method: str = "auto", force: bool = False, jobs: int = 8,
              source: Path | str | None = None, log: Callable[[str], None] = print, **kwargs) -> dict[str, Status]:
    """Fetches several repositories; a failure is reported for that repository and the others continue."""
    out: dict[str, Status] = {}
    for entry in entries:
        started = time.time()
        try:
            status = fetch(entry, root, method=method, force=force, jobs=jobs, source=source, log=log, **kwargs)
        except FetchError as exc:
            status = verify(entry, root)
            status.state = "failed" if status.ok or status.state == "missing" else status.state
            status.action = f"failed: {exc}"
            log(f"{entry.name:18s} FAILED {exc}")
        else:
            log(f"{entry.name:18s} {status.summary()} [{time.time() - started:.1f}s]")
        out[entry.name] = status
    return out
