"""Helpers for benchmark preparation and for reading the benchmarks' official code (ported unchanged).

Preparation: save embedded images as RGB PNG, extract media from zip/tar archives once (a marker file makes it
idempotent), read parquet tables, letter the options of a multiple-choice question.

Official code: several benchmarks score with their authors' own functions (Video-MME's answer regex, TOMATO's
pre-parser and judge prompt, TimeBlind's extractor, VideoReasonBench's judge, Video-SafetyBench's prompt). That code
is fetched at a pinned commit by `streamreplace fetch-upstream` into the upstream root ($STREAMREPLACE_HOME/upstream,
see paths.py) and read from source here, so scoring needs no API clients. `relative` paths are
"<upstream name>/<path inside that repository>", e.g. "timeblind/utils.py"."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile

from ...models.upstream import upstream_path


def save_image(data, path: Path) -> Path:
    """Writes embedded image bytes (or a PIL image) as RGB PNG; skips files that already exist."""
    from PIL import Image
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, dict):
            data = data.get("bytes")
        image = data if isinstance(data, Image.Image) else Image.open(BytesIO(data))
        tmp = path.with_suffix(".tmp.png")
        image.convert("RGB").save(tmp)
        tmp.rename(path)
    return path


def extract_zip(archive: Path, destination: Path,
                suffixes=(".mp4", ".mkv", ".webm", ".avi", ".mov", ".png", ".jpg", ".jpeg", ".srt")):
    """Extracts media members once; a marker file makes it idempotent across re-prepares."""
    marker = destination / f".extracted-{Path(archive).name}"
    if marker.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            if info.is_dir() or not info.filename.lower().endswith(suffixes) or "__MACOSX" in info.filename:
                continue
            target = destination / info.filename
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(info) as source, open(target, "wb") as out:
                while chunk := source.read(1 << 22):
                    out.write(chunk)
    marker.write_text("ok\n")


def extract_tar(archive: Path, destination: Path, suffixes=(".mp4", ".mkv", ".webm", ".avi", ".mov")):
    """extract_zip for a (gzipped) tar: media members only, streamed in one pass, idempotent via the marker."""
    import tarfile
    marker = destination / f".extracted-{Path(archive).name}"
    if marker.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r|*") as handle:
        for info in handle:
            if (not info.isfile() or not info.name.lower().endswith(suffixes) or info.name.startswith("/")
                    or ".." in Path(info.name).parts):
                continue
            target = destination / info.name
            if target.exists() and target.stat().st_size == info.size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            with handle.extractfile(info) as source, open(tmp, "wb") as out:
                while chunk := source.read(1 << 22):
                    out.write(chunk)
            tmp.rename(target)
    marker.write_text("ok\n")


def read_parquet(path: Path) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(f"reading {Path(path).name} needs pyarrow: pip install -e \".[prepare]\" (or pip install "
                          "pyarrow)") from exc
    return pq.read_table(path).to_pylist()


def letters(options) -> dict[str, str]:
    return {chr(ord("A") + i): str(v).strip() for i, v in enumerate(options)}


# ---------- official evaluation code (fetched upstream repositories, pinned in manifests/upstream.yaml) ----------

def upstream_module(relative: str):
    """Import an upstream file that only needs the standard library, by path (cached)."""
    import importlib.util
    import sys
    path = upstream_path(relative)
    name = "_upstream_" + relative.replace("/", "_").removesuffix(".py")
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules[name]


def upstream_functions(relative: str, *names: str, scope: dict | None = None) -> dict:
    """Top-level functions from an upstream file whose module-level imports (API clients) the scoring env lacks:
    only those definitions are compiled, verbatim, into `scope`. "Class.method" compiles a method as a plain function
    (call it with a stand-in for self)."""
    import ast
    path = upstream_path(relative)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    nodes, keys = [], {}
    for name in names:
        owner, _, function = name.rpartition(".")
        body = classes[owner].body if owner else tree.body
        node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == function)
        nodes.append(node)
        keys[name] = function
    scope = dict(scope or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return {name: scope[function] for name, function in keys.items()}


def upstream_constant(relative: str, name: str):
    """The value of a module-level assignment (e.g. a prompt-template dict) in an upstream file, compiled from source."""
    import ast
    path = upstream_path(relative)
    node = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    return eval(compile(ast.Expression(node.value), str(path), "eval"), {})


def upstream_assignment(relative: str, function: str, variable: str):
    """The expression assigned to `variable` inside an upstream function (a prompt f-string, a constant), compiled from
    the source so the wording is the official one; returns a callable taking that function's local names."""
    import ast
    path = upstream_path(relative)
    func = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(n, ast.FunctionDef) and n.name == function)
    node = next(n for n in ast.walk(func) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == variable for t in n.targets))
    code = compile(ast.Expression(node.value), str(path), "eval")
    return lambda **names: eval(code, {}, names)
