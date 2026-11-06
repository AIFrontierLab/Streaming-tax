"""Python environments: build them with uv from the pinned locks, apply the checked patches, and check an existing
environment against its lock (`streamreplace env list | build | check | patch`).

Every model family needs its own torch / transformers stack, so the paper's runs used one environment per family
plus an orchestrator environment and a judge environment. Each is defined in the repository by
`envs/<name>/requirements.lock` (the exact packages of the paper's runs), `envs/<name>/requirements.in` (the direct
requirements, for re-locking) and `envs/<name>/python-version`. A built environment lives at `<envs root>/<name>`
(paths.py: STREAMREPLACE_ENVS, default $STREAMREPLACE_HOME/envs), its interpreter at `<envs root>/<name>/bin/python`.

build: `uv venv --python <version> --seed`, then `uv pip sync <lock>` (installs exactly the lock, nothing resolved
anew), then the patches listed for the environment in envs/patches/manifest.yaml (each checked by the file's sha256
before and after), then `streamreplace-env.json` in the environment records the lock's sha256 and the patches. The
core environment also gets this repository installed in editable mode (model environments do not: the workers put
the repository's src/ on PYTHONPATH). The special name `ffmpeg` builds the FFmpeg 6 shared libraries that torchcodec
needs (`<envs root>/ffmpeg/lib`), as a conda-forge prefix with micromamba, mamba or conda.

check: the interpreter runs, its Python version matches, the installed distributions equal the lock (missing,
different and extra packages are listed), the patches are in place, and the key modules import (import probes)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Callable, Iterable

from ..paths import project_root

FFMPEG = "ffmpeg"
FFMPEG_SPEC = "ffmpeg=6.1.2=gpl_hdfc89ed_706"     # the conda-forge build of the paper's runs (FFmpeg 6.1.2)
ENV_FILE = "streamreplace-env.json"
# extra `uv pip sync` options: the judge lock mixes the PyTorch cu129 index with PyPI
UV_OPTIONS: dict[str, list[str]] = {"judge": ["--index-strategy", "unsafe-best-match"]}
# environments that get this repository installed (editable, without dependencies)
INSTALL_SELF = {"core"}
# modules each environment's code imports (checked by `env check`; failures are reported, not fixed)
PROBES: dict[str, tuple[str, ...]] = {
    "core": ("numpy", "yaml", "PIL", "av", "decord", "pyarrow"),
    "modern": ("torch", "transformers", "qwen_vl_utils", "torchcodec", "decord", "av", "cv2", "peft", "flash_attn",
               "flashinfer"),
    "llava": ("torch", "transformers", "peft", "decord", "av", "cv2", "timm", "open_clip"),
    "livestar": ("torch", "transformers", "peft", "decord", "av", "cv2", "timm", "sentencepiece", "flash_attn"),
    "mmduet2": ("torch", "transformers", "qwen_vl_utils", "decord", "av", "cv2", "flash_attn"),
    "aura": ("vllm", "transformers", "openai", "decord", "av", "cv2"),
    "judge": ("vllm", "torch", "transformers", "accelerate", "openai", "google.genai", "spacy", "datasketch",
              "en_core_web_sm"),
    "lingo": ("torch", "transformers", "sentencepiece", "pandas", "datasets"),
}


class PatchError(RuntimeError):
    """A patch does not apply to the installed file exactly (another package version?)."""


def env_definitions_dir() -> Path:
    return project_root() / "envs"


def normalize(name: str) -> str:
    """Distribution name in PEP 503 form (lower case, runs of '-', '_', '.' become '-')."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_lock(text: str) -> tuple[dict[str, str], list[str]]:
    """({normalized name: "==<version>" or "@ <url>"}, [option lines such as --extra-index-url ...])."""
    packages, options = {}, []
    for raw in text.splitlines():
        line = raw.split(" #", 1)[0].strip() if not raw.lstrip().startswith("#") else ""
        if not line:
            continue
        if line.startswith("-"):
            options.append(line)
        elif " @ " in line:
            name, url = line.split(" @ ", 1)
            packages[normalize(name.strip())] = "@ " + url.strip()
        elif "==" in line:
            name, version = line.split("==", 1)
            packages[normalize(name.strip())] = "==" + version.strip()
        else:
            raise ValueError(f"lock line is not pinned: {raw!r}")
    return packages, options


@dataclass(frozen=True)
class EnvDefinition:
    name: str
    folder: Path

    @property
    def lock(self) -> Path:
        return self.folder / "requirements.lock"

    @property
    def requirements(self) -> Path:
        return self.folder / "requirements.in"

    @property
    def python(self) -> str:
        return (self.folder / "python-version").read_text(encoding="utf-8").strip()

    def packages(self) -> dict[str, str]:
        return parse_lock(self.lock.read_text(encoding="utf-8"))[0]

    def lock_sha256(self) -> str:
        return hashlib.sha256(self.lock.read_bytes()).hexdigest()


def definitions(folder: Path | None = None) -> dict[str, EnvDefinition]:
    """Every environment defined in the repository: envs/<name>/ with a requirements.lock."""
    folder = Path(folder) if folder is not None else env_definitions_dir()
    return {p.name: EnvDefinition(p.name, p) for p in sorted(folder.iterdir())
            if p.is_dir() and (p / "requirements.lock").is_file()}


def names(folder: Path | None = None) -> list[str]:
    """Buildable names: every defined environment, plus ffmpeg (first: other environments' probes need it)."""
    return [FFMPEG, *definitions(folder)]


def expand(requested: Iterable[str], folder: Path | None = None) -> list[str]:
    """Names (comma lists allowed) with "all" = every buildable name; unknown names are an error."""
    known = names(folder)
    out: list[str] = []
    for item in requested:
        for name in (n for n in str(item).split(",") if n):
            if name == "all":
                out += known
            elif name in known:
                out.append(name)
            else:
                raise ValueError(f"unknown environment {name!r} (known: {', '.join(known)})")
    return list(dict.fromkeys(out))


# Patches ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Patch:
    name: str
    env: str
    file: Path
    target: str                      # relative to the environment's site-packages
    sha256_before: str
    sha256_after: str
    package: str = ""
    why: str = ""


def patches(folder: Path | None = None, env: str | None = None) -> list[Patch]:
    """The patches of envs/patches/manifest.yaml (those of one environment when `env` is given)."""
    import yaml
    root = (Path(folder) if folder is not None else env_definitions_dir()) / "patches"
    manifest = root / "manifest.yaml"
    if not manifest.is_file():
        return []
    entries = (yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}).get("patches") or []
    out = [Patch(name=e["name"], env=e["env"], file=root / e["file"], target=e["target"],
                 sha256_before=e["sha256_before"], sha256_after=e["sha256_after"], package=e.get("package", ""),
                 why=" ".join(str(e.get("why", "")).split())) for e in entries]
    return [p for p in out if env is None or p.env == env]


def _hunks(diff: str) -> tuple[str, list[tuple[int, int, list[str]]]]:
    """(target path of the +++ line, [(old start, old length, hunk lines)]) of a single-file unified diff; text
    before the --- line is a comment."""
    lines = diff.splitlines(keepends=True)
    index = next((i for i, line in enumerate(lines) if line.startswith("--- ")), None)
    if index is None or index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
        raise PatchError("not a unified diff (no ---/+++ header)")
    target = lines[index + 1][4:].strip().split("\t")[0]
    hunks: list[tuple[int, int, list[str]]] = []
    for line in lines[index + 2:]:
        if line.startswith("@@"):
            match = re.match(r"@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", line)
            if not match:
                raise PatchError(f"bad hunk header {line.strip()!r}")
            hunks.append((int(match.group(1)), int(match.group(2) or 1), []))
        elif line.startswith(("--- ", "+++ ")):
            raise PatchError("one file per patch")
        elif hunks and line[:1] in (" ", "-", "+"):
            hunks[-1][2].append(line)
        elif hunks and line.startswith("\\"):
            continue                              # "\ No newline at end of file"
        elif line.strip():
            raise PatchError(f"unexpected line in diff: {line.rstrip()!r}")
    if not hunks:
        raise PatchError("the diff has no hunks")
    return target, hunks


def apply_unified_diff(original: str, diff: str) -> str:
    """Applies a single-file unified diff exactly: every context and removed line must match at its stated position
    (no fuzz, no offsets)."""
    lines = original.splitlines(keepends=True)
    out: list[str] = []
    position = 0
    for start, length, body in _hunks(diff)[1]:
        first = start - 1 if length else start
        if first < position:
            raise PatchError("overlapping or unordered hunks")
        out.extend(lines[position:first])
        position = first
        for line in body:
            tag, text = line[0], line[1:]
            if tag in " -":
                if position >= len(lines) or lines[position].rstrip("\r\n") != text.rstrip("\r\n"):
                    found = lines[position].rstrip() if position < len(lines) else "<end of file>"
                    raise PatchError(f"line {position + 1}: expected {text.rstrip()!r}, found {found!r}")
                if tag == " ":
                    out.append(lines[position])
                position += 1
            else:
                out.append(text if text.endswith("\n") else text + "\n")
    out.extend(lines[position:])
    return "".join(out)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patch_status(patch: Patch, site_packages: Path) -> str:
    """'applied', 'unpatched' (the expected original), 'missing' or 'unexpected' (another file version)."""
    target = Path(site_packages) / patch.target
    if not target.is_file():
        return "missing"
    digest = _sha256(target)
    return {patch.sha256_after: "applied", patch.sha256_before: "unpatched"}.get(digest, "unexpected")


def apply_patch(patch: Patch, site_packages: Path) -> str:
    """Applies `patch` under `site_packages` if needed; returns 'applied' or 'already applied'. The file must be the
    expected original (sha256_before) and the result must be sha256_after, otherwise nothing is written."""
    target = Path(site_packages) / patch.target
    status = patch_status(patch, site_packages)
    if status == "applied":
        return "already applied"
    if status != "unpatched":
        raise PatchError(f"{patch.name}: {target} is {status} (expected sha256 {patch.sha256_before}"
                         f"{' of ' + patch.package if patch.package else ''})")
    diff = patch.file.read_text(encoding="utf-8")
    declared = _hunks(diff)[0]
    if not declared.endswith(patch.target):
        raise PatchError(f"{patch.file.name} patches {declared}, the manifest says {patch.target}")
    result = apply_unified_diff(target.read_text(encoding="utf-8"), diff)
    if hashlib.sha256(result.encode("utf-8")).hexdigest() != patch.sha256_after:
        raise PatchError(f"{patch.name}: the patched file does not have the expected sha256 {patch.sha256_after}")
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp.write_text(result, encoding="utf-8")
    shutil.copymode(target, tmp)
    tmp.replace(target)
    return "applied"


# Running an environment's interpreter ----------------------------------------------------------------------------

Runner = Callable[..., subprocess.CompletedProcess]

_INSTALLED = r"""
import json
from importlib.metadata import distributions
out = {}
for dist in distributions():                    # the first one per name is the one imports resolve to
    if dist.metadata["Name"] in out:
        continue
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    out[dist.metadata["Name"]] = {"version": dist.version, "url": direct.get("url"),
                                  "editable": bool(direct.get("dir_info", {}).get("editable"))}
print(json.dumps(out))
"""

_PROBE = r"""
import importlib, json, sys
out = {}
for name in sys.argv[1:]:
    try:
        module = importlib.import_module(name)
        out[name] = {"ok": True, "version": str(getattr(module, "__version__", ""))}
    except BaseException as exc:
        out[name] = {"ok": False, "error": (type(exc).__name__ + ": " + str(exc))[:300]}
print(json.dumps(out))
"""


def _python_json(python: Path, code: str, args: list[str], run: Runner, environ: dict | None = None,
                 timeout: int = 900) -> dict:
    env = dict(os.environ)
    env.update(environ or {})
    done = run([str(python), "-I", "-c", code, *args], capture_output=True, text=True, timeout=timeout, env=env)
    if done.returncode != 0:
        raise RuntimeError(f"{python} failed: {(done.stderr or '').strip()[-500:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def python_version(python: Path, run: Runner = subprocess.run) -> str:
    code = "import json, platform; print(json.dumps({'version': platform.python_version()}))"
    return str(_python_json(python, code, [], run)["version"])


def site_packages(python: Path, run: Runner = subprocess.run) -> Path:
    code = "import json, sysconfig; print(json.dumps({'purelib': sysconfig.get_paths()['purelib']}))"
    return Path(_python_json(python, code, [], run)["purelib"])


def installed(python: Path, run: Runner = subprocess.run) -> dict[str, dict]:
    """{distribution name: {"version", "url", "editable"}} of an environment."""
    return _python_json(python, _INSTALLED, [], run)


def probe(python: Path, modules: Iterable[str], run: Runner = subprocess.run, environ: dict | None = None) -> dict:
    return _python_json(python, _PROBE, list(modules), run, environ)


def compare_packages(lock: dict[str, str], found: dict[str, dict]) -> dict[str, list]:
    """Differences between a lock and an environment: {"missing", "different", "extra"}; editable installs (this
    repository in the core environment) are not counted as extra."""
    have: dict[str, str] = {}
    for name, entry in found.items():
        if entry.get("editable"):
            continue
        have[normalize(name)] = f"@ {entry['url']}" if entry.get("url") else f"=={entry['version']}"
    missing = sorted(set(lock) - set(have))
    extra = sorted(set(have) - set(lock))
    different = sorted([name, lock[name], have[name]] for name in set(lock) & set(have) if lock[name] != have[name])
    return {"missing": missing, "different": different, "extra": extra}


# Build -------------------------------------------------------------------------------------------------------------

@dataclass
class Plan:
    """What `env build` does for one environment: commands to run, then patches, then the record."""
    name: str
    target: Path
    commands: list[list[str]] = field(default_factory=list)
    patches: list[Patch] = field(default_factory=list)


def conda_tool() -> str | None:
    for tool in ("micromamba", "mamba", "conda"):
        if shutil.which(tool):
            return tool
    return None


def plan(name: str, envs_root: Path, uv: str = "uv", conda: str | None = None, recreate: bool = False,
         folder: Path | None = None, repo: Path | None = None, dry_run: bool = False) -> Plan:
    """The build plan of one environment (nothing is run)."""
    target = Path(envs_root) / name
    if name == FFMPEG:
        tool = conda or conda_tool() or ("conda" if dry_run else None)
        if tool is None:
            raise RuntimeError("building the FFmpeg libraries needs micromamba, mamba or conda on PATH")
        commands = [] if (target / "lib").is_dir() and not recreate else \
            [[tool, "create", "-y", "-p", str(target), "-c", "conda-forge", FFMPEG_SPEC]]
        return Plan(name, target, commands)
    definition = definitions(folder)[name]
    python = target / "bin" / "python"
    commands = []
    if recreate or not python.exists():
        commands.append([uv, "venv", *(["--clear"] if recreate else []), "--python", definition.python, "--seed",
                         str(target)])
    commands.append([uv, "pip", "sync", "--python", str(python), *UV_OPTIONS.get(name, []), str(definition.lock)])
    if name in INSTALL_SELF:
        commands.append([uv, "pip", "install", "--python", str(python), "--no-deps", "-e",
                         str(repo or project_root())])
    return Plan(name, target, commands, patches(folder, name))


def build(name: str, envs_root: Path, uv: str = "uv", conda: str | None = None, recreate: bool = False,
          dry_run: bool = False, run: Runner = subprocess.run, folder: Path | None = None,
          log: Callable[[str], None] = print) -> dict:
    """Builds (or updates) one environment; returns its record. dry_run=True only logs the commands."""
    steps = plan(name, envs_root, uv, conda, recreate, folder, dry_run=dry_run)
    for command in steps.commands:
        log("$ " + " ".join(command))
        if not dry_run:
            done = run(command)
            if done.returncode != 0:
                raise RuntimeError(f"{name}: command failed ({done.returncode}): {' '.join(command)}")
    if name == FFMPEG:
        return {"name": name, "spec": FFMPEG_SPEC, "dry_run": dry_run}
    python = steps.target / "bin" / "python"
    applied = []
    for patch in steps.patches:
        log(f"patch {patch.name} -> {patch.target}")
        if not dry_run:
            applied.append({"name": patch.name, "status": apply_patch(patch, site_packages(python, run)),
                            "sha256": patch.sha256_after})
    definition = definitions(folder)[name]
    record = {"name": name, "python": definition.python, "lock_sha256": definition.lock_sha256(), "patches": applied,
              "built": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if not dry_run:
        (steps.target / ENV_FILE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return {**record, "dry_run": dry_run}


# Check -------------------------------------------------------------------------------------------------------------

def check(name: str, envs_root: Path, run: Runner = subprocess.run, probes: bool = True,
          ffmpeg_lib: Path | None = None, folder: Path | None = None) -> dict:
    """The state of one built environment: {"ok", "problems", "warnings", ...details}."""
    target = Path(envs_root) / name
    report: dict = {"name": name, "path": str(target), "problems": [], "warnings": []}
    if name == FFMPEG:
        libs = sorted(p.name for p in (target / "lib").glob("libavcodec.so*")) if (target / "lib").is_dir() else []
        report["libavcodec"] = libs
        if not libs:
            report["problems"].append(f"no FFmpeg libraries in {target / 'lib'} (streamreplace env build ffmpeg)")
        report["ok"] = not report["problems"]
        return report
    definition = definitions(folder)[name]
    python = target / "bin" / "python"
    if not python.exists():
        report["problems"].append(f"{python} does not exist (streamreplace env build {name})")
        report["ok"] = False
        return report
    try:
        version = python_version(python, run)
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        report["problems"].append(f"the interpreter does not run: {exc}")
        report["ok"] = False
        return report
    report["python"] = version
    if version.split(".")[:2] != definition.python.split(".")[:2]:
        report["problems"].append(f"Python {version}, the lock is for {definition.python}")
    elif version != definition.python:
        report["warnings"].append(f"Python {version} (the paper's runs used {definition.python})")
    differences = compare_packages(definition.packages(), installed(python, run))
    report["packages"] = differences
    for key in ("missing", "different"):
        if differences[key]:
            report["problems"].append(f"{len(differences[key])} {key} package(s): "
                                      + ", ".join(d if isinstance(d, str) else d[0] for d in differences[key][:8]))
    if differences["extra"]:
        report["warnings"].append(f"{len(differences['extra'])} package(s) not in the lock: "
                                  + ", ".join(differences["extra"][:8]))
    env_patches = patches(folder, name)
    if env_patches:
        where = site_packages(python, run)
        report["patches"] = {p.name: patch_status(p, where) for p in env_patches}
        for patch_name, status in report["patches"].items():
            if status != "applied":
                report["problems"].append(f"patch {patch_name}: {status} (streamreplace env patch {name})")
    record = target / ENV_FILE
    if record.is_file():
        built = json.loads(record.read_text(encoding="utf-8"))
        if built.get("lock_sha256") != definition.lock_sha256():
            report["warnings"].append("the lock changed since this environment was built")
    if probes and name in PROBES:
        environ = {}
        if ffmpeg_lib is not None and Path(ffmpeg_lib).is_dir():
            environ["LD_LIBRARY_PATH"] = os.pathsep.join(
                p for p in (str(ffmpeg_lib), os.environ.get("LD_LIBRARY_PATH", "")) if p)
        results = probe(python, PROBES[name], run, environ)
        report["probes"] = results
        failed = [module for module, result in results.items() if not result.get("ok")]
        if failed:
            report["problems"].append("modules that do not import: " + ", ".join(failed))
    report["ok"] = not report["problems"]
    return report


def patch_env(name: str, envs_root: Path, run: Runner = subprocess.run, folder: Path | None = None) -> list[dict]:
    """Applies the environment's patches to an existing environment (e.g. one installed by hand)."""
    python = Path(envs_root) / name / "bin" / "python"
    where = site_packages(python, run)
    return [{"name": p.name, "status": apply_patch(p, where)} for p in patches(folder, name)]
