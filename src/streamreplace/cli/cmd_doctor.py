"""`streamreplace doctor`: what this installation can run.

Shows the resolved roots (and where each came from), validates the configuration, and lists which model
environments, fetched upstream repositories (checked against the pinned blobs of manifests/upstream.yaml),
checkpoints, prepared benchmarks and adapter/benchmark modules exist. Exit code 1 when the configuration (or a paths
file) is invalid; with --strict also when anything is missing or an upstream tree does not verify."""
from __future__ import annotations

import json
from pathlib import Path
import platform
import subprocess
import sys

from .. import __version__
from . import context


def register(subparsers) -> None:
    parser = subparsers.add_parser("doctor", help="check paths, configuration, environments, upstream code and data",
                                   description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--probe-envs", action="store_true",
                        help="start each model environment's interpreter once and report its Python version")
    parser.add_argument("--strict", action="store_true", help="exit 1 when anything is missing")
    parser.set_defaults(func=run)


def _envs(config) -> set[str]:
    names = {block.env for model in config.models.values() for block in (model.offline, model.stream) if block}

    def walk(value):                                     # judge environments, wherever judges.yaml names them
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "env" and isinstance(item, str):
                    names.add(item)
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(config.section("judges"))
    return names - {"local"}


def _upstream(project: Path, root: Path) -> tuple[dict, bool]:
    """(report, something missing or wrong) for the third-party code under the upstream root.

    The expected repositories are the fetchable entries of the source checkout's manifests/upstream.yaml, read through
    provisioning.upstream (the reference-only entries, AURA and MOSS-VL, are cited and never fetched). Each tree is
    checked offline against its pinned blobs: ok | missing (no tree) | incomplete (files absent) | modified (wrong
    content). Without a manifest (no source checkout) only the folders present are listed."""
    import yaml
    from ..provisioning import upstream as up
    present = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")) \
        if root.is_dir() else []
    report: dict = {"root": str(root), "present": present, "expected": None, "states": {}, "missing": []}
    path = up.manifest_path(project)
    if not path.is_file():
        return report, False
    report["manifest"] = str(path)
    try:
        manifest = up.load_manifest(path)
    except (up.ManifestError, ValueError, TypeError, OSError, yaml.YAMLError) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}" if not isinstance(exc, up.ManifestError) else str(exc)
        return report, True
    expected = up.fetchable(manifest)
    states = {name: up.verify(manifest[name], root).state for name in expected}
    report.update(expected=expected, states=states,
                  missing=[name for name, state in states.items() if state == "missing"])
    return report, any(state != "ok" for state in states.values())


def _probe(python: Path) -> str:
    try:
        out = subprocess.run([str(python), "-c", "import sys; print(sys.version.split()[0])"], capture_output=True,
                             text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"failed: {exc}"
    return out.stdout.strip() if out.returncode == 0 else f"failed: {out.stderr.strip()[-200:]}"


def collect(args) -> tuple[dict, bool, bool]:
    """(report, fatal problem found, something missing)."""
    ctx = context(args)
    report: dict = {"streamreplace": __version__, "python": platform.python_version(),
                    "executable": sys.executable}
    fatal = missing = False
    try:
        paths = ctx.paths
    except (ValueError, OSError) as exc:
        report["paths"] = {"error": str(exc)}
        return report, True, True
    report["paths"] = {name: {"path": str(paths.root(name)), "source": paths.sources.get(name, ""),
                              "exists": paths.root(name).is_dir()}
                       for name in ("home", "data", "checkpoints", "envs", "outputs", "cache", "upstream")}
    try:
        config = ctx.config
    except (ValueError, OSError) as exc:
        report["config"] = {"dir": str(ctx.config_dir), "error": str(exc)}
        return report, True, True
    report["config"] = {"dir": str(ctx.config_dir), "files": [str(p) for p in config.sources],
                        "models": len(config.models), "pairs": len(config.pairs),
                        "benchmarks": len(config.benchmarks), "suites": sorted(config.suites),
                        "sections": sorted(config.sections)}

    envs = {}
    for env in sorted(_envs(config)):
        python = paths.env_python(env)
        entry = {"python": str(python), "exists": python.exists()}
        if args.probe_envs and entry["exists"]:
            entry["version"] = _probe(python)
        envs[env] = entry
        missing |= not entry["exists"]
    report["envs"] = envs

    report["upstream"], incomplete = _upstream(Path(ctx.config_dir).resolve().parent, paths.upstream)
    missing |= incomplete

    checkpoints = {}
    for model in config.models.values():
        for role, ref in model.checkpoints().items():
            folder = paths.checkpoint_dir(ref)
            entry = {"model": model.id, "role": role, "dir": str(folder), "exists": folder.is_dir()}
            marker = folder / ".revision"
            if marker.is_file():
                entry["revision_matches"] = marker.read_text(encoding="utf-8").strip() == (ref.revision or "")
            checkpoints[str(ref)] = checkpoints.get(str(ref)) or entry
            missing |= not entry["exists"]
    report["checkpoints"] = checkpoints

    from ..benchmarks import registry as benchmarks
    from ..models import registry as adapters
    data = {}
    for name, spec in config.benchmarks.items():
        folder = paths.dataset_dir(name)
        data[name] = {"dir": str(folder), "prepared": (folder / "prepared" / "samples.jsonl").is_file(),
                      "module": benchmarks.available(name) if name in benchmarks.names() else False}
        missing |= not data[name]["prepared"]
    report["data"] = data
    report["adapters"] = {name: adapters.available(name) for name in adapters.names()}
    return report, fatal, missing


def _print(report: dict) -> None:
    print(f"streamreplace {report['streamreplace']} (python {report['python']}: {report['executable']})")
    print("\npaths:")
    if "error" in report["paths"]:
        print(f"  ERROR {report['paths']['error']}")
        return
    for name, entry in report["paths"].items():
        print(f"  {name:12s} {'ok     ' if entry['exists'] else 'missing'} {entry['path']}  [{entry['source']}]")
    config = report.get("config", {})
    print(f"\nconfiguration: {config.get('dir')}")
    if "error" in config:
        print(f"  ERROR {config['error']}")
        return
    print(f"  {config['models']} models, {config['pairs']} pairs, {config['benchmarks']} benchmarks; suites: "
          f"{', '.join(config['suites'])}; other sections: {', '.join(config['sections']) or '-'}")
    print("\nenvironments:")
    for env, entry in report["envs"].items():
        version = f" (python {entry['version']})" if "version" in entry else ""
        print(f"  {env:12s} {'ok     ' if entry['exists'] else 'missing'} {entry['python']}{version}")
    upstream = report["upstream"]
    print(f"\nupstream code: {upstream['root']}")
    print(f"  present: {', '.join(upstream['present']) or '-'}")
    if "error" in upstream:
        print(f"  ERROR {upstream['error']}")
    elif upstream["expected"] is not None:
        states = upstream["states"]
        verified = [name for name, state in states.items() if state == "ok"]
        print(f"  verified: {len(verified)}/{len(states)} pinned repositories")
        print(f"  missing: {', '.join(upstream['missing']) or '-'} (streamreplace fetch-upstream all)")
        for name, state in states.items():
            if state not in ("ok", "missing"):
                print(f"  {state}: {name} (streamreplace fetch-upstream {name} --force)")
    print("\ncheckpoints:")
    for ref, entry in report["checkpoints"].items():
        note = "" if entry.get("revision_matches", True) else "  (other revision on disk)"
        print(f"  {'ok     ' if entry['exists'] else 'missing'} {ref}{note}")
    print("\nbenchmarks (prepared data / module):")
    for name, entry in report["data"].items():
        print(f"  {name:18s} {'prepared' if entry['prepared'] else 'missing '}  "
              f"{'module' if entry['module'] else 'no module'}")
    ready = [name for name, ok in report["adapters"].items() if ok]
    print(f"\nadapter modules present: {', '.join(ready) or '-'}")


def run(args) -> int:
    report, fatal, missing = collect(args)
    if args.json:
        print(json.dumps(report, indent=1))
    else:
        _print(report)
    return 1 if fatal or (args.strict and missing) else 0
