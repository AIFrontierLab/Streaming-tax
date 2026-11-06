"""Raw benchmark data -> the prepared samples every run reads (`streamreplace prepare`).

prepare_dataset() calls the benchmark module's prepare(raw, media) with raw = `<dataset dir>/raw` and media =
`<dataset dir>/media` (some modules store paths relative to that media folder, e.g. Video-MME's subtitles) and writes

    <dataset dir>/prepared/samples.jsonl   the test split, one Sample per line, in prepared order
    <dataset dir>/prepared/train.jsonl     the training split, for benchmarks with prepare_train (fine-tuning sets)
    <dataset dir>/prepared/manifest.json   counts, answer kinds, sha256 of each split file, preparation time (UTC)
    <dataset dir>/media/                   media the module extracts or generates

where the dataset directory is `<data>/<benchmark>` (paths.py). Rows are Sample.to_dict() written with
core.io.write_jsonl. Media paths are stored relative to the dataset directory (os.path.abspath, not resolve: raw/ and
media/ may be symlinks out of the tree, and some videos are read in place under raw/), so the tree can move. An
existing split is never rebuilt unless force=True. The test split must have the module's EXPECTED_COUNT samples
(the size the paper used); otherwise prepare stops before writing samples.jsonl.

Checks. Run identities do not hash the data, so these checks are what tells a reproduction of the paper apart from an
evaluation on other data:
- check_prepared(): the sample count against the module's EXPECTED_COUNT and manifests/prepared.yaml, and the sha256
  of samples.jsonl and the prompt fingerprint (data/prompts.prompt_map_sha256 of every rendered prompt) against
  manifests/prepared.yaml. A different samples file usually means another revision of the raw data. The manifest
  records every benchmark of the paper; a benchmark without a row (one added later) gets a note, not a problem.
- check_videos(): the benchmark's videos against manifests/video_meta/<benchmark>.jsonl (media/video_meta.py): the
  same set of videos and, for all or a sampled fraction of them, the same decodable frame count, fps and header
  frame count when measured with the local libraries. Measuring also fills the decodable-frame-count cache the runs
  read, so a full check before a run saves decoding every video inside the model workers.

manifests/prepared.yaml holds {version: 1, prepared: {<benchmark>: {count, samples_sha256, prompts_sha256}}} (each
field optional): samples_sha256 is the sha256 of samples.jsonl exactly as prepare writes it, prompts_sha256 the
prompt fingerprint above."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import os
from pathlib import Path
import random
from typing import TYPE_CHECKING

from ..core.io import file_digest, read_json, read_jsonl, write_json, write_jsonl
from ..core.schema import Sample, utc_now
from .prompts import prompt_map_sha256, render_prompts
from .samples import FILES, dataset_dir, load_samples, samples_file

if TYPE_CHECKING:
    from ..paths import Paths

EXPECTED_KEYS = ("count", "samples_sha256", "prompts_sha256")


def _bench(name: str, bench=None):
    if bench is not None:
        return bench
    from ..benchmarks import registry
    return registry.get(name)


def _rows(name: str, samples: list[Sample], base: Path) -> list[dict]:
    ids = [s.id for s in samples]
    if not samples or len(ids) != len(set(ids)):
        raise ValueError(f"{name}: prepared samples must be nonempty with unique ids")

    def relative(path: str) -> str:
        path = Path(os.path.abspath(path))       # abspath, not resolve: media/raw may be symlinks out of the tree
        if not path.is_file():
            raise ValueError(f"{name}: missing media {path}")
        return str(path.relative_to(os.path.abspath(base)))

    return [replace(s, images=tuple(relative(p) for p in s.images),
                    video=relative(s.video) if s.video else None).to_dict() for s in samples]


def prepare_dataset(name: str, force: bool = False, paths: Paths | None = None, bench=None) -> dict:
    """Prepares the missing splits of benchmark `name` (all of them with force=True); returns the manifest plus
    "status": "prepared" or "exists"."""
    base = dataset_dir(name, paths)
    raw, out = base / "raw", base / "prepared"
    bench = _bench(name, bench)
    splits = ["test"] + (["train"] if bench.has_train else [])
    if all((out / FILES[s]).exists() for s in splits) and not force:
        manifest = read_json(out / "manifest.json") if (out / "manifest.json").exists() else {}
        return {"dataset": name, "status": "exists", **manifest}
    if out.exists() and not (out / FILES["test"]).exists() and any(out.iterdir()):
        raise ValueError(f"{out} holds data in an unknown layout; move it away first")
    manifest = read_json(out / "manifest.json") if (out / "manifest.json").exists() and not force else {}
    for split in splits:
        if (out / FILES[split]).exists() and not force:
            continue
        try:
            media = base / "media"
            samples = bench.prepare(raw, media) if split == "test" else bench.prepare_train(raw, media)
        except FileNotFoundError as exc:
            raise ValueError(f"{name}: raw data not found ({exc}); download it into {raw} first "
                             f"(streamreplace download datasets {name})") from exc
        want = getattr(bench, "EXPECTED_COUNT", None)
        if split == "test" and want is not None and len(samples) != int(want):
            raise ValueError(f"{name}: prepare() returned {len(samples)} samples, the benchmark expects {want} "
                             "(EXPECTED_COUNT): the raw data is not the release the paper used; samples.jsonl was "
                             "not written")
        write_jsonl(out / FILES[split], _rows(name, samples, base))
        digest = file_digest(out / FILES[split])
        if split == "test":
            manifest.update(samples=len(samples), kinds=dict(Counter(s.kind for s in samples)), samples_sha256=digest)
        else:
            manifest.update(train_samples=len(samples), train_sha256=digest)
        manifest.update(dataset=name, media=bench.media, prepared_at=utc_now())
    write_json(out / "manifest.json", manifest)
    return {"status": "prepared", **manifest}


def expected_path() -> Path:
    from ..paths import project_root
    return project_root() / "manifests" / "prepared.yaml"


def load_expected(path: str | Path | None = None) -> dict[str, dict]:
    """manifests/prepared.yaml ({version: 1, prepared: {benchmark: {count, samples_sha256, prompts_sha256}}}) as
    {benchmark: entry}; {} when the file is absent."""
    import yaml
    path = Path(path) if path is not None else expected_path()
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or set(data) - {"version", "prepared"} or not isinstance(data.get("prepared"), dict):
        raise ValueError(f"{path}: expected {{version: 1, prepared: {{<benchmark>: {{{', '.join(EXPECTED_KEYS)}}}}}}}")
    out = {}
    for name, entry in data["prepared"].items():
        entry = dict(entry or {})
        unknown = sorted(set(entry) - set(EXPECTED_KEYS))
        if unknown:
            raise ValueError(f"{path}: {name}: unknown fields {unknown} (allowed: {', '.join(EXPECTED_KEYS)})")
        out[str(name)] = entry
    return out


def check_prepared(name: str, paths: Paths | None = None, bench=None, expected: dict | None = None,
                   prompts: bool = True) -> dict:
    """Count, samples sha256 and prompt fingerprint of a prepared benchmark, compared with the expected values.
    Returns {"count", "samples_sha256", "prompts_sha256" (with prompts=True), "expected", "problems": [...],
    "notes": [...]}; notes say what could not be compared (no manifests/prepared.yaml row)."""
    path = samples_file(name, paths=paths)
    if not path.is_file():
        raise ValueError(f"{name} is not prepared: {path} is missing; run: streamreplace prepare {name}")
    bench = _bench(name, bench)
    rows = read_jsonl(path)
    result: dict = {"count": len(rows), "samples_sha256": file_digest(path)}
    problems = []
    want = getattr(bench, "EXPECTED_COUNT", None)
    if want is not None and len(rows) != int(want):
        problems.append(f"{len(rows)} samples; the benchmark module expects {want}")
    entry = (load_expected() if expected is None else expected).get(name) or {}
    if "count" in entry and int(entry["count"]) != len(rows):
        problems.append(f"{len(rows)} samples; manifests/prepared.yaml expects {entry['count']}")
    if "samples_sha256" in entry and entry["samples_sha256"] != result["samples_sha256"]:
        problems.append(f"samples.jsonl sha256 {result['samples_sha256'][:12]}... differs from the expected "
                        f"{str(entry['samples_sha256'])[:12]}... (another revision of the raw data?)")
    if prompts:
        try:
            rendered = render_prompts(name, load_samples(name, paths=paths, bench=bench), bench=bench)
        except Exception as exc:                  # noqa: BLE001 - e.g. a prompt that reads not yet fetched upstream code
            problems.append(f"prompts not checked: {type(exc).__name__}: {exc}")
        else:
            result["prompts_sha256"] = prompt_map_sha256(rendered)
            if "prompts_sha256" in entry and entry["prompts_sha256"] != result["prompts_sha256"]:
                problems.append(f"rendered prompts sha256 {result['prompts_sha256'][:12]}... differ from the "
                                f"expected {str(entry['prompts_sha256'])[:12]}...")
    result["expected"] = entry
    result["problems"] = problems
    result["notes"] = [] if entry else [
        "no row in manifests/prepared.yaml: count, samples and prompts were not compared with recorded values"]
    return result


def check_videos(name: str, paths: Paths | None = None, fraction: float = 0.0, workers: int = 1,
                 seed: int = 0) -> dict:
    """The prepared benchmark's videos against manifests/video_meta/<name>.jsonl; with fraction > 0, that fraction
    of the videos (seeded random choice; 1 = all) is measured with the local libraries and compared."""
    from ..media import video_meta
    manifest = video_meta.load_manifest(name)
    if not manifest:
        return {"manifest": None, "problems": []}
    root = dataset_dir(name, paths)
    local = video_meta.dataset_videos(root)
    missing, extra = sorted(set(manifest) - set(local)), sorted(set(local) - set(manifest))
    result: dict = {"manifest": len(manifest), "videos": len(local)}
    problems = []
    if missing:
        problems.append(f"{len(missing)} videos of the manifest are not in the prepared samples (e.g. {missing[0]})")
    if extra:
        problems.append(f"{len(extra)} prepared videos are not in the manifest (e.g. {extra[0]})")
    common = sorted(set(manifest) & set(local))
    if fraction > 0 and common:
        chosen = common if fraction >= 1 else sorted(
            random.Random(seed).sample(common, min(len(common), max(1, round(len(common) * fraction)))))
        versions = video_meta.library_versions()
        rows = video_meta.measure_videos(root, chosen, workers=workers,
                                         reader="decord" if versions.get("decord") else None,
                                         torchcodec=versions.get("torchcodec") is not None)
        diffs = {row["video"]: video_meta.compare(manifest[row["video"]], row) for row in rows}
        diffs = {video: d for video, d in diffs.items() if d}
        result.update(measured=len(rows), differ=len(diffs),
                      readers={k: v for k, v in versions.items()})
        problems += [f"{video}: {'; '.join(d)}" for video, d in list(diffs.items())[:10]]
        if len(diffs) > 10:
            problems.append(f"... and {len(diffs) - 10} more videos differ")
    result["problems"] = problems
    return result
