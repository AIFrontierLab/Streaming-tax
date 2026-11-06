"""Which samples the fine-tuning experiment uses, decided once per dataset by `streamreplace finetune prepare <dataset>`
and shared by every model and budget (ported unchanged; the paper's lists are in paper/frozen/finetune/):

  <root>/subset.json       test ids: all of them up to test_size; above it a stratified sample over the benchmark's
                           STRATUM_KEYS (largest-remainder allocation). It is the runner's subset file of the folder
  <root>/train_order.json  the training order: a seeded permutation of the training pool (round-robin over the values
                           of INTERLEAVE when the module sets it, e.g. LingoQA Action / Scenery 1:1), cut at the largest
                           budget; budget b = the first b ids, so the budgets are nested
  <root>/selection.json    counts, what was dropped and why, sha256 of both lists

The training pool excludes samples sharing the benchmark's GROUP (trajectory / segment / surgery case) with any test
sample, and for CHECK_NEAR_DUPLICATES benchmarks (LingoQA) samples with a frame whose 32x20 grayscale thumbnail
correlates >= NEAR with a test frame's (random training frames reach at most 0.93, frames of one test clip 0.93-0.98;
an 8x8 average hash flagged half of all driving frames as "duplicates").
Orders come from sha256(seed:id), so they need no RNG state and are identical on every machine. Once written, an entry
is never changed (re-running only verifies it).

The blank-frame control's questions (control_subset) are the same stratified rule applied to the test subset with the
control's size (500 per dataset in the paper; it reproduces the paper's control questions of all three datasets)."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
from pathlib import Path

from ...core.io import read_json, write_json
from . import Experiment

NEAR = 0.95
FILES = ("subset", "train_order", "selection")
CONTROLS_FILE = "controls.json"      # paper/frozen/finetune: the control folders' questions, {control: {dataset: ids}}


def _key(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def _digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def stratified(samples, size: int, keys: tuple, seed: int) -> list[str]:
    """`size` ids, each stratum's share by largest remainder, within a stratum in seeded hash order."""
    if len(samples) <= size:
        return sorted(s.id for s in samples)
    strata: dict[tuple, list] = {}
    for s in samples:
        strata.setdefault(tuple(s.metadata.get(k) for k in keys), []).append(s)
    quota = {k: size * len(v) / len(samples) for k, v in strata.items()}
    take = {k: int(q) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: (take[k] - quota[k], str(k)))[:size - sum(take.values())]:
        take[k] += 1
    chosen = [s.id for k, v in strata.items() for s in sorted(v, key=lambda s: _key(seed, s.id))[:take[k]]]
    return sorted(chosen)


def ordered(samples, interleave: str | None, seed: int) -> list:
    """Seeded permutation; with `interleave`, round-robin over that metadata field's values (each in hash order)."""
    if not interleave:
        return sorted(samples, key=lambda s: _key(seed, s.id))
    groups: dict[str, list] = {}
    for s in sorted(samples, key=lambda s: _key(seed, s.id)):
        groups.setdefault(s.metadata[interleave], []).append(s)
    names, out = sorted(groups), []
    for i in range(max(len(v) for v in groups.values())):
        out += [groups[n][i] for n in names if i < len(groups[n])]
    return out


def thumbnail(path: str):
    """Zero-mean, unit-norm 32x20 grayscale thumbnail (dot product = normalised correlation)."""
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        image.draft("L", (128, 80))                  # JPEG: decode at reduced scale
        x = np.asarray(image.convert("L").resize((32, 20), Image.BILINEAR), dtype=np.float32).ravel()
    x -= x.mean()
    return x / (np.linalg.norm(x) + 1e-6)


def _frame_paths(video: str) -> list[str]:
    from ...media.video_meta import read_frame_list
    return read_frame_list(video)[0]


def _thumbnails(videos: list[str], workers: int) -> dict:
    """video -> (frames x 640) array of its frames' thumbnails."""
    import numpy as np
    frames = {v: _frame_paths(v) for v in videos}
    flat = [f for fs in frames.values() for f in fs]
    with ProcessPoolExecutor(workers) as pool:
        values = dict(zip(flat, pool.map(thumbnail, flat, chunksize=64)))
    return {v: np.stack([values[f] for f in fs]) for v, fs in frames.items()}


def _bench(dataset: str):
    from ...benchmarks import registry
    return registry.get(dataset)


def _load(dataset: str, split: str = "test", subset=None) -> list:
    from ...data.samples import load_samples
    return load_samples(dataset, split=split, subset=subset)


def prepare(exp: Experiment, dataset: str, workers: int = 16, redo: bool = False) -> dict:
    """Writes (or re-verifies) the dataset's test subset and training order in the experiment root. redo=True (only
    before anything trained or evaluated on the old selection): archive this dataset's selection to
    <root>/_old/<time>/ and select again, e.g. after prepare dropped samples whose media turned out unavailable."""
    from datetime import datetime
    bench = _bench(dataset)
    seed, size, budget = exp.seed, exp.test_size, exp.top_budget
    root = exp.root
    subset_path, order_path, record_path = root / "subset.json", root / "train_order.json", root / "selection.json"
    subsets = read_json(subset_path) if subset_path.exists() else {}
    orders = read_json(order_path) if order_path.exists() else {}
    records = read_json(record_path) if record_path.exists() else {}
    if redo and dataset in records:
        used = [p for p in (root / dataset).glob("*.jsonl")] + list((root / "adapters" / dataset).glob("*/*/DONE.json"))
        if used:
            raise ValueError(f"{dataset}: outputs or adapters already use the old selection ({used[0]}); archive them first")
        old = {"subset": subsets.pop(dataset, None), "train_order": orders.pop(dataset, None), "selection": records.pop(dataset)}
        write_json(root / "_old" / datetime.now().strftime("%Y%m%d-%H%M%S") / f"selection_{dataset}.json", old)
    test = _load(dataset)                             # the whole prepared test split (no subset applied)
    train = _load(dataset, split="train")
    group = getattr(bench.module, "GROUP", None)
    held = {s.metadata[group] for s in test} if group else set()
    pool = [s for s in train if not group or s.metadata[group] not in held]
    dropped_group = len(train) - len(pool)
    candidates = ordered(pool, getattr(bench.module, "INTERLEAVE", None), seed)
    dropped_near: list[str] = []
    if getattr(bench.module, "CHECK_NEAR_DUPLICATES", False):
        import numpy as np
        reference = np.concatenate(list(_thumbnails(sorted({s.video for s in test}), workers).values()))
        kept, start = [], 0
        while len(kept) < budget and start < len(candidates):
            batch = candidates[start:start + 2 * (budget - len(kept))]
            thumbs = _thumbnails(sorted({s.video for s in batch}), workers)
            for s in batch:
                if float((thumbs[s.video] @ reference.T).max()) >= NEAR:
                    dropped_near.append(s.id)
                else:
                    kept.append(s)
            start += len(batch)
        candidates = kept
    order = [s.id for s in candidates[:budget]]
    chosen = stratified(test, size, tuple(getattr(bench.module, "STRATUM_KEYS", ())), seed)
    if len(order) < budget:
        raise ValueError(f"{dataset}: only {len(order)} training samples for budget {budget}")
    record = {"test": len(test), "test_subset": len(chosen), "train": len(train), "pool": len(pool),
              "dropped_group": dropped_group, "dropped_near_duplicate": len(dropped_near),
              "near_duplicate_ids": dropped_near[:200], "budget": budget,
              "test_sha256": _digest(chosen), "train_sha256": _digest(order), "seed": seed}
    for name, store, value in (("subset", subsets, chosen), ("train_order", orders, order)):
        if dataset in store and store[dataset] != value:
            raise ValueError(f"{dataset}: {name} differs from the recorded one; it is fixed once written (move "
                             f"{root}/{name}.json aside to redo the selection)")
        store[dataset] = value
    records[dataset] = record
    root.mkdir(parents=True, exist_ok=True)
    write_json(subset_path, subsets)
    write_json(order_path, orders)
    write_json(record_path, records)
    return {"dataset": dataset, **{k: v for k, v in record.items() if k != "near_duplicate_ids"}}


def selection(root: Path, dataset: str) -> tuple[list[str], dict]:
    """The training order and its record, as written by prepare (training units read nothing else)."""
    root = Path(root)
    order = read_json(root / "train_order.json")[dataset]
    record = read_json(root / "selection.json")[dataset]
    if _digest(order) != record["train_sha256"]:
        raise ValueError(f"{dataset}: train_order.json does not match selection.json")
    return order, record


def subset_ids(root: Path, dataset: str) -> list[str]:
    path = Path(root) / "subset.json"
    if not path.is_file():
        raise ValueError(f"{path} is missing; run: streamreplace finetune prepare {dataset}")
    subset = read_json(path)
    if dataset not in subset:
        raise ValueError(f"{dataset} is not prepared in {path}; run: streamreplace finetune prepare {dataset}")
    return subset[dataset]


def control_subset(exp: Experiment, dataset: str, size: int | None) -> list[str]:
    """A control folder's questions: the test subset, or `size` of them chosen by stratified() with the experiment's
    seed (the paper's blank-frame controls: 500 per dataset)."""
    ids = subset_ids(exp.root, dataset)
    if size is None:
        return list(ids)
    keep = set(ids)
    samples = [s for s in _load(dataset) if s.id in keep]
    return stratified(samples, int(size), tuple(getattr(_bench(dataset).module, "STRATUM_KEYS", ())), exp.seed)


def write_subset(folder: Path, chosen: dict[str, list[str]]) -> Path:
    """Adds benchmarks to a folder's subset.json; an existing entry must be identical (outputs were generated from it)."""
    path = Path(folder) / "subset.json"
    subsets = read_json(path) if path.exists() else {}
    for dataset, ids in chosen.items():
        if dataset in subsets and subsets[dataset] != ids:
            raise ValueError(f"{path}: {dataset} differs from the recorded ids; its outputs were generated from them")
        subsets[dataset] = ids
    write_json(path, subsets)
    return path


def frozen_dir() -> Path:
    from ...paths import project_root
    return project_root() / "paper" / "frozen" / "finetune"


def compare_frozen(root: Path, datasets: list[str], frozen: Path | None = None) -> list[str]:
    """Differences between the experiment root's selection and the paper's (paper/frozen/finetune/)."""
    frozen = Path(frozen) if frozen is not None else frozen_dir()
    problems = []
    for name in FILES:
        mine, paper = Path(root) / f"{name}.json", frozen / f"{name}.json"
        if not mine.is_file() or not paper.is_file():
            continue
        a, b = read_json(mine), read_json(paper)
        for dataset in datasets:
            if dataset in a and dataset in b and a[dataset] != b[dataset]:
                problems.append(f"{name}.json: {dataset} differs from the paper's (paper/frozen/finetune/{name}.json)")
    return problems


def compare_controls(exp: Experiment, frozen: Path | None = None) -> list[str]:
    """Differences between the control folders' subset.json files (those already written) and the paper's control
    questions (paper/frozen/finetune/controls.json: {control: {dataset: ids}})."""
    path = (Path(frozen) if frozen is not None else frozen_dir()) / CONTROLS_FILE
    if not path.is_file():
        return []
    problems = []
    for name, subsets in read_json(path).items():
        mine = exp.folder(name) / "subset.json" if name in exp.controls else None
        if mine is None or not mine.is_file():
            continue
        current = read_json(mine)
        problems += [f"{name}/subset.json: {dataset} differs from the paper's control questions "
                     f"(paper/frozen/finetune/{CONTROLS_FILE})"
                     for dataset, ids in subsets.items() if dataset in current and current[dataset] != ids]
    return problems


def use_frozen(root: Path, datasets: list[str], frozen: Path | None = None) -> list[str]:
    """Copies the paper's selection of `datasets` into the experiment root (write-once rules as in prepare)."""
    frozen = Path(frozen) if frozen is not None else frozen_dir()
    root = Path(root)
    written = []
    for name in FILES:
        paper = read_json(frozen / f"{name}.json")
        path = root / f"{name}.json"
        store = read_json(path) if path.exists() else {}
        for dataset in datasets:
            if dataset not in paper:
                raise ValueError(f"paper/frozen/finetune/{name}.json has no {dataset}")
            if dataset in store and store[dataset] != paper[dataset]:
                raise ValueError(f"{path}: {dataset} differs from the paper's selection; move the file aside first")
            store[dataset] = paper[dataset]
        root.mkdir(parents=True, exist_ok=True)
        write_json(path, store)
        written.append(str(path))
    return written
