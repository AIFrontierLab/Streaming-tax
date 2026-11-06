"""Generation-cap reruns: the offline rows that stopped at the generation cap, generated again at a larger cap.

The formal offline runs cap generation at the benchmark's max_new_tokens (2,048 for MathVista and MMMU, 512 for
MMStar). A reply cut there may lose its final answer, and when a parent and its child are cut on different items the
C-P contrast is biased. For the pairs and benchmarks of configs/analysis.yaml (overlays.truncation_rerun) the paper
therefore generated those items again at 8,192 tokens, for every member of the pair, and the contrasts read the rerun
rows on them (analysis/arms.py; `streamreplace stats`). The absolute-score tables keep the formal rows.

  select   per pair and benchmark, the ids on which ANY member's formal offline row hit the cap:
           <root>/<pair>/subset.json (the rerun folder's items) and <root>/capped.json (per member: its cap and its
           own capped ids; the contrast of a parent and a child reruns the union of THEIR two lists). A row hit the cap
           if its finish is "length". A row with no recorded stop reason (finish missing or "unknown") is counted
           with the model's own tokenizer, in the model's environment on CPU, and counts as capped at >= cap - 8
           tokens (decode -> encode is not exact).
  plan     the rerun jobs: every member of the pair offline (conditions P and C) on the selected ids, settings
           unchanged except generation.max_new_tokens, in the folder <root>/<pair>/ (an ordinary outputs folder:
           score it with `streamreplace --outputs <root>/<pair> score`)
  report   <root>/TRUNC.md and trunc.json: per pair and benchmark the formal and the merged headline metric of parent
           and child, delta = child - parent with a paired cluster-bootstrap 95% interval (the integer-index engine of
           analysis/stats_index.py), TOST at +-3 points and McNemar; per member how many rows were capped, how many
           still are at the new cap, how many verdicts the rerun changed, and how many rerun replies reproduce the
           formal text where that member had not been capped (greedy decoding: rows that ended before the old cap
           should end identically under a larger one).

`python -m streamreplace.run.rerun_capped SOURCE TARGET` is the token counter run inside a model environment."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

MARGIN = 8            # decode -> encode is not exact: re-tokenised replies within 8 tokens of the cap count as capped
TOST_PP = 3.0         # equivalence margin (points) of "the child keeps the parent's accuracy"
CLUSTER = {"dynamath": "seed"}    # bootstrap clusters of the report: DynaMath's 10 variants of a seed; else the item
# Adapters whose tokenizer ships with the upstream model code rather than with the checkpoint.
TOKENIZER_UPSTREAM = {"internvl_offline": "livestar/inference"}
CAPPED_FILE = "capped.json"


# ---------- selection ----------

def unfinished(row: dict) -> bool:
    """A reply without a recorded stop reason (no finish, or "unknown"): its length has to be measured."""
    return not row.get("error") and row.get("finish") not in {"eos", "length"}


def capped(rows: dict[str, dict], cap: int, counts: dict[str, int], margin: int = MARGIN) -> list[str]:
    """Sorted ids whose reply stopped at the cap (errored rows never count)."""
    ids = []
    for i, row in rows.items():
        if row.get("error"):
            continue
        if row.get("finish") == "length":
            ids.append(i)
        elif unfinished(row):
            if i not in counts:
                raise ValueError(f"{i}: no finish recorded and no token count")
            if counts[i] >= cap - margin:
                ids.append(i)
    return sorted(ids)


def formal_cap(config, outputs: Path, dataset: str, model: str) -> int:
    """max_new_tokens of the formal offline job: from its meta.json, else from the configuration."""
    from ..core.identity import read_meta
    meta = read_meta(Path(outputs) / dataset / f"{model}.offline.meta.json") or {}
    settings = (meta.get("payload") or {}).get("settings") or meta.get("settings")
    if not settings:
        settings = config.settings(dataset, model, "offline")
    return int(settings["generation"]["max_new_tokens"])


def tokenizer_dir(config, paths, model: str) -> Path:
    spec = config.model(model)
    relative = TOKENIZER_UPSTREAM.get(spec.offline.adapter if spec.offline else "")
    return Path(paths.upstream) / relative if relative else Path(paths.checkpoint_dir(spec.checkpoint))


def count_tokens(config, paths, model: str, rows: dict[str, dict]) -> dict[str, int]:
    """Token counts of the replies without a recorded finish, with the model's tokenizer in its own environment
    (CPU only; no weights are loaded)."""
    from .envs import worker_env
    texts = {i: r.get("raw") or "" for i, r in rows.items() if unfinished(r)}
    if not texts:
        return {}
    spec = config.model(model)
    with tempfile.TemporaryDirectory() as tmp:
        source, target = Path(tmp) / "texts.json", Path(tmp) / "counts.json"
        source.write_text(json.dumps({"tokenizer": str(tokenizer_dir(config, paths, model)), "texts": texts}),
                          encoding="utf-8")
        env = worker_env(paths, spec.offline.env, gpu="", threads=1)
        subprocess.run([str(paths.env_python(spec.offline.env)), "-m", "streamreplace.run.rerun_capped",
                        str(source), str(target)], env=env, check=True)
        return {str(k): int(v) for k, v in json.loads(target.read_text(encoding="utf-8")).items()}


def tokens_main(source: str, target: str) -> int:
    """Runs in the model's environment: number of tokens of each reply (no special tokens)."""
    from transformers import AutoTokenizer
    job = json.loads(Path(source).read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(job["tokenizer"], trust_remote_code=True, local_files_only=True)
    counts = {i: len(tokenizer(t, add_special_tokens=False).input_ids) for i, t in job["texts"].items()}
    Path(target).write_text(json.dumps(counts), encoding="utf-8")
    return 0


def read_capped(path: Path) -> dict:
    """A capped.json ({"cap_margin", "rerun_cap", "pairs": {pair: {dataset: {model: {"cap", "rows_without_finish",
    "ids"}}}}}); an empty selection when the file does not exist."""
    path = Path(path)
    if not path.is_file():
        return {"cap_margin": MARGIN, "rerun_cap": None, "pairs": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def select(config, paths, outputs: str | Path, root: str | Path, pairs: list[str], datasets: list[str],
           rerun_cap: int, margin: int = MARGIN, token_counts: dict | None = None) -> dict:
    """Writes <root>/<pair>/subset.json and <root>/capped.json from the formal offline rows in `outputs`; returns
    a summary {"<pair>/<dataset>": {"rerun ids": n, <model>: n}}. Benchmarks already selected for a pair must come out
    the same (the ids of a folder never change under a running job); new ones are added. token_counts: optional
    {model: {id: tokens}} for rows without a recorded finish (otherwise counted in the model's environment)."""
    from ..core.io import latest, read_jsonl, write_json
    from ..data.samples import SUBSET_FILE, read_subset
    outputs, root = Path(outputs), Path(root)
    record = read_capped(root / CAPPED_FILE)
    record.update(cap_margin=margin, rerun_cap=rerun_cap)
    summary = {}
    for pair in pairs:
        members = list(config.pair(pair).members)
        folder = root / pair
        subset = read_subset(folder) or {}
        per_pair = record["pairs"].setdefault(pair, {})
        for dataset in datasets:
            per_model = {}
            for model in members:
                path = outputs / dataset / f"{model}.offline.jsonl"
                if not path.is_file():
                    raise ValueError(f"{path} not found: the formal offline job of {model} on {dataset}")
                rows = latest(read_jsonl(path))
                cap = formal_cap(config, outputs, dataset, model)
                if token_counts and model in token_counts:
                    counts = {i: int(n) for i, n in token_counts[model].items()}
                else:
                    counts = count_tokens(config, paths, model, rows)
                per_model[model] = {"cap": cap, "rows_without_finish": sum(unfinished(r) for r in rows.values()),
                                    "ids": capped(rows, cap, counts, margin)}
            ids = sorted(set().union(*(m["ids"] for m in per_model.values())))
            if dataset in subset and subset[dataset] != ids:
                raise ValueError(f"{folder / SUBSET_FILE}: {dataset} ids differ from this selection; use a new --root")
            subset[dataset] = ids
            per_pair[dataset] = per_model
            summary[f"{pair}/{dataset}"] = {"rerun ids": len(ids), **{m: len(v["ids"]) for m, v in per_model.items()}}
        write_json(folder / SUBSET_FILE, subset)
    write_json(root / CAPPED_FILE, record)
    return summary


# ---------- reruns ----------

def plan(config, paths, root: str | Path, pairs: list[str], datasets: list[str], rerun_cap: int) -> list:
    """The rerun jobs (run.plan.Job) of every member of each pair, in <root>/<pair>/, on the selected ids."""
    from ..config.settings import parse_sets
    from ..data.samples import read_subset
    from .plan import plan as plan_jobs
    jobs = []
    for pair in pairs:
        folder = Path(root) / pair
        subset = read_subset(folder)
        if subset is None:
            raise ValueError(f"{folder}: no subset.json; run `streamreplace rerun-capped select` first")
        chosen = [d for d in datasets if subset.get(d)]          # benchmarks with no capped id need no rerun
        if chosen:
            jobs += plan_jobs(config, paths, benchmarks=chosen, pairs=[pair], protocols=["P", "C"],
                              sets=parse_sets([f"generation.max_new_tokens={int(rerun_cap)}"]), outputs=folder)
    return jobs


# ---------- report ----------

def metric_row(policy, sample, row: dict) -> dict:
    """The fields benchmark metrics read. `answered` is the rule prediction only, as the reported table had it."""
    from ..scoring.outcome import outcome
    return {"id": sample.id, "correct": outcome(policy, row), "kind": sample.kind, "metadata": sample.metadata,
            "pred": row.get("pred"), "judge_pred": row.get("judge_pred"), "parse": row.get("parse"),
            "silent": row.get("parse") == "no_answer", "answered": row.get("pred") is not None, "video": sample.video}


def scored(row: dict | None) -> bool:
    return bool(row) and (bool(row.get("error")) or row.get("correct") is not None)


def paired(values: list[int], clusters: list) -> dict:
    """delta (points) of per-question child - parent values with 95% / 90% cluster-bootstrap intervals."""
    from ..analysis.stats_index import bootstrap_ci, mcnemar_p
    strata = [0] * len(values)
    low, high = bootstrap_ci(values, clusters, strata)
    low90, high90 = bootstrap_ci(values, clusters, strata, level=0.90)
    return {"delta": 100 * sum(values) / len(values), "ci": [100 * low, 100 * high],
            "tost": -TOST_PP < 100 * low90 and 100 * high90 < TOST_PP,
            "mcnemar_p": mcnemar_p(values.count(-1), values.count(1))}


def compare(dataset: str, bench, policy, samples: dict, members: tuple[str, str], formal: dict, rerun: dict,
            ids: list[str]) -> dict:
    """One pair on one benchmark: formal vs merged (rerun rows on `ids` for both members). Missing rows count wrong."""
    order = sorted(samples)
    key = CLUSTER.get(dataset)
    clusters = [samples[i].metadata[key] if key else i for i in order]
    result = {"rerun_ids": len(ids)}
    for name, source in (("formal", lambda m, i: formal[m].get(i)),
                         ("merged", lambda m, i: (rerun[m] if i in ids else formal[m]).get(i))):
        rows = {m: [metric_row(policy, samples[i], source(m, i) or {"error": "missing"}) for i in order]
                for m in members}
        metrics = {m: bench.metrics(rows[m]) for m in members}
        values = [int(c["correct"]) - int(p["correct"]) for p, c in zip(rows[members[0]], rows[members[1]])]
        result[name] = {"metrics": metrics, **paired(values, clusters)}
    return result


def member_stats(policy, model: str, info: dict, formal: dict, rerun: dict, ids: list[str]) -> dict:
    from ..scoring.outcome import outcome
    mine = set(info["ids"])
    others = [i for i in ids if i not in mine and i in rerun and i in formal]
    flips = [(outcome(policy, formal[i]), outcome(policy, rerun[i])) for i in mine if i in rerun]
    return {"model": model, "cap": info["cap"], "capped": len(mine),
            "still_capped": sum(rerun[i].get("finish") == "length" for i in mine if i in rerun),
            "to_right": sum(not f and r for f, r in flips), "to_wrong": sum(f and not r for f, r in flips),
            "same_text": [sum(rerun[i].get("raw") == formal[i].get("raw") for i in others), len(others)]}


def report(config, outputs: str | Path, root: str | Path, catalog=None, capped_file: str | Path | None = None) -> Path:
    """Writes <root>/TRUNC.md and <root>/trunc.json (see the module docstring); returns the TRUNC.md path."""
    from ..analysis.records import Catalog, joined_rows
    from ..core.identity import read_meta
    from ..core.io import atomic_text, write_json
    outputs, root = Path(outputs), Path(root)
    catalog = catalog or Catalog()
    record = read_capped(Path(capped_file) if capped_file else root / CAPPED_FILE)
    margin = record.get("cap_margin", MARGIN)
    lines = [f"# Generation-cap sensitivity (`{root.name}`)", "",
             "Rows that hit the formal generation cap (finish = length, or a re-tokenised reply of >= cap − "
             f"{margin} tokens where no finish was recorded) are generated again at the rerun cap for every member of "
             "the pair (docs/statistics.md, generation-cap reruns). **Merged** = the rerun rows of both members on "
             "every id either of them capped, formal rows elsewhere; scoring = rule parse + the judge fallback, as the "
             "formal scorer. Δ = child − parent in pp (+ = child better, − = child worse), paired cluster bootstrap "
             f"95% CI; TOST ✓ = the 90% CI lies inside ±{TOST_PP:g} pp; McNemar exact. Errored or missing rows count "
             "wrong.", ""]
    data = {}
    for pair, datasets in sorted(record["pairs"].items()):
        folder = root / pair
        members = list(config.pair(pair).members)
        lines += [f"## {pair}: parent `{members[0]}`", ""]
        for dataset, per_model in datasets.items():
            bench, policy = catalog.bench(dataset), catalog.policy(dataset)
            samples = catalog.samples(dataset)
            formal = {m: joined_rows(outputs, dataset, f"{m}.offline") or {} for m in members}
            rerun = {m: joined_rows(folder, dataset, f"{m}.offline") or {} for m in members}
            meta = read_meta(folder / dataset / f"{members[0]}.offline.meta.json") or {}
            settings = (meta.get("payload") or {}).get("settings") or meta.get("settings")
            new_cap = settings["generation"]["max_new_tokens"] if settings else (record.get("rerun_cap") or "?")
            union = sorted(set().union(*(set(v["ids"]) for v in per_model.values())))
            missing = {m: sum(not scored(rerun[m].get(i)) for i in union) for m in members}
            lines += [f"### {dataset} (cap {per_model[members[0]]['cap']} → {new_cap}; {len(union)} rerun ids)", ""]
            if any(missing.values()):
                lines += ["pending: " + ", ".join(f"{m} {len(union) - n}/{len(union)}" for m, n in missing.items()), ""]
                data[f"{pair}/{dataset}"] = {"pending": missing}
                continue
            entry = {"cap": per_model[members[0]]["cap"], "rerun_cap": new_cap, "pairs": {}, "members": []}
            lines += ["| child | ids | parent formal → merged | child formal → merged | Δ formal | Δ merged [95% CI] "
                      "| TOST ±3 | McNemar p |", "|---|---|---|---|---|---|---|---|"]
            for child in members[1:]:
                ids = sorted(set(per_model[members[0]]["ids"]) | set(per_model[child]["ids"]))
                result = compare(dataset, bench, policy, samples, (members[0], child), formal, rerun, ids)
                entry["pairs"][child] = result
                f, m = result["formal"], result["merged"]
                head = lambda block, model: next(iter(block["metrics"][model].values()))  # noqa: E731
                p_value = "–" if m["mcnemar_p"] is None else f"{m['mcnemar_p']:.3g}"
                lines.append(f"| {child} | {len(ids)} | {head(f, members[0]):.1f} → {head(m, members[0]):.1f} | "
                             f"{head(f, child):.1f} → {head(m, child):.1f} | {f['delta']:+.1f} | {m['delta']:+.1f} "
                             f"[{m['ci'][0]:+.1f}, {m['ci'][1]:+.1f}] | {'✓' if m['tost'] else '✗'} | {p_value} |")
            lines += ["", "| model | capped | still capped | verdict → right / → wrong | same text where not capped |",
                      "|---|---|---|---|---|"]
            for model in members:
                stats = member_stats(policy, model, per_model[model], formal[model], rerun[model], union)
                entry["members"].append(stats)
                lines.append(f"| {model} | {stats['capped']} | {stats['still_capped']} | {stats['to_right']} / "
                             f"{stats['to_wrong']} | {stats['same_text'][0]}/{stats['same_text'][1]} |")
            lines.append("")
            data[f"{pair}/{dataset}"] = entry
    target = root / "TRUNC.md"
    atomic_text(target, "\n".join(lines) + "\n")
    write_json(root / "trunc.json", data)
    return target


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: python -m streamreplace.run.rerun_capped SOURCE.json TARGET.json  (token counter; runs in a "
              "model environment)", file=sys.stderr)
        return 2
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return tokens_main(*args)


if __name__ == "__main__":
    raise SystemExit(main())
