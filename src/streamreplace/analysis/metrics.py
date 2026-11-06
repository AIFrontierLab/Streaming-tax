"""Per-job metrics and completion counts: metrics.json, the source of every absolute score in the paper.

For each job (dataset x model x mode) the benchmark's own metrics() is evaluated on the job's counted rows (the
same rules as records.py). Benchmarks return percentages rounded to 2 decimals (MME per task, its sums to 1); the
paper's absolute-score tables print these values with 1 decimal. A job is "judged" when any row carries a judge
decision (or the benchmark is always judged); its metrics then use the judge's decision where there is one, and the
rule-only metrics are kept beside them.

Counts per job: expected, done (scored, not errored), errors, no_answer (silent stream rows), forced (MMDuet's forced
stream answers), parse_fail (no rule prediction and not silent), truncated (finish "length"), unscored, latency."""
from __future__ import annotations

from pathlib import Path
from statistics import mean

from .records import Catalog, expected_count, final, joined_rows, outcome, parse_job, scoped_samples

SKIP_FOLDERS = {"logs", "_old", "_archive", "judges", "analysis", "tables"}


def job_files(outputs: Path):
    """(dataset, job) of every `<dataset>/<model>[@<variant>].<offline|stream>.jsonl` under `outputs`."""
    for path in sorted(Path(outputs).glob("*/*.*.jsonl")):
        if path.parent.name in SKIP_FOLDERS:
            continue
        stem = path.name[:-len(".jsonl")]
        if stem.rsplit(".", 1)[1] in {"offline", "stream"}:
            yield path.parent.name, stem


def collect_job(outputs: Path, dataset: str, job: str, catalog: Catalog) -> dict | None:
    """Metrics and counts of one job, with its counted rows (id -> joined row) and their benchmark strata."""
    joined = joined_rows(outputs, dataset, job)
    if joined is None:
        return None
    bench = catalog.bench(dataset)
    samples = scoped_samples(catalog, outputs, dataset)       # the folder's subset.json, if it has one
    model, variant, mode = parse_job(job)
    policy = catalog.policy(dataset)
    rows = [r for r in joined.values() if r["id"] in samples]
    scored = [r for r in rows if not r.get("error") and r.get("correct") is not None]
    failed = [r for r in rows if r.get("error")]
    counted = scored + failed        # an errored sample counts (wrong; safe under the safety judges)
    judged_required = policy == "always"                       # only meaningful once judged (VideoReasonBench)

    def metric_rows(pick):
        return [{"id": r["id"], "correct": outcome(policy, r, pick), "kind": samples[r["id"]].kind,
                 "answered": r.get("pred") is not None or r.get("judge_pred") is not None, "pred": r.get("pred"),
                 "judge_pred": r.get("judge_pred") if pick is final else None,
                 "parse": r.get("parse"),
                 "silent": r.get("parse") == "no_answer",
                 "metadata": samples[r["id"]].metadata, "video": samples[r["id"]].video} for r in counted]

    rule = bench.metrics(metric_rows(lambda r: r["correct"])) if counted else {}
    judged = any(r.get("judge_correct") is not None for r in scored) or judged_required
    return {
        "dataset": dataset, "job": job, "model": f"{model}@{variant}" if variant else model, "mode": mode,
        "metrics": bench.metrics(metric_rows(final)) if judged else rule,
        "rule_metrics": rule, "judged": judged,
        "expected": expected_count(samples, mode), "done": len(scored), "errors": len(failed),
        "parse_fail": sum(r.get("pred") is None and r.get("parse") != "no_answer" for r in scored),
        "no_answer": sum(r.get("parse") == "no_answer" for r in scored),
        "forced": sum(bool((r.get("info") or {}).get("forced")) for r in scored),
        "truncated": sum(r.get("finish") == "length" for r in scored),
        "latency": round(mean(r["latency"] for r in scored), 2) if scored else None,
        "unscored": len(rows) - len(scored) - len(failed),
        "rows": {r["id"]: r for r in counted},
        "strata": {r["id"]: catalog.stratum(dataset, samples[r["id"]]) for r in counted},
        "metric_rows": metric_rows(final if judged else (lambda r: r["correct"])) if counted else None,
    }


def collect(outputs: Path, catalog: Catalog, datasets: list[str] | None = None) -> dict:
    """(dataset, model, mode) -> collect_job result for every job under `outputs` (optionally only `datasets`)."""
    results = {}
    for dataset, job in job_files(outputs):
        if datasets is not None and dataset not in datasets:
            continue
        try:
            catalog.bench(dataset)
        except (KeyError, ValueError, ImportError):
            continue                                   # a folder that is not a benchmark
        result = collect_job(Path(outputs), dataset, job, catalog)
        if result is not None:
            results[(dataset, result["model"], result["mode"])] = result
    return results


PUBLIC = ("dataset", "model", "mode", "job", "metrics", "rule_metrics", "judged", "expected", "done", "errors",
          "no_answer", "forced", "parse_fail", "truncated", "unscored", "latency")


def to_json(results: dict) -> list[dict]:
    """The metrics.json rows (no per-item data), sorted by (dataset, model, mode)."""
    return [{k: r[k] for k in PUBLIC} for _, r in sorted(results.items(), key=lambda kv: kv[0])]


def value(metrics_rows: list[dict], dataset: str, model: str, mode: str, name: str):
    """One metric value from metrics.json rows (None when absent)."""
    for r in metrics_rows:
        if (r["dataset"], r["model"], r["mode"]) == (dataset, model, mode):
            return r["metrics"].get(name)
    return None
