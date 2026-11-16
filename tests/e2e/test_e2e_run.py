"""End-to-end runs with the mock models: outputs, resume, identity refusal and --fresh, prompt drift, shards and
concurrency, error rows in the counts, experiment folders (subset, prompt overrides, --limit), load-time grouping,
the plugin variant arms and fine-tuned model ids."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
import socket
import subprocess
import sys

from e2e_project import JOBS, MODELS, dump, make_project, rows_of, run_all, score_rule_layer
import pytest

from streamreplace.core.identity import run_identity
from streamreplace.core.io import file_digest, file_lock, fingerprint, read_jsonl, write_jsonl
from streamreplace.core.schema import GEN_KEYS
from streamreplace.data import apply_edits, load_samples, render_prompts, samples_file
from streamreplace.run.dispatch import make_units
from streamreplace.run.envs import worker_env
from streamreplace.run.plan import lock_path, shard_lock_path


def test_end_to_end_outputs_are_flat_and_readable(project):
    jobs, status = run_all(project)
    assert [j.name for j in jobs] == JOBS
    assert status["demo/parent.offline"] == "complete"
    assert "1 left" in status["demo/cand.offline"] and "1 left" in status["demo/cand.stream"]   # yn-2 fails
    out = project.outputs
    assert sorted(p.name for p in (out / "demo").iterdir() if not p.name.startswith(".")) == [
        "cand.offline.jsonl", "cand.offline.meta.json", "cand.stream.jsonl", "cand.stream.meta.json",
        "parent.offline.jsonl", "parent.offline.meta.json"]
    samples = {s.id: s for s in load_samples("demo", paths=project.paths)}
    prompts = render_prompts("demo", list(samples.values()))
    rows = project.rows("demo/parent.offline")
    assert len(rows) == 6 and all(list(r) == list(GEN_KEYS) for r in rows)
    assert all(r["raw"] == f"The answer is {samples[r['id']].answer}." and r["prompt"] == prompts[r["id"]]
               for r in rows)                                    # accuracy 1.0
    failed = [r for r in project.rows("demo/cand.offline") if r["error"]]
    assert [r["id"] for r in failed] == ["yn-2"] and failed[0]["raw"] is None
    assert failed[0]["error"] == "RuntimeError: configured mock failure"
    assert all(r["info"]["stream_fps"] == 1 for r in project.rows("demo/cand.stream") if not r["error"])
    text = (out / "demo/parent.offline.meta.json").read_text()
    meta = json.loads(text)
    assert meta["identity"] == jobs[0].identity == run_identity(meta["payload"]) and meta["payload"] == jobs[0].payload
    segment, = meta["segments"]
    assert segment["load_seconds"] is not None and segment["shard"] == [0, 1] and segment["pending"] == 6
    assert segment["samples_sha256"] == file_digest(samples_file("demo", paths=project.paths))
    assert str(project.root) not in text                        # no paths in metas
    host = socket.gethostname()
    assert len(host) < 4 or host not in text                    # no host names
    assert sorted(p.name for p in (out / "logs").glob("*.log")) == [
        "cand.offline.log", "cand.stream.log", "parent.offline.log"]


def test_resume_skips_done_and_retries_errors(project):
    run_all(project)
    before = project.rows("demo/cand.offline")
    _, status = run_all(project)
    after = project.rows("demo/cand.offline")
    assert status["demo/parent.offline"] == "complete"
    assert after[:len(before)] == before                         # append-only: earlier rows untouched
    assert len(after) == len(before) + 1 and after[-1]["id"] == "yn-2" and after[-1]["error"]   # only yn-2 retried
    assert len(project.rows("demo/parent.offline")) == 6         # a complete job starts no worker
    assert len(json.loads((project.outputs / "demo/cand.offline.meta.json").read_text())["segments"]) == 2
    assert len(json.loads((project.outputs / "demo/parent.offline.meta.json").read_text())["segments"]) == 1


def test_rows_of_other_ids_are_ignored(project):
    run_all(project)
    path = project.outputs / "demo/parent.offline.jsonl"
    write_jsonl(path, project.rows("demo/parent.offline") + [{"id": "not-a-sample", "prompt": "x", "error": "boom"}])
    assert project.run(project.plan(pairs=(), models="parent")) == {"demo/parent.offline": "complete"}


def test_changed_settings_are_refused_unless_fresh(project):
    run_all(project)
    changed = project.plan(pairs=(), models="parent", sets={"generation": {"max_new_tokens": 5}})
    with pytest.raises(ValueError, match="identity changed: settings.generation.max_new_tokens: 16 -> 5"):
        project.run(changed)
    assert project.run(changed, fresh=True)["demo/parent.offline"] == "complete"
    archive = project.outputs / "_archive"
    assert len(list(archive.glob("*/demo/parent.offline.jsonl"))) == 1
    assert len(list(archive.glob("*/demo/parent.offline.meta.json"))) == 1
    archived = len(list(archive.rglob("*.jsonl")))
    project.run(changed, fresh=True)                             # settings now match: nothing more is archived
    assert len(list(archive.rglob("*.jsonl"))) == archived


def test_fresh_archives_scores_too(project):
    run_all(project)
    score_rule_layer(project, "demo", "parent.offline")
    project.run(project.plan(pairs=(), models="parent", sets={"generation": {"max_new_tokens": 5}}), fresh=True)
    assert list((project.outputs / "_archive").glob("*/demo/parent.offline.scores.jsonl"))
    assert not (project.outputs / "demo/parent.offline.scores.jsonl").exists()


def test_a_refused_run_writes_no_metas(project):
    project.run(project.plan(pairs=(), models="parent"))
    jobs = project.plan(sets={"generation": {"max_new_tokens": 5}})   # parent conflicts; the child's jobs are new
    with pytest.raises(ValueError, match="identity changed"):
        project.run(jobs)
    assert not (project.outputs / "demo/cand.offline.meta.json").exists()
    assert not (project.outputs / "demo/cand.stream.meta.json").exists()


@pytest.mark.parametrize("model", ["parent", "cand"])          # parent's job is complete, cand's has a row left
def test_prompt_drift_is_refused_unless_fresh(project, model):
    run_all(project)
    path = project.outputs / f"demo/{model}.offline.jsonl"
    rows = read_jsonl(path)
    rows[0]["prompt"] = "an older prompt"
    write_jsonl(path, rows)
    jobs = project.plan(pairs=(), models=model, protocols="P,C")
    with pytest.raises(ValueError, match="prompt changed"):
        project.run(jobs)
    project.run(jobs, fresh=True)
    assert list((project.outputs / "_archive").glob(f"*/demo/{model}.offline.jsonl"))
    assert "an older prompt" not in path.read_text()


@pytest.mark.parametrize("shards, concurrency", [(3, 1), (1, 4), (2, 3)])
def test_shards_and_concurrency_give_the_same_rows(tmp_path, shards, concurrency):
    """Sharded workers share one output file; concurrent samples append whole lines. Either way every sample gets
    exactly one row, identical to a single in-order worker."""
    results = []
    for name, (n, c) in {"serial": (1, 1), "parallel": (shards, concurrency)}.items():
        models = deepcopy(MODELS)
        models["cand"]["options"]["concurrency"] = c
        project = make_project(tmp_path / name, models=models)
        status = project.run(project.plan(), shards=n)
        assert status["demo/parent.offline"] == "complete" and "1 left" in status["demo/cand.offline"]
        results.append(rows_of(project))
        if n > 1:
            logs = sorted(p.name for p in (project.outputs / "logs").glob("parent.offline*.log"))
            assert logs == [f"parent.offline.shard{k}.log" for k in range(n)]
    assert results[0] == results[1]
    assert all(len({r[0] for r in rows}) == len(rows) for rows in results[1].values())   # no duplicate ids


def test_errored_samples_count_as_wrong(project):
    """Official denominators: a sample whose generation failed is counted, as wrong (and listed under errors)."""
    from streamreplace.analysis.metrics import collect_job
    from streamreplace.analysis.records import Catalog, load_job
    run_all(project)
    for job in ("parent.offline", "cand.offline", "cand.stream"):
        score_rule_layer(project, "demo", job)
    catalog = Catalog(samples=lambda dataset: load_samples(dataset, paths=project.paths))
    result = collect_job(project.outputs, "demo", "cand.offline", catalog)
    assert result["expected"] == 6 and result["errors"] == 1 and "yn-2" in result["rows"]
    assert result["done"] == len(result["rows"]) - 1 == 5
    job = load_job(project.outputs, "demo", "cand.offline", catalog)
    assert len(job.rows) == job.expected == 6 and job.rows["yn-2"].error and not job.rows["yn-2"].correct
    parent = load_job(project.outputs, "demo", "parent.offline", catalog)
    assert all(r.correct for i, r in parent.rows.items() if i != "open-1")


def test_outputs_folder_with_subset_runs_only_those_ids(project):
    """--outputs puts a side experiment in its own folder; its subset.json restricts every job to those ids and
    enters the identity, while the main folder stays untouched and unrestricted."""
    exp = project.root / "exp"
    exp.mkdir()
    (exp / "subset.json").write_text(json.dumps({"demo": ["yn-1", "open-1"]}))
    code = project.cli("--outputs", str(exp), "run", "--pairs", "p", "--benchmarks", "demo", "--gpus", "0")
    assert code == 0                                             # yn-2 (the configured failure) is not in the subset
    for job in JOBS:
        assert sorted(r["id"] for r in project.rows(job, exp)) == ["open-1", "yn-1"]
    assert not (project.outputs / "demo").exists()
    side, main = project.plan(outputs=exp), project.plan()
    assert side[1].payload["subset"] == fingerprint(["open-1", "yn-1"]) and main[1].payload["subset"] is None
    assert side[1].identity != main[1].identity


def test_prompt_overrides_edit_every_prompt(project):
    exp = project.root / "robust"
    samples = load_samples("demo", paths=project.paths)
    edits = {s.id: [[s.question, s.question.upper()]] for s in samples}
    write_jsonl(exp / "prompt_overrides.jsonl", [{"dataset": "demo", "id": i, "edits": e} for i, e in edits.items()])
    jobs = project.plan(outputs=exp, pairs=(), models="parent")
    assert jobs[0].payload["prompt_overrides"] == fingerprint(edits)
    assert project.run(jobs) == {"demo/parent.offline": "complete"}
    base = render_prompts("demo", samples)
    assert {r["id"]: r["prompt"] for r in project.rows("demo/parent.offline", exp)} == {
        i: apply_edits(p, edits[i]) for i, p in base.items()}
    partial = project.root / "partial"
    write_jsonl(partial / "prompt_overrides.jsonl", [{"dataset": "demo", "id": i, "edits": e}
                                                    for i, e in list(edits.items())[:-1]])
    with pytest.raises(ValueError, match="no row for open-1"):
        project.run(project.plan(outputs=partial, pairs=(), models="parent"))
    assert not (partial / "demo").exists()                       # refused before any meta is written


def test_cli_limit_run_uses_the_smoke_order(project):
    quick = project.root / "quick"
    code = project.cli("--outputs", str(quick), "run", "--pairs", "p", "--benchmarks", "demo", "--limit", "1",
                       "--gpus", "0")
    assert code == 0                                             # 1 sample per job; no configured failure
    assert sorted(p.name for p in (quick / "logs").glob("*.log")) == [
        "cand.offline.log", "cand.stream.log", "parent.offline.log"]          # one worker (log) per arm
    for job in JOBS:
        assert [r["id"] for r in project.rows(job, quick)] == ["mcq-1"]   # the demo's smoke order starts with mcq-1
    assert json.loads((quick / "demo/parent.offline.meta.json").read_text())["payload"]["limit"] == 1
    assert not (project.outputs / "demo").exists()
    code = project.cli("--outputs", str(quick), "run", "--pairs", "p", "--benchmarks", "demo", "--gpus", "0")
    assert code == 1                                             # a full run is refused in the limited folder


def test_jobs_with_different_load_time_settings_get_separate_workers(tmp_path):
    """A process loads its model once: benchmarks whose load-time settings (the adapter's LOAD_KEYS; the mock
    declares stream.mock_load) differ are never served by one process."""
    benchmarks = {"demo": {"media": "video"}, "demo2": {"media": "video", "settings": {"stream": {"mock_load": 2}}}}
    project = make_project(tmp_path / "split", benchmarks=benchmarks)
    jobs = project.plan(benchmarks="demo,demo2", pairs=(), models="cand", protocols="S")
    assert [[j.name for j in u.jobs] for u in make_units(jobs)] == [["demo/cand.stream"], ["demo2/cand.stream"]]
    project.run(jobs)
    assert sorted(p.name for p in (project.outputs / "logs").glob("*.log")) == [
        "cand.stream.part1.log", "cand.stream.part2.log"]
    assert all("load" not in r["info"] for r in project.rows("demo/cand.stream") if not r["error"])
    assert all(r["info"]["load"] == 2 for r in project.rows("demo2/cand.stream") if not r["error"])
    same = {"demo": {"media": "video"}, "demo2": {"media": "video", "settings": {"generation": {"max_new_tokens": 8}}}}
    project = make_project(tmp_path / "same", benchmarks=same)
    jobs = project.plan(benchmarks="demo,demo2", pairs=(), models="cand", protocols="S")
    assert [[j.name for j in u.jobs] for u in make_units(jobs)] == [["demo/cand.stream", "demo2/cand.stream"]]


def test_dry_run_and_status_write_nothing(project, capsys):
    assert project.cli("run", "--pairs", "p", "--benchmarks", "demo", "--dry-run") == 0
    out = capsys.readouterr().out
    assert "3 jobs planned" in out and "demo/cand.stream" in out and "new" in out
    assert not project.outputs.exists()
    run_all(project)
    capsys.readouterr()
    assert project.cli("status", "--pairs", "p", "--benchmarks", "demo") == 2
    out = capsys.readouterr().out
    assert "1 of 3 jobs complete" in out and "5/6" in out and "1 errors" in out and "partial" in out


def test_print_launcher_writes_payloads_a_worker_can_run(project, capsys):
    code = project.cli("run", "--pairs", "p", "--benchmarks", "demo", "--launcher", "print", "--gpus", "3")
    out = capsys.readouterr().out
    assert code == 0 and out.count("-m streamreplace.run.worker") == 3 and "CUDA_VISIBLE_DEVICES=3" in out
    assert not (project.outputs / "demo/parent.offline.jsonl").exists()
    payload = project.outputs / "logs/parent.offline.payload.json"
    requests = json.loads(payload.read_text())["jobs"][0]["requests"]
    assert len(requests) == 6 and all("reference" in r for r in requests)    # the mock is the only adapter told
    subprocess.run([sys.executable, "-m", "streamreplace.run.worker", str(payload)], check=True, capture_output=True,
                   env=worker_env(project.paths, "local", 0, 1))
    assert len(project.rows("demo/parent.offline")) == 6


def test_requests_carry_the_model_safe_view_only(project):
    """Workers get the sample's view: no answer (except for the test mock) and no private metadata."""
    from streamreplace.benchmarks import registry
    from streamreplace.run.dispatch import Inputs
    job = project.plan(pairs=(), models="parent")[0]
    assert all("reference" in r for r in Inputs(project.paths).requests(job))      # the mock scores itself
    job.adapter = "qwen_offline"
    registry.get("demo").PRIVATE_METADATA = frozenset({"group"})
    requests = Inputs(project.paths).requests(job)
    assert len(requests) == 6 and not any("reference" in r or r["metadata"] for r in requests)
    assert set(requests[0]) == {"id", "prompt", "images", "video", "kind", "question", "choices", "metadata"}


def test_a_locked_output_file_is_left_alone(project):
    """A worker never writes a file the scorer holds (exclusive job lock), nor a shard another worker runs."""
    assert project.cli("run", "--pairs", "p", "--benchmarks", "demo", "--launcher", "print", "--gpus", "0") == 0
    payload = project.outputs / "logs/parent.offline.payload.json"
    out = project.outputs / "demo/parent.offline.jsonl"
    for lock in (lock_path(out), shard_lock_path(out, 0)):
        with file_lock(lock, blocking=False):
            proc = subprocess.run([sys.executable, "-m", "streamreplace.run.worker", str(payload)],
                                  capture_output=True, text=True, env=worker_env(project.paths, "local", 0, 1))
        assert proc.returncode == 2 and "is locked" in proc.stdout + proc.stderr and not out.exists()


def test_plugin_variant_arms(tmp_path):
    """Plugin arms are stream jobs <child>@<variant>.stream: the Recall delta joins the settings (and reaches the
    model at load time), the plugin block joins the identity, and the worker passes the arm's hooks to stream()."""
    pytest.importorskip("streamreplace.plugin.hooks")
    plugin = {"arms": {"sfix": {"recall": "budget", "respond": False, "gate": None, "prs": False},
                       "inert": {"recall": "native", "respond": False, "gate": None, "prs": False}},
              "plug_arm": "sfix", "recall": {"cand": {"stream": {"mock_load": 7}}}, "children": ["cand"]}
    project = make_project(tmp_path, sections={"plugin": plugin})
    jobs = project.plan(protocols="", variants="sfix,inert")
    assert [j.name for j in jobs] == ["demo/cand@sfix.stream", "demo/cand@inert.stream"]
    by_name = project.plan(pairs=(), models="cand@sfix,cand", protocols="S")    # one arm by its name
    assert [j.name for j in by_name] == ["demo/cand@sfix.stream", "demo/cand.stream"]
    assert by_name[0].payload == jobs[0].payload
    sfix, inert = jobs
    assert sfix.settings["stream"] == {"fps": 1, "mock_load": 7} and inert.settings["stream"] == {"fps": 1}
    assert sfix.payload["job"] == {"model": "cand", "mode": "stream", "variant": "sfix"}
    assert sfix.payload["model"]["plugin"]["settings"] == {"stream": {"mock_load": 7}}
    assert sfix.payload["model"]["plugin"]["recipe"]["name"] == "sfix"
    assert project.run(jobs) == {"demo/cand@sfix.stream": "1 left (see logs/cand@sfix.stream.log)",
                                 "demo/cand@inert.stream": "1 left (see logs/cand@inert.stream.log)"}
    rows = [r for r in project.rows("demo/cand@sfix.stream") if not r["error"]]
    assert len(rows) == 5 and all(r["info"]["load"] == 7 and r["info"]["plugin"]["native"] == r["raw"] for r in rows)


def test_finetuned_model_from_an_extra_config(project):
    """A fine-tuned id from an extra config file: its LoRA folder is handed to the adapter, and the identity holds the
    sha256 of the adapter weights, never the path."""
    adapter = project.outputs / "adapters/demo/cand/1k"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights v1")
    extra = dump(project.root / "finetuned.yaml", {"models": {"cand@demo-1k": {
        "role": "child", "checkpoint": "test/cand@0", "offline": {"adapter": "mock", "env": "local"},
        "options": {"accuracy": 1.0, "lora": "adapters/demo/cand/1k", "deterministic": True}}}})
    jobs = project.plan(extra=[extra], pairs=(), models="cand@demo-1k")
    assert [j.name for j in jobs] == ["demo/cand@demo-1k.offline"]
    job = jobs[0]
    assert job.payload["model"]["options"]["lora"] == {"sha256": hashlib.sha256(b"weights v1").hexdigest()}
    assert "adapters" not in json.dumps(job.payload) and job.deterministic and job.lora == adapter
    assert project.run(jobs) == {"demo/cand@demo-1k.offline": "complete"}
    assert all(r["info"]["lora"] == str(adapter) for r in project.rows("demo/cand@demo-1k.offline"))
    (adapter / "adapter_model.safetensors").write_bytes(b"weights v2")         # retrained: never reuse old rows
    with pytest.raises(ValueError, match="identity changed: model.options.lora.sha256"):
        project.run(project.plan(extra=[extra], pairs=(), models="cand@demo-1k"))
    missing = dump(project.root / "missing.yaml", {"models": {"cand@demo-2k": {
        "role": "child", "checkpoint": "test/cand@0", "offline": {"adapter": "mock", "env": "local"},
        "options": {"lora": "adapters/demo/cand/2k"}}}})
    with pytest.raises(ValueError, match="has no adapter_model.safetensors"):
        project.plan(extra=[missing], pairs=(), models="cand@demo-2k")


def test_worker_imports_only_light_dependencies():
    """The worker is imported inside every model environment (Python 3.10-3.12): only the standard library, this
    package and PyYAML / numpy / Pillow / PyAV may be imported, never torch or the model stacks."""
    from streamreplace.run.envs import src_dir
    code = ("import json, sys\nbefore = {m.split('.')[0] for m in sys.modules}\nimport streamreplace.run.worker\n"
            "after = {m.split('.')[0] for m in sys.modules}\n"
            "print(json.dumps(sorted(m for m in after - before if m not in sys.stdlib_module_names\n"
            "                        and m != 'streamreplace' and not m.startswith('_'))))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                            env={**os.environ, "PYTHONPATH": str(src_dir())})
    third_party = set(json.loads(result.stdout)) - {"cython_runtime"}   # a pseudo-module of PyYAML's C extension
    assert third_party <= {"yaml", "numpy", "PIL", "av"}, third_party
