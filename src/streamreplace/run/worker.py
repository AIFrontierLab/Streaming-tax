"""Generation worker: runs inside a model's own Python environment, one process per (arm, shard), launched by
run/dispatch.py as

    python -m streamreplace.run.worker <payload.json>

The payload names the model, the adapter and its constants, and lists the jobs (one per benchmark) with their
resolved settings and this shard's requests. For each job in turn the worker
1. takes the job lock (shared; the scorer takes it exclusively and skips busy files) and the shard's lock
   (exclusive: a second worker on the same shard fails at once);
2. checks that meta.json still holds the planned identity and that stored rows carry the same prompts;
3. generates the pending requests (no row yet, or the last row is an error, which is retried) and appends one row per
   request, a whole line at a time under an exclusive file lock with fsync (core/io.py), so shards and threads share
   one output file;
4. records a run segment in meta.json: package and library versions, GPU names, code version, load time, the sha256
   of the prepared samples and of the upstream manifest. Never host names or paths.

The model is loaded once, lazily, for the first job with pending samples, with that job's settings (the dispatcher
groups only jobs whose load-time settings, the adapter's LOAD_KEYS, agree). Offline jobs (conditions P and C) call
adapter.generate(request), stream jobs (condition S) adapter.stream(request); a plugin arm passes the arm's hooks to
stream() and calls hooks.begin(request) first. An adapter whose `concurrency` is above 1 (AURA's vLLM client) gets
that many requests in flight on a thread pool; with plugin hooks requests run one at a time, because the hooks keep
per-request state. A failing sample becomes an error row (raw/tokens/finish/info null, the message in `error`) and
never stops the job, and a failing job never stops the others. Exit code 2 when anything failed.

A fine-tuned model id names its LoRA adapter folder in options.lora: the worker calls adapter.load_lora(path) right
after loading and refuses an adapter without that method, so a fine-tuned id can never silently evaluate its base
model. With options.deterministic, torch's deterministic algorithms are switched on before loading.

Only the standard library and this package's light modules are imported here (configs need PyYAML); torch and the
model stack are imported by the adapter, inside the model's environment."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import traceback

from .. import paths as paths_module
from ..config.schema import ModelSpec
from ..core.identity import add_segment, read_meta, run_segment
from ..core.io import append_jsonl, file_lock, read_jsonl
from ..core.schema import Generation, Request, SampleView, generation_row
from ..models import registry
from ..models.upstream import Upstream
from .plan import lock_path, meta_lock_path, shard_lock_path

PAYLOAD_VERSION = 1


class Worker:
    def __init__(self, payload: dict):
        if payload.get("version") != PAYLOAD_VERSION:
            raise ValueError(f"payload version {payload.get('version')!r}; this worker reads version {PAYLOAD_VERSION}")
        self.payload = payload
        arm = payload["arm"]
        self.stream = arm["mode"] == "stream"
        self.label = f"{arm['model']}{'@' + arm['variant'] if arm.get('variant') else ''}.{arm['mode']}"
        self.shard = tuple(payload.get("shard") or (0, 1))
        model = payload["model"]
        self.spec = replace(ModelSpec.parse(model["id"], model["spec"]), pair=model.get("pair"))
        self.adapter_name = payload["adapter"]
        self.adapter = None
        self.hooks = None
        self.load_seconds: float | None = None

    # ---------- model ----------

    def load(self, settings: dict) -> None:
        """Loads the adapter once, with the settings of the first job that has pending samples."""
        if self.adapter is not None:
            return
        cls = registry.load_class(self.adapter_name)
        have = {"PROTOCOL_VERSION": cls.PROTOCOL_VERSION, "LOAD_KEYS": list(cls.LOAD_KEYS)}
        if have != self.payload["constants"]:
            raise RuntimeError(f"adapter {self.adapter_name}: the class has {have}, the plan used "
                               f"{self.payload['constants']}; plan again with streamreplace run")
        if self.payload.get("deterministic"):
            try:
                import torch
            except ImportError:                  # only the test mock runs without torch
                torch = None
            if torch is not None:
                torch.use_deterministic_algorithms(True)
        started = time.time()
        adapter = cls(self.spec, settings, Upstream())
        try:
            lora = self.spec.options.get("lora")
            if lora:
                loader = getattr(adapter, "load_lora", None)
                if loader is None:
                    raise RuntimeError(f"adapter {self.adapter_name} cannot load LoRA adapters; {self.spec.id} would "
                                       "silently be its base model")
                loader(lora)
            if self.payload.get("plugin"):
                self.hooks = self._hooks()
        except BaseException:
            adapter.close()
            raise
        self.adapter = adapter
        self.load_seconds = round(time.time() - started, 1)
        print(f"loaded {self.label} ({self.adapter_name}) in {self.load_seconds}s", flush=True)

    def _hooks(self):
        """The plugin arm's AnswerTurnHooks, rebuilt from the same plugin configuration the plan used."""
        from ..plugin.hooks import make_hooks
        from ..plugin.recipe import load_plugin_config
        block = self.payload["plugin"]
        hooks = make_hooks(block["child"], block["variant"], load_plugin_config(section=block.get("section") or None))
        planned = self.payload.get("plugin_identity")
        if planned is not None and (json.dumps(hooks.identity(), sort_keys=True, default=str)
                                    != json.dumps(planned, sort_keys=True, default=str)):
            raise RuntimeError("the plugin arm changed since planning (artefacts or configs/plugin.yaml); plan again "
                               "with streamreplace run")
        return hooks

    # ---------- jobs ----------

    def request(self, job: dict, item: dict) -> Request:
        choices = dict(item.get("choices") or {})
        view = SampleView(item["id"], item["question"], item["kind"], dict(choices), dict(item.get("metadata") or {}))
        return Request(id=item["id"], dataset=job["dataset"], prompt=item["prompt"],
                       images=list(item.get("images") or []), video=item.get("video"), settings=job["settings"],
                       kind=item["kind"], choices=choices, sample=view, reference=item.get("reference"))

    def one(self, job: dict, item: dict, out: Path) -> dict:
        """Generates one request and appends its row (an error row if anything raised)."""
        request = self.request(job, item)
        started = time.perf_counter()
        try:
            if not self.stream:
                result = self.adapter.generate(request)
            elif self.hooks is None:
                result = self.adapter.stream(request)
            else:
                begin = getattr(self.hooks, "begin", None)
                if begin is not None:
                    begin(request)
                result = self.adapter.stream(request, hooks=self.hooks)
            generation = result if isinstance(result, Generation) else Generation.from_dict(result)
            row = generation_row(item["id"], item["prompt"], generation, round(time.perf_counter() - started, 3))
        except Exception as exc:                  # one bad sample must not stop the job
            row = generation_row(item["id"], item["prompt"], None, round(time.perf_counter() - started, 3),
                                 error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            torch = sys.modules.get("torch")
            if "out of memory" in str(exc).lower() and torch is not None:
                torch.cuda.empty_cache()
        append_jsonl(out, row)
        return row

    def run_job(self, job: dict) -> int:
        """Returns the number of failed samples; raises when the job cannot run at all."""
        out = Path(job["out"])
        shard = self.shard[0]
        with ExitStack() as stack:
            try:
                stack.enter_context(file_lock(lock_path(out), shared=True, blocking=False))
                stack.enter_context(file_lock(shard_lock_path(out, shard), blocking=False))
            except BlockingIOError:
                raise RuntimeError(f"{out.name} is locked: being scored, or another worker runs shard "
                                   f"{shard}") from None
            return self._run_job(job, out)

    def _run_job(self, job: dict, out: Path) -> int:
        name = f"{job['dataset']}/{job['job']}"
        meta_path = Path(job["meta"])
        meta = read_meta(meta_path)
        if meta is None or meta.get("identity") != job["identity"]:
            raise RuntimeError(f"{name}: meta.json does not hold the planned identity {job['identity']}; plan again "
                               "with streamreplace run")
        requests = job["requests"]
        prompts = {item["id"]: item["prompt"] for item in requests}
        rows = read_jsonl(out)
        drift = [row["id"] for row in rows if row["id"] in prompts and row.get("prompt") != prompts[row["id"]]]
        if drift:
            raise RuntimeError(f"prompt changed for {len(drift)} saved rows (e.g. {drift[0]}); rerun with --fresh")
        done = {row["id"]: row for row in rows}                 # the last row per id wins
        pending = [item for item in requests if item["id"] not in done or done[item["id"]].get("error")]
        k, n = self.shard
        part = f" (shard {k + 1}/{n})" if n > 1 else ""
        print(f"== {name}: {len(pending)} pending of {len(requests)}{part}", flush=True)
        if not pending:
            return 0
        self.load(job["settings"])
        with file_lock(meta_lock_path(meta_path)):              # shards update meta.json in turn
            add_segment(meta_path, run_segment(
                adapter=self.adapter_name, env=self.payload.get("env"), shard=[k, n], pending=len(pending),
                load_seconds=self.load_seconds, samples_sha256=job.get("samples_sha256"),
                upstream_manifest_sha256=self.payload.get("upstream_manifest_sha256")))
        concurrency = 1 if self.hooks is not None else max(1, int(getattr(self.adapter, "concurrency", 1) or 1))
        errors = 0
        pool = ThreadPoolExecutor(concurrency) if concurrency > 1 else None   # 1: in order, on the main thread
        try:
            results = (pool.map(lambda item: self.one(job, item, out), pending) if pool
                       else (self.one(job, item, out) for item in pending))
            for number, (item, row) in enumerate(zip(pending, results), 1):
                errors += bool(row["error"])
                print(f"[{number}/{len(pending)}] {job['dataset']} {item['id']} {row['latency']}s"
                      f"{' ERROR ' + row['error'] if row['error'] else ''}", flush=True)
        finally:
            if pool is not None:
                pool.shutdown()
        return errors

    def run(self) -> int:
        failed = 0
        try:
            for job in self.payload["jobs"]:
                try:
                    failed += self.run_job(job)
                except Exception:                 # a broken job must not stop the arm's other benchmarks
                    failed += 1
                    print(f"== job {job['dataset']}/{job['job']} failed", flush=True)
                    traceback.print_exc()
        finally:
            if self.adapter is not None:
                self.adapter.close()
        return 2 if failed else 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m streamreplace.run.worker <payload.json>", file=sys.stderr)
        return 1
    payload = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    block = payload.get("paths")
    if block:   # roots come from the STREAMREPLACE_* variables; per-checkpoint / per-dataset folders from the plan
        paths_module.set_current(paths_module.resolve(config_paths=block))
    return Worker(payload).run()


if __name__ == "__main__":
    raise SystemExit(main())
