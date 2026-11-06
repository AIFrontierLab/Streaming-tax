"""One fine-tuning training unit (model, dataset, budget[, epochs, tag]), run inside the model's own environment on one
GPU: `python -m streamreplace.generalization.finetune.train <payload.json>` (started by `streamreplace finetune train`,
launch.py, which writes the payload: the zero-shot model spec, resolved settings, LoRA and optimiser configuration).
Writes to the unit's folder <root>/adapters/<dataset>/<model>/<label>/:
  unit.json     what this unit is (model, settings, LoRA + optimiser configuration, data hashes); a resume must match it
  loss.jsonl    one line per optimiser step (loss, lr, grad norm, samples, tokens, seconds, memory)
  ckpt.pt       LoRA weights + AdamW state + step, rewritten atomically every save_every_steps / save_every_minutes
  adapter/      the result (PEFT safetensors + adapter_config.json), then DONE.json; ckpt.pt is removed

Recipe (configs/generalization/finetune.yaml `lora` / `train`; ported unchanged): the first `budget` ids of
train_order.json; epoch 0 in that order, later epochs in a seeded reshuffle (a seed-replication unit has its own seed,
and with `shuffle` epoch 0 is a seeded reshuffle of the same ids too); per optimiser step grad_accum samples, each
sample's loss = mean token cross-entropy over its answer + <|im_end|> (fp32), the step's loss = their mean; AdamW
(0.9, 0.999, eps 1e-8), linear warmup then cosine to 0, gradient clipping. Inputs come from the model adapter's
encode(), the same code that builds its evaluation inputs (models/base.py Trainable).

Determinism and resuming: torch's deterministic algorithms are on (the launcher also sets CUBLAS_WORKSPACE_CONFIG and
FLASH_ATTENTION_DETERMINISTIC), nothing in a step is random (LoRA dropout 0, fixed data order), so restarting from
ckpt.pt at step k gives the same run as never stopping."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from ...core.io import append_jsonl, fingerprint, read_json, read_jsonl, write_json, write_jsonl
from ...core.schema import Request
from . import label

PAYLOAD_VERSION = 1


def epoch_order(ids: list[str], epoch: int, seed: int, shuffle: bool = False) -> list[str]:
    """Epoch 0 keeps train_order (unless `shuffle`); other epochs are a reshuffle seeded by (seed, epoch)."""
    if epoch == 0 and not shuffle:
        return list(ids)
    return sorted(ids, key=lambda i: hashlib.sha256(f"{seed}:{epoch}:{i}".encode()).hexdigest())


def schedule(step: int, total: int, warmup: int) -> float:
    """Learning-rate multiplier: linear warmup over `warmup` steps, then cosine to 0 at `total`."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def _cuda(torch) -> bool:
    return torch.cuda.is_available()


class Unit:
    def __init__(self, payload: dict, load_adapter=None):
        """payload: written by launch.training_payload(). A pilot (throughput, memory, resume checks) has its own
        `dir` and `max_steps`: it stops after max_steps with a checkpoint but no adapter; the learning-rate schedule
        is still the full unit's. load_adapter(spec, settings) -> adapter replaces the registry (tests)."""
        from .data import selection
        if payload.get("version") != PAYLOAD_VERSION:
            raise ValueError(f"payload version {payload.get('version')!r}; this trainer reads {PAYLOAD_VERSION}")
        self.payload, unit = payload, payload["unit"]
        self.unit = unit
        self.dir = Path(payload["dir"])
        self.max_steps = payload.get("max_steps")
        self.dataset = unit["dataset"]
        self.settings = payload["settings"]
        self.family = payload["family"]
        self.load_adapter = load_adapter
        order, record = selection(Path(payload["selection"]), self.dataset)
        self.ids = order[:unit["budget"]]
        if len(self.ids) != unit["budget"]:
            raise ValueError(f"train_order has {len(self.ids)} ids < budget {unit['budget']}")
        train = payload["train"]
        self.train = train
        self.accum = int(train["grad_accum"])
        self.seed, self.shuffle = int(unit.get("seed", train["seed"])), bool(unit.get("shuffle"))
        self.samples_total = unit["budget"] * unit["epochs"]
        self.steps = math.ceil(self.samples_total / self.accum)
        self.warmup = max(1, math.ceil(float(train["warmup_ratio"]) * self.steps))
        self.identity = {"unit": unit, "model": payload["model"]["spec"], "adapter": payload["adapter"],
                         "settings": self.settings, "lora": payload["lora"], "train": train, "steps": self.steps,
                         "train_sha256": record["train_sha256"], "test_sha256": record["test_sha256"]}
        self.hash = fingerprint(self.identity)

    def sample(self, k: int):
        """The k-th training sample of the run (k counts across epochs)."""
        epoch, position = divmod(k, self.unit["budget"])
        return self.by_id[epoch_order(self.ids, epoch, self.seed, self.shuffle)[position]]

    def _adapter(self):
        if self.load_adapter is not None:
            return self.load_adapter(self.payload["model"], self.settings)
        from ...config.schema import ModelSpec
        from ...models import registry
        spec = ModelSpec.parse(self.payload["model"]["id"], self.payload["model"]["spec"])
        return registry.create(self.payload["adapter"], spec, self.settings)

    def run(self) -> dict:
        import torch
        from ...benchmarks import registry
        from ...data.samples import load_samples
        from . import lora
        self.dir.mkdir(parents=True, exist_ok=True)
        if (self.dir / "unit.json").exists() and read_json(self.dir / "unit.json")["hash"] != self.hash:
            raise ValueError(f"{self.dir}: unit settings changed since this unit started; use --fresh")
        write_json(self.dir / "unit.json", {"hash": self.hash, **self.identity})
        bench = registry.get(self.dataset)
        wanted = set(self.ids)
        self.by_id = {s.id: s for s in load_samples(self.dataset, split="train") if s.id in wanted}
        family = self.family
        started = time.time()
        torch.use_deterministic_algorithms(True)       # strict: SDPA memory-efficient backward is deterministic only then
        adapter = self._adapter()
        model = adapter.model
        lora.inject(model, family, int(self.payload["lora"]["r"]), int(self.payload["lora"]["alpha"]), self.seed)
        lora.train_mode(model, family)
        params = list(lora.lora_parameters(model).values())
        train = self.train
        optimizer = torch.optim.AdamW(params, lr=float(train["lr"]), betas=(0.9, 0.999), eps=1e-8,
                                      weight_decay=float(train["weight_decay"]))
        step = 0
        ckpt = self.dir / "ckpt.pt"
        if ckpt.exists():
            state = torch.load(ckpt, map_location="cpu")
            if state["hash"] != self.hash:
                raise ValueError(f"{ckpt} belongs to other settings; use --fresh")
            named = lora.lora_parameters(model)
            for name, value in state["lora"].items():
                named[name].data.copy_(value)
            optimizer.load_state_dict(state["optimizer"])
            step = int(state["step"])
            kept = [r for r in read_jsonl(self.dir / "loss.jsonl") if r["step"] <= step]   # steps after the checkpoint redo
            write_jsonl(self.dir / "loss.jsonl", kept)
            print(f"resumed at step {step}/{self.steps}", flush=True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: schedule(s, self.steps, self.warmup),
                                                      last_epoch=step - 1)

        def encode(k: int):
            sample = self.sample(k)
            view = bench.view(sample)
            request = Request(id=sample.id, dataset=self.dataset, prompt=bench.prompt(view), images=list(sample.images),
                              video=sample.video, settings=self.settings, kind=sample.kind,
                              choices=dict(sample.choices or {}), sample=view)
            return adapter.encode(request, sample.answer)

        cuda = _cuda(torch)
        peak = lambda: round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else None
        prefetch = ThreadPoolExecutor(1)       # one thread: MOSS's encode changes its processor in place
        pending = {k: prefetch.submit(encode, k) for k in range(step * self.accum, min(self.samples_total, step * self.accum + 2))}
        last_save, losses = time.time(), []
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        while step < self.steps:
            if self.max_steps and step >= self.max_steps:
                self._checkpoint(torch, lora, model, optimizer, step)
                prefetch.shutdown(cancel_futures=True)
                return {"pilot_stopped_at": step, "peak_mem_gb": peak()}
            t0, step_losses, tokens = time.time(), [], 0
            first, last = step * self.accum, min(self.samples_total, (step + 1) * self.accum)
            for k in range(first, last):
                enc = pending.pop(k).result()
                for nxt in range(k + 1, min(self.samples_total, k + 3)):
                    if nxt not in pending:
                        pending[nxt] = prefetch.submit(encode, nxt)
                n = enc["n"]
                logits = adapter.train_forward(enc)[0, :-1].float()
                targets = enc["inputs"]["input_ids"][0, -n:].to(logits.device)
                loss = torch.nn.functional.cross_entropy(logits[-n:], targets)
                (loss / (last - first)).backward()
                step_losses.append(loss.detach().item())
                tokens += int(enc["inputs"]["input_ids"].shape[1])
            norm = float(torch.nn.utils.clip_grad_norm_(params, float(train["clip"])))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            losses.append(sum(step_losses) / len(step_losses))
            append_jsonl(self.dir / "loss.jsonl", {"step": step, "loss": round(losses[-1], 5), "lr": scheduler.get_last_lr()[0],
                                                   "grad_norm": round(norm, 4), "samples": last - first, "tokens": tokens,
                                                   "seconds": round(time.time() - t0, 2), "max_mem_gb": peak()})
            if step < self.steps and (step % int(train["save_every_steps"]) == 0
                                      or time.time() - last_save > 60 * float(train["save_every_minutes"])):
                self._checkpoint(torch, lora, model, optimizer, step)
                last_save = time.time()
        prefetch.shutdown()
        sha = lora.save(model, self.dir / "adapter")
        history = [r["loss"] for r in read_jsonl(self.dir / "loss.jsonl")]
        from ...core.identity import code_version
        done = {"unit": self.unit, "hash": self.hash, "steps": self.steps, "samples": self.samples_total,
                "adapter_sha256": sha, "train_sha256": self.identity["train_sha256"],
                "test_sha256": self.identity["test_sha256"], "loss_first10": history[:10], "loss_last10": history[-10:],
                "seconds_this_run": round(time.time() - started), "peak_mem_gb": peak(),
                "lora": lora.describe(model), "code": code_version(), "python": sys.version.split()[0],
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if cuda else None,
                "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
        write_json(self.dir / "DONE.json", done)
        ckpt.unlink(missing_ok=True)
        return done

    def _checkpoint(self, torch, lora, model, optimizer, step: int):
        state = {"hash": self.hash, "step": step, "optimizer": optimizer.state_dict(),
                 "lora": {k: v.detach().cpu() for k, v in lora.lora_parameters(model).items()}}
        tmp = self.dir / f".ckpt.pt.{os.getpid()}.tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.dir / "ckpt.pt")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m streamreplace.generalization.finetune.train <payload.json>", file=sys.stderr)
        return 1
    payload = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    block = payload.get("paths")
    if block:   # roots come from the STREAMREPLACE_* variables; per-checkpoint / per-dataset folders from the payload
        from ... import paths as paths_module
        paths_module.set_current(paths_module.resolve(config_paths=block))
    unit = payload["unit"]
    trainer = Unit(payload)
    if (trainer.dir / "DONE.json").exists():
        print(f"{trainer.dir} done", flush=True)
        return 0
    trainer.dir.mkdir(parents=True, exist_ok=True)
    with (trainer.dir / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"{trainer.dir} is being trained by another process", flush=True)
            return 3
        print(f"== train {unit['model']} {unit['dataset']} {label(unit['budget'], unit['tag'])}: "
              f"{trainer.samples_total} samples, {trainer.steps} steps", flush=True)
        done = trainer.run()
    print(json.dumps({k: done[k] for k in ("steps", "adapter_sha256", "peak_mem_gb", "loss_last10", "pilot_stopped_at")
                      if k in done}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
