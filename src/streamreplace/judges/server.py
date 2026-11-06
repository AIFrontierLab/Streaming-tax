"""The extraction judge's model server: local vLLM OpenAI-compatible servers (one replica per GPU, data parallel; the
judge fits on one card), or any OpenAI-compatible endpoint you already run.

Settings (configs/judges.yaml, extract.server / extract.client) default to what the paper's scoring used:
`vllm serve <checkpoint> --served-model-name judge --max-model-len 32768 --gpu-memory-utilization 0.9
--limit-mm-per-prompt '{"image": 0, "video": 0}' --max-num-seqs 64` with VLLM_ENGINE_READY_TIMEOUT_S=3600 and
VLLM_USE_FLASHINFER_SAMPLER=0; requests with temperature 0 and thinking disabled
(chat_template_kwargs.enable_thinking = false), at most 32 in flight per replica, 5 attempts 5 s apart.
Runs inside the judge environment (needs the `openai` client and, for local servers, `vllm`)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from typing import Mapping

SERVER_DEFAULTS = {"served_model_name": "judge", "port": 8031, "max_model_len": 32768, "gpu_memory_utilization": 0.9,
                   "limit_mm_per_prompt": {"image": 0, "video": 0},       # text-only judging
                   "max_num_seqs": 64,             # the hybrid-attention cache limits concurrency; we send <= 32
                   "ready_polls": 900, "ready_interval_s": 2.0,
                   # first start compiles and weight loading from network storage is slow; greedy only, so no
                   # FlashInfer sampler (and no FlashInfer JIT against an older system CUDA)
                   "env": {"VLLM_ENGINE_READY_TIMEOUT_S": "3600", "VLLM_USE_FLASHINFER_SAMPLER": "0"}}
CLIENT_DEFAULTS = {"temperature": 0, "enable_thinking": False, "in_flight_per_replica": 32, "attempts": 5,
                   "retry_sleep_s": 5.0}


def free_port(preferred: int) -> int:
    """`preferred` if nothing is bound to it, else a port the OS assigns. Two judges on one host must not share a
    port: the second vLLM server would exit, and its client could talk to the other job's server."""
    for candidate in (preferred, 0):
        with socket.socket() as probe:
            try:
                probe.bind(("", candidate))
                return probe.getsockname()[1]
            except OSError:
                continue
    raise OSError("no free port")


class JudgeServer:
    """Chat completions against local vLLM replicas (one per visible GPU) or an external endpoint (server.url)."""

    def __init__(self, checkpoint: str | None, server: Mapping | None = None, client: Mapping | None = None,
                 log: Path | None = None):
        from openai import OpenAI
        self.server = {**SERVER_DEFAULTS, **(server or {})}
        self.client_options = {**CLIENT_DEFAULTS, **(client or {})}
        self.processes: list[subprocess.Popen] = []
        self.clients = []
        url = self.server.get("url")
        if url:                                     # an endpoint someone else runs
            self.model = self.server.get("model") or self.server["served_model_name"]
            self.clients.append(OpenAI(base_url=url, api_key=os.environ.get("OPENAI_API_KEY", "local")))
        else:
            if not checkpoint:
                raise ValueError("extraction judge: no checkpoint folder and no server.url")
            self.model = self.server["served_model_name"]
            self._start(checkpoint, log or Path("judge-vllm.log"))
        self._next, self._lock = 0, threading.Lock()

    def _start(self, checkpoint: str, log: Path) -> None:
        from openai import OpenAI
        devices = [d for d in (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",") if d != ""]
        vllm = str(Path(sys.executable).parent / "vllm")
        log.parent.mkdir(parents=True, exist_ok=True)
        for index, device in enumerate(devices):
            port = free_port(int(self.server["port"]) + index)
            out = log.with_name(f"{log.stem}{'' if index == 0 else f'-{index}'}{log.suffix}").open("a")
            self.processes.append(subprocess.Popen(
                [vllm, "serve", checkpoint, "--served-model-name", self.model, "--port", str(port),
                 "--max-model-len", str(self.server["max_model_len"]),
                 "--gpu-memory-utilization", str(self.server["gpu_memory_utilization"]),
                 "--limit-mm-per-prompt", json.dumps(self.server["limit_mm_per_prompt"]),
                 "--max-num-seqs", str(self.server["max_num_seqs"])],
                stdout=out, stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": device,
                     **{k: str(v) for k, v in (self.server.get("env") or {}).items()}}))
            self.clients.append(OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="local"))
        for process, client in zip(self.processes, self.clients):          # the replicas load in parallel
            for _ in range(int(self.server["ready_polls"])):
                try:
                    client.models.list()
                    break
                except Exception:
                    if process.poll() is not None:
                        self.close()
                        raise RuntimeError(f"judge server exited; see {log.parent}/{log.stem}*.log")
                    time.sleep(float(self.server["ready_interval_s"]))
            else:
                self.close()
                raise RuntimeError("judge server did not start")

    @property
    def replicas(self) -> int:
        return len(self.clients)

    @property
    def in_flight(self) -> int:
        return int(self.client_options["in_flight_per_replica"]) * self.replicas

    @property
    def client(self):
        with self._lock:
            self._next += 1
            return self.clients[self._next % len(self.clients)]

    def chat(self, contents, max_tokens: int = 64) -> str:
        """One user turn (`contents`: a string or a list of content parts)."""
        return self.complete([{"role": "user", "content": contents}], max_tokens)

    def complete(self, messages, max_tokens: int = 64) -> str:
        """The stripped reply text; retried, then raises (the caller leaves the request uncached)."""
        attempts = int(self.client_options["attempts"])
        for attempt in range(attempts):
            try:
                out = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=self.client_options["temperature"],
                    max_tokens=max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": bool(self.client_options["enable_thinking"])}})
                return (out.choices[0].message.content or "").strip()
            except Exception:
                if attempt == attempts - 1:
                    raise
                time.sleep(float(self.client_options["retry_sleep_s"]))
        return ""

    def close(self) -> None:
        for process in self.processes:
            process.terminate()
        for process in self.processes:
            process.wait(120)
