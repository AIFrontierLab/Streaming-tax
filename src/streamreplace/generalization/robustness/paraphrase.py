"""Paraphrases for the `para` condition of the prompt-robustness experiment (Supp. Table eval_robustness, the
paraphrase column of Table 2): back-translation through a pivot language, verified by rules and by a language model.

Per unit (PromptRobust's semantic level writes the prompt in another language and translates it back to English):
  1. translate the question stem into the pivot language (Chinese in the paper), then back into English;
  2. the candidate must pass perturb.paraphrase_problems (too similar, numbers changed, length, adds the correct
     option's text) and then get "yes" to: do the two versions ask the same thing, with the same correct option?
  3. if the back-translation fails either check, one direct "reword" request goes through the same two checks;
  4. if that also fails, the unit is skipped for every model (paper: 21 of 600 units, leaving 294 Video-MME and 285
     MMBench units; 319 back-translations and 260 direct rewordings were accepted).
Results are appended to `<root>/paraphrase_cache.jsonl` as each request finishes (an interrupted run resumes):
{key, ok, text, reason, method, pivot_text, back, direct, problems, verdicts, verdict}; edits.py reads them by key.

The model is any OpenAI-compatible chat endpoint (`paraphrase.endpoint` in the specification, e.g. a local
`vllm serve <checkpoint> --served-model-name judge`), asked with temperature 0, thinking disabled
(chat_template_kwargs.enable_thinking = false) and at most 1,024 new tokens, one user message per request. The paper
used the same Qwen3.5-family 27B FP8 instruct checkpoint as the extraction judge (configs/judges.yaml) on a local
vLLM server. NOT EXACTLY REGENERABLE: another model, checkpoint, serving stack or GPU gives other paraphrases; the
paper's prompts are fixed by the frozen override files (their sha256 in paper/frozen/robustness/edits.json)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading
import time
from typing import Callable
import urllib.request

from . import perturb

TO_PIVOT = ("Translate the following question into {pivot}. Keep names, numbers and quoted text unchanged. "
            "Output only the translation.\n\n{text}")
TO_ENGLISH = "Translate the following text into English. Output only the translation.\n\n{text}"
REWORD = ("Rewrite the following question using different words so that it asks exactly the same thing. Keep names, "
          "numbers and quoted text unchanged, do not add or remove information, and do not answer it. Output only the "
          "rewritten question.\n\n{text}")
SAME = ("Below are two versions of a question about the same image or video, followed by its answer options.\n\n"
        "Version 1: {a}\n\nVersion 2: {b}\n\nOptions:\n{options}\n\nDo the two versions ask exactly the same thing, so "
        "that the correct option is the same for both? Answer yes or no.")
DEFAULTS = {"min_change": 0.2, "endpoint": None, "model": "judge", "api_key_env": "OPENAI_API_KEY",
            "max_tokens": 1024, "temperature": 0, "enable_thinking": False, "threads": 32, "attempts": 5,
            "retry_sleep_s": 5.0, "timeout_s": 600}


def settings(spec: dict) -> dict:
    return {**DEFAULTS, **(spec.get("paraphrase") or {})}


def requests(exp) -> list[dict]:
    """One request per (para condition, dataset, unit) of the condition folders' subsets."""
    from .edits import options_text, paraphrase_key, read_cache
    from .make import DATASETS
    min_change = settings(exp.spec)["min_change"]
    out = {}
    for condition, c in exp.conditions.items():
        if c.get("edit") != "para":
            continue
        if exp.subset(condition) is None:
            raise ValueError(f"{exp.folder(condition)}: no subset.json; run `streamreplace robustness subset` first")
        for dataset in exp.datasets(condition):
            for unit, samples in exp.units(dataset, condition).items():
                s = samples[0]
                stem = DATASETS[dataset].stem(s)
                k = paraphrase_key(dataset, unit, stem, c["pivot"])
                out[k] = {"key": k, "stem": stem, "pivot": c["pivot"], "options": options_text(s),
                          "answer_text": (s.choices or {}).get(s.answer, ""), "min_change": min_change}
    done = set(read_cache(exp))
    return [r for r in out.values() if r["key"] not in done]


def clean_reply(text: str) -> str:
    """Drops a 'Translation:'-style label and wrapping quotes the model sometimes adds."""
    text = text.strip()
    for label in ("Translation:", "Rewritten question:", "Question:"):
        if text.lower().startswith(label.lower()):
            text = text[len(label):].strip()
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'“”":
        text = text[1:-1].strip()
    return text.strip("“”").strip()


def paraphrase(chat: Callable[[str], str], request: dict) -> dict:
    """chat(prompt) -> reply. The full procedure for one request (no I/O; testable with a stub chat)."""
    stem, answer = request["stem"], request["answer_text"]
    pivot_text = clean_reply(chat(TO_PIVOT.format(pivot=request["pivot"], text=stem)))
    candidates = {"back_translation": lambda: clean_reply(chat(TO_ENGLISH.format(text=pivot_text))),
                  "direct": lambda: clean_reply(chat(REWORD.format(text=stem)))}
    result = {"key": request["key"], "pivot_text": pivot_text, "back": None, "direct": None, "problems": {},
              "verdicts": {}, "verdict": None}
    for method, ask in candidates.items():
        text = ask()
        result[{"back_translation": "back", "direct": "direct"}[method]] = text
        result["problems"][method] = problems = perturb.paraphrase_problems(stem, text, answer, request["min_change"])
        if problems:
            continue
        reply = chat(SAME.format(a=stem, b=text, options=request["options"])).strip().lower()
        result["verdicts"][method] = result["verdict"] = "yes" if reply.startswith("yes") else "no"
        if result["verdict"] == "yes":
            return {**result, "method": method, "ok": True, "text": text, "reason": None}
    last = result["problems"]["direct"]
    reason = "rules: " + ", ".join(last) if last else "judged not equivalent"
    return {**result, "method": "direct", "ok": False, "text": None, "reason": reason}


class Endpoint:
    """chat(prompt) -> the stripped reply of an OpenAI-compatible /chat/completions endpoint (stdlib HTTP)."""

    def __init__(self, url: str, model: str = "judge", api_key: str | None = None, max_tokens: int = 1024,
                 temperature: float = 0, enable_thinking: bool = False, attempts: int = 5, retry_sleep_s: float = 5.0,
                 timeout_s: float = 600):
        self.url = url.rstrip("/") + "/chat/completions"
        self.model, self.api_key, self.max_tokens, self.temperature = model, api_key, max_tokens, temperature
        self.enable_thinking, self.attempts, self.retry_sleep_s, self.timeout_s = (enable_thinking, attempts,
                                                                                  retry_sleep_s, timeout_s)

    @classmethod
    def from_settings(cls, options: dict) -> Endpoint:
        if not options.get("endpoint"):
            raise ValueError("no paraphrase endpoint: set paraphrase.endpoint in the specification (or configs/"
                             "local.yaml `robustness: {paraphrase: {endpoint: ...}}`) or pass --endpoint")
        return cls(options["endpoint"], options["model"], os.environ.get(options["api_key_env"] or ""),
                   options["max_tokens"], options["temperature"], options["enable_thinking"], options["attempts"],
                   options["retry_sleep_s"], options["timeout_s"])

    def body(self, prompt: str) -> dict:
        """The request body: one user message, greedy, thinking disabled through the chat template."""
        return {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature, "max_tokens": self.max_tokens,
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking}}

    def __call__(self, prompt: str) -> str:
        data = json.dumps(self.body(prompt)).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for attempt in range(self.attempts):
            try:
                request = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    reply = json.loads(response.read().decode())
                return (reply["choices"][0]["message"].get("content") or "").strip()
            except Exception:
                if attempt == self.attempts - 1:
                    raise
                time.sleep(self.retry_sleep_s)
        return ""


def run(pending: list[dict], cache_path: Path, chat: Callable[[str], str], threads: int = 32,
        log: Callable[[str], None] = print) -> dict:
    """Paraphrases every pending request in `threads` parallel requests, appending each result to the cache as it
    finishes; a request that raises is reported and left for the next run."""
    lock, failed = threading.Lock(), []
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    def handle(request):
        try:
            result = paraphrase(chat, request)
        except Exception as exc:                        # asked again on the next run
            log(f"paraphrase error {request['key']}: {exc}")
            failed.append(request["key"])
            return
        with lock, cache_path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(result, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max(1, threads)) as pool:
        for number, _ in enumerate(pool.map(handle, pending), 1):
            if number % 100 == 0:
                log(f"paraphrase: {number}/{len(pending)}")
    return {"requests": len(pending), "errors": len(failed)}


def generate(exp, endpoint: str | None = None, model: str | None = None, threads: int | None = None,
             log: Callable[[str], None] = print) -> dict:
    """Paraphrases every unit of the para conditions that the cache does not hold yet; returns counts."""
    from ...core.io import read_jsonl
    from .edits import PARAPHRASE_CACHE
    options = settings(exp.spec)
    options.update({k: v for k, v in (("endpoint", endpoint), ("model", model), ("threads", threads)) if v})
    pending = requests(exp)
    cache_path = Path(exp.root) / PARAPHRASE_CACHE
    status = {"new": len(pending)}
    if pending:
        log(f"paraphrase: {len(pending)} requests to {options['endpoint']}")
        status.update(run(pending, cache_path, Endpoint.from_settings(options), int(options["threads"]), log))
    rows = read_jsonl(cache_path)
    return {**status, "cached": len(rows), "ok": sum(bool(r["ok"]) for r in rows),
            "failed": sum(not r["ok"] for r in rows),
            "methods": {m: sum(1 for r in rows if r["ok"] and r["method"] == m) for m in ("back_translation", "direct")}}
