"""The extraction judge (configs/judges.yaml `extract`): an instruction-tuned LLM served by vLLM, thinking disabled,
temperature 0. It serves the benchmarks with JUDGE "fallback" (only rows whose rule parse found no answer) and
"always" (VideoReasonBench, every parsed answer). Three request kinds, all cached under the original keys
{kind, question, choices, response, answer, meta}:

  extract  the generic extraction prompt below (one user message, 64 tokens); the reply is the extracted answer
           ("none" or empty = no answer), scored by the benchmark's own rules as "Final answer: <reply>"
  chat     a benchmark's official extraction prompt (TOMATO's gpt_parser, TimeBlind's llm_judge, the VMDT fairness
           fallback) sent verbatim with the benchmark's max_tokens; the raw reply goes to its judge_result()
  vrb      VideoReasonBench's official judge and state checkers (judges/vrb.py); the verdict decides correctness

Gating (the paper's rule): a request is made only for a row without error, with non-empty raw text, not silent (parse
"no_answer" is wrong and never judged) and either FALLBACK with no rule prediction or ALWAYS with one. The judge always
sees the raw, uncleaned reply (VideoReasonBench: the extracted final answer). A judge answer then fills judge_pred /
judge_correct; the benchmark's final correctness is judge_correct when set, else the rule's `correct`.

The paper used a Qwen3.5-architecture 27B FP8 instruct model (configs/judges.yaml records its Hugging Face id and
revision). The cache key does not name the judge; the cache file name (config hash) does, and with `server.url` it
also covers that endpoint and the model name it is asked for."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ..benchmarks.base import JudgePolicy
from .base import EXTRACT_KEY_FIELDS, JudgeRequest, cache_key
from .registry import ConfiguredJudge

EXTRACT_TOKENS = 64
VRB_META = ("task", "states", "coords", "visible_time", "num_state")


def extraction_prompt(request: dict) -> str:
    """The generic extraction prompt (unchanged since the paper's first scoring run)."""
    q = request["question"]
    if request["choices"]:
        options = "\n".join(f"{k}. {v}" for k, v in request["choices"].items())
        return ("You are given a multiple-choice question, its options and a model's response. Identify the single option "
                "the response chooses as its final answer. Reply with only the option letter. If the response does not "
                f"clearly choose exactly one option, reply None.\n\nQuestion: {q}\nOptions:\n{options}\n\n"
                f"Response: {request['response']}\n\nAnswer:")
    if request["meta"].get("kind") == "yesno":
        return ("You are given a yes/no question and a model's response. Reply with only yes or no according to the "
                f"response's final answer, or None if it gives neither.\n\nQuestion: {q}\n\nResponse: {request['response']}\n\nAnswer:")
    return ("You are given a question and a model's response. Extract the final answer the response gives (a number, "
            "expression, list or short phrase) exactly as written, with no explanation. If there is no final answer, "
            f"reply None.\n\nQuestion: {q}\n\nResponse: {request['response']}\n\nFinal answer:")


def request_key(request: dict) -> str:
    return cache_key({k: request[k] for k in EXTRACT_KEY_FIELDS})


def make_request(bench, sample, gen: dict, score: dict) -> dict | None:
    """The judge request one scored row needs, with its "key", or None (gating in the module docstring)."""
    # "no_answer" = stream silence or an official NA reply: wrong, never judged. A rule-parse miss ("no_marker") is.
    if sample is None or gen.get("error") or not gen.get("raw") or score.get("parse") == "no_answer":
        return None
    policy = JudgePolicy.of(bench.JUDGE)
    if policy is JudgePolicy.ALWAYS:
        if score["pred"] is None:
            return None
        request = {"kind": "vrb", "question": sample.question, "choices": {}, "response": score["pred"],
                   "answer": sample.answer, "meta": {k: sample.metadata.get(k) for k in VRB_META}}
    elif policy is JudgePolicy.FALLBACK and score["pred"] is None:
        official = bench.judge_messages(sample, gen["raw"])
        if official:                     # the benchmark's own extraction prompt, sent verbatim; reply post-processed
            messages, max_tokens = official
            request = {"kind": "chat", "question": sample.question, "choices": sample.choices, "response": gen["raw"],
                       "answer": "", "meta": {"messages": messages, "max_tokens": max_tokens}}
        else:
            request = {"kind": "extract", "question": sample.question, "choices": sample.choices,
                       "response": gen["raw"], "answer": "", "meta": {"kind": sample.kind}}
    else:
        return None
    return {**request, "key": request_key(request)}


def apply_result(bench, sample, score: dict, result: dict) -> dict:
    """The score row with a cached answer applied (the benchmark parses the extracted answer itself)."""
    score = dict(score)
    if "verdict" in result:
        score["judge_correct"] = bool(result["verdict"])
    elif "reply" in result:
        prediction = bench.judge_result(sample, result["reply"])
        score.update(judge_pred=prediction, judge_correct=bench.correct(sample, prediction))
    else:
        extracted = result["extracted"]
        prediction = bench.parse(bench.view(sample), f"Final answer: {extracted}")[0] if extracted else None
        score.update(judge_pred=prediction, judge_correct=bench.correct(sample, prediction))
    return score


class ExtractionJudge(ConfiguredJudge):
    NAME = "extract"
    ENV = "judge"
    PROMPT_VERSION = 1

    def identity(self) -> dict:
        """What the answers depend on. With server.url (an endpoint you run) the endpoint and the model name it is
        asked for are part of it: its answers are never filed under the configured checkpoint's cache."""
        ident = {"judge": self.NAME, "checkpoint": self.spec.checkpoint, "backend": self.spec.backend,
                 "decoding": {"temperature": self.spec.option("client", "temperature", default=0),
                              "enable_thinking": bool(self.spec.option("client", "enable_thinking", default=False))},
                 "prompts": self.PROMPT_VERSION}
        url = self.spec.option("server", "url")
        if url:
            from .server import SERVER_DEFAULTS
            model = (self.spec.option("server", "model") or self.spec.option("server", "served_model_name")
                     or SERVER_DEFAULTS["served_model_name"])          # the name server.JudgeServer asks for
            ident["endpoint"] = {"url": str(url), "model": str(model)}
        return ident

    def requests(self, bench, sample, gen: dict, score: dict) -> list[JudgeRequest]:
        request = make_request(bench, sample, gen, score)
        return [] if request is None else [JudgeRequest(self.NAME, request["key"], request)]

    def apply(self, bench, sample, score: dict, result: dict) -> dict:
        return apply_result(bench, sample, score, result)

    def process_spec(self) -> dict:
        url = self.spec.option("server", "url")
        return {"checkpoint_dir": None if url else str(self.checkpoint_dir()),
                "server": dict(self.spec.option("server", default={}) or {}),
                "client": dict(self.spec.option("client", default={}) or {}),
                "log": str(self.paths.outputs / "logs" / "judges" / f"{self.NAME}.{self.config_hash()}.vllm.log")}

    def run(self, reqs: list[JudgeRequest], cache, gpus: list[int]) -> None:
        self.launch([r.payload for r in reqs], cache, gpus, self.process_spec())

    def self_test(self, gpus: list[int]) -> None:
        """The extraction judge has no frozen reference set, so there is nothing to check (the scoring never depends
        on re-asking it: cached answers are always applied)."""
        return None


def serve(requests: list[dict], cache, spec: dict) -> None:
    """Judge-process side (judges/entry.py): answers every request and appends {"key", "extracted" | "reply" |
    "verdict"} to the cache; a request that fails after the retries is reported and not cached."""
    import os
    from pathlib import Path
    from .server import JudgeServer
    from .vrb import setup, verdict
    # VideoReasonBench's utilities read these before any call we route; never used for a real API
    os.environ.setdefault("OPENAI_API_KEY", "local")
    os.environ.setdefault("OPENAI_API_BASE", "http://127.0.0.1/v1")
    server = JudgeServer(spec.get("checkpoint_dir"), spec.get("server"), spec.get("client"),
                         Path(spec["log"]) if spec.get("log") else None)
    try:
        u = setup(server) if any(r["kind"] == "vrb" for r in requests) else None

        def handle(request):
            try:
                if request["kind"] == "vrb":
                    result = {"verdict": verdict(u, request)}
                elif request["kind"] == "chat":        # a benchmark's official extraction prompt; raw reply returned
                    result = {"reply": server.complete(request["meta"]["messages"], request["meta"]["max_tokens"])}
                else:
                    text = server.chat([{"type": "text", "text": extraction_prompt(request)}], max_tokens=EXTRACT_TOKENS)
                    result = {"extracted": None if text.strip().lower() in {"none", ""} else text.strip()}
            except Exception as exc:        # not cached: asked again on the next scoring pass
                print(f"judge error {request['key']}: {exc}", flush=True)
                return
            cache.append({"key": request["key"], **result})

        with ThreadPoolExecutor(server.in_flight) as pool:
            for number, _ in enumerate(pool.map(handle, requests), 1):
                if number % 200 == 0:
                    print(f"judge: {number}/{len(requests)}", flush=True)
    finally:
        server.close()
