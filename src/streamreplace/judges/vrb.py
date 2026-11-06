"""VideoReasonBench's official judge (supplement benchmark; JUDGE "always"): answer questions go to the official
LLM judge prompt (get_judge_prompt / llm_judge_api, reply "Correct" = right), operation questions to the official
state checkers (eval_op_hrd / grid / cup / file / card / chip, whose move extraction also asks the judge). The code
is the fetched llyx97/video_reason_bench utils/eval_utils.py (manifests/upstream.yaml, with the pinned patch that
parses move lists with ast.literal_eval), used unchanged except that its OpenAI call is routed to the extraction
judge's server (thinking off, temperature 0). Runs inside the judge environment."""
from __future__ import annotations

import ast


def setup(server):
    """Imports the official utilities and routes their OpenAI calls to `server` (judges/server.JudgeServer)."""
    from ..models.upstream import use_upstream
    with use_upstream("videoreasonbench"):
        from utils import eval_utils

    def completion(model, contents, max_tokens=1024, max_try=5, temperature=0.):
        return server.chat(contents, max_tokens=max_tokens), None
    eval_utils.test_chat_completion_openai = completion
    return eval_utils


def verdict(u, request: dict) -> bool:
    """Mirrors the official evaluation: answer-based questions via llm_judge; operation questions via state checkers."""
    response, meta = request["response"], request["meta"]
    if request["answer"]:
        return bool(u.llm_judge_api(request["question"], response, request["answer"], "judge"))
    states = ast.literal_eval(meta["states"]) if isinstance(meta["states"], str) else meta["states"]
    coords = ast.literal_eval(meta["coords"]) if isinstance(meta["coords"], str) else meta["coords"]
    index = 0 if meta["visible_time"] == "end" else -1
    task, n = meta["task"], int(meta["num_state"])
    if task == "hrd":
        return bool(u.eval_op_hrd(response, request["question"], states[index], "judge", None))
    if task == "grid":
        return bool(u.eval_op_grid(response, request["question"], states[index], coords[index], "judge", None))
    if task == "cup":
        return bool(u.eval_op_cup(response, request["question"], states[index], "judge", None))
    if task == "file_sys":
        # official questions/file_sys.json: path -> {command: files}; the parquet release keeps only the file lists
        # (same order), and eval_op_file only reads the values
        src = {f"path{i}": (dict([list(states[f"path{i}"].items())[index]]) if isinstance(states[f"path{i}"], dict)
                            else {"state": states[f"path{i}"][index]}) for i in range(n)}
        return bool(u.eval_op_file(response, request["question"], src, "judge", None))
    prefix = {"card": "pile", "chip": "cup"}[task]
    src = {f"{prefix}{i}": states[f"{prefix}{i}"][index] for i in range(n)}
    return bool({"card": u.eval_op_card, "chip": u.eval_op_chip}[task](response, request["question"], src, "judge", None))
