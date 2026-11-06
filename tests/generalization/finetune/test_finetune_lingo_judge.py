"""The LingoQA judge (judges/lingo.py) without its environment: cache keys byte-identical to the paper's runs, requests
for pending rows only, the score-row update, the judge process with a stand-in for LingoQA's benchmark/judge.py, and
LingoQA rows scored end to end from a cache."""
import sys
import types

import pytest

from streamreplace import paths as paths_module
from streamreplace.core.io import read_jsonl, write_json, write_jsonl
from streamreplace.core.schema import Sample, score_row
from streamreplace.judges import entry, lingo
from streamreplace.judges.base import JudgeRequest
from streamreplace.judges.cache import JudgeCache
from streamreplace.judges.registry import JudgeSpec, build, load_specs

SPEC = JudgeSpec(name="lingo", env="lingo", backend="hf", checkpoint=lingo.JUDGE)
LIGHTS = Sample("q1", "Lights?", "Yes, green.", "open", metadata={"segment_id": "s1", "references": ["Yes, green.", "Green."]})

# A stand-in for LingoQA's benchmark/judge.py: the same class name, constructor and compute() signature.
STUB = '''
class LingoJudge:
    loaded = []

    def __init__(self, pretrained_model):
        LingoJudge.loaded.append(pretrained_model)
        self.device = None

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def compute(self, questions, references, predictions):
        assert len(questions) == len(references) == len(predictions) == 1      # batch size 1, as evaluate.py
        return [max(1.5 if predictions[0].lower() in r.lower() and predictions[0] else -2.0 for r in references[0])]
'''


def test_cache_keys_equal_the_paper_runs():
    assert lingo.request_key("Lights?", ["Yes, green.", "Green."], "Green") == "a8bc58ffdd6b8bfc"
    assert lingo.request_key("Is the road wet?", ["Yes.", "Yes, it is raining."], "") == "01ab4319ecbc0ac0"
    assert lingo.request_key("Ampel?", ["Grün – ja."], "grün") == "aaf192643a48a63d"   # non-ASCII as is


def test_the_shipped_configuration_builds_the_judge():
    spec = load_specs()["lingo"]
    assert spec.checkpoint == lingo.JUDGE and spec.env == "lingo"
    judge = build("lingo")
    assert isinstance(judge, lingo.LingoJudge) and judge.env == "lingo"
    assert judge.identity() == {"judge": "lingo", "checkpoint": lingo.JUDGE, "backend": "hf", "dtype": "float32",
                                "batch_size": 1, "prompts": 1}
    assert len(judge.config_hash()) == 16 and judge.config_hash() == lingo.LingoJudge(SPEC).config_hash()
    with pytest.raises(ValueError, match="only 'hf'"):
        lingo.LingoJudge(JudgeSpec(name="lingo", env="lingo", backend="vllm", checkpoint=lingo.JUDGE))


def test_requests_are_made_for_pending_rows_only():
    judge = lingo.LingoJudge(SPEC)
    pending = {**score_row("q1"), "pred": "Green", "parse": "pending"}
    [request] = judge.requests(None, LIGHTS, {"id": "q1", "raw": "Green"}, pending)
    assert request.judge == "lingo" and request.key == "a8bc58ffdd6b8bfc"
    assert request.payload == {"key": "a8bc58ffdd6b8bfc", "question": "Lights?", "references": ["Yes, green.", "Green."],
                               "prediction": "Green"}
    [empty] = judge.requests(None, LIGHTS, {"id": "q1", "raw": ""}, {**pending, "pred": None})
    assert empty.payload["prediction"] == ""                      # an empty reply is judged as an empty answer
    assert judge.requests(None, LIGHTS, {"id": "q1", "error": "x"}, score_row("q1")) == []     # errored: no request
    assert judge.requests(None, None, {"id": "q1"}, pending) == []


def test_the_judge_result_sets_the_official_decision():
    judge = lingo.LingoJudge(SPEC)
    pending = {**score_row("q1"), "pred": "Green", "parse": "pending"}
    row = judge.apply(None, LIGHTS, pending, {"key": "k", "score": 2.512349})
    assert row == {**pending, "judge_pred": 2.5123, "correct": True, "parse": "lingo"}
    assert judge.apply(None, LIGHTS, pending, {"key": "k", "score": 0.0})["correct"] is False     # > 0, not >= 0
    assert judge.apply(None, LIGHTS, pending, {"key": "k", "score": 0.00001})["judge_pred"] == 0.0
    assert judge.apply(None, LIGHTS, pending, {"key": "k", "score": 0.00001})["correct"] is True  # the unrounded score


def test_run_starts_one_process_on_the_first_gpu(tmp_path, monkeypatch):
    paths = paths_module.resolve(home=tmp_path, environ={})
    (paths.upstream / "lingoqa" / "benchmark").mkdir(parents=True)
    judge = lingo.LingoJudge(SPEC, paths)
    launched = []
    monkeypatch.setattr(judge, "launch", lambda payloads, cache, gpus, spec: launched.append((payloads, gpus, spec)))
    request = JudgeRequest("lingo", "k", {"key": "k", "question": "q", "references": ["r"], "prediction": "p"})
    monkeypatch.setattr(paths_module, "_current", paths)
    judge.run([request], judge.cache(tmp_path / "out"), [2, 3])
    [(payloads, gpus, spec)] = launched
    assert payloads == [request.payload] and gpus == [2]
    assert spec == {"checkpoint_dir": str(tmp_path / "checkpoints" / "wayveai" / "Lingo-Judge"),
                    "code": str(paths.upstream / "lingoqa" / "benchmark")}


def test_run_needs_the_fetched_lingoqa_code(tmp_path, monkeypatch):
    from streamreplace.models.upstream import UpstreamMissing
    paths = paths_module.resolve(home=tmp_path, environ={})
    monkeypatch.setattr(paths_module, "_current", paths)
    with pytest.raises(UpstreamMissing, match="fetch-upstream lingoqa"):
        lingo.LingoJudge(SPEC, paths).process_spec()


@pytest.fixture
def stub_judge(tmp_path, monkeypatch):
    """The stand-in judge.py in a benchmark/ folder and a torch module without CUDA; sys.path and sys.modules are
    restored afterwards."""
    code = tmp_path / "benchmark"
    code.mkdir()
    (code / "judge.py").write_text(STUB, encoding="utf-8")
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(sys, "path", list(sys.path))
    yield code
    sys.modules.pop("judge", None)


def test_the_judge_process_scores_with_the_official_class(tmp_path, stub_judge):
    requests = [lingo.request_for(LIGHTS, "green"), lingo.request_for(LIGHTS, "red"), lingo.request_for(LIGHTS, "")]
    cache = JudgeCache(tmp_path / "judges" / "lingo.x.jsonl")
    lingo.serve(requests, cache, {"checkpoint_dir": str(tmp_path / "ckpt"), "code": str(stub_judge)})
    judge_module = sys.modules["judge"]
    assert judge_module.LingoJudge.loaded == [str((tmp_path / "ckpt").resolve())]
    scores = {r["key"]: r["score"] for r in read_jsonl(cache.path)}
    assert scores == {requests[0]["key"]: 1.5, requests[1]["key"]: -2.0, requests[2]["key"]: -2.0}


def test_the_judge_process_entry_serves_uncached_requests_once(tmp_path, monkeypatch):
    cache = JudgeCache(tmp_path / "judges" / "lingo.x.jsonl")
    cache.add([{"key": "a", "score": 1.0}])
    write_jsonl(tmp_path / "requests.jsonl", [{"key": "a"}, {"key": "b"}, {"key": "b"}])
    write_json(tmp_path / "job.json", {"judge": "lingo", "requests": str(tmp_path / "requests.jsonl"),
                                       "cache": str(cache.path), "spec": {}})
    served = []
    monkeypatch.setattr(lingo, "serve", lambda reqs, c, spec: served.append(reqs))
    assert entry.main([str(tmp_path / "job.json")]) == 0 and served == [[{"key": "b"}]]


def test_lingoqa_rows_are_scored_from_the_judge_cache(tmp_path):
    from streamreplace.scoring.score import CacheMiss, score
    outputs = tmp_path / "outputs"
    samples = [LIGHTS, Sample("q2", "Wet road?", "Yes.", "open", metadata={"segment_id": "s1", "references": ["Yes."]}),
               Sample("q3", "Speed?", "Slow.", "open", metadata={"segment_id": "s2", "references": ["Slow."]})]
    write_jsonl(outputs / "lingoqa" / "m@lingoqa-0.offline.jsonl", [
        {"id": "q1", "raw": "<think>hm</think> Green", "error": None, "info": {}},
        {"id": "q2", "raw": "", "error": None, "info": {}},
        {"id": "q3", "raw": None, "error": "CUDA out of memory", "info": {}}])
    judge = lingo.LingoJudge(SPEC)
    run = lambda **kw: score(outputs, samples=lambda dataset: samples, judges={"lingo": judge}, log=None, **kw)
    with pytest.raises(CacheMiss):
        run(cache_only=True)
    judge.cache(outputs).add([{"key": lingo.request_key("Lights?", ["Yes, green.", "Green."], "Green"), "score": 2.0},
                              {"key": lingo.request_key("Wet road?", ["Yes."], ""), "score": -0.7}])
    report = run(cache_only=True)
    assert report.requests == {"lingo": 2} and report.misses == {"lingo": 0}
    rows = {r["id"]: r for r in read_jsonl(outputs / "lingoqa" / "m@lingoqa-0.offline.scores.jsonl")}
    assert rows["q1"] == {"id": "q1", "pred": "Green", "parse": "lingo", "correct": True, "judge_pred": 2.0,
                          "judge_correct": None, "judge": {"name": "lingo", "key": "a8bc58ffdd6b8bfc"}}
    assert rows["q2"]["pred"] is None and rows["q2"]["correct"] is False and rows["q2"]["judge_pred"] == -0.7
    assert rows["q3"] == score_row("q3")                          # errored: never judged, counted wrong
    assert report.jobs[0]["correct"] == 1 and report.jobs[0]["counted"] == 3
