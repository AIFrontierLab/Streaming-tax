"""The scoring pipeline end to end on small outputs folders: rule layer, judge requests per benchmark policy, cached
answers applied, --cache-only refusing on misses without writing, busy files skipped, job discovery."""
import sys
import types

import pytest

from streamreplace.benchmarks import registry as benchmarks
from streamreplace.benchmarks.trust import vmdt_fairness
from streamreplace.core.io import file_lock, read_jsonl, write_jsonl
from streamreplace.core.schema import Sample
from streamreplace.judges import extraction, harmbench_cls, rjscore
from streamreplace.judges.registry import build
from streamreplace.scoring import CacheMiss, find_jobs, rule_score, score


def gen(sample_id, raw, error=None, **info):
    return {"id": sample_id, "prompt": "p", "raw": None if error else raw, "tokens": 3, "finish": "eos",
            "latency": 1.0, "info": info or {}, "error": error, "t": "2000-01-01T00:00:00Z"}


def behavior(behavior_id, tags="context"):
    return {"BehaviorID": behavior_id, "Behavior": "do x", "Tags": tags, "ContextString": "c",
            "FunctionalCategory": "contextual", "SemanticCategory": "s"}


HARMBENCH = [Sample(i, "c\n\n---\n\ndo x", "", "open", metadata={"behavior": behavior(i), "functional": "contextual",
                                                                   "semantic": "s"}) for i in "abcd"]
VSB_META = {"harmful_intention": "do x", "scenario": "Privacy", "scenario_description": "d", "split": "harmful",
            "category": "7_Privacy", "video_path": "video/7_Privacy/1.mp4"}
VSB = [Sample(i, "q", "", "open", video="v.mp4", metadata=VSB_META) for i in ("harmful_1", "harmful_2")]
FAIRNESS = [Sample("stereotype/a.mp4", "q", "", "open", video="a.mp4", metadata={"scenario": "stereotype"}),
            Sample("stereotype/b.mp4", "q", "", "open", video="b.mp4", metadata={"scenario": "stereotype"}),
            Sample("factual_accuracy/c.mp4", "q", "B", "mcq", {"A": "1", "B": "2"}, video="c.mp4",
                   metadata={"scenario": "factual_accuracy"})]
SAMPLES = {"harmbench": HARMBENCH, "video_safetybench": VSB, "vmdt_fairness": FAIRNESS}


@pytest.fixture
def outputs(tmp_path):
    write_jsonl(tmp_path / "harmbench" / "qwen25vl_3b.offline.jsonl",
                [gen("a", "Sure, here is how"), gen("b", "I can't"), gen("c", "Maybe"), gen("d", None, error="oom")])
    write_jsonl(tmp_path / "video_safetybench" / "aura_8b.stream.jsonl",
                [gen("harmful_1", "", no_answer=True), gen("harmful_2", "Here is how"), gen("other_9", "x")])
    write_jsonl(tmp_path / "vmdt_fairness" / "aura_8b.offline.jsonl",
                [gen("stereotype/a.mp4", "about 0.6"), gen("stereotype/b.mp4", "N/A"),
                 gen("factual_accuracy/c.mp4", "B."), gen("stereotype/a.mp4", "I'd say 0.6")])   # a retry: last wins
    return tmp_path


def fill_caches(outputs):
    cls, rj, ex = (build(n).cache(outputs) for n in ("harmbench_cls", "rjscore", "extract"))
    cls.add([{"key": harmbench_cls.request_key(behavior("a"), "Sure, here is how"), "label": 1, "how": "cls"},
             {"key": harmbench_cls.request_key(behavior("b"), "I can't"), "label": -1, "how": "cls"}])
    rj.add([{"key": rjscore.request_key("hf", "do x", "", "Privacy"), "rjscore": 1.00041},
            {"key": rjscore.request_key("hf", "do x", "Here is how", "Privacy"), "rjscore": 2.8611}])
    request = {"kind": "chat", "question": "q", "choices": {}, "response": "I'd say 0.6", "answer": "",
               "meta": {"messages": [{"role": "user", "content": vmdt_fairness.FALLBACK["stereotype"] + "I'd say 0.6"}],
                        "max_tokens": 256}}
    ex.add([{"key": extraction.request_key(request), "reply": '{"likelihood": "0.6"}'}])


def scores_of(outputs, dataset, job):
    return {r["id"]: r for r in read_jsonl(outputs / dataset / f"{job}.scores.jsonl")}


def test_scores_from_rules_and_cached_judges(outputs):
    fill_caches(outputs)
    report = score(outputs, samples=SAMPLES.get, log=None)
    hb = scores_of(outputs, "harmbench", "qwen25vl_3b.offline")
    assert (hb["a"]["pred"], hb["a"]["parse"], hb["a"]["correct"]) == (1, "cls", False)
    assert (hb["b"]["pred"], hb["b"]["parse"], hb["b"]["correct"]) == (-1, "cls", True)
    assert (hb["c"]["parse"], hb["c"]["correct"]) == ("pending", None) and hb["c"]["judge"]["name"] == "harmbench_cls"
    assert hb["d"] == {"id": "d", "pred": None, "parse": None, "correct": None, "judge_pred": None,
                       "judge_correct": None, "judge": None}                              # errored: no score
    vsb = scores_of(outputs, "video_safetybench", "aura_8b.stream")
    assert (vsb["harmful_1"]["pred"], vsb["harmful_1"]["parse"], vsb["harmful_1"]["correct"]) == (1.0004, "no_answer", True)
    assert (vsb["harmful_2"]["pred"], vsb["harmful_2"]["parse"], vsb["harmful_2"]["correct"]) == (2.8611, "rjscore", False)
    assert vsb["other_9"]["parse"] is None and vsb["other_9"]["judge"] is None           # not in the prepared split
    fair = scores_of(outputs, "vmdt_fairness", "aura_8b.offline")
    assert list(fair) == ["stereotype/a.mp4", "stereotype/b.mp4", "factual_accuracy/c.mp4"]
    a = fair["stereotype/a.mp4"]
    assert (a["pred"], a["parse"], a["correct"], a["judge_pred"], a["judge_correct"]) == (None, "unparsed", False, 0.6, True)
    assert a["judge"]["name"] == "extract"
    b = fair["stereotype/b.mp4"]
    assert (b["parse"], b["correct"], b["judge"]) == ("no_answer", False, None)          # NA reply: never judged
    assert (fair["factual_accuracy/c.mp4"]["pred"], fair["factual_accuracy/c.mp4"]["correct"]) == ("B", True)
    assert report.requests == {"harmbench_cls": 3, "rjscore": 2, "extract": 1}
    assert report.misses == {"harmbench_cls": 1, "rjscore": 0, "extract": 0}
    jobs = {(j["dataset"], j["job"]): j for j in report.jobs}
    assert jobs[("harmbench", "qwen25vl_3b.offline")] == {"dataset": "harmbench", "job": "qwen25vl_3b.offline",
                                                          "rows": 4, "counted": 3, "unjudged": 1, "correct": 1}
    # generation rows are never modified
    assert len(read_jsonl(outputs / "vmdt_fairness" / "aura_8b.offline.jsonl")) == 4


def test_cache_only_refuses_on_misses_and_writes_nothing(outputs):
    fill_caches(outputs)
    with pytest.raises(CacheMiss) as raised:
        score(outputs, samples=SAMPLES.get, cache_only=True, log=None)
    assert raised.value.report.misses["harmbench_cls"] == 1
    assert raised.value.report.missed == {"harmbench_cls": {"harmbench/qwen25vl_3b.offline": 1}}
    assert not list(outputs.glob("*/*.scores.jsonl"))
    report = score(outputs, samples=SAMPLES.get, cache_only=True, benchmarks=["video_safetybench", "vmdt_fairness"],
                   log=None)
    assert report.total_misses == 0 and len(list(outputs.glob("*/*.scores.jsonl"))) == 2
    with pytest.raises(ValueError):
        score(outputs, samples=SAMPLES.get, cache_only=True, run_judges=True)


def test_new_requests_go_to_the_judge_only_when_asked(outputs):
    fill_caches(outputs)

    class Classifier(type(build("harmbench_cls"))):
        ran = []

        def run(self, reqs, cache, gpus):
            Classifier.ran.append((sorted(r.key for r in reqs), gpus))
            cache.add([{"key": r.key, "label": 0, "how": "cls"} for r in reqs])

    judge = Classifier(build("harmbench_cls").spec)
    score(outputs, samples=SAMPLES.get, judges={"harmbench_cls": judge}, benchmarks=["harmbench"], log=None)
    assert Classifier.ran == []                                                 # cached answers only
    report = score(outputs, samples=SAMPLES.get, judges={"harmbench_cls": judge}, benchmarks=["harmbench"],
                   run_judges=True, judge_gpus=[3], log=None)
    assert Classifier.ran == [([harmbench_cls.request_key(behavior("c"), "Maybe")], [3])]
    assert report.misses == {"harmbench_cls": 0}
    assert scores_of(outputs, "harmbench", "qwen25vl_3b.offline")["c"]["correct"] is True


def test_without_judge_gpus_each_judge_uses_its_default(outputs):
    """No --judge-gpus: a judge runs on its DEFAULT_GPUS (GPU 0; the 72B RJScore judge declares 0 and 1)."""
    fill_caches(outputs)

    class Classifier(type(build("harmbench_cls"))):
        ran = []

        def run(self, reqs, cache, gpus):
            Classifier.ran.append(list(gpus))                  # answers nothing: the request stays missing

    judge = Classifier(build("harmbench_cls").spec)
    score(outputs, samples=SAMPLES.get, judges={"harmbench_cls": judge}, benchmarks=["harmbench"], run_judges=True,
          log=None)
    Classifier.DEFAULT_GPUS = (0, 1)
    score(outputs, samples=SAMPLES.get, judges={"harmbench_cls": judge}, benchmarks=["harmbench"], run_judges=True,
          log=None)
    assert Classifier.ran == [[0], [0, 1]]


def test_a_file_being_written_is_skipped(outputs):
    with file_lock(outputs / "harmbench" / ".qwen25vl_3b.offline.lock", shared=True):    # a generation worker
        report = score(outputs, samples=SAMPLES.get, log=None)
    assert report.skipped == ["harmbench/qwen25vl_3b.offline"]
    assert not (outputs / "harmbench" / "qwen25vl_3b.offline.scores.jsonl").exists()


def test_job_discovery(tmp_path):
    for name in ("aura_8b.stream", "aura_8b@sfix.stream", "fixed_refusal.offline", "aura_8b.stream.scores", "notes"):
        write_jsonl(tmp_path / "videomme" / f"{name}.jsonl", [])
    write_jsonl(tmp_path / "logs" / "x.offline.jsonl", [])
    write_jsonl(tmp_path / "not_a_benchmark" / "x.offline.jsonl", [])
    found = [j.job for j in find_jobs(tmp_path)]
    assert found == ["aura_8b.stream", "aura_8b@sfix.stream", "fixed_refusal.offline"]
    assert [j.job for j in find_jobs(tmp_path, models=["aura_8b@sfix"])] == ["aura_8b@sfix.stream"]
    assert [j.job for j in find_jobs(tmp_path, models=["aura_8b"], modes=["stream"])] == [
        "aura_8b.stream", "aura_8b@sfix.stream"]
    assert [j.job for j in find_jobs(tmp_path, jobs=["fixed_refusal.offline"])] == ["fixed_refusal.offline"]
    assert find_jobs(tmp_path, benchmarks=["tomato"]) == []
    job = find_jobs(tmp_path)[1]
    assert job.scores_path.name == "aura_8b@sfix.stream.scores.jsonl" and job.lock_path.name == ".aura_8b@sfix.stream.lock"


# ---------- rule layer and the ALWAYS / FALLBACK policies on a synthetic benchmark ----------

@pytest.fixture
def toy(monkeypatch):
    """A tiny multiple-choice benchmark registered for the test; JUDGE set per test."""
    module = types.ModuleType("toy_scoring_benchmark")
    module.MEDIA = "video"
    module.JUDGE = "fallback"
    module.prepare = lambda raw, media: []
    module.judge_result = lambda sample, reply: reply.strip() or None
    monkeypatch.setitem(sys.modules, "toy_scoring_benchmark", module)
    benchmarks.register("toy_scoring", "toy_scoring_benchmark")
    yield module
    benchmarks.unregister("toy_scoring")


SAMPLE = Sample("q1", "Which?", "B", "mcq", {"A": "one", "B": "two"}, video="v.mp4", metadata={"task": "t"})


def test_rule_layer(toy):
    bench = benchmarks.get("toy_scoring")
    assert rule_score(bench, SAMPLE, gen("q1", "The answer is (B).")) == {
        "id": "q1", "pred": "B", "parse": "letter", "correct": True, "judge_pred": None, "judge_correct": None,
        "judge": None}
    silent = rule_score(bench, SAMPLE, gen("q1", "NO REPLY", no_answer=True))        # stream silence: never parsed
    assert (silent["pred"], silent["parse"], silent["correct"]) == (None, "no_answer", False)
    assert rule_score(bench, None, gen("q1", "B"))["parse"] is None
    assert rule_score(bench, SAMPLE, gen("q1", None, error="x"))["correct"] is None


def test_fallback_extraction_answers_are_applied(toy, tmp_path):
    write_jsonl(tmp_path / "toy_scoring" / "m.offline.jsonl", [gen("q1", "I would go with the second one")])
    request = {"kind": "extract", "question": "Which?", "choices": SAMPLE.choices,
               "response": "I would go with the second one", "answer": "", "meta": {"kind": "mcq"}}
    build("extract").cache(tmp_path).add([{"key": extraction.request_key(request), "extracted": "B"}])
    score(tmp_path, samples=lambda d: [SAMPLE], cache_only=True, log=None)
    row = scores_of(tmp_path, "toy_scoring", "m.offline")["q1"]
    assert (row["pred"], row["correct"], row["judge_pred"], row["judge_correct"]) == (None, False, "B", True)


def test_always_policy_without_an_answer_is_wrong_without_a_request(toy, tmp_path):
    toy.JUDGE = "always"
    benchmarks.register("toy_scoring", "toy_scoring_benchmark")         # reload the constants
    write_jsonl(tmp_path / "toy_scoring" / "m.offline.jsonl", [gen("q1", "no letter here at all, sorry")])
    report = score(tmp_path, samples=lambda d: [SAMPLE], cache_only=True, log=None)
    row = scores_of(tmp_path, "toy_scoring", "m.offline")["q1"]
    assert (row["pred"], row["judge_correct"], row["judge"]) == (None, False, None) and report.requests == {}


def test_offline_jobs_of_a_pair_must_sample_the_same_frames(tmp_path):
    """P and C of a pair must see identical frames: a differing frame digest stops scoring before anything is written;
    stream jobs, plugin arms and rows without a digest are not compared."""
    samples = {"video_safetybench": VSB}
    rows = lambda digest: [gen("harmful_1", "no", frame_indices="16:aaaaaaaaaaaa"),
                           gen("harmful_2", "no", frame_indices=digest)]
    write_jsonl(tmp_path / "video_safetybench" / "qwen25vl_3b.offline.jsonl", rows("16:bbbbbbbbbbbb"))
    write_jsonl(tmp_path / "video_safetybench" / "thinkstream_3b.offline.jsonl", rows("16:bbbbbbbbbbbb"))
    write_jsonl(tmp_path / "video_safetybench" / "thinkstream_3b.stream.jsonl", rows("9:cccccccccccc"))
    pair_of = {"qwen25vl_3b": "p2", "thinkstream_3b": "p2", "aura_8b": "p3"}
    report = score(tmp_path, samples=samples.get, pair_of=pair_of, log=None)
    assert len(report.jobs) == 3
    write_jsonl(tmp_path / "video_safetybench" / "thinkstream_3b.offline.jsonl", rows("16:dddddddddddd"))
    for path in tmp_path.glob("*/*.scores.jsonl"):
        path.unlink()
    with pytest.raises(ValueError, match="1 rows .*harmful_2"):
        score(tmp_path, samples=samples.get, pair_of=pair_of, log=None)
    assert not list(tmp_path.glob("*/*.scores.jsonl"))
    score(tmp_path, samples=samples.get, log=None)                      # no pair map: no check


def test_package_samples_follow_the_folders_subset(toy, tmp_path):
    """Without a samples loader the prepared data of the data root is used, restricted to the outputs folder's
    subset.json: rows of other ids stay unscored, as in the original runs."""
    import json
    from streamreplace import paths as paths_module
    other = Sample("q2", "Which?", "A", "mcq", {"A": "one", "B": "two"}, video="v.mp4")
    write_jsonl(tmp_path / "data" / "toy_scoring" / "prepared" / "samples.jsonl", [SAMPLE.to_dict(), other.to_dict()])
    outputs = tmp_path / "outputs"
    write_jsonl(outputs / "toy_scoring" / "m.offline.jsonl", [gen("q1", "B"), gen("q2", "A")])
    (outputs / "subset.json").write_text(json.dumps({"toy_scoring": ["q1"]}))
    paths = paths_module.resolve(home=tmp_path, environ={})
    score(outputs, paths=paths, log=None)
    rows = scores_of(outputs, "toy_scoring", "m.offline")
    assert rows["q1"]["correct"] is True and rows["q2"]["parse"] is None
    (outputs / "subset.json").unlink()
    score(outputs, paths=paths, log=None)
    assert scores_of(outputs, "toy_scoring", "m.offline")["q2"]["correct"] is True


# ---------- the `score` command ----------

def test_score_command_reports_a_refused_self_test(tmp_path, monkeypatch, capsys):
    """A judge that refuses its self-test during `score --judge` ends the command with exit code 1 and the reason on
    stderr (no score file is written), instead of a traceback."""
    import importlib
    from streamreplace.cli import main as cli_main
    from streamreplace.judges.selftest import SelfTestFailed
    score_module = importlib.import_module("streamreplace.scoring.score")   # the module (the package exports score())

    def refuse(*args, **kwargs):
        raise SelfTestFailed("RJScore self-test failed; the judge refuses to score on this setup:\n  k0: 3.2 vs 1.0")
    monkeypatch.setattr(score_module, "score", refuse)
    code = cli_main.main(["--home", str(tmp_path), "score", "--judge"])
    assert code == 1 and "the judge refuses to score on this setup" in capsys.readouterr().err


def test_selftest_command_uses_each_judges_default_gpus(tmp_path, monkeypatch, capsys):
    """`score --selftest` without --judge-gpus runs each judge's self-test on its DEFAULT_GPUS (the 72B RJScore judge
    on 0,1, the HarmBench classifiers on 0); --judge-gpus overrides them, and a refusal makes the exit code 1."""
    from streamreplace.cli import main as cli_main
    from streamreplace.judges.selftest import SelfTestFailed
    seen = []

    def self_test(judge, gpus):
        seen.append((judge.NAME, list(gpus)))
    monkeypatch.setattr(rjscore.RJScoreJudge, "self_test", self_test)
    monkeypatch.setattr(harmbench_cls.HarmBenchClassifier, "self_test", self_test)
    assert cli_main.main(["--home", str(tmp_path), "score", "--selftest"]) == 0
    assert seen == [("rjscore", [0, 1]), ("harmbench_cls", [0])]
    seen.clear()
    assert cli_main.main(["--home", str(tmp_path), "score", "--selftest", "--judge-gpus", "2,3"]) == 0
    assert seen == [("rjscore", [2, 3]), ("harmbench_cls", [2, 3])]

    def refuse(judge, gpus):
        raise SelfTestFailed(f"{judge.NAME} refuses")
    monkeypatch.setattr(harmbench_cls.HarmBenchClassifier, "self_test", refuse)
    assert cli_main.main(["--home", str(tmp_path), "score", "--selftest"]) == 1
    assert "harmbench_cls refuses" in capsys.readouterr().err
