"""Judge caches (append-only, last line wins, one file per judge configuration) and configs/judges.yaml."""
from dataclasses import replace
import threading

import pytest

from streamreplace.benchmarks.base import JudgePolicy
from streamreplace.judges.base import JUDGES
from streamreplace.judges.cache import JudgeCache, cache_path
from streamreplace.judges.registry import JudgeSpec, build, judge_name, load_specs


def test_cache_appends_and_the_last_line_wins(tmp_path):
    cache = JudgeCache(tmp_path / "judges" / "rjscore.abc.jsonl")
    assert len(cache) == 0 and cache.get("k1") is None and cache.missing(["k1", "k1", "k2"]) == ["k1", "k2"]
    assert cache.add([{"key": "k1", "rjscore": 1.0}, {"key": "k2", "rjscore": 2.0}]) == 2
    cache.append({"key": "k1", "rjscore": 3.0})
    assert cache.get("k1")["rjscore"] == 3.0 and "k2" in cache and cache.missing(["k1", "k3"]) == ["k3"]
    fresh = JudgeCache(cache.path)                     # another reader sees the same, last line per key wins
    assert fresh.get("k1")["rjscore"] == 3.0 and len(fresh) == 2
    assert cache.path.read_text(encoding="utf-8").count("\n") == 3
    with pytest.raises(ValueError, match="key"):
        cache.add([{"rjscore": 1.0}])


def test_cache_appends_from_threads(tmp_path):
    cache = JudgeCache(tmp_path / "c.jsonl")
    threads = [threading.Thread(target=lambda k=k: cache.append({"key": f"k{k}", "reply": "x" * 1000}))
               for k in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(JudgeCache(cache.path)) == 50


def test_policies_name_their_judges():
    names = {p: judge_name(type("B", (), {"JUDGE": p})()) for p in JudgePolicy}
    assert names == {JudgePolicy.NONE: None, JudgePolicy.FALLBACK: "extract", JudgePolicy.ALWAYS: "extract",
                     JudgePolicy.CLASSIFIER: "harmbench_cls", JudgePolicy.RJSCORE: "rjscore", JudgePolicy.LINGO: "lingo"}


def test_configured_judges_and_pinned_checkpoints():
    specs = load_specs()
    assert set(specs) <= set(JUDGES) and {"extract", "harmbench_cls", "rjscore"} <= set(specs)
    assert specs["rjscore"].checkpoint == "Qwen/Qwen2.5-72B-Instruct@495f39366efef23836d0cfae4fbe635880d2be31"
    assert dict(specs["harmbench_cls"].checkpoints) == {
        "text": "cais/HarmBench-Llama-2-13b-cls@bda705349d1144fa618770bea64d99ce54e3835b",
        "multimodal": "cais/HarmBench-Llama-2-13b-cls-multimodal-behaviors@928eaf3333b75a0eba1d7632d26256a56223b949"}
    assert specs["extract"].checkpoint.startswith("Qwen/") and "@" in specs["extract"].checkpoint
    for spec in specs.values():                        # every checkpoint is pinned to a full revision
        for ref in [spec.checkpoint, *spec.checkpoints.values()]:
            if ref:
                assert len(ref.split("@")[1]) == 40
    assert specs["extract"].option("client", "temperature") == 0
    assert specs["extract"].option("client", "enable_thinking") is False
    assert specs["extract"].option("server", "missing", default=7) == 7


def test_the_cache_file_name_changes_with_the_judge_configuration(tmp_path):
    for name in ("extract", "harmbench_cls", "rjscore"):
        judge = build(name)
        spec = judge.spec
        other = (replace(spec, checkpoint=spec.checkpoint.replace("@", "@0")) if spec.checkpoint
                 else replace(spec, checkpoints={**spec.checkpoints, "text": "org/other@" + "0" * 40}))
        changed = type(judge)(other)
        assert changed.config_hash() != judge.config_hash()
        assert judge.cache(tmp_path).path == cache_path(tmp_path, name, judge.config_hash())
        assert judge.cache(tmp_path).path.name == f"{name}.{judge.config_hash()}.jsonl"
        assert type(judge)(spec).config_hash() == judge.config_hash()          # deterministic
    extract = build("extract")
    thinking = replace(extract.spec, options={**extract.spec.options, "client": {"enable_thinking": True}})
    assert type(extract)(thinking).config_hash() != extract.config_hash()
    # settings that cannot change an answer do not rename the cache
    port = replace(extract.spec, options={**extract.spec.options, "server": {"port": 9999}})
    assert type(extract)(port).config_hash() == extract.config_hash()


def test_an_external_extraction_endpoint_gets_its_own_cache():
    """With server.url the answers come from whatever that endpoint serves: they must not be filed under (and later
    applied as) the configured checkpoint's answers. Without it the paper's configuration hashes as before."""
    extract = build("extract")
    assert "endpoint" not in extract.identity() and extract.spec.option("server", "url") is None

    def with_server(**server):
        options = {**extract.spec.options, "server": {**extract.spec.option("server", default={}), **server}}
        return type(extract)(replace(extract.spec, options=options))
    endpoint = with_server(url="http://127.0.0.1:8000/v1")
    assert endpoint.identity()["endpoint"] == {"url": "http://127.0.0.1:8000/v1", "model": "judge"}
    assert endpoint.config_hash() != extract.config_hash()
    other_model = with_server(url="http://127.0.0.1:8000/v1", model="another-model")
    assert other_model.config_hash() not in (endpoint.config_hash(), extract.config_hash())


def test_default_judge_gpus():
    from streamreplace.judges.rjscore import RJScoreJudge, gpu_memory_problem
    assert build("extract").DEFAULT_GPUS == build("harmbench_cls").DEFAULT_GPUS == (0,)
    assert RJScoreJudge.DEFAULT_GPUS == (0, 1)                     # the 72B judge over two GPUs, as the paper's runs
    gib = 2 ** 30
    assert gpu_memory_problem([int(95.6 * gib)] * 2) is None       # two 96 GB GPUs
    assert gpu_memory_problem([int(79.3 * gib)] * 2) is None       # two 80 GB GPUs
    assert "--judge-gpus" in gpu_memory_problem([int(95.6 * gib)])  # one: the rest would be offloaded to CPU memory
    assert gpu_memory_problem([int(45.0 * gib)] * 2) and gpu_memory_problem([])


def test_rjscore_supports_only_the_official_backend():
    from streamreplace.judges.rjscore import RJScoreJudge
    spec = JudgeSpec.parse("rjscore", {"checkpoint": "Qwen/Qwen2.5-72B-Instruct@" + "1" * 40, "backend": "vllm"})
    with pytest.raises(ValueError, match="hf"):
        RJScoreJudge(spec)


def test_spec_parsing_errors():
    with pytest.raises(ValueError):
        JudgeSpec.parse("extract", ["not", "a", "mapping"])
    with pytest.raises(ValueError):
        JudgeSpec.parse("harmbench_cls", {"checkpoints": ["a"]})
    with pytest.raises(ValueError, match="unknown judges"):
        load_specs({"mystery": {}})
