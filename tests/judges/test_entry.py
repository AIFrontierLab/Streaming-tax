"""The judge process entry point: cached requests are skipped, the environment a judge process gets, missing
environments refused, and the extraction server's port fallback."""
import socket

import pytest

from streamreplace import paths as paths_module
from streamreplace.core.io import write_json, write_jsonl
from streamreplace.judges import entry
from streamreplace.judges.cache import JudgeCache


def test_cached_requests_are_never_asked_again(tmp_path, monkeypatch):
    cache = JudgeCache(tmp_path / "judges" / "extract.x.jsonl")
    cache.add([{"key": "a", "extracted": "B"}, {"key": "b", "extracted": None}])
    write_jsonl(tmp_path / "requests.jsonl", [{"key": "a", "kind": "extract"}, {"key": "b", "kind": "extract"},
                                              {"key": "a", "kind": "extract"}])
    write_json(tmp_path / "job.json", {"judge": "extract", "requests": str(tmp_path / "requests.jsonl"),
                                       "cache": str(cache.path), "spec": {}})
    served = []
    monkeypatch.setattr("streamreplace.judges.extraction.serve", lambda reqs, c, spec: served.append(reqs))
    assert entry.main([str(tmp_path / "job.json")]) == 0 and served == []
    write_jsonl(tmp_path / "requests.jsonl", [{"key": "c", "kind": "extract"}, {"key": "c", "kind": "extract"},
                                              {"key": "a", "kind": "extract"}])
    assert entry.main([str(tmp_path / "job.json")]) == 0
    assert served == [[{"key": "c", "kind": "extract"}]]                  # new requests only, each once
    assert entry.main([]) == 2


def test_judge_process_environment(tmp_path):
    paths = paths_module.resolve(home=tmp_path, environ={})
    env = entry.judge_environ(paths, [2, 3], tmp_path / "envs" / "judge" / "bin" / "python", {"EXTRA": 1})
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3" and env["HF_HUB_OFFLINE"] == "1" and env["EXTRA"] == "1"
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert env["PATH"].startswith(str(tmp_path / "envs" / "judge" / "bin"))
    assert env["STREAMREPLACE_OUTPUTS"] == str(paths.outputs) and str(entry.SRC) in env["PYTHONPATH"]


def test_a_missing_judge_environment_is_refused(tmp_path):
    paths = paths_module.resolve(home=tmp_path, environ={})
    with pytest.raises(ValueError, match="environment 'judge' is missing"):
        entry.launch("rjscore", [{"key": "k"}], tmp_path / "c.jsonl", {}, [0], env="judge", paths=paths)
    assert not (tmp_path / "outputs" / "logs").exists()


def test_free_port_falls_back_when_taken():
    from streamreplace.judges.server import free_port
    with socket.socket() as busy:
        busy.bind(("", 0))
        busy.listen()
        taken = busy.getsockname()[1]
        port = free_port(taken)
        assert port != taken and port > 0


def test_every_judge_has_a_process_module():
    from streamreplace.judges.base import JUDGES
    assert set(entry.SERVERS) == set(JUDGES)
