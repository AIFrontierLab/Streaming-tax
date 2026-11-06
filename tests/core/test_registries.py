"""Registries and interfaces: adapter, benchmark and judge names (models/registry.py, benchmarks/registry.py,
judges/base.py), the benchmark module wrapper with its shared defaults (benchmarks/base.py), the upstream resolver
(models/upstream.py) and the CPU-only helpers of models/{base,decoding,lm_head}.py."""
from __future__ import annotations

import json
import sys
import types

import pytest

from streamreplace.benchmarks import common
from streamreplace.benchmarks import registry as benchmarks
from streamreplace.benchmarks.base import JudgePolicy, ParseClass, parse_class
from streamreplace.core.schema import Request, Sample
from streamreplace.judges.base import JUDGES, load_judge_class
from streamreplace.models import registry as adapters
from streamreplace.models.base import (BaseAdapter, OfflineAdapter, StreamAdapter, captures_letter_logits,
                                       is_trainable, supervised_length)
from streamreplace.models.decoding import check_repetition_penalty, greedy, sample_seed
from streamreplace.models.upstream import Upstream, UpstreamMissing

ADAPTERS = {"qwen_offline", "llava_offline", "internvl_offline", "moss_offline", "mmduet_stream",
            "thinkstream_stream", "mmduet2_stream", "aura_stream", "aura_stream_hf", "livestar_stream",
            "mossrt_stream", "mock"}
BENCHMARKS = {"image": ["dynamath", "mathvista", "mmbench", "mme", "mmmu", "mmstar", "scienceqa", "visulogic"],
              "video": ["videomme", "tomato", "timeblind", "videoreasonbench"],
              "safety": ["harmbench", "video_safetybench", "mcv_safetybench"],
              "trust": ["vmdt_fairness", "vmdt_privacy"], "finetune": ["lingoqa", "eyepcr", "openuav_qa"]}


def test_adapter_registry():
    assert set(adapters.names()) == ADAPTERS
    assert adapters.modes("qwen_offline") == {"offline"} and adapters.modes("aura_stream_hf") == {"stream"}
    assert adapters.modes("mock") == {"offline", "stream"}
    assert adapters.target("mmduet_stream") == ("streamreplace.models.mmduet_stream", "MMDuetStream")
    with pytest.raises(ValueError, match="Unknown model adapter"):
        adapters.target("nope")
    assert all(adapters.available(name) for name in ADAPTERS)          # every module exists (imported lazily)


def test_create_passes_constructor_keywords(monkeypatch):
    calls = []

    class Fake(BaseAdapter):
        def __init__(self, spec, settings, upstream, load=True):
            super().__init__(spec, settings, upstream)
            calls.append((spec, settings, type(upstream).__name__, load))

    module = types.ModuleType("fake_adapter_module")
    module.Fake = Fake
    monkeypatch.setitem(sys.modules, "fake_adapter_module", module)
    monkeypatch.setitem(adapters.ADAPTERS, "fake", "fake_adapter_module:Fake")
    adapter = adapters.create("fake", "spec", {"a": 1}, load=False)
    assert isinstance(adapter, Fake) and calls == [("spec", {"a": 1}, "Upstream", False)]
    adapters.create("fake", "spec", {}, Upstream())
    assert calls[-1][3] is True and adapter.concurrency == 1 and adapter.close() is None


def test_benchmark_registry_and_runtime_registration():
    assert benchmarks.names()[:20] == [n for group in BENCHMARKS.values() for n in group]
    for group, names in BENCHMARKS.items():
        for name in names:
            assert benchmarks.module_path(name) == f"streamreplace.benchmarks.{group}.{name}"
    with pytest.raises(ValueError, match="Unknown benchmark"):
        benchmarks.module_path("nope")
    module = types.ModuleType("fake_benchmark_module")
    module.MEDIA = "video"
    sys.modules["fake_benchmark_module"] = module
    try:
        benchmarks.register("fakebench", "fake_benchmark_module")
        assert "fakebench" in benchmarks.names() and benchmarks.get("fakebench").module is module
    finally:
        benchmarks.unregister("fakebench")
        del sys.modules["fake_benchmark_module"]
    assert "fakebench" not in benchmarks.names()


def test_benchmark_module_defaults():
    module = types.ModuleType("bare")
    module.MEDIA = "image"
    module.prepare = lambda raw, media: []
    module.EXTRA = 3
    bench = benchmarks.BenchmarkModule("bare", module)
    assert (bench.NAME, bench.JUDGE, bench.LOWER_IS_BETTER, bench.METRICS) == ("bare", JudgePolicy.FALLBACK, False, [])
    assert bench.EXPECTED_COUNT is None and bench.PRIVATE_METADATA == frozenset() and not bench.has_train
    assert bench.media == "image" and bench.EXTRA == 3 and not bench.safe_on_error
    sample = Sample("1", "Which?", "B", "mcq", {"A": "x", "B": "y"}, metadata={"secret": 1})
    assert bench.prompt(bench.view(sample)) == common.prompt(sample)
    assert bench.parse(bench.view(sample), "B") == ("B", "letter") and bench.correct(sample, "B")
    assert bench.stratum(sample) is None and bench.judge_messages(sample, "x") is None
    assert bench.metrics([{"correct": True, "metadata": {}}]) == {"acc": 100.0}
    module.PRIVATE_METADATA = ("secret",)
    module.JUDGE = "RJScore"
    bench = benchmarks.BenchmarkModule("bare", module)
    assert bench.view(sample).metadata == {} and bench.JUDGE is JudgePolicy.RJSCORE and bench.safe_on_error
    module.MEDIA = "audio"
    with pytest.raises(ValueError, match="MEDIA"):
        benchmarks.BenchmarkModule("bare", module)


def test_judge_policy_and_parse_classes():
    assert JudgePolicy.of(None) is JudgePolicy.NONE and JudgePolicy.of("ALWAYS") is JudgePolicy.ALWAYS
    assert parse_class("no_answer", None) is ParseClass.SILENT and parse_class("pending", None) is ParseClass.PENDING
    assert parse_class("no_marker", None) is ParseClass.UNPARSED and parse_class("letter", "B") is ParseClass.PARSED
    assert set(JUDGES) == {"extract", "harmbench_cls", "rjscore", "lingo"}
    with pytest.raises(ValueError, match="Unknown judge"):
        load_judge_class("nope")


def test_upstream_resolver_restores_sys_path(tmp_path):
    (tmp_path / "repo" / "pkg").mkdir(parents=True)
    (tmp_path / "repo" / "pkg" / "upstream_probe_mod.py").write_text("VALUE = 7\n")
    upstream = Upstream(tmp_path)
    before = list(sys.path)
    with upstream.use("repo/pkg") as added:
        assert sys.path[0] == str(tmp_path / "repo" / "pkg") and added == [tmp_path / "repo" / "pkg"]
        import upstream_probe_mod
        assert upstream_probe_mod.VALUE == 7
    assert sys.path == before and upstream.available("repo") and not upstream.available("other")
    sys.modules.pop("upstream_probe_mod", None)
    with pytest.raises(UpstreamMissing, match="fetch-upstream other"):
        upstream.path("other/file.py")
    assert upstream.path("other/file.py", require=False) == tmp_path / "other" / "file.py"


def test_decoding_helpers(tmp_path):
    assert greedy({"generation": {"max_new_tokens": "512"}}) == {
        "do_sample": False, "temperature": None, "top_p": None, "top_k": None, "num_beams": 1, "max_new_tokens": 512}
    assert sample_seed("019-3") == Request("019-3", "d", "p").seed
    check_repetition_penalty(1.0, tmp_path)                           # no generation_config.json: transformers' 1.0
    (tmp_path / "generation_config.json").write_text(json.dumps({"repetition_penalty": 1.05}))
    check_repetition_penalty(1.05, tmp_path)
    check_repetition_penalty(None, tmp_path)
    with pytest.raises(ValueError, match="repetition_penalty=1.0"):
        check_repetition_penalty(1.0, tmp_path, "qwen25vl_3b")


def test_adapter_protocols_and_capabilities():
    class Offline(BaseAdapter):
        CAPTURES_LETTER_LOGITS = True

        def build_inputs(self, req):
            return None

        def generate(self, req):
            return None

        def encode(self, req, answer=None):
            return {}

        def train_forward(self, enc):
            return None

    class Stream(BaseAdapter):
        def schedule(self, req):
            return {}

        def stream(self, req, hooks=None):
            return None

    offline, stream = Offline(None, {}, None), Stream(None, {}, None)
    assert isinstance(offline, OfflineAdapter) and not isinstance(offline, StreamAdapter)
    assert isinstance(stream, StreamAdapter) and is_trainable(offline) and not is_trainable(stream)
    assert captures_letter_logits(offline) and not captures_letter_logits(stream)


def test_supervised_length_refuses_merges_across_the_prompt_boundary():
    def tokenize(text):                                               # one token per character, EOT as one token
        out, i = [], 0
        while i < len(text):
            if text.startswith("<|im_end|>", i):
                out.append(-1)
                i += len("<|im_end|>")
            else:
                out.append(ord(text[i]))
                i += 1
        return out

    assert supervised_length(tokenize, "Q: ", "yes") == 4                # "y", "e", "s" + EOT
    merging = lambda text: tokenize(text.replace(" y", "▁y"))      # noqa: E731 - " y" becomes one token
    with pytest.raises(ValueError, match="clean continuation"):
        supervised_length(merging, "Q: ", "yes")


def test_lm_head_wrapper_keeps_last_positions_and_captures_the_first_step():
    torch = pytest.importorskip("torch")
    from streamreplace.models.lm_head import counted_stop, head_wrapper, letter_logits
    inner = torch.nn.Linear(4, 6, bias=False)
    head = head_wrapper(inner)
    hidden = torch.randn(1, 5, 4)
    assert torch.equal(head(hidden), inner(hidden)) and head.calls == 1
    head.keep, head.capture = 1, True
    out = head(hidden)
    assert out.shape == (1, 1, 6) and torch.allclose(out[0, -1], inner(hidden)[0, -1])
    assert torch.allclose(head.captured, inner(hidden)[0, -1].float())
    head(torch.randn(1, 1, 4))                                          # later steps never overwrite the capture
    assert torch.allclose(head.captured, inner(hidden)[0, -1].float())
    assert counted_stop(head, 3) == {"tokens": 3, "finish": "length"}

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return {"A": [2], "B": [5], "C": [1, 2]}[text]

    req = Request("1", "d", "p", kind="mcq", choices={"A": "x", "B": "y", "C": "z"})
    logits = letter_logits(head, Tokenizer(), req)
    assert set(logits) == {"A", "B"} and logits["A"] == round(float(head.captured[2]), 4)
    assert letter_logits(head, Tokenizer(), Request("1", "d", "p", kind="open")) is None
