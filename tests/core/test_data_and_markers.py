"""Data preparation rules that decide what every run reads (data/prepare.py, data/prompts.py), and the marker skipping
of the shared test configuration (tests/conftest.py)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import types

import pytest

from streamreplace import paths
from streamreplace.benchmarks.registry import BenchmarkModule
from streamreplace.core.io import read_jsonl
from streamreplace.core.schema import Sample
from streamreplace.data import prompt_map_digest, prompt_map_sha256
from streamreplace.data.prepare import check_prepared, prepare_dataset

ROOT_CONFTEST = Path(__file__).resolve().parents[1] / "conftest.py"


def test_prompt_map_sha256_is_the_manifest_definition():
    prompts = {"b": "Was ist das?\nA. ä", "a": "Which?"}
    text = json.dumps(prompts, sort_keys=True, ensure_ascii=False)
    assert prompt_map_sha256(prompts) == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert prompt_map_sha256(dict(reversed(prompts.items()))) == prompt_map_sha256(prompts)
    assert prompt_map_digest is prompt_map_sha256


def fake_benchmark(count: int, expected: int, seen: dict) -> BenchmarkModule:
    module = types.ModuleType("fake_prepared_benchmark")
    module.MEDIA, module.EXPECTED_COUNT = "video", expected

    def prepare(raw: Path, media: Path) -> list[Sample]:
        seen.update(raw=raw, media=media)
        return [Sample(f"q{i}", f"Question {i}?", "yes", "yesno", video=str(raw / "clip.mp4")) for i in range(count)]
    module.prepare = prepare
    return BenchmarkModule("fakeprep", module)


def test_prepare_reads_raw_writes_relative_paths_and_enforces_the_expected_count(tmp_path):
    outside = tmp_path / "downloads"                       # raw/ is a symlink to a download folder elsewhere
    outside.mkdir()
    (outside / "clip.mp4").write_bytes(b"\0")
    p = paths.resolve(home=tmp_path / "home", environ={}, project=tmp_path)
    base = p.dataset_dir("fakeprep")
    base.mkdir(parents=True)
    (base / "raw").symlink_to(outside, target_is_directory=True)
    seen: dict = {}
    with pytest.raises(ValueError, match="expects 3"):
        prepare_dataset("fakeprep", paths=p, bench=fake_benchmark(2, 3, seen))
    assert not (base / "prepared" / "samples.jsonl").exists()
    assert seen == {"raw": base / "raw", "media": base / "media"}
    result = prepare_dataset("fakeprep", paths=p, bench=fake_benchmark(3, 3, seen))
    rows = read_jsonl(base / "prepared" / "samples.jsonl")
    assert result["status"] == "prepared" and result["samples"] == 3 and len(rows) == 3
    assert rows[0] == Sample("q0", "Question 0?", "yes", "yesno", video="raw/clip.mp4").to_dict() | {"images": []}
    check = check_prepared("fakeprep", p, bench=fake_benchmark(3, 3, seen), expected={})
    assert check["problems"] == [] and check["count"] == 3
    assert check["prompts_sha256"] == prompt_map_sha256({f"q{i}": f"Question {i}?\nAnswer with yes or no."
                                                         for i in range(3)})
    assert prepare_dataset("fakeprep", paths=p, bench=fake_benchmark(3, 3, seen))["status"] == "exists"


@pytest.fixture
def root_conftest(tmp_path):
    spec = importlib.util.spec_from_file_location("streamreplace_root_conftest_under_test", ROOT_CONFTEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._roots["value"] = paths.resolve(home=tmp_path / "home", environ={}, project=tmp_path)
    return module


class Item:
    def __init__(self, *marks):
        self.marks = [m.mark for m in marks]

    def get_closest_marker(self, name):
        return next((m for m in self.marks if m.name == name), None)

    def iter_markers(self, name):
        return [m for m in self.marks if m.name == name]


def test_marker_skip_reasons(root_conftest, tmp_path):
    reason = root_conftest.skip_reason
    home = tmp_path / "home"
    assert reason(Item()) is None
    assert "no data root" in reason(Item(pytest.mark.requires_data))
    assert "mme is not prepared" in reason(Item(pytest.mark.requires_data("mme")))
    (home / "data" / "mme" / "prepared").mkdir(parents=True)
    (home / "data" / "mme" / "prepared" / "samples.jsonl").write_text("{}\n")
    assert reason(Item(pytest.mark.requires_data("mme"))) is None and reason(Item(pytest.mark.requires_data)) is None
    assert "tomato" in reason(Item(pytest.mark.requires_data("mme", "tomato")))
    assert "fetch-upstream vmdt" in reason(Item(pytest.mark.requires_upstream("vmdt")))
    (home / "upstream" / "vmdt").mkdir(parents=True)
    assert reason(Item(pytest.mark.requires_upstream("vmdt"), pytest.mark.requires_data("mme"))) is None
    assert "mmduet2/demo" in reason(Item(pytest.mark.requires_upstream("mmduet2/demo")))
    root_conftest._roots["value"] = ValueError("bad paths file")
    assert "bad paths file" in reason(Item(pytest.mark.requires_upstream("vmdt")))
