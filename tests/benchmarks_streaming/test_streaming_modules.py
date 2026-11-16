"""OVO-Bench and StreamingBench (benchmarks/streaming/): scope, ids, prompts and the official scoring rules.

The tests marked requires_upstream read the benchmarks' own annotations, prompt templates and scorers from the
fetched code (`streamreplace fetch-upstream ovo_bench streamingbench`); the others need nothing."""
from __future__ import annotations

from collections import Counter
import io
import tarfile

import pytest

from streamreplace.benchmarks.streaming import ovo_bench, streamingbench
from streamreplace.core.schema import Sample


# ---------- without upstream code ----------

def test_both_are_registered_as_the_streaming_suite():
    from streamreplace.benchmarks import registry
    from streamreplace.config.load import load_config
    config = load_config()
    assert config.expand_benchmarks("streaming") == ["ovo_bench", "streamingbench"]
    for name, module in (("ovo_bench", ovo_bench), ("streamingbench", streamingbench)):
        assert registry.get(name).module is module and config.benchmark(name).media == "video"
        assert config.settings(name, "aura_8b", "stream")["generation"]["max_new_tokens"] == 128
        assert module.MAX_SECONDS == 120
    assert (ovo_bench.EXPECTED_COUNT, streamingbench.EXPECTED_COUNT) == (1576, 700)


def test_ovo_modes_and_official_aggregation():
    assert [ovo_bench.mode_of(t) for t in ("EPM", "OCR", "CRR")] == ["backward", "realtime", "forward"]
    rows = ([{"metadata": {"task": "EPM"}, "correct": c} for c in (True, False)]            # backward 50
            + [{"metadata": {"task": "STU"}, "correct": True}]                              # realtime 100, 0 -> 50
            + [{"metadata": {"task": "OCR"}, "correct": False}]
            + [{"metadata": {"task": "REC"}, "correct": c} for c in (True, True, True, False)])   # forward 75
    got = ovo_bench.official(rows)
    assert got["tasks"] == {"EPM": 50, "STU": 100, "OCR": 0, "REC": 75}
    assert got["modes"] == {"backward": 50, "realtime": 50, "forward": 75}
    assert got["total"] == pytest.approx(175 / 3)
    assert ovo_bench.official(rows[:2])["total"] is None            # no total while a mode has no rows
    metrics = ovo_bench.metrics([{**r, "answered": True} for r in rows])
    assert metrics["overall"] == 58.33 and metrics["acc_rows"] == 62.5 and metrics["mode/forward"] == 75


def test_ovo_clips_are_read_out_of_the_split_tar(tmp_path):
    files = {f"chunked_videos/{name}": bytes([65 + i]) * (4000 + 700 * i)
             for i, name in enumerate(("1.mp4", "2_0.mp4", "3.mp4", "4.mp4"))}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    blob, raw, media = buffer.getvalue(), tmp_path / "raw", tmp_path / "media"
    raw.mkdir()
    for k, suffix in enumerate(("aa", "ab", "ac")):                # members cross the part boundaries
        (raw / f"chunked_videos.tar.part{suffix}").write_bytes(blob[k * 7000:(k + 1) * 7000 if k < 2 else None])
    ovo_bench.extract(raw, media, {"2_0.mp4", "4.mp4"})
    got = {p.name: p.read_bytes() for p in (media / "chunked_videos").iterdir()}
    assert got == {"2_0.mp4": files["chunked_videos/2_0.mp4"], "4.mp4": files["chunked_videos/4.mp4"]}
    for path in raw.iterdir():                                     # everything wanted is there: nothing is read
        path.unlink()
    ovo_bench.extract(raw, media, {"2_0.mp4", "4.mp4"})
    with pytest.raises(ValueError, match="no chunked_videos.tar.part"):
        ovo_bench.extract(raw, media, {"1.mp4"})


def test_streamingbench_time_stamps_and_counting_rule():
    assert streamingbench.seconds("00:03:10") == 190 and streamingbench.seconds("01:00:05") == 3605
    sample = Sample("sample_1_1", "q", "B", "mcq", {"A": "x", "B": "y", "C": "z", "D": "w"}, video="v.mp4")
    assert streamingbench.correct(sample, "B. y") and streamingbench.correct(sample, "B")
    assert not streamingbench.correct(sample, "The answer is B") and not streamingbench.correct(sample, None)
    assert streamingbench.parse(sample, "<think>hm</think>B") == ("B", "official")
    assert streamingbench.parse(sample, "  ") == (None, "empty")
    rows = [{"metadata": {"task": "t"}, "correct": True, "answered": True},
            {"metadata": {"task": "t"}, "correct": False, "answered": True},
            {"metadata": {"task": "u"}, "correct": False, "answered": False}]
    got = streamingbench.metrics(rows)
    assert got["acc"] == pytest.approx(33.33, abs=0.01) and got["acc_official"] == 50 and got["task/t"] == 50
    assert "task/u" not in got                                     # the official accuracy skips empty replies


# ---------- with the benchmarks' own code ----------

def ovo_sample(sid, f):
    meta = {k: f[k] for k in ("item", "task", "mode", "source", "t", "probe", "type") if k in f}
    if f["mode"] != "forward":
        from streamreplace.benchmarks import common
        return Sample(sid, f["question"], f["answer"], "mcq", common.letters(f["options"]), video="v.mp4", metadata=meta)
    return Sample(sid, f["question"], f["answer"], "numeric" if f["task"] == "REC" else "yesno", video="v.mp4",
                  metadata=meta)


@pytest.mark.requires_upstream("ovo_bench")
def test_ovo_bench_rows_prompts_and_scores():
    rows = ovo_bench.items()
    assert len(rows) == 3035 and len({sid for sid, _ in rows}) == 3035
    by_mode = Counter(f["mode"] != "forward" for _, f in rows)
    assert by_mode == {True: 1468, False: 1567}
    assert sum(f["t"] <= ovo_bench.MAX_SECONDS for _, f in rows) == ovo_bench.EXPECTED_COUNT
    first = {}
    for sid, f in rows:
        first.setdefault(f["task"], ovo_sample(sid, f))
    assert set(first) == set(ovo_bench.BACKWARD + ovo_bench.REALTIME + ovo_bench.FORWARD)
    for task, sample in first.items():
        text = ovo_bench.prompt(sample)
        assert sample.question in text, task
        if task not in ovo_bench.FORWARD:
            assert all(choice in text for choice in sample.choices.values())
            wrong = next(letter for letter in sample.choices if letter != sample.answer)
            assert ovo_bench.correct(sample, sample.answer) and not ovo_bench.correct(sample, wrong)
        elif task == "REC":
            assert ovo_bench.correct(sample, f"It happened {sample.answer} times")
            assert not ovo_bench.correct(sample, str(int(sample.answer) + 1))
        else:
            other = "No" if sample.answer == "Yes" else "Yes"
            assert ovo_bench.correct(sample, sample.answer) and not ovo_bench.correct(sample, other)
        assert not ovo_bench.correct(sample, None)                 # no reply is wrong
        assert ovo_bench.parse(sample, "") == (None, "empty")


@pytest.mark.requires_upstream("streamingbench")
def test_streamingbench_rows_and_prompt():
    rows = streamingbench.items()
    assert len(rows) == 2500 and len({sid for sid, _ in rows}) == 2500
    assert sum(f["t"] <= streamingbench.MAX_SECONDS for _, f in rows) == streamingbench.EXPECTED_COUNT
    assert len({f["video"] for _, f in rows}) == 499 and len({f["task"] for _, f in rows}) == 10
    sid, f = rows[0]
    assert sid.endswith("_1") and f["clip"] == f"{f['video'][:-4]}_0_{f['t']}.mp4"
    sample = Sample(sid, f["question"], f["answer"], "mcq", {o[0]: o[2:].strip() for o in f["options"]}, video="v.mp4",
                    metadata={"task": f["task"], "options": f["options"]})
    text = streamingbench.prompt(sample)
    assert text.endswith("\n\nThe best option is:") and f["question"] in text
    assert all(option in text for option in f["options"])


@pytest.mark.requires_upstream("ovo_bench", "streamingbench")
def test_prompt_fingerprints_match_the_manifest():
    """The id -> prompt map of the rows in scope, as manifests/prepared.yaml records it (prompts do not depend on
    the media, so the annotations and the official templates are enough)."""
    from streamreplace.data.prepare import load_expected
    from streamreplace.data.prompts import prompt_map_sha256
    expected = load_expected()
    ovo = {sid: ovo_bench.prompt(ovo_sample(sid, f)) for sid, f in ovo_bench.items() if f["t"] <= ovo_bench.MAX_SECONDS}
    assert len(ovo) == expected["ovo_bench"]["count"]
    assert prompt_map_sha256(ovo) == expected["ovo_bench"]["prompts_sha256"]
    real = {}
    for sid, f in streamingbench.items():
        if f["t"] <= streamingbench.MAX_SECONDS:
            sample = Sample(sid, f["question"], f["answer"], "mcq", {o[0]: o[2:].strip() for o in f["options"]},
                            video="v.mp4", metadata={"options": f["options"]})
            real[sid] = streamingbench.prompt(sample)
    assert len(real) == expected["streamingbench"]["count"]
    assert prompt_map_sha256(real) == expected["streamingbench"]["prompts_sha256"]
