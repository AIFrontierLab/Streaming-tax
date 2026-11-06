"""CPU-only stream schedules: what each child's native stream is fed (frames, chunks, segments, turns, question time).
The full check against every formal stream row of the paper's runs is a maintainer script; these are the rules."""
from __future__ import annotations

import pytest

from streamreplace.media.digest import frame_digest
from streamreplace.media.frames import frame_indices
from streamreplace.models import aura_stream, livestar_stream, mmduet2_stream, mmduet_stream, mossrt_stream
from streamreplace.models import thinkstream_stream
from streamreplace.models.mmduet_stream import load_video_indices
from streamreplace.models.readouts import stream_seed
from streamreplace.models.thinkstream_stream import chunking
from stream_fakes import make_video, request


# ------------------------------------------------------------------------------------------- MMDuet

def test_mmduet_load_video_keeps_frames_by_accumulated_float_time():
    """Upstream sums 1 / fps frame by frame: at 30 fps the 0.5 s target is reached at frame 16, not 15 (exact
    arithmetic would give 15). The paper's runs saw these frames, so the accumulation is part of the rule."""
    assert load_video_indices(300, 30.0, 300)[:8] == [0, 16, 31, 45, 60, 75, 91, 106]
    assert load_video_indices(300, 24.0, 300)[:6] == [0, 12, 25, 36, 48, 61]
    assert load_video_indices(300, 30000 / 1001, 300)[:6] == [0, 15, 30, 45, 60, 75]
    assert len(load_video_indices(300, 30.0, 300)) == 20                       # floor(10 s * 2)
    assert load_video_indices(12, 4.0, 12) == [0, 2, 4, 6, 8, 10]


def test_mmduet_load_video_stops_at_the_last_decoded_frame_and_at_400():
    # a header that counts frames the decoder never returns (edit list): fewer frames than floor(duration * 2)
    assert load_video_indices(14, 4.0, 12) == [0, 2, 4, 6, 8, 10]
    assert len(load_video_indices(30 * 250, 30.0, 30 * 250)) == 400
    assert load_video_indices(30, 30.0, 0) == []                             # nothing decodes: upstream then fails


def test_mmduet_schedule_and_its_200_second_limit():
    meta = {"frames": 300, "fps": 30.0, "header_frames": 300}
    plan = mmduet_stream.schedule(request(), meta)
    assert plan["frames"] == 20 and plan["indices"] == load_video_indices(300, 30.0, 300)
    assert plan["frame_indices"] == frame_digest({"indices": plan["indices"]})
    assert plan["question_before_frame"] == 19 and plan["question_time"] == 9.5
    with_opencv = mmduet_stream.schedule(request(), {**meta, "opencv": {"frames": 280.0, "fps": 30.0}})
    assert with_opencv["frames"] == 18                                        # OpenCV's header wins when measured
    assert mmduet_stream.schedule(request(), {"frames": 6000, "fps": 30.0, "header_frames": 6000})["frames"] == 400
    with pytest.raises(ValueError, match="400 frames"):
        mmduet_stream.schedule(request(), {"frames": 6001, "fps": 30.0, "header_frames": 6001})


def test_mmduet_schedule_measures_a_real_file(tmp_path):
    video = make_video(tmp_path / "clip.mp4", frames=12, fps=4)
    plan = mmduet_stream.schedule(request(video=str(video)))
    assert plan["frames"] == 6 and plan["indices"] == [0, 2, 4, 6, 8, 10]


# ------------------------------------------------------------------------------------------- ThinkStream

@pytest.mark.parametrize("duration, chunks, seconds", [(3.0, 3, 1.0), (59.9, 59, 1.0), (120.0, 120, 1.0),
                                                       (121.0, 60, 2.0), (300.0, 100, 3.0), (0.6, 1, 1.0)])
def test_thinkstream_chunks_of_ceil_duration_over_120_seconds(duration, chunks, seconds):
    plan = chunking(duration, 30.0, int(duration * 30) + 1)
    assert (plan["chunks"], plan["chunk_seconds"], plan["frames"]) == (chunks, seconds, 2 * chunks)
    assert plan["windows"][0][0] == 0.0 and plan["windows"][-1][1] == duration


def test_thinkstream_low_fps_clip_gets_fewer_longer_chunks():
    plan = chunking(10.0, 0.5, 5)            # 5 frames for 10 one-second chunks of 2 frames
    assert (plan["chunks"], plan["frames"], plan["chunk_seconds"]) == (2, 4, 5.0)


def test_thinkstream_schedule_question_seed_and_window():
    meta = {"torchcodec": {"frames": 360, "fps": 30.0, "duration": 12.0}}
    plan = thinkstream_stream.schedule(request(sample_id="v-1"), meta)
    assert plan["chunks"] == 12 and plan["frames"] == 24 and len(plan["indices"]) == 24
    assert plan["indices"][0] == 0 and plan["indices"][-1] == 359 and plan["indices"] == sorted(plan["indices"])
    assert plan["question_chunk"] == 11 and plan["question_time"] == 12.0
    assert plan["seed"] == stream_seed("v-1") and plan["window_chunks"] == 20 and plan["max_len"] == 24576
    recall = request(sample_id="v-1", settings={"generation": {"max_new_tokens": 8},
                                                "stream": {"fps": 1, "window": "full", "max_len": 81920, "seed": 1}})
    full = thinkstream_stream.schedule(recall, meta)
    assert (full["window_chunks"], full["max_len"], full["seed"]) == (120, 81920, stream_seed("v-1", 1))
    assert full["indices"] == plan["indices"] and full["digest"] != plan["digest"]


# ------------------------------------------------------------------------------------------- AURA

def test_aura_spans_official_full_and_sub_second():
    spans = aura_stream.spans(95.4)
    assert len(spans) == 30 and spans[-1][0] == pytest.approx(94.4) and spans[0][0] == pytest.approx(65.4)
    full = aura_stream.spans(95.4, "full")
    assert len(full) == 95 and full[0][0] == pytest.approx(0.4) and all(length == 1.0 for _, length in full)
    assert aura_stream.spans(95.4, 40)[0][0] == pytest.approx(55.4)
    assert aura_stream.spans(0.6) == [(0.0, 0.6)] and aura_stream.spans(0.6, "full") == [(0.0, 0.6)]


def test_aura_schedule_window_start_and_recall():
    meta = {"frames": 954, "fps": 10.0}
    plan = aura_stream.schedule(request(), meta)
    assert (plan["segments"], plan["window_start"], plan["evicted"], plan["question_segment"]) == (30, 65.4, 65, 29)
    full = aura_stream.schedule(request(settings={"generation": {"max_new_tokens": 8},
                                                  "stream": {"fps": 1, "window": "full"}}), meta)
    assert (full["segments"], full["window_start"], full["evicted"]) == (95, 0.4, 0)
    assert aura_stream.schedule(request(), meta, adapter="aura_stream_hf")["digest"] != plan["digest"]


# ------------------------------------------------------------------------------------------- 1-fps children

def test_mmduet2_turns_of_two_frames_with_the_question_in_the_last():
    assert mmduet2_stream.turns_of(5) == [[0, 1], [2, 3], [4]] and mmduet2_stream.turns_of(4) == [[0, 1], [2, 3]]
    plan = mmduet2_stream.schedule(request(), {"frames": 151, "fps": 30.0})
    assert plan["indices"] == frame_indices({"frames": 151, "fps": 30.0}, fps=1) == [0, 30, 60, 90, 120]
    assert (plan["frames"], plan["turns"], plan["turn_frames"], plan["question_turn"]) == (5, 3, [2, 2, 1], 2)
    assert plan["question_time"] == 4.0


def test_livestar_and_moss_feed_the_fixed_step_frames():
    meta = {"frames": 151, "fps": 30.0}
    live = livestar_stream.schedule(request(), meta)
    moss = mossrt_stream.schedule(request(), meta)
    assert live["indices"] == moss["indices"] == [0, 30, 60, 90, 120]
    assert moss["timestamps"] == [0.0, 1.0, 2.0, 3.0, 4.0] and moss["question_with_frame"] == 4
    assert moss["question_not_before"] == pytest.approx(151 / 30)
    assert live["frame_indices"] == moss["frame_indices"] == frame_digest({"indices": [0, 30, 60, 90, 120]})


def test_schedules_measure_real_files_and_are_deterministic(tmp_path):
    video = str(make_video(tmp_path / "clip.mp4", frames=12, fps=4))
    for module in (mmduet2_stream, livestar_stream, mossrt_stream, aura_stream):
        first, second = module.schedule(request(video=video)), module.schedule(request(video=video))
        assert first == second and len(first["digest"]) == 16
    assert mmduet2_stream.schedule(request(video=video))["indices"] == [0, 4, 8]
    assert aura_stream.schedule(request(video=video))["spans"] == [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
