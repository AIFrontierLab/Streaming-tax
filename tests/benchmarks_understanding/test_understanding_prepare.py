"""prepare() on tiny synthetic raw folders laid out like the official downloads: sample ids (they key judge caches,
frozen id lists and TOMATO's seeded fallback), answers, options, metadata and media paths. The parquet-based
benchmarks need pyarrow (skipped without it). Expected sizes are lowered to the synthetic sizes."""
from __future__ import annotations

import base64
import csv
from io import BytesIO
import json
from pathlib import Path
import zipfile

import pytest
from PIL import Image

from streamreplace.benchmarks.registry import get


def png(color=(255, 0, 0)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
    return buffer.getvalue()


def write_parquet(path: Path, rows: list[dict]) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def write_zip(path: Path, members: dict[str, bytes] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as handle:
        for name, data in (members or {}).items():
            handle.writestr(name, data)


def prepare(name, raw, media, monkeypatch, count):
    bench = get(name)
    monkeypatch.setattr(bench.module, "EXPECTED_COUNT", count)
    return bench.prepare(raw, media)


def test_mmbench_rotations_hints_and_image_pointers(tmp_path):
    raw, media = tmp_path / "raw", tmp_path / "media"
    raw.mkdir()
    image = base64.b64encode(png()).decode()
    fields = ["index", "question", "hint", "A", "B", "C", "D", "answer", "category", "l2-category", "image"]
    rows = [["0", "Which color?", "A hint", "red", "blue", "", "nan", "B", "cat", "l2a", image],
            ["1000000", "Which color?", "A hint", "blue", "red", "", "nan", "A", "cat", "l2a", "0"],   # -> row 0's image
            ["1", "Q2 ", "nan", "x", "y", "z", "", "C", "cat", "l2b", image]]
    with open(raw / "MMBench_DEV_EN_legacy.tsv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(fields)
        writer.writerows(rows)
    samples = get("mmbench").prepare(raw, media)
    assert [s.id for s in samples] == ["0", "1000000", "1"]
    first, rotated, other = samples
    assert first.question == "A hint\nWhich color?" and first.choices == {"A": "red", "B": "blue"}
    assert first.metadata == {"question": 0, "rotation": 0, "rotations": 2, "category": "cat", "l2": "l2a",
                              "hint": "A hint", "raw_question": "Which color?"}
    assert rotated.metadata["rotation"] == 1 and rotated.answer == "A" and rotated.images == first.images
    assert Path(first.images[0]) == media / "0.png" and (media / "0.png").is_file()
    assert other.metadata["hint"] is None and other.question == "Q2" and other.metadata["rotations"] == 1
    assert get("mmbench").prompt(get("mmbench").view(other)) == (
        "Q2  There are several options:\nA. x\nB. y\nC. z\n\nAnswer with the option's letter from the given choices "
        "directly.")


def test_visulogic_jsonl_and_image_archive(tmp_path, monkeypatch):
    raw, media = tmp_path / "raw", tmp_path / "media"
    raw.mkdir()
    lines = [{"id": "00000", "question": "Q?\nA: A\nB: B\nC: C\nD: D ", "label": "b", "tag": "Quantitative",
              "image_path": "images/00000.png"},
             {"id": "00001", "question": "R?", "label": "D", "tag": "Spatial", "image_path": "images/00001.png"}]
    (raw / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
    write_zip(raw / "images.zip", {"images/00000.png": png(), "images/00001.png": png()})
    samples = prepare("visulogic", raw, media, monkeypatch, 2)
    assert [(s.id, s.answer, s.question, s.metadata) for s in samples] == [
        ("00000", "B", "Q?\nA: A\nB: B\nC: C\nD: D", {"category": "Quantitative"}),
        ("00001", "D", "R?", {"category": "Spatial"})]
    assert samples[0].choices == {k: k for k in "ABCD"} and Path(samples[0].images[0]).is_file()


def test_timeblind_roles_options_and_videos_read_in_place(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [{"index": 0, "type": "yes_no", "question": "Does it stop?", "answer": "Yes ", "video_path": "TimeBlind/videos/v0.mp4"},
            {"index": 1, "type": "yes_no", "question": "Does it stop?", "answer": "no", "video_path": "TimeBlind/videos/v1.mp4"},
            {"index": 2, "type": "multiple_choice", "question": "Which first? Option A: the boy  Option B: the girl",
             "answer": "b", "video_path": "TimeBlind/videos/v0.mp4"},
            {"index": 3, "type": "multiple_choice", "question": "Which first? Option A: the boy Option B: the girl",
             "answer": "A", "video_path": "TimeBlind/videos/v1.mp4"}]
    (raw / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (raw / "tags.json").write_text(json.dumps({"Order": [0, 2], "Speed": [2]}), encoding="utf-8")
    samples = get("timeblind").prepare(raw, tmp_path / "media")
    assert [(s.id, s.kind, s.answer) for s in samples] == [("0", "yesno", "yes"), ("1", "yesno", "no"),
                                                           ("2", "mcq", "B"), ("3", "mcq", "A")]
    assert [s.metadata["role"] for s in samples] == [0, 1, 2, 3] and {s.metadata["instance"] for s in samples} == {0}
    assert samples[2].choices == {"A": "the boy", "B": "the girl"}
    assert samples[2].metadata["tags"] == ["Order", "Speed"] and samples[1].metadata["tags"] == []
    assert samples[0].video == str(raw / "videos/v0.mp4")


def test_mme_question_ids_and_hashed_images(tmp_path):
    raw, media = tmp_path / "raw", tmp_path / "media"
    rows = [{"question_id": "code_reasoning/0020.png", "question": "Is it python? Please answer yes or no. ",
             "answer": "Yes", "category": "code_reasoning", "image": {"bytes": png(), "path": None}},
            {"question_id": "code_reasoning/0020.png", "question": "Is it java? Please answer yes or no.",
             "answer": "No", "category": "code_reasoning", "image": {"bytes": png(), "path": None}}]
    write_parquet(raw / "data/test-00000-of-00001.parquet", rows)
    samples = get("mme").prepare(raw, media)
    assert [(s.id, s.answer, s.kind) for s in samples] == [("code_reasoning/0020.png#1", "yes", "yesno"),
                                                           ("code_reasoning/0020.png#2", "no", "yesno")]
    assert samples[0].question == "Is it python? Please answer yes or no."
    assert Path(samples[0].images[0]) == media / "5f6cabef8c9d01ad.png" and samples[1].images == samples[0].images
    assert samples[0].metadata == {"image": "code_reasoning/0020.png", "category": "code_reasoning"}


def test_scienceqa_ids_count_every_row(tmp_path, monkeypatch):
    raw, media = tmp_path / "raw", tmp_path / "media"
    image = {"bytes": png(), "path": None}
    rows = [{"image": image, "hint": "H", "question": "Q1 ", "choices": ["a", "b"], "answer": 1, "subject": "s",
             "grade": "g", "category": "c"},
            {"image": None, "hint": "", "question": "no image", "choices": ["a", "b"], "answer": 0, "subject": "s",
             "grade": "g", "category": "c"},
            {"image": image, "hint": "", "question": "Q3", "choices": ["x", "y", "z"], "answer": 2, "subject": "s",
             "grade": "g", "category": "c"}]
    write_parquet(raw / "ScienceQA-IMG/test-00000-of-00001.parquet", rows)
    samples = prepare("scienceqa", raw, media, monkeypatch, 2)
    assert [(s.id, s.question, s.answer) for s in samples] == [("00000-0", "Context: H\nQ1", "B"), ("00000-2", "Q3", "C")]
    assert samples[1].choices == {"A": "x", "B": "y", "C": "z"} and Path(samples[1].images[0]) == media / "00000-2.png"


def test_tomato_ids_are_deduplicated_with_the_row_number(tmp_path, monkeypatch):
    raw, media = tmp_path / "raw", tmp_path / "media"
    write_zip(raw / "videos.zip")
    row = {"key": "0209-03", "question": "How many? ", "options": ["2", "1"], "answer": 1,
           "demonstration_type": "human", "motion_type": "count"}
    write_parquet(raw / "data/count-00000-of-00001.parquet", [row, {**row, "answer": 0}])
    samples = prepare("tomato", raw, media, monkeypatch, 2)
    assert [(s.id, s.answer) for s in samples] == [("count:0209-03", "B"), ("count:0209-03:1", "A")]
    assert samples[0].question == "How many? " and samples[0].choices == {"A": "2", "B": "1"}
    assert samples[0].video == str(media / "videos/human/0209-03.mp4")
    assert samples[0].metadata == {"reasoning": "count", "demonstration": "human", "motion": "count", "video": "0209-03"}


def test_mmstar_reads_both_option_layouts(tmp_path):
    raw, media = tmp_path / "raw", tmp_path / "media"
    image = {"bytes": png(), "path": None}
    rows = [{"index": 0, "question": "Which one?\nOptions: A: x, B: y, C: z", "answer": "A", "category": "cp",
             "l2_category": "l2", "image": image},
            {"index": 1, "question": "Hint: read it.\nQuestion: Which one?\nChoices:\n(A) p\n(B) q", "answer": "B",
             "category": "cp", "l2_category": "l2", "image": image}]
    write_parquet(raw / "mmstar.parquet", rows)
    first, second = get("mmstar").prepare(raw, media)
    assert (first.question, first.choices) == ("Which one?", {"A": "x", "B": "y", "C": "z"})
    assert (second.question, second.choices) == ("Which one?", {"A": "p", "B": "q"})
    assert get("mmstar").prompt(get("mmstar").view(second)) == (
        "Hint: read it.\nQuestion: Which one?\nChoices:\n(A) p\n(B) q\nAnswer with the option's letter from the given "
        "choices directly")


def test_videomme_short_split_and_relative_subtitle_path(tmp_path, monkeypatch):
    raw, media = tmp_path / "raw", tmp_path / "media"
    write_zip(raw / "videos_chunked_01.zip")
    write_zip(raw / "subtitle.zip")
    row = {"question_id": "001-1", "videoID": "fFjv93ACGo8", "duration": "short", "domain": "Knowledge",
           "task_type": "Counting Problem", "question": "How many? ", "options": ["A. one.", "B. two.", "C. 3", "D) 4"],
           "answer": "C"}
    write_parquet(raw / "videomme/test-00000-of-00001.parquet", [row, {**row, "question_id": "301-1", "duration": "medium"}])
    (sample,) = prepare("videomme", raw, media, monkeypatch, 1)
    assert (sample.id, sample.question, sample.answer) == ("001-1", "How many?", "C")
    assert sample.choices == {"A": "one.", "B": "two.", "C": "3", "D": "4"}
    assert sample.video == str(media / "data/fFjv93ACGo8.mp4")
    assert sample.metadata == {"video": "fFjv93ACGo8", "duration": "short", "domain": "Knowledge",
                               "task": "Counting Problem", "subtitle": "media/subtitle/fFjv93ACGo8.srt"}


def test_dynamath_stable_ids_kinds_and_choices(tmp_path, monkeypatch):
    raw, media = tmp_path / "raw", tmp_path / "media"
    for variant in range(1, 11):
        write_parquet(raw / f"data/sample{variant}-00000-of-00001.parquet", [
            {"id": 6, "question": "Period? ", "ground_truth": "2.094 ", "answer_type": "float",
             "decoded_image": {"bytes": png(), "path": None}, "subject": "algebra", "knowledge_level": "high school"},
            {"id": 7, "question": "Which? Choices: (A) one (B) two", "ground_truth": "B", "answer_type": "multiple choice",
             "decoded_image": {"bytes": png(), "path": None}, "subject": "graph", "knowledge_level": "college"}])
    samples = prepare("dynamath", raw, media, monkeypatch, 20)
    assert samples[0].id == "4632251fe0dcaf354fd88f5b"                 # seed 6, variant 1
    assert [(s.kind, s.question, s.answer) for s in samples[:2]] == [("numeric", "Period?", "2.094"),
                                                                     ("mcq", "Which?", "B")]
    assert samples[1].choices == {"A": "one", "B": "two"}
    assert samples[1].metadata == {"seed": "7", "variant": 1, "subject": "graph", "knowledge_level": "college",
                                   "raw_question": "Which? Choices: (A) one (B) two"}
    assert Path(samples[-1].images[0]) == media / "variant10/7.png"


def test_mathvista_letters_and_tolerances(tmp_path):
    raw, media = tmp_path / "raw", tmp_path / "media"
    image = {"bytes": png(), "path": None}
    base = {"decoded_image": image, "unit": None, "metadata": {"task": "figure question answering"}}
    rows = [{**base, "pid": "1", "question": "Which? ", "question_type": "multi_choice", "choices": ["3", "4 "],
             "answer": "4", "answer_type": "text", "precision": None, "query": "Hint: ...\nQuestion: Which?"},
            {**base, "pid": "2", "question": "How many?", "question_type": "free_form", "choices": None, "answer": "7",
             "answer_type": "integer", "precision": None, "query": "q2"},
            {**base, "pid": "3", "question": "How long?", "question_type": "free_form", "choices": None, "answer": "1.5",
             "answer_type": "float", "precision": 1.0, "query": "q3"}]
    write_parquet(raw / "data/testmini-00000-of-00001.parquet", rows)
    first, second, third = get("mathvista").prepare(raw, media)
    assert (first.kind, first.answer, first.choices, first.question) == ("mcq", "B", {"A": "3", "B": "4"}, "Which?")
    assert (second.kind, second.metadata["tolerance"]) == ("numeric", 0)
    assert third.kind == "numeric" and third.metadata["tolerance"] == 0.5 * 10 ** -1 + 1e-9
    assert get("mathvista").prompt(get("mathvista").view(first)) == "Hint: ...\nQuestion: Which?"


def test_mmmu_options_and_image_order(tmp_path, monkeypatch):
    raw, media = tmp_path / "raw", tmp_path / "media"
    rows = [{"id": "validation_Art_1", "question": "<image 1> <image 2> Which?", "options": "['x', 'y']", "answer": "B",
             "question_type": "multiple-choice", "image_1": {"bytes": png(), "path": None},
             "image_2": {"bytes": png((0, 0, 255)), "path": None}, "subfield": "sf", "topic_difficulty": "Easy"},
            {"id": "validation_Art_2", "question": "Value?", "options": "[]", "answer": "5", "question_type": "open",
             "image_1": {"bytes": png(), "path": None}, "image_2": None, "subfield": "sf", "topic_difficulty": "Hard"}]
    write_parquet(raw / "Art/validation-00000-of-00001.parquet", rows)
    first, second = prepare("mmmu", raw, media, monkeypatch, 2)
    assert (first.kind, first.choices, first.answer) == ("mcq", {"A": "x", "B": "y"}, "B")
    assert [Path(p).name for p in first.images] == ["validation_Art_1_1.png", "validation_Art_1_2.png"]
    assert (second.kind, len(second.images)) == ("text", 1)
    assert second.metadata == {"subject": "Art", "subfield": "sf", "difficulty": "Hard"}


def test_videoreasonbench_open_answers_and_state_metadata(tmp_path, monkeypatch):
    raw, media = tmp_path / "raw", tmp_path / "media"
    write_zip(raw / "videos.zip")
    write_parquet(raw / "cup/test-00000-of-00001.parquet", [
        {"id": "cup_1", "question": "Q? Final Answer:", "answer": None, "video": "cup/1.mp4", "dim": "order_state",
         "visible_time": "end", "num_state": 3, "num_operation": 2, "states": [[1, 2], [2, 1]], "coords": [0, 1],
         "moves": ["swap"]}])
    (sample,) = prepare("videoreasonbench", raw, media, monkeypatch, 1)
    assert (sample.kind, sample.answer, sample.video) == ("open", "", str(media / "cup/1.mp4"))
    assert sample.metadata["task"] == "cup" and sample.metadata["states"] == [[1, 2], [2, 1]]
