"""The fine-tuning benchmarks: LingoQA (references grouped per question, training deduplicated), EyePCR (the option
reorder, the official prompt wording, the option-text parser with its post hoc rules and the first-round SENSITIVITY
parser) and OpenUAV-QA (frame lists from the ModelScope listing; the official template and extractor, which need the
fetched upstream code)."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional
import zipfile

import numpy as np
from PIL import Image
import pytest

from streamreplace.benchmarks import common, registry
from streamreplace.benchmarks.finetune import eyepcr, lingoqa
from streamreplace.benchmarks.finetune import openuav_qa as uav
from streamreplace.core.schema import Sample
from streamreplace.media.frames import offline_frames


# ---------- LingoQA ----------

def write_split(raw: Path, part: str, rows: list[dict], kind: str):
    import pyarrow as pa
    import pyarrow.parquet as pq
    for row in rows:
        for frame in ast.literal_eval(row["images"]) if isinstance(row["images"], str) else row["images"]:
            (raw / part / frame).parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 10)).save(raw / part / frame)
    (raw / part).mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), raw / part / f"{kind}.parquet")


def lingo_frames(split, segment):
    return [f"images/{split}/{segment}/{i}.jpg" for i in range(5)]


def test_lingoqa_groups_references_and_dedupes_training(tmp_path):
    pytest.importorskip("pyarrow")
    raw, media = tmp_path / "raw", tmp_path / "media"
    val = [{"question_id": "q1", "segment_id": "s1", "images": str(lingo_frames("val", "s1")), "question": "Lights?",
            "answer": a} for a in ("Yes, green.", "Green.")]
    write_split(raw, "evaluation", val, "val")
    train = [{"question_id": f"t{i}", "segment_id": "s2", "images": lingo_frames("train", "s2"), "question": q,
              "answer": "x"} for i, q in enumerate(["What now?", "Why?", "What now?"])]   # (segment, question) twice
    write_split(raw, "action", train, "train")
    write_split(raw, "scenery", [{"question_id": "t9", "segment_id": "s3", "images": lingo_frames("train", "s3"),
                                  "question": "Road?", "answer": "wet"}], "train")
    test = lingoqa.prepare(raw, media)
    assert [s.id for s in test] == ["q1"] and test[0].metadata["references"] == ["Yes, green.", "Green."]
    images, meta = offline_frames(test[0].video, {"sampler": "all"})
    assert len(images) == 5 and meta["fps"] == 1.0 and meta["files"][0] == "../../raw/evaluation/images/val/s1/0.jpg"
    training = lingoqa.prepare_train(raw, media)
    assert [s.id for s in training] == ["t0", "t1", "t9"] and training[2].metadata["subset"] == "scenery"
    bench = registry.get("lingoqa")
    assert bench.has_train and bench.judge.value == "lingo" and "references" not in bench.view(test[0]).metadata


def test_lingoqa_sends_the_cleaned_reply_to_the_judge():
    sample = Sample("q1", "Lights?", "Green.", "open", metadata={"references": ["Yes, green.", "Green."]})
    assert lingoqa.parse(sample, "<think>hm</think> Green") == ("Green", "text")
    assert lingoqa.parse(sample, "<think>only thinking</think>") == (None, "empty")
    assert lingoqa.metrics([{"correct": True}, {"correct": False}]) == {"acc": 50.0}


# ---------- EyePCR ----------

OPTIONS = ["Phacoemulsification", "Vitrectomy", "Intraocular Lens (IOL) Implantation", "Trabeculectomy"]


def conversation(case, clip, questions, mcq=True):
    turns = [{"role": "user", "content": "<video>" + (eyepcr.INTRO if mcq else "Answer concisely.")},
             {"role": "assistant", "content": eyepcr.REPLY if mcq else "Sure."}]
    for question, options, answer in questions:
        turns += [{"role": "user", "content": f"{question}{eyepcr.CHOICE}{options}" if mcq else question},
                  {"role": "assistant", "content": answer}]
    return {"messages": turns, "videos": [f"videos_resized_slice/{case}/{clip}.mp4"]}


def write_eyepcr(raw):
    raw.mkdir()
    test = [conversation("case_0001", "0.00-10.00", [("What surgery is shown?", OPTIONS, OPTIONS[2]),
                                                     ("Which tool?", ["Forceps", "Hook", "Blade", "Cannula"], "hook."),
                                                     ("Which zone?", ["Central", "Peripheral", "Anterior", "Posterior"], "Periphery")]),
            conversation("case_0001", "10.00-20.00", [("What is described?", [], "An iris repair.")], mcq=False)]
    train = [conversation("case_0002", "5.00-9.50", [("What surgery is shown?", OPTIONS, OPTIONS[0]),
                                                     ("Size?", ["Small", "small", "Large", "Medium"], "Small")])]
    (raw / "test.json").write_text(json.dumps(test))
    (raw / "train.json").write_text(json.dumps(train))
    (raw / "test_cases.txt").write_text("case_0001\n")
    with zipfile.ZipFile(raw / "MCQ.zip", "w") as archive:
        archive.writestr("Surgical_Operation/case_0001.json", json.dumps({"0.00-10.00": [
            {"Q": "What surgery is shown?", "options": OPTIONS, "answer": OPTIONS[2]}]}))
        for category in ("Intraoperative_Anatomy", "Pathological_Diagnosis"):    # filed twice, the second schema
            archive.writestr(f"{category}/case_0001.json", json.dumps({"0.00-10.00": [
                {"Q": "Which tool?", "Options": ["Forceps", "Hook", "Blade", "Cannula"], "Answer": "hook."}]}))
        archive.writestr("Surgical_Operation/case_0002.json", json.dumps({"5.00-9.50": [
            {"Q": "What surgery is shown?", "options": OPTIONS, "answer": OPTIONS[0]},
            {"Q": "Size?", "options": ["Small", "small", "Large", "Medium"], "answer": "Small"}]}))
    with zipfile.ZipFile(raw / eyepcr.CLIPS, "w") as archive:
        for path in ("case_0001/0.00-10.00", "case_0001/10.00-20.00", "case_0002/5.00-9.50"):
            archive.writestr(f"videos_resized_slice/{path}.mp4", b"\0")


def test_eyepcr_keeps_the_multiple_choice_questions_of_the_official_split(tmp_path):
    raw = tmp_path / "raw"
    write_eyepcr(raw)
    test, train = eyepcr.prepare(raw, tmp_path / "media"), eyepcr.prepare_train(raw, tmp_path / "media")
    # the open-ended conversation is skipped; "Periphery" is no option; "hook." is respelled as its option
    assert [s.id for s in test] == ["case_0001_0.00-10.00_0", "case_0001_0.00-10.00_1"]
    assert [s.answer for s in test] == [OPTIONS[2], "Hook"] and all(s.kind == "text" and not s.choices for s in test)
    assert test[0].metadata["category"] == "Surgical_Operation"
    assert test[1].metadata["category"] == "Intraoperative_Anatomy+Pathological_Diagnosis"
    assert test[0].video == str(raw / "videos_resized_slice/case_0001/0.00-10.00.mp4") and (raw / "videos_resized_slice").is_dir()
    assert [s.id for s in train] == ["case_0002_5.00-9.50_0"]        # two equal options ("Small", "small"): dropped
    bench = registry.get("eyepcr")
    assert bench.has_train and bench.judge.value == "none" and bench.GROUP == "case"


def test_eyepcr_split_files_must_follow_the_test_cases(tmp_path):
    raw = tmp_path / "raw"
    write_eyepcr(raw)
    (raw / "test_cases.txt").write_text("case_0002\n")
    with pytest.raises(AssertionError, match="official case split"):
        eyepcr.prepare(raw, tmp_path / "media")


def test_eyepcr_options_are_reordered_per_question_and_the_prompt_keeps_the_official_wording():
    orders = {tuple(eyepcr.reorder(f"case_{i}", OPTIONS)) for i in range(40)}
    assert len(orders) > 10 and all(sorted(o) == sorted(OPTIONS) for o in orders)
    assert eyepcr.reorder("case_7", OPTIONS) == eyepcr.reorder("case_7", list(reversed(OPTIONS)))
    sample = Sample("x", "What surgery is shown?", OPTIONS[0], "text", metadata={"options": OPTIONS})
    assert eyepcr.prompt(sample) == (eyepcr.INTRO + "\nWhat surgery is shown?\nCHOICE: ['Phacoemulsification', "
                                     "'Vitrectomy', 'Intraocular Lens (IOL) Implantation', 'Trabeculectomy']")


@pytest.mark.parametrize("reply, expected, how", [
    ("Vitrectomy", "Vitrectomy", "option_text"),
    ("vitrectomy.", "Vitrectomy", "option_text"),
    ("The answer is Trabeculectomy.", "Trabeculectomy", "option_text"),
    ("<think>maybe vitrectomy</think>Phacoemulsification", "Phacoemulsification", "option_text"),
    ("This segment shows intraocular lens (IOL) implantation into the capsular bag.",
     "Intraocular Lens (IOL) Implantation", "option_in_reply"),
    ("It is vitrectomy rather than trabeculectomy.", None, "several_options"),
    ("A cataract operation.", None, "no_option"),
    ("C", "Intraocular Lens (IOL) Implantation", "letter"),
    ("The answer is B.", "Vitrectomy", "letter"),
    ("Answer: (D)", "Trabeculectomy", "letter"),
    ("**A**", "Phacoemulsification", "letter"),
    ("E", None, "no_option"),                                           # no fifth option
    ("B. Laminar flow", None, "no_option"),                             # a letter with text that is no option
    ("The correct answer is 'Trabeculectomy'. Unlike vitrectomy, which works on the vitreous, it",
     "Trabeculectomy", "marked_option"),
    ("Vitrectomy is unlikely; trabeculectomy fits.", None, "several_options"),   # no marker: no leading-option reading
    ("", None, "empty"),
])
def test_eyepcr_replies_are_matched_to_one_option(reply, expected, how):
    sample = Sample("x", "q", OPTIONS[1], "text", metadata={"options": OPTIONS})
    assert eyepcr.parse(sample, reply) == (expected, how)
    assert eyepcr.correct(sample, expected) == (expected == OPTIONS[1])


def test_eyepcr_an_option_inside_another_named_option_does_not_count():
    options = ["Cataract", "Cataract surgery", "Glaucoma", "Retina"]
    sample = Sample("x", "q", "Cataract surgery", "text", metadata={"options": options})
    assert eyepcr.parse(sample, "This is cataract surgery on the left eye.") == ("Cataract surgery", "option_in_reply")
    assert eyepcr.parse(sample, "A dense cataract is visible.") == ("Cataract", "option_in_reply")
    assert eyepcr.match(["No", "Yes", "Unclear", "Partly"], "Nothing is visible") == (None, "no_option")   # whole words


def test_eyepcr_the_first_round_parser_stays_available_for_the_sensitivity_tables():
    sample = Sample("x", "q", OPTIONS[1], "text", metadata={"options": OPTIONS})
    assert eyepcr.parse(sample, "The answer is B.") == ("Vitrectomy", "letter")
    [(heading, first_round)] = eyepcr.SENSITIVITY.items()
    assert heading == "with the first-round parser (no marked-option or letter rule)"   # the report's table heading
    assert first_round(sample, "The answer is B.") == (None, "no_option")
    assert first_round(sample, "The answer is Vitrectomy. Not trabeculectomy.") == (None, "several_options")


# ---------- OpenUAV-QA ----------

def frames_dir(root: Path, n: int) -> Path:
    root.mkdir(parents=True)
    for i in range(n):                                   # a distinct red value per frame (n <= 333 < 3 * 256)
        Image.new("RGB", (8, 8), (i % 256, i // 256, 0)).save(root / f"{i + 1:05d}.png")
    return root


def item(i, split=None):
    row = {"id": f"TravelUAV_Train_{i:05d}", "video_name": f"TravelUAV_Train_{i:05d}", "question": f"q{i}",
           "options": {"D": "d", "A": "a", "C": "c", "B": "b"}, "answer": "C", "source_map": "M",
           "source_traj": f"t{i}", "total_frames": 40, "task_type": "path_planning"}
    return {**row, "split": split} if split else row


def test_openuav_prepare_writes_frame_lists_and_skips_trajectories_without_frames(tmp_path):
    raw, media = tmp_path / "raw", tmp_path / "media"
    raw.mkdir()
    (raw / "TravelUAV_test.jsonl").write_text("\n".join(json.dumps(item(i, "seen_map")) for i in (1, 2)) + "\n\n")
    (raw / "TravelUAV_train.jsonl").write_text(json.dumps(item(3)) + "\n\n")
    frames_dir(raw / "frames/TravelUAV_Train_00001", 40)
    frames_dir(raw / "frames/TravelUAV_Train_00003", 12)      # shorter than total_frames: kept, counted
    tree = [{"folder": f"TravelUAV_Train_0000{i}", "files": [{"name": p.name, "size": 1, "sha256": ""}
                                                              for p in sorted((raw / f"frames/TravelUAV_Train_0000{i}").iterdir())]}
            for i in (1, 3)]                                   # 00002 has no folder on ModelScope: skipped
    (raw / "modelscope_tree.jsonl").write_text("".join(json.dumps(e) + "\n" for e in tree))
    with pytest.raises(ValueError, match="incomplete"):
        uav.prepare(raw, media)
    (raw / ".listing_complete").write_text("2 folders\n")
    test, train = uav.prepare(raw, media), uav.prepare_train(raw, media)
    assert [s.id for s in test] == ["TravelUAV_Train_00001"] and [s.id for s in train] == ["TravelUAV_Train_00003"]
    sample = test[0]
    assert list(sample.choices) == ["A", "B", "C", "D"] and sample.metadata["split"] == "seen_map"
    frames, meta = offline_frames(sample.video, {"sampler": "all"})
    assert len(frames) == 32 and meta["fps"] == 2.0 and meta["files"][0] == "../raw/frames/TravelUAV_Train_00001/00001.png"
    assert train[0].metadata["frames_listed"] == 12 and train[0].metadata["split"] == "train"
    assert registry.get("openuav_qa").has_train


def test_openuav_trajectories_with_frames_served_empty_are_skipped(tmp_path):
    raw = tmp_path / "raw"
    frames_dir(raw / "frames/TravelUAV_Train_00007", 5)
    row = {"id": "TravelUAV_Train_00007", "video_name": "TravelUAV_Train_00007", "question": "q", "answer": "A",
           "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "source_map": "M", "source_traj": "t", "total_frames": 5}
    (raw / "TravelUAV_train.jsonl").write_text(json.dumps(row) + "\n")
    (raw / "modelscope_tree.jsonl").write_text(json.dumps({"folder": "TravelUAV_Train_00007", "files": [
        {"name": f"{i:05d}.png", "size": 1, "sha256": ""} for i in range(1, 6)]}) + "\n")
    (raw / ".listing_complete").write_text("1\n")
    assert len(uav.prepare_train(raw, tmp_path / "media")) == 1
    (raw / "modelscope_unavailable.json").write_text(json.dumps(["TravelUAV_Train_00007/00003.png"]))
    assert uav.prepare_train(raw, tmp_path / "media") == []


@pytest.mark.requires_upstream("openuav_qa")
@pytest.mark.parametrize("n", [10, 32, 33, 54, 333])
def test_openuav_frames_are_the_official_selection(tmp_path, n):
    folder = frames_dir(tmp_path / "TravelUAV_Train_00001", n)
    load = common.upstream_functions(uav.DATASET, "SISBenchDataset._load_images",
                                     scope={"np": np, "Image": Image, "List": List, "Optional": Optional})["SISBenchDataset._load_images"]
    stub = SimpleNamespace(max_frames=32, image_size=None, min_pixels=None, max_pixels=None,
                           _resolve_video_input=lambda item: folder)
    official = [im.getpixel((0, 0)) for im in load(stub, {"video_name": folder.name})]
    ours = [Image.open(folder / f).convert("RGB").getpixel((0, 0)) for f in uav.select_frames([p.name for p in folder.iterdir()])]
    assert ours == official and len(ours) == min(n, 32)


@pytest.mark.requires_upstream("openuav_qa")
def test_openuav_prompt_is_the_official_template():
    sample = Sample("x", "How should the drone fly to reach the red truck?", "B", "mcq",
                    {"B": "Cruise forward.", "A": "Ascend.", "D": "Hover.", "C": "Land."})
    assert uav.prompt(sample) == (
        "Watch the video carefully and answer the following multiple choice question.\n\n"
        "Question:\nHow should the drone fly to reach the red truck?\n\n"
        "Options:\nA. Ascend.\nB. Cruise forward.\nC. Land.\nD. Hover.\n\n"
        "Please select the correct answer from the options above. Only output the letter (A, B, C, or D) "
        "corresponding to your answer.")


@pytest.mark.requires_upstream("openuav_qa")
@pytest.mark.parametrize("text, letter", [
    ("C", "C"), ("C.", "C"), ("<think>\nmaybe A\n</think>\n\nD", "D"), ("The answer is b", "B"),
    ("\u7b54\u6848\u662fA", "A"), ("I pick\nB", "B"), ("A drone should cruise", "A"),   # upstream quirk kept
    ("<|im_end|>", None), ("no idea", None)])
def test_openuav_answer_extraction_is_the_official_cascade(text, letter):
    sample = Sample("x", "q", "A", "mcq", {k: k for k in "ABCD"})
    assert uav.parse(sample, text)[0] == letter
