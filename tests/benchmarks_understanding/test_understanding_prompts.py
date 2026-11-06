"""Golden prompt strings of the 12 understanding benchmarks. The prompt is part of the request identity: the expected
strings below were rendered by the code of the paper's runs for the same synthetic samples, and every formal row of
the paper's runs was checked against this code (manifests/prepared.yaml holds the per-dataset prompt-map hashes).
Prompts are rendered from a SampleView: no answer, no private metadata."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from streamreplace.benchmarks.registry import get
from streamreplace.core.schema import Sample, SampleView

GUIDE = ("\n## Answer Instruction Please provide an answer to the question outlined above. Your response should adhere "
         "to the following JSON format, which includes two keys: 'solution' and 'short answer'. The 'solution' key can "
         "contain detailed steps needed to solve the question, and the 'short answer' key should provide a concise "
         "response. ")
EXAMPLE = ("\n\nExample of expected JSON response format:\n\n{\n    \"solution\": \"[Detailed step-by-step explanation]\","
           "\n    \"short answer\": \"[Concise Answer]\"\n}")
MCQ = "Answer with the option's letter from the given choices directly."

CASES = [
    ("dynamath", SampleView("d1", "What is the period of y = sin(2x)?", "numeric",
                            metadata={"raw_question": "What is the period of y = sin(2x)?", "seed": "1"}),
     "## Question\n What is the period of y = sin(2x)?" + GUIDE + "Format the answer as a three-digit floating-point "
     "number and provide it in the 'short answer' key." + EXAMPLE),
    ("dynamath", SampleView("d2", "Which is larger?", "mcq", {"A": "1", "B": "2"},
                            {"raw_question": "Which is larger? Choices: (A) 1 (B) 2", "seed": "2"}),
     "## Question\n Which is larger? Choices: (A) 1 (B) 2" + GUIDE + "Provide the corresponing choice option in the "
     "'short answer' key, such as 'A', 'B', 'C', or 'D'." + EXAMPLE),
    ("dynamath", SampleView("d3", "Name the shape.", "text", metadata={"raw_question": "Name the shape.", "seed": "3"}),
     "## Question\n Name the shape." + GUIDE + "Float numbers in the answer should be formatted as three-digit "
     "floating-point numbers." + EXAMPLE),
    ("mathvista", SampleView("1", "How many?", "numeric", metadata={
        "query": "Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, "
                 "3, at the end.\nQuestion: How many?"}),
     "Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the "
     "end.\nQuestion: How many?"),
    ("mmbench", SampleView("0", "A hint\nWhich color?", "mcq", {"A": "red", "B": "blue"},
                           {"hint": "A hint", "raw_question": "Which color?"}),
     "A hint Which color? There are several options:\nA. red\nB. blue\n\n" + MCQ),
    ("mmbench", SampleView("1", "Which color?", "mcq", {"A": "red", "B": "blue", "C": "green"},
                           {"hint": None, "raw_question": "Which color?"}),
     "Which color? There are several options:\nA. red\nB. blue\nC. green\n\n" + MCQ),
    ("mme", SampleView("x#1", "Is there a dog? Please answer yes or no.", "yesno"),
     "Is there a dog? Please answer yes or no."),
    ("mme", SampleView("x#2", "Is there a cat?", "yesno"), "Is there a cat?\nAnswer with yes or no."),
    ("mmmu", SampleView("m1", "<image 1> What is shown?", "mcq", {"A": "$6", "B": "$7"}),
     "<image 1> What is shown?\n(A) $6\n(B) $7\n\n" + MCQ),
    ("mmmu", SampleView("m2", "Compute the value.", "text"),
     "Compute the value.\nAnswer the question using a single word or phrase."),
    ("mmstar", SampleView("0", "Which option?", "mcq", {"A": "x", "B": "y"},
                          {"raw_question": " Which option?\nOptions: A: x, B: y "}),
     "Which option?\nOptions: A: x, B: y\nAnswer with the option's letter from the given choices directly"),
    ("scienceqa", SampleView("00000-0", "Context: Some context.\nWhich is a metal?", "mcq", {"A": "iron", "B": "wood"}),
     "Context: Some context.\nWhich is a metal?\nA. iron\nB. wood\n" + MCQ),
    ("visulogic", SampleView("00000", "Pick one:\nA: A  \nB: B  \nC: C  \nD: D", "mcq", {k: k for k in "ABCD"}),
     "Pick one:\nA: A  \nB: B  \nC: C  \nD: DJust Answer the question following this format: Answer:\\boxed{$LETTER}"),
    ("videomme", SampleView("001-1", "What happens first?", "mcq", {"A": "a.", "B": "b.", "C": "c.", "D": "d."}),
     "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond "
     "with only the letter (A, B, C, or D) of the correct option.\nWhat happens first?\nA. a.\nB. b.\nC. c.\nD. d.\n\n"
     + MCQ),
    ("tomato", SampleView("count:k", "How many times does it turn?", "mcq", {"A": "1", "B": "2", "C": "3"}),
     "You will be provided with frames uniformly sampled from a video, the frames are provided in chronological order "
     "of the video. Analyze these frames and provide the answer to the question about the video content. Answer the "
     "multiple-choice question about the video content.\n\nYou must use these frames to answer the multiple-choice "
     "question; do not rely on any externel knowledge or commonsense. \n\n<question> \nHow many times does it turn? \n"
     "</question>\n\n<options> \n{'A': '1', 'B': '2', 'C': '3'} \n</options>\n\nEven if the information in these "
     "separate frames is not enough to answer the question, PLEASE TRY YOUR BEST TO GUESS AN ANSWER WHICH YOU THINK "
     "WOULD BE THE MOST POSSIBLE ONE BASED ON THE QUESTION. \n\nDO NOT GENERATE ANSWER SUCH AS 'NOT POSSIBLE TO "
     "DETERMINE.' \n"),
    ("timeblind", SampleView("0", "Does the boy eat first?", "yesno"), "Does the boy eat first? Please output Yes or No."),
    ("timeblind", SampleView("1", "Which is first? Option A: the boy Option B: the girl", "mcq",
                             {"A": "the boy", "B": "the girl"}),
     "Which is first? Option A: the boy Option B: the girl Please output A or B."),
    ("videoreasonbench", SampleView("v1", "Watch the video.\nProvide a summary of the final answer after 'Final Answer:'",
                                    "open"),
     "Watch the video.\nProvide a summary of the final answer after 'Final Answer:'"),
]


@pytest.mark.parametrize("name, view, expected", CASES, ids=[f"{c[0]}-{c[1].id}" for c in CASES])
def test_prompt_matches_the_original_rendering(name, view, expected):
    assert get(name).prompt(view) == expected


@pytest.mark.parametrize("name", ["videomme", "tomato", "timeblind"])
def test_video_prompts_need_only_question_choices_and_kind(name):
    """The plugin's calibration renders these prompts from bare (question, choices, kind) objects."""
    view = next(c[1] for c in CASES if c[0] == name)
    bare = SimpleNamespace(question=view.question, choices=view.choices, kind=view.kind)
    assert get(name).prompt(bare) == get(name).prompt(view)


def test_prompts_never_contain_the_reference_or_private_metadata():
    bench = get("timeblind")
    sample = Sample("6", "Does it stop? zzz-question", "no", "yesno", video="v.mp4",
                    metadata={"instance": 1, "role": 2, "type": "yes_no", "tags": []})
    view = bench.view(sample)
    assert not hasattr(view, "answer") and "role" not in view.metadata and view.metadata["instance"] == 1
    assert bench.prompt(view) == "Does it stop? zzz-question Please output Yes or No."
    vrb = get("videoreasonbench")
    sample = Sample("v", "q", "secret", "open", video="v.mp4",
                    metadata={"task": "cup", "states": [[1]], "coords": [0], "moves": ["x"], "visible_time": "end"})
    assert set(vrb.view(sample).metadata) == {"task", "visible_time"}
