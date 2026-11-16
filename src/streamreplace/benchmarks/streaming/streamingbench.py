"""StreamingBench, Real-Time Visual Understanding (THUNLP-MT/StreamingBench): 500 videos, 2,500 four-option questions
in 10 tasks, each asked at a time stamp t. This is the part streaming-model papers usually report; Omni-Source
(audio) and Contextual Understanding are not run.

Paper use: native streaming ability with and without Stream++ (Supp. D.4), on the questions asked within the first
120 s (MAX_SECONDS: 700 of the 2,500). Reported: accuracy with unanswered questions counted wrong (`acc`).

Protocol = the official one (src/benchmark/StreamingBench.py, context_time -1 as in scripts/eval.sh): the clip [0, t]
cut by utils/video_execution.py split_video (ffmpeg -ss 0 -t t, libx264 + aac, re-encoded), the question at its end.
Questions: the official src/data/questions_real.json (the evaluation's input; the dataset's CSV is another revision
with edited texts, so it is not used). Ids = <sample>_<k>, k = the question's position among that video's questions in
the file. Prompt: PROMPT_TEMPLATE + "\\n\\nThe best option is:", verbatim. Scoring: src/data/count.py "real", verbatim
logic: right iff the reply's first character is the answer letter; the official accuracy skips empty replies
(`acc_official`), `acc` counts them wrong (as Video-MME's two denominators).
Annotations and code: the fetched repository (manifests/upstream.yaml `streamingbench`). Media: the real-time zips ->
media/videos/sample_<n>_real.mp4 (src/data/move_video.py's naming); clips -> media/clips/<video>_0_<t>.mp4, cut with
the FFmpeg of the `ffmpeg` environment (STREAMREPLACE_FFMPEG overrides it; STREAMREPLACE_CLIP_WORKERS = parallel
ffmpeg processes, default cores // 6). Re-encoding costs about 3 CPU-seconds per second of 1080p video."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import zipfile

from .. import common
from ..base import JudgePolicy
from ...core.schema import Sample
from ...models.upstream import upstream_path

NAME = "streamingbench"
MEDIA = "video"
JUDGE = JudgePolicy.NONE
MAX_SECONDS = 120                        # the paper's scope: questions asked within the first 120 s
EXPECTED_COUNT = 700                     # of 2,500 questions
QUESTIONS = "streamingbench/src/data/questions_real.json"
TEMPLATE = "streamingbench/src/benchmark/StreamingBench.py"
MEMBER = re.compile(r"^sample_(\d+)/video\.mp4$")


def ffmpeg() -> str:
    """The FFmpeg 6 binary of the `ffmpeg` environment (libx264), or $STREAMREPLACE_FFMPEG."""
    from ... import paths
    return os.environ.get("STREAMREPLACE_FFMPEG") or str(Path(paths.current().envs) / "ffmpeg" / "bin" / "ffmpeg")


def seconds(stamp: str) -> int:
    """StreamingBench.py: "00:03:10" -> 190."""
    return sum(int(x) * 60 ** i for i, x in enumerate(reversed(stamp.split(":"))))


def items() -> list[tuple[str, dict]]:
    out, seen = [], {}
    for entry in json.loads(upstream_path(QUESTIONS).read_text(encoding="utf-8")):
        video = Path(entry["video_path"]).name                     # sample_<n>_real.mp4
        sample = video.removesuffix("_real.mp4")
        for q in entry["questions"]:
            seen[sample] = seen.get(sample, 0) + 1
            t = seconds(q["time_stamp"])
            out.append((f"{sample}_{seen[sample]}",
                        {"video": video, "t": t, "question": q["question"], "options": q["options"],
                         "answer": q["answer"], "task": q["task_type"], "ability": q.get("required_ability"),
                         "clip": f"{Path(video).stem}_0_{t}.mp4"}))
    return out


def extract(raw: Path, media: Path, wanted: set[str]) -> None:
    """The wanted source videos out of the real-time zips, renamed as move_video.py does (sample_<n>/video.mp4 ->
    sample_<n>_real.mp4)."""
    missing = {name for name in wanted if not (media / "videos" / name).is_file()}
    for archive in sorted(raw.glob("Real-Time Visual Understanding_*.zip")):
        if not missing:
            break
        with zipfile.ZipFile(archive) as handle:
            for info in handle.infolist():
                match = MEMBER.match(info.filename)
                name = f"sample_{match[1]}_real.mp4" if match else None
                if name not in missing:
                    continue
                target = media / "videos" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + ".part")
                with handle.open(info) as source, open(partial, "wb") as out:
                    while chunk := source.read(1 << 22):
                        out.write(chunk)
                partial.rename(target)
                missing.discard(name)
    if missing:
        raise ValueError(f"StreamingBench: {len(missing)} source videos not in the downloaded zips, e.g. "
                         f"{sorted(missing)[:3]}")


def cut(source: Path, clip: Path, t: int) -> None:
    """split_video(video, 0, t) as the official ffmpeg-python call renders it: -ss 0 before -i, -t t, libx264 + aac."""
    clip.parent.mkdir(parents=True, exist_ok=True)
    partial = clip.with_name(clip.stem + ".part.mp4")
    subprocess.run([ffmpeg(), "-v", "error", "-y", "-ss", "0", "-i", str(source), "-acodec", "aac", "-t", str(t),
                    "-vcodec", "libx264", str(partial)], check=True)
    partial.rename(clip)


def prepare(raw: Path, media: Path) -> list[Sample]:
    from concurrent.futures import ThreadPoolExecutor
    rows = [(sid, f) for sid, f in items() if f["t"] <= MAX_SECONDS]
    extract(raw, media, {f["video"] for _, f in rows})
    todo = list({f["clip"]: (media / "videos" / f["video"], media / "clips" / f["clip"], f["t"]) for _, f in rows
                 if not (media / "clips" / f["clip"]).is_file()}.values())    # questions at the same time share a clip
    workers = int(os.environ.get("STREAMREPLACE_CLIP_WORKERS", max(1, (os.cpu_count() or 6) // 6)))
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda job: cut(*job), todo))
    samples = []
    for sid, f in rows:
        choices = {o[0]: o[2:].strip() for o in f["options"]}           # "A. text" (every option is lettered)
        meta = {"task": f["task"], "ability": f["ability"], "source": f["video"], "t": f["t"], "options": f["options"]}
        samples.append(Sample(sid, f["question"], f["answer"], "mcq", choices, video=str(media / "clips" / f["clip"]),
                              metadata=meta))
    return samples


def prompt(sample) -> str:
    """StreamingBenchEval: options lettered when they are not, PROMPT_TEMPLATE, then "\\n\\nThe best option is:"."""
    template = common.upstream_constant(TEMPLATE, "PROMPT_TEMPLATE")
    options = sample.metadata["options"]
    if not options[0].startswith("A."):
        options = [f"A. {options[0]}", f"B. {options[1]}", f"C. {options[2]}", f"D. {options[3]}"]
    return template.format(sample.question, *options) + "\n\nThe best option is:"


def parse(sample, text):
    reply = common.clean(text)
    return (reply, "official") if reply else (None, "empty")


def correct(sample, prediction) -> bool:
    """count.py: model_answer = response[0]; right iff model_answer == the answer letter."""
    return bool(prediction) and prediction[0] == sample.answer


def metrics(rows) -> dict:
    answered = [r for r in rows if r.get("answered")]
    tasks: dict[str, list] = {}
    for r in answered:
        tasks.setdefault(r["metadata"]["task"], []).append(r["correct"])
    return {"acc": common.pct(r["correct"] for r in rows), "acc_official": common.pct(r["correct"] for r in answered),
            "answered": common.pct(r.get("answered") for r in rows),
            **{f"task/{t}": common.pct(v) for t, v in sorted(tasks.items())}}


def stratum(sample) -> str:
    return sample.metadata["task"]


def smoke(samples):
    return common.smoke(samples, group="source", n=10)
