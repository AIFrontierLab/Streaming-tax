"""OVO-Bench (JoeLeelyf/OVO-Bench): 644 videos, 1,640 queries in three modes, Backward Tracing (EPM, ASI, HLD),
Real-Time Visual Perception (STU, OJR, ATR, ACR, OCR, FPD) and Forward Active Responding (REC, SSR, CRR).

Paper use: native streaming ability with and without Stream++ (Supp. D.4), on the rows whose clip lasts at most 120 s
(MAX_SECONDS: 1,576 of the 3,035 rows; several children's native streams cannot run much longer clips). Reported:
mean question accuracy (`acc_rows`, unanswered = wrong) and the official total average (`overall`).

Protocol = the official one (utils/OVOBench.py), which the streaming models' own OVO evaluations also follow: each
backward / real-time query gets the official pre-chunked clip [0, t] (chunked_videos/{id}.mp4) and each forward query
is asked again at every probe time (chunked_videos/{id}_{i}.mp4, 1,567 probes). One sample per query or probe; in
stream mode the clip plays and the question arrives at its end.
Prompts: constant.py's templates through the official build_prompt, verbatim. Scoring: utils/OVOBenchScore.py,
verbatim, no judge: backward / real-time right iff the option letter occurs in the reply; REC iff the reply's digits
joined equal the count; SSR / CRR iff the reply is exactly "Y" / "N" for the probe's type or contains "Yes" / "No".
`overall` = the official Total Avg (the mean of the three modes' averages, each the mean of its tasks' accuracies; a
forward task's accuracy is over its probes). The reply is cleaned of think blocks and control tokens first
(common.clean); an empty reply is the official None (0).
Annotations and code: the fetched repository (manifests/upstream.yaml `ovo_bench`). Media: the prepared rows' clips
are streamed out of raw/chunked_videos.tar.part* into media/chunked_videos/ in one pass."""
from __future__ import annotations

import json
from pathlib import Path

from .. import common
from ..base import JudgePolicy
from ...core.schema import Sample
from ...models.upstream import upstream_path

NAME = "ovo_bench"
MEDIA = "video"
JUDGE = JudgePolicy.NONE
MAX_SECONDS = 120                        # the paper's scope: clips of at most 120 s up to the question
EXPECTED_COUNT = 1576                    # of 3,035 rows (1,468 backward / real-time queries + 1,567 forward probes)
BACKWARD, REALTIME, FORWARD = ("EPM", "ASI", "HLD"), ("STU", "OJR", "ATR", "ACR", "OCR", "FPD"), ("REC", "SSR", "CRR")
MODES = {"backward": BACKWARD, "realtime": REALTIME, "forward": FORWARD}
ANNOTATIONS = "ovo_bench/data/ovo_bench_new.json"


def mode_of(task: str) -> str:
    return next(mode for mode, tasks in MODES.items() if task in tasks)


class _Parts:
    """The split tar (chunked_videos.tar.parta[a-o]) read as one stream, in order (members cross part boundaries)."""

    def __init__(self, paths):
        self.paths, self.handle = list(paths), None

    def read(self, size=-1):
        while True:
            if self.handle is None:
                if not self.paths:
                    return b""
                self.handle = open(self.paths.pop(0), "rb")
            data = self.handle.read(size)
            if data:
                return data
            self.handle.close()
            self.handle = None


def extract(raw: Path, media: Path, wanted: set[str]) -> None:
    """Streams the wanted clips (file names) out of the split tar in one pass; nothing is read when all are there."""
    import tarfile
    if all((media / "chunked_videos" / name).is_file() for name in wanted):
        return
    parts = sorted(raw.glob("chunked_videos.tar.part*"))
    if not parts:
        raise ValueError(f"no chunked_videos.tar.part* in {raw}")
    try:
        with tarfile.open(fileobj=_Parts(parts), mode="r|") as handle:
            for info in handle:
                name = Path(info.name)
                if not info.isfile() or name.name not in wanted or name.is_absolute() or ".." in name.parts:
                    continue
                target = media / "chunked_videos" / name.name
                if target.exists() and target.stat().st_size == info.size:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + ".part")
                with handle.extractfile(info) as source, open(partial, "wb") as out:
                    while chunk := source.read(1 << 22):
                        out.write(chunk)
                partial.rename(target)
    except (tarfile.ReadError, EOFError) as exc:
        raise ValueError(f"OVO-Bench archive incomplete or corrupt ({exc}); finish the download first") from exc


def items() -> list[tuple[str, dict]]:
    """(sample id, fields) for every query and forward probe, in the annotation's order."""
    out = []
    for anno in json.loads(upstream_path(ANNOTATIONS).read_text(encoding="utf-8")):
        task = anno["task"]
        base = {"item": anno["id"], "task": task, "mode": mode_of(task), "source": anno["video"]}
        if task not in FORWARD:
            out.append((str(anno["id"]), {**base, "file": f"{anno['id']}.mp4", "t": anno["realtime"],
                                           "question": anno["question"], "options": anno["options"],
                                           "answer": chr(65 + anno["gt"])}))
            continue
        for i, probe in enumerate(anno["test_info"]):
            fields = {**base, "file": f"{anno['id']}_{i}.mp4", "t": probe["realtime"], "probe": i}
            if task == "REC":
                fields.update(question=anno["activity"], answer=str(probe["count"]))
            elif task == "SSR":
                fields.update(question=probe["step"], answer="No" if probe["type"] == 0 else "Yes", type=probe["type"])
            else:
                fields.update(question=anno["question"], answer="No" if probe["type"] == 0 else "Yes",
                              type=probe["type"])
            out.append((f"{anno['id']}_{i}", fields))
    return out


def prepare(raw: Path, media: Path) -> list[Sample]:
    rows = [(sid, f) for sid, f in items() if f["t"] <= MAX_SECONDS]
    extract(raw, media, {f["file"] for _, f in rows})
    samples = []
    for sid, f in rows:
        video = media / "chunked_videos" / f["file"]
        if not video.is_file():
            raise ValueError(f"OVO-Bench: missing clip {video}")
        meta = {k: f[k] for k in ("item", "task", "mode", "source", "t", "probe", "type") if k in f}
        if f["mode"] != "forward":
            samples.append(Sample(sid, f["question"], f["answer"], "mcq", common.letters(f["options"]),
                                  video=str(video), metadata=meta))
        else:
            samples.append(Sample(sid, f["question"], f["answer"], "numeric" if f["task"] == "REC" else "yesno",
                                  video=str(video), metadata=meta))
    return samples


def _build_prompt():
    templates = {name: common.upstream_constant("ovo_bench/constant.py", name)
                 for name in ("BR_PROMPT_TEMPLATE", "REC_PROMPT_TEMPLATE", "SSR_PROMPT_TEMPLATE", "CRR_PROMPT_TEMPLATE")}
    return common.upstream_functions("ovo_bench/utils/OVOBench.py", "OVOBenchOffline.build_prompt",
                                     scope=templates)["OVOBenchOffline.build_prompt"]


def prompt(sample) -> str:
    """OVOBenchOffline.build_prompt with the annotation fields it reads (activity, the probe's step, the question)."""
    task = sample.metadata["task"]
    if task not in FORWARD:
        return _build_prompt()(None, task, sample.question, list(sample.choices.values()), None, None)
    probe = sample.metadata["probe"]
    anno = {"activity": sample.question, "question": sample.question,
            "test_info": [{"step": sample.question} if i == probe else {} for i in range(probe + 1)]}
    return _build_prompt()(None, task, None, None, anno, probe)


def parse(sample, text):
    reply = common.clean(text)
    return (reply, "official") if reply else (None, "empty")


def _scorer():
    return common.upstream_module("ovo_bench/utils/OVOBenchScore.py").OVOBenchOfflineScore(None, None)


def correct(sample, prediction) -> bool:
    """The official per-row score on the (cleaned) reply; None = no reply = 0."""
    task, scorer = sample.metadata["task"], _scorer()
    if task not in FORWARD:
        results, _ = scorer.calculate_score_backward_realtime(
            [{"task": task, "response": prediction, "ground_truth": sample.answer}])
        return bool(results[0]["score"])
    probe = {"response": prediction}
    if task == "REC":
        probe["count"] = int(sample.answer)
    else:
        probe["type"] = sample.metadata["type"]
    _, scores = scorer.calculate_score_forward([{"task": task, "test_info": [probe]}])
    return bool(scores[task][0])


def official(rows) -> dict:
    """The official aggregation (OVOBenchOfflineScore.score): task accuracy, mode average over tasks, Total Avg over
    the three modes (None while a mode has no rows)."""
    tasks: dict[str, list] = {}
    for r in rows:
        tasks.setdefault(r["metadata"]["task"], []).append(bool(r["correct"]))
    acc = {task: 100 * sum(v) / len(v) for task, v in tasks.items()}
    modes = {mode: sum(acc[t] for t in names if t in acc) / sum(t in acc for t in names)
             for mode, names in MODES.items() if any(t in acc for t in names)}
    total = sum(modes.values()) / 3 if len(modes) == 3 else None
    return {"total": total, "modes": modes, "tasks": acc}


def metrics(rows) -> dict:
    result = official(rows)
    rnd = lambda v: None if v is None else round(v, 2)                 # noqa: E731
    return {"overall": rnd(result["total"]), **{f"mode/{m}": rnd(v) for m, v in result["modes"].items()},
            **{f"task/{t}": rnd(v) for t, v in sorted(result["tasks"].items())},
            "acc_rows": common.pct(r["correct"] for r in rows), "answered": common.pct(r.get("answered") for r in rows)}


def stratum(sample) -> str:
    return sample.metadata["task"]


def smoke(samples):
    return common.smoke(samples, group="item", n=24)
