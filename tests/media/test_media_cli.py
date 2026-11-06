"""`streamreplace video-meta build|verify` (cli/cmd_media.py) on a small prepared benchmark of generated videos."""
from __future__ import annotations

import argparse
import json

import pytest

from streamreplace.cli import cmd_media
from streamreplace.media.video_meta import library_versions, write_frame_list

HAS_DECORD = library_versions()["decord"] is not None


def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="streamreplace")
    cmd_media.register(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(argv)
    return args.func(args)


@pytest.fixture
def dataset(tmp_path, new_video, new_images):
    """A prepared benchmark: two videos (one used by two samples), a frame list and an image-only sample."""
    root = tmp_path / "data" / "demo"
    new_video(root / "media/a.mp4", frames=12, fps=4)
    new_video(root / "media/b.mp4", frames=8, fps=2)
    write_frame_list(root / "media/c.frames.json", new_images(root / "media/c", 5), 1.0)
    samples = [{"id": "1", "video": "media/a.mp4"}, {"id": "2", "video": "media/b.mp4"},
               {"id": "3", "video": "media/a.mp4"}, {"id": "4", "video": "media/c.frames.json"},
               {"id": "5", "images": ["media/c/00000.png"]}]
    (root / "prepared").mkdir()
    (root / "prepared/samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    return root


@pytest.fixture
def options(dataset, tmp_path):
    def make(*extra: str) -> list[str]:
        return ["demo", "--dataset-dir", str(dataset), "--manifests", str(tmp_path / "manifests"),
                "--frame-counts", str(tmp_path / "frame_counts"), "--workers", "1", *extra]
    return make


def build(options, tmp_path) -> list[dict]:
    reader = [] if HAS_DECORD else ["--allow-pyav-fps"]
    assert run(["video-meta", "build", *options("--no-torchcodec", *reader)]) == 0
    return [json.loads(line) for line in (tmp_path / "manifests/demo.jsonl").read_text().splitlines()]


def test_build_writes_one_relative_row_per_video(options, tmp_path):
    assert build(options, tmp_path) == [
        {"video": "media/a.mp4", "frames": 12, "fps": 4.0, "header_frames": 12},
        {"video": "media/b.mp4", "frames": 8, "fps": 2.0, "header_frames": 8},
        {"video": "media/c.frames.json", "frames": 5, "fps": 1.0, "header_frames": 5}]


@pytest.mark.skipif(HAS_DECORD, reason="decord is installed")
def test_build_without_decord_needs_an_explicit_fps_reader(options):
    with pytest.raises(SystemExit, match="decord"):
        run(["video-meta", "build", *options("--no-torchcodec")])


def test_verify_passes_on_the_same_data_and_reports_differences(options, tmp_path, capsys):
    rows = build(options, tmp_path)
    assert run(["video-meta", "verify", *options()]) == 0
    assert run(["video-meta", "verify", *options("--recount", "--sample", "0.5", "--seed", "3")]) == 0
    capsys.readouterr()
    rows[1]["frames"] = 9                                                  # a manifest the data does not match
    rows.append({"video": "media/z.mp4", "frames": 3, "fps": 1.0, "header_frames": 3})
    (tmp_path / "manifests/demo.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert run(["video-meta", "verify", *options()]) == 1
    out = capsys.readouterr().out
    assert "missing 1, extra 0; measured 3: 1 differ" in out
    assert "missing locally: media/z.mp4" in out and "media/b.mp4: frames: manifest 9, measured 8" in out


def test_verify_without_a_manifest_fails(options):
    assert run(["video-meta", "verify", *options()]) == 1


def test_the_command_is_registered_in_the_main_cli(options, tmp_path):
    from streamreplace.cli.main import main
    build(options, tmp_path)
    assert main(["--home", str(tmp_path / "home"), "video-meta", "verify", *options()]) == 0
