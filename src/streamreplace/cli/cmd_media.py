"""`streamreplace video-meta build|verify`: the per-benchmark video_meta manifests (docs/frames.md).

build   measures every video of a prepared benchmark (decodable frames with PyAV, fps and header count with decord,
        torchcodec's exact-mode metadata when torchcodec loads) and writes manifests/video_meta/<benchmark>.jsonl.
        Run it with a model environment's python (decord is a requirement there; the paper's runs used decord 0.6.0,
        and torchcodec 0.7.0 with the FFmpeg 6 libraries).
verify  compares a local copy of a benchmark with the shipped manifest: the same videos, and, for all or a random
        fraction of them, the same values when measured with the local libraries. Fields whose library is missing
        are skipped and reported. Exit code 1 on any difference.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import sys

from ..media import video_meta


def register(subparsers) -> None:
    parser = subparsers.add_parser("video-meta", help="build or verify the per-benchmark video_meta manifests",
                                   description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    actions = parser.add_subparsers(dest="video_meta_action", required=True)

    def common(sub):
        sub.add_argument("benchmarks", nargs="+", help="benchmark names (one manifest each)")
        sub.add_argument("--dataset-dir", type=Path,
                         help="the benchmark's dataset directory (holds prepared/samples.jsonl); one benchmark only. "
                              "Default: the benchmark's directory under the data root")
        sub.add_argument("--manifests", type=Path, default=None,
                         help="manifest directory (default: manifests/video_meta of this repository)")
        sub.add_argument("--frame-counts", type=Path, default=None,
                         help="decodable-frame-count cache to use (default: <cache root>/frame_counts)")
        sub.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))

    build = actions.add_parser("build", help="measure every video and write the manifest")
    common(build)
    build.add_argument("--no-torchcodec", action="store_true", help="do not record torchcodec metadata")
    build.add_argument("--allow-pyav-fps", action="store_true",
                       help="without decord, record PyAV's header fps (not the paper's reader; for other data only)")
    build.set_defaults(func=_build)

    verify = actions.add_parser("verify", help="check a local copy of the data against the manifest")
    common(verify)
    verify.add_argument("--sample", type=float, default=1.0,
                        help="fraction of the videos to measure again (random, seeded; default: all)")
    verify.add_argument("--seed", type=int, default=0)
    verify.add_argument("--recount", action="store_true",
                        help="decode again instead of reading cached decodable-frame counts")
    verify.set_defaults(func=_verify)


def _dataset_dir(args, benchmark: str) -> Path:
    if args.dataset_dir is not None:
        if len(args.benchmarks) != 1:
            raise SystemExit("--dataset-dir needs exactly one benchmark")
        return args.dataset_dir
    from ..paths import dataset_dir
    return Path(dataset_dir(benchmark))


def _progress(name: str):
    def report(done: int, total: int) -> None:
        if done % 200 == 0 or done == total:
            print(f"  {name}: {done}/{total}", file=sys.stderr, flush=True)
    return report


def _build(args) -> int:
    versions = video_meta.library_versions()
    if versions["decord"] is None and not args.allow_pyav_fps:
        raise SystemExit("decord is not installed: run with a model environment's python (the paper's fps reader), "
                         "or pass --allow-pyav-fps for data outside the paper")
    torchcodec = not args.no_torchcodec and versions["torchcodec"] is not None
    if not args.no_torchcodec and not torchcodec:
        print("note: torchcodec cannot be loaded (missing, or no FFmpeg libraries); its metadata is not recorded",
              file=sys.stderr)
    print("readers: " + ", ".join(f"{k} {v or 'missing'}" for k, v in versions.items()), file=sys.stderr)
    for benchmark in args.benchmarks:
        root = _dataset_dir(args, benchmark)
        videos = video_meta.dataset_videos(root)
        rows = video_meta.measure_videos(root, videos, workers=args.workers, progress=_progress(benchmark),
                                         cache_dir=args.frame_counts,
                                         reader="decord" if versions["decord"] else "auto", torchcodec=torchcodec)
        path = video_meta.write_manifest(video_meta.manifest_path(benchmark, args.manifests), rows)
        hidden = sum(row["header_frames"] > row["frames"] for row in rows)
        print(f"{benchmark}: {len(rows)} videos ({hidden} with frames hidden by an edit list) -> {path}")
    return 0


def _verify(args) -> int:
    versions = video_meta.library_versions()
    reader = "decord" if versions["decord"] else None
    torchcodec = versions["torchcodec"] is not None
    checked = ["frames"] + (["fps", "header_frames"] if reader else []) + (["torchcodec"] if torchcodec else [])
    print(f"checking {', '.join(checked)} (decord {versions['decord'] or 'missing'}, torchcodec "
          f"{versions['torchcodec'] or 'missing'}, av {versions['av'] or 'missing'})", file=sys.stderr)
    failed = False
    for benchmark in args.benchmarks:
        manifest = video_meta.load_manifest(benchmark, args.manifests)
        if not manifest:
            print(f"{benchmark}: no manifest at {video_meta.manifest_path(benchmark, args.manifests)}")
            failed = True
            continue
        root = _dataset_dir(args, benchmark)
        local = video_meta.dataset_videos(root)
        missing, extra = sorted(set(manifest) - set(local)), sorted(set(local) - set(manifest))
        common = sorted(set(manifest) & set(local))
        chosen = common if args.sample >= 1 else sorted(
            random.Random(args.seed).sample(common, max(1, round(len(common) * args.sample))))
        rows = video_meta.measure_videos(root, chosen, workers=args.workers, progress=_progress(benchmark),
                                         cache_dir=args.frame_counts, reader=reader, torchcodec=torchcodec,
                                         recount=args.recount)
        diffs = {row["video"]: video_meta.compare(manifest[row["video"]], row) for row in rows}
        diffs = {video: d for video, d in diffs.items() if d}
        print(f"{benchmark}: manifest {len(manifest)}, local {len(local)}, missing {len(missing)}, extra {len(extra)}; "
              f"measured {len(rows)}: {len(diffs)} differ")
        for video in missing[:10]:
            print(f"  missing locally: {video}")
        for video in extra[:10]:
            print(f"  not in the manifest: {video}")
        for video, d in list(diffs.items())[:20]:
            print(f"  {video}: {'; '.join(d)}")
        failed |= bool(missing or extra or diffs)
    return 1 if failed else 0
