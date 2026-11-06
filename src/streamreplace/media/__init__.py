"""Video frames shared by every model: decodable frame counts and fps, deterministic frame sampling, frame digests,
and the per-benchmark video_meta manifests (paper conditions P, C and S; docs/frames.md).

- video_meta: decodable frame count (PyAV, cached), decord fps, frame-list videos, manifests.
- frames: frame_indices / sample_frames / offline_frames, the sampling rules of the paper's runs.
- digest: frame_digest, the per-row fingerprint `info.frame_indices`.
- qwen_sampling: a CPU mirror of qwen-vl-utils' frame selection (the Qwen pairs' offline sampler), for verification.
"""
from .digest import frame_digest
from .frames import frame_indices, offline_frames, offline_indices, offline_sampling, open_rgb, sample_frames
from .video_meta import (decodable_frames, is_frame_list, load_manifest, manifest_info, read_frame_list, video_info,
                         write_frame_list)

__all__ = ["decodable_frames", "frame_digest", "frame_indices", "is_frame_list", "load_manifest", "manifest_info",
           "offline_frames", "offline_indices", "offline_sampling", "open_rgb", "read_frame_list", "sample_frames",
           "video_info", "write_frame_list"]
