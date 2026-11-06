"""build_inputs() with the real processors and tokenizers on a CPU (adapters built with load=False, no weights).

Each case runs inside the model's own environment (recognized by its transformers version) when the checkpoint's
processor files are under the checkpoints root (`streamreplace download models --processors-only`) and, for LLaVA
and LiveStar, the upstream code is fetched; it is skipped otherwise. The request is a synthetic 4-second clip with
Video-SafetyBench's offline sampling (16 uniform frames, so no torchcodec is needed) and a synthetic image."""
from __future__ import annotations

from fractions import Fraction

import pytest

from streamreplace import paths
from streamreplace.config.load import load_config
from streamreplace.core.schema import Request
from streamreplace.models import registry
from streamreplace.models.upstream import Upstream
from streamreplace.provisioning import envs

UPSTREAM = {"llava_offline": "llava", "internvl_offline": "livestar"}
MODELS = ["qwen25vl_3b", "thinkstream_3b", "mmduet2_3b", "qwen3vl_8b", "aura_8b", "moss_vl", "moss_vl_rt",
          pytest.param("llava_ov_7b", marks=pytest.mark.requires_upstream("llava")),
          pytest.param("mmduet_7b", marks=pytest.mark.requires_upstream("llava")),
          pytest.param("livestar_iv_8b", marks=pytest.mark.requires_upstream("livestar")),
          pytest.param("livestar_8b", marks=pytest.mark.requires_upstream("livestar"))]


def clip(path, frames: int = 48, fps: int = 12, size: int = 96):
    import av
    import numpy as np
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = stream.height = size
        stream.pix_fmt, stream.time_base = "yuv420p", Fraction(1, fps)
        for i in range(frames):
            array = np.full((size, size, 3), (5 * i) % 255, np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


@pytest.mark.parametrize("model_id", MODELS)
def test_build_inputs_on_a_cpu(model_id, tmp_path):
    transformers = pytest.importorskip("transformers")
    config = load_config()
    spec = config.model(model_id)
    adapter_name, env = spec.offline.adapter, spec.offline.env
    if envs.definitions()[env].packages().get("transformers") != "==" + transformers.__version__:
        pytest.skip(f"{model_id} runs in the {env} environment")
    folders = [paths.checkpoint_dir(ref) for ref in spec.checkpoints().values()]
    if not all(folder.is_dir() for folder in folders):
        pytest.skip(f"checkpoint files of {model_id} are not under the checkpoints root")
    if adapter_name in UPSTREAM and not Upstream().available(UPSTREAM[adapter_name]):
        pytest.skip(f"upstream {UPSTREAM[adapter_name]} is not fetched")
    settings = config.settings("video_safetybench", model_id, "offline")
    adapter = registry.load_class(adapter_name)(spec, settings, Upstream(), load=False)
    video = Request("v1", "video_safetybench", "What happens?", [], str(clip(tmp_path / "clip.mp4")), settings,
                    kind="open")
    info = adapter.build_inputs(video).info
    assert info["frame_indices"].startswith("16:") and info["frames"] == 16 and info["duration"] == 4.0
    from PIL import Image
    Image.new("RGB", (640, 360), (40, 80, 120)).save(tmp_path / "image.png")
    image = Request("i1", "mmbench", "Which colour?", [str(tmp_path / "image.png")], None,
                    config.settings("mmbench", model_id, "offline"), kind="mcq", choices={"A": "blue", "B": "red"})
    info = adapter.build_inputs(image).info
    assert info["input_tokens"] > 0
    kwargs = adapter.generation_kwargs(image)
    assert kwargs["do_sample"] is False and kwargs["repetition_penalty"] == spec.decoding.repetition_penalty
    adapter.close()
