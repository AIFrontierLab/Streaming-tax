"""Temporal-position fix for Qwen2.5-VL video (offline conditions P and C of pair p2: Qwen2.5-VL-3B, ThinkStream-3B,
MMDuet2-3B).

In transformers 4.5x, `Qwen2_5_VLModel.get_rope_index` converts `second_per_grid_t` (seconds per temporal grid step,
2 / sample fps) to the integer dtype of its position range before multiplying it by `tokens_per_second`. Any value
that is not a whole number is truncated: temporal positions collapse to 0 above 2 fps and are compressed otherwise.
transformers 4.49 multiplied in floating point. The paper's runs (transformers 4.57.3) re-bound the method with the
conversion done to float32 instead, and every p2 offline video row records `info.rope_fix = true`.

apply(model) recompiles the installed method's source with that one change and binds the result to the given
Qwen2_5_VLModel instance only (the class is untouched). The patch is version-checked: the source must contain the
converted expression exactly once, as in transformers 4.57.3. If it does not, apply() raises, because a different
transformers version may compute positions differently from the paper's runs."""
from __future__ import annotations

import inspect
import logging
import textwrap
import types

LOG = logging.getLogger(__name__)

BUG = "second_per_grid_t, dtype=range_tensor.dtype, device=range_tensor.device"
FIX = "second_per_grid_t, dtype=torch.float32, device=range_tensor.device"
TESTED_TRANSFORMERS = "4.57.3"        # the modern environment's version (envs/modern/requirements.lock)


def _source() -> tuple[types.ModuleType, str]:
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as module
    return module, textwrap.dedent(inspect.getsource(module.Qwen2_5_VLModel.get_rope_index))


def needs_fix() -> bool:
    """True when the installed transformers has the integer conversion this module fixes (no model needed)."""
    return _source()[1].count(BUG) == 1


def apply(model) -> bool:
    """Re-binds `model.get_rope_index` (model: a Qwen2_5_VLModel, i.e. `<Qwen2_5_VLForConditionalGeneration>.model`)
    with the float32 conversion. Returns True; raises RuntimeError when the installed source is not the one the
    paper's runs patched."""
    import transformers
    module, source = _source()
    if source.count(BUG) != 1:
        raise RuntimeError(
            f"transformers {transformers.__version__}: Qwen2_5_VLModel.get_rope_index does not contain "
            f"{BUG!r} exactly once, so the temporal-position fix of the paper's runs (made on transformers "
            f"{TESTED_TRANSFORMERS}) cannot be applied; install the modern environment from "
            "envs/modern/requirements.lock")
    if transformers.__version__ != TESTED_TRANSFORMERS:
        LOG.warning("Qwen2.5-VL temporal-position fix applied on transformers %s (checked on %s)",
                    transformers.__version__, TESTED_TRANSFORMERS)
    scope: dict = {}
    code = compile(source.replace(BUG, FIX), module.__file__, "exec")
    exec(code, dict(vars(module)), scope)          # the module's globals, so the method resolves names as before
    model.get_rope_index = types.MethodType(scope["get_rope_index"], model)
    return True
