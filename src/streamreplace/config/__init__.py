"""Configuration: configs/*.yaml loaded into validated dataclasses (schema.py, load.py) and the per-job settings
resolution (settings.py). Typical use:

    from streamreplace.config import load_config
    config = load_config()                                   # configs/ of the checkout (+ configs/local.yaml)
    config.settings("videomme", "thinkstream_3b", "stream")  # resolved settings of one job"""
from .load import load_config, read_yaml, validate
from .schema import AdapterSpec, BenchmarkSpec, CheckpointRef, Config, DecodingSpec, ModelSpec, PairSpec
from .settings import SAMPLING_KEYS, SECTIONS, compose_settings, merge, parse_sets, resolve_settings

__all__ = ["load_config", "read_yaml", "validate", "AdapterSpec", "BenchmarkSpec", "CheckpointRef", "Config",
           "DecodingSpec", "ModelSpec", "PairSpec", "SAMPLING_KEYS", "SECTIONS", "compose_settings", "merge",
           "parse_sets", "resolve_settings"]
