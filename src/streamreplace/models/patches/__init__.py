"""Version-checked runtime patches of third-party code that the paper's runs depended on.

Each module changes one function of an installed package or of fetched upstream code, and only after checking that
the code it changes is exactly the code the paper's runs changed; otherwise it raises instead of silently computing
something else:
- qwen25_rope: the float32 cast in transformers' Qwen2.5-VL temporal positions (offline Qwen2.5-VL family, pair p2).
- mmduet2_client: MMDuet2's proactive client with the benchmark's token budget (MMDuet2 stream).

A patch of an installed file that must be in place before another program reads it (AURA's video processor, read by
the vLLM server of the AURA stream) is applied when the environment is built instead: see envs/patches/ and
`streamreplace env build`."""
