"""Video understanding benchmarks, evaluated under all three conditions (P: parent offline, C: child in the same
offline pipeline, S: child through its native streaming interface with the question at the end of the video):
Video-MME (short split), TOMATO, TimeBlind, and VideoReasonBench (supplement). One module per benchmark; use them
through `streamreplace.benchmarks.registry.get(name)`. The official scorers (Video-MME's answer regex, TOMATO's
pre-parser and judge prompt, TimeBlind's extractor and judge prompt, VideoReasonBench's answer extractor) are read
from the upstream repositories fetched by `streamreplace fetch-upstream`. docs/benchmarks/<name>.md describes each."""
