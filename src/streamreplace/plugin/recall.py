"""Recall (part of S_fix): what the stream keeps when the question arrives.

Retention never reads the question, an answer or the video's remaining duration: chunks are kept in arrival order up
to a budget; past it, a dense recent window plus a hash-priority reservoir of older chunks. On every set evaluated in
the paper the budget is never reached, so Recall equals full history there, and it changes only the two children whose
native protocol evicts:
- ThinkStream: its KV window of 20 chunks / 24,576 tokens becomes 120 chunks / 81,920 tokens (settings
  stream.window "full", stream.max_len 81920; configs/plugin.yaml recall). That window is its budget; past it
  ThinkStream's own eviction applies (no reservoir).
- AURA: its benchmark stream keeps the last min(30, floor(d)) one-second segments; Recall keeps all floor(d)
  segments, still aligned to the video's end (settings stream.window "full"; aura_spans below). AURA's plugin arms
  run on models/aura_stream_hf.py, which reads that setting and cuts the segments with models/aura_stream.spans():
  starts d - floor(d) + i for i < floor(d), the same floats as aura_spans(d, "full"). The paper's runs applied this
  policy inside the plugin's engine; here it is the arm's settings delta, so it is also part of the arm's identity.
  A budget is the further setting stream.budget B (no shipped arm sets one): aura_stream_hf then keeps
  aura_spans(d, B), every segment while floor(d) <= B, else the last min(30, B) plus a Reservoir of B - min(30, B)
  older ones, with the video's file name as the stream id (the paper's plugin engine used the video path).
MMDuet, MMDuet2, LiveStar and MOSS-VL-Realtime already keep their whole history (no change)."""
from __future__ import annotations

from collections import deque
import hashlib
import heapq
import math


class Reservoir:
    """`recent` chunks densely + `archive` older ones chosen by a deterministic hash priority (stream_id, arrival
    index): the archive keeps the chunks with the smallest first 8 bytes of sha256(f"{seed}:{stream_id}:{index}")."""

    def __init__(self, recent: int, archive: int, stream_id: str = "", seed: int = 0):
        if recent < 1 or archive < 0:
            raise ValueError("recent must be positive and archive nonnegative")
        self.recent, self.archive, self.stream_id, self.seed = recent, archive, stream_id, seed
        self.arrivals = 0
        self._recent: deque = deque()
        self._archive: list = []

    def push(self, chunk) -> None:
        self._recent.append((self.arrivals, chunk))
        self.arrivals += 1
        if len(self._recent) > self.recent:
            index, evicted = self._recent.popleft()
            if self.archive:
                digest = hashlib.sha256(f"{self.seed}:{self.stream_id}:{index}".encode()).digest()
                heapq.heappush(self._archive, (-int.from_bytes(digest[:8], "big"), index, evicted))
                if len(self._archive) > self.archive:
                    heapq.heappop(self._archive)

    def items(self) -> list:
        """Retained (arrival index, chunk) pairs in chronological order."""
        old = [(index, chunk) for _, index, chunk in self._archive]
        return sorted(old + list(self._recent), key=lambda item: item[0])


def aura_spans(duration: float, policy="official", recent: int = 30,
               stream_id: str = "") -> tuple[list[tuple[float, float]], int]:
    """AURA's one-second segments aligned to the video's end (start = d - floor(d) + i), under a retention policy:
    "official" (AURA's benchmark window, the last min(30, floor(d))), "full" (every segment; the plugin's Recall) or
    an int budget (all segments while they fit, then the last `recent` + a reservoir). A video shorter than one second
    is one span (0, d). Returns (spans as (start, length), number of segments evicted)."""
    count = math.floor(duration)
    if count < 1:
        return [(0.0, duration)], 0
    starts = [duration - count + i for i in range(count)]
    if policy == "official":
        keep = starts[-min(30, count):]
    elif policy == "full" or count <= int(policy):
        keep = starts
    else:
        memory = Reservoir(min(recent, int(policy)), int(policy) - min(recent, int(policy)), stream_id)
        for start in starts:
            memory.push(start)
        keep = [start for _, start in memory.items()]
    return [(start, 1.0) for start in keep], count - len(keep)


def aura_policy(settings: dict):
    """The aura_spans() policy of an AURA plugin arm's stream settings, as aura_stream_hf applies it: no stream.window
    = "official", stream.window "full" = "full", and with stream.budget B as well, the budget B. An int stream.window
    N is AURA's own last-N window (models/aura_stream.spans), not a Recall policy: it raises here."""
    stream = settings.get("stream") or {}
    window, budget = stream.get("window"), stream.get("budget")
    if window == "full":
        return "full" if budget is None else int(budget)
    if window not in (None, "official") or budget is not None:
        raise ValueError(f"stream.window {window!r} / stream.budget {budget!r}: not a Recall policy (a budget needs "
                         "stream.window full)")
    return "official"
