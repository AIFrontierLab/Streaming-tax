"""Recall (Stream++'s memory component): what the stream keeps when the question arrives.

Recall replaces the child's memory update without enlarging the memory: the budget B is the child's own native
visual budget. Frames are kept in arrival order while they fit; once the budget is reached, the `recent` newest
chunks are kept densely and the other B - recent slots hold a uniform sample of everything older (Sec. 6.1:
m_t = {v_(t-k+1), ..., v_t} U A_t). The sample is a bottom-k reservoir: every chunk gets a deterministic priority
from sha256(f"{seed}:{stream id}:{arrival index}") when it arrives, and an older chunk stays iff its priority is
among the lowest B - recent of all chunks that have left the recent window. Retention therefore never reads the
question, an answer or the video's remaining duration, and every machine keeps the same chunks (the stream id is
the video's file name).

It changes only the two children whose native protocol evicts (configs/plugin.yaml `recall`):
- AURA: its benchmark stream keeps the last min(30, floor(d)) one-second segments. Recall (settings stream.window
  full, stream.budget 30, stream.recent 10) keeps `aura_spans(d, 30, recent=10)`: every segment while floor(d) <= 30,
  else the newest 10 plus 20 sampled older ones. AURA's plugin arms run on models/aura_stream_hf.py, which reads
  these settings.
- ThinkStream: its KV cache holds the last 20 video chunks. Recall (setting stream.reservoir 10) keeps the newest 10
  chunks plus 10 sampled older ones: models/thinkstream_stream.py evicts inside the KV engine with `evict_index`,
  which keeps exactly the set `Reservoir(10, 10)` keeps (tests/plugin/test_recall.py).
MMDuet, MMDuet2, LiveStar and MOSS-VL-Realtime discard no frames natively, so Recall does not act on them.

The memory controls of Sec. 5.1 and Supp. D use the same settings: stream.window full without a budget is the
full-history control S_full (ThinkStream: with stream.max_len 81920), an int stream.window N the recent-only memory
of N chunks, and stream.budget 2B / stream.reservoir with stream.window 2B the doubled budget."""
from __future__ import annotations

from collections import deque
import hashlib
import heapq
import math


def priority(stream_id: str, index: int, seed: int = 0) -> int:
    """Deterministic priority of arrival `index` in one stream (the first 8 bytes of
    sha256(f"{seed}:{stream_id}:{index}")); lower = kept longer."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{stream_id}:{index}".encode()).digest()[:8], "big")


class Reservoir:
    """`recent` chunks densely + `archive` older ones: the archive keeps the chunks with the lowest priority()."""

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
                heapq.heappush(self._archive, (-priority(self.stream_id, index, self.seed), index, evicted))
                if len(self._archive) > self.archive:
                    heapq.heappop(self._archive)

    def items(self) -> list:
        """Retained (arrival index, chunk) pairs in chronological order."""
        old = [(index, chunk) for _, index, chunk in self._archive]
        return sorted(old + list(self._recent), key=lambda item: item[0])

    def indices(self) -> list[int]:
        return [index for index, _ in self.items()]


def evict_index(labels: list[int], arriving: int, recent: int, stream_id: str, seed: int = 0) -> int:
    """For an engine that evicts one chunk when its budget is full, just before chunk `arriving` is added: the
    position in `labels` (retained arrival indices, oldest first) of the chunk to drop so that the retained set stays
    the one `Reservoir(recent, budget - recent)` holds. The newest recent - 1 retained chunks and the arriving one are
    never dropped; among the older ones the highest priority goes (the oldest if none is older, which a budget above
    `recent` never reaches)."""
    older = [i for i, label in enumerate(labels) if label <= arriving - recent]
    if not older:
        return 0
    return max(older, key=lambda i: priority(stream_id, labels[i], seed))


def aura_spans(duration: float, policy="official", recent: int = 30, stream_id: str = "",
               seed: int = 0) -> tuple[list[tuple[float, float]], int]:
    """AURA's one-second segments aligned to the video's end (start = d - floor(d) + i), under a retention policy:
    "official" (AURA's benchmark window, the last min(30, floor(d))), "full" (every segment) or an int budget (all
    segments while they fit, then the newest `recent` + a reservoir of older ones). A video shorter than one second
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
        memory = Reservoir(min(recent, int(policy)), int(policy) - min(recent, int(policy)), stream_id, seed)
        for start in starts:
            memory.push(start)
        keep = [start for _, start in memory.items()]
    return [(start, 1.0) for start in keep], count - len(keep)


def aura_policy(settings: dict) -> tuple:
    """(policy, recent, seed) of aura_spans() for an AURA stream's settings, as aura_stream_hf applies them: no
    stream.window = "official"; stream.window "full" = "full", or with stream.budget B the budget B, whose newest
    stream.recent segments (default 30) stay dense and whose reservoir uses stream.memory_seed (default 0). An int
    stream.window N is AURA's own last-N window (models/aura_stream.spans), not a Recall policy: it raises here."""
    stream = settings.get("stream") or {}
    window, budget = stream.get("window"), stream.get("budget")
    recent, seed = int(stream.get("recent", 30)), int(stream.get("memory_seed", 0))
    if window == "full":
        return ("full" if budget is None else int(budget)), recent, seed
    if window not in (None, "official") or budget is not None:
        raise ValueError(f"stream.window {window!r} / stream.budget {budget!r}: not a Recall policy (a budget needs "
                         "stream.window full)")
    return "official", recent, seed
