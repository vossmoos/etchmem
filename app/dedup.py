"""
Stage 1 — signal dedup (the cheap "garbage collector"), embedding-based.

Near-duplicate signals (same meaning) are grouped by cosine distance. One
member becomes the *canonical* representative; the others point at it via
`canonical_id`. Crucially this PRESERVES provenance — every original signal
and its distinct source survives, so corroboration is never lost.

A second, optional condition guards historical corpora: two signals whose
event times are further apart than `max_time_gap` are never the same signal,
however similar the text. Support records, order confirmations and status
reports are formulaic, so two of them years apart can sit well inside the
distance threshold while describing different things.

No LLM here: finding duplicates is what embeddings do cheaply and
deterministically.
"""
from __future__ import annotations

import numpy as np

from app.stores import Signal


def group_duplicates(
    signals: list[Signal], dedup_distance: float, max_time_gap: float = 0.0
) -> list[list[Signal]]:
    """Single-link grouping of near-identical signals. Returns groups.

    `max_time_gap` (seconds, 0 = off) additionally requires two signals to be
    close in event time before they can be treated as duplicates.
    """
    if not signals:
        return []
    if len(signals) == 1:
        return [signals]

    vecs = np.array([s.embedding for s in signals], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vecs / norms
    dist = 1.0 - (unit @ unit.T)

    n = len(signals)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    times = [s.event_at for s in signals] if max_time_gap > 0 else None

    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] > dedup_distance:
                continue
            if times is not None and abs(times[i] - times[j]) > max_time_gap:
                continue
            parent[find(i)] = find(j)

    groups: dict[int, list[Signal]] = {}
    for idx, sig in enumerate(signals):
        groups.setdefault(find(idx), []).append(sig)
    return list(groups.values())
