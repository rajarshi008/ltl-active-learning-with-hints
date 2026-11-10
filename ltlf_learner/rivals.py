"""NextRival (Section V-B): the order in which rivals are compared with the hypothesis.

An order is a callable order(state, version_space) -> rivals, best first. It
decides only which formulas are compared, never which are in the version space.
With Level-1 hints (state.guidance), rivals are then sorted by their distance
to the hinted classes (ties: by text for the enumeration order, otherwise by
the order's own ranking).

    enumeration    the version space's own order (smallest first)
    semantic       closest to the hypothesis first, by d_sem on a probe set
    semantic_far   farthest first, by the same distance
    split          rivals whose short witness divides the version space most evenly first
"""
import random

from .traces import PackedSample, random_trace

ORDERS = ("enumeration", "semantic", "semantic_far", "split")


class RivalOrder:
    """One of ORDERS.

    probe_traces: traces for d_sem (by default 200 random traces with lengths in
        `length_range`, drawn with their own seed so other random choices stay the same).
    pool: the ShortTracePool, for the split order.
    """

    def __init__(self, name, alphabet=None, length_range=(1, 6), seed=0, pool=None,
                 probe_traces=None, probe_size=200, split_limit=200):
        if name not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS}")
        self.name, self.pool, self.split_limit = name, pool, split_limit
        if name in ("semantic", "semantic_far"):
            if probe_traces is None:
                rng = random.Random((seed or 0) * 1000003 + 29)
                probe_traces = [random_trace(rng, alphabet, length_range) for _ in range(probe_size)]
            self.probe = PackedSample([(t, False) for t in probe_traces])
            self.probe_memo, self.probe_size = {}, len(probe_traces)

    def semantic_distance(self, first, second):
        """d_sem: the fraction of probe traces on which the two formulas disagree."""
        s, memo = self.probe, self.probe_memo
        return ((s.value(first, memo) ^ s.value(second, memo)) & s.starts).bit_count() / self.probe_size

    def __call__(self, state, version_space):
        rivals = version_space.rivals(state)
        if self.name in ("semantic", "semantic_far"):
            sign = 1 if self.name == "semantic" else -1
            rivals = sorted(rivals, key=lambda f: sign * self.semantic_distance(state.current, f))
        elif self.name == "split":
            rivals = self._split(state, list(rivals))
        if state.guidance.classes:
            distance = state.guidance.distance
            if self.name == "enumeration":   # closest to the hinted classes, ties by text
                rivals = sorted(rivals, key=lambda f: (distance(f), str(f)))
            else:                            # stable: ties keep this order's ranking
                rivals = sorted(rivals, key=distance)
        return rivals

    def _split(self, state, rivals):
        """Rivals ordered by how evenly their first short witness divides the version space."""
        pool = self.pool
        if pool is None or pool.depth < 0:
            return rivals
        sample, memo = pool.sample, pool.memo
        values = [sample.value(c, memo) for c in state.candidates]
        current = sample.value(state.current, memo)
        scored, rest = [], []
        for index, rival in enumerate(rivals):
            differ = (current ^ sample.value(rival, memo)) & sample.starts if index < self.split_limit else 0
            if not differ:
                rest.append(rival)
                continue
            bit = (differ & -differ).bit_length() - 1
            accepted = sum((v >> bit) & 1 for v in values)
            scored.append((abs(2 * accepted - len(values)), index, rival))
        scored.sort(key=lambda item: item[:2])
        return [rival for _, _, rival in scored] + rest
