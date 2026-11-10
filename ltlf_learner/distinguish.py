"""Distinguish (Section V-C): a trace on which two formulas disagree.

A distinguisher is a callable d(phi, psi) that returns
    a trace       on which exactly one of the formulas holds,
    None          when the formulas are equivalent (over all traces, or over
                  the feasible traces of an environment),
    UNKNOWN       when an environment search stopped at its horizon or budget.

All searches are breadth-first with letters in a fixed order, so they return a
shortest witness, first in that order. `PoolFirst` first checks a pool of short
traces with bit operations (the heuristic of Section VI) and only then runs the
complete search; it returns the same witness, only faster.
"""
from functools import lru_cache
from itertools import product

from .automata import distinguish_by_automata
from .progression import Progressor, distinguish_by_progression
from .traces import PackedSample

UNKNOWN = "unknown"


class ShortTracePool:
    """All traces up to some length, packed into one PackedSample, in breadth-first order.

    The length is the largest one with at most `max_traces` traces, so the pool
    never holds more than that. find() returns the first pool trace on which two
    formulas disagree, which is the witness a breadth-first search would return.
    """

    def __init__(self, alphabet, max_traces=5000, max_product_states=100000):
        self.depth, self.sample, self.trace_at = _pool(tuple(alphabet), max_traces,
                                                       max_product_states)
        self.memo = {}

    def find(self, first, second):
        """The first short trace on which the formulas disagree, or None."""
        if self.depth < 0:
            return None
        s = self.sample
        differ = (s.value(first, self.memo) ^ s.value(second, self.memo)) & s.starts
        if not differ:
            return None
        return self.trace_at[(differ & -differ).bit_length() - 1]


@lru_cache(maxsize=8)
def _pool(alphabet, max_traces, max_product_states):
    """(depth, PackedSample of all traces up to depth, start bit -> trace). Cached; do not modify."""
    def up_to(k):
        return sum(len(alphabet) ** n for n in range(k + 1))
    depth = -1
    while up_to(depth + 1) <= max_traces and up_to(depth + 2) < max_product_states:
        depth += 1
    traces = [t for n in range(depth + 1) for t in product(alphabet, repeat=n)]
    trace_at, offset = {}, 0
    for t in traces:
        trace_at[offset + len(t)] = t
        offset += len(t) + 1
    return depth, PackedSample([(t, False) for t in traces]), trace_at


class Progression:
    """Complete search over pairs of progressed formulas (no automata)."""

    def __init__(self, alphabet, max_product_states=100000):
        self.alphabet, self.max_states = alphabet, max_product_states
        self.progressor = Progressor()

    def __call__(self, phi, psi):
        return distinguish_by_progression(phi, psi, self.alphabet,
                                          max_product_states=self.max_states,
                                          progressor=self.progressor)


class Automata:
    """Complete search over the product of the two formulas' MONA automata."""

    def __init__(self, atoms, alphabet, max_product_states=100000):
        self.atoms, self.alphabet, self.max_states = atoms, alphabet, max_product_states

    def __call__(self, phi, psi):
        return distinguish_by_automata(phi, psi, self.atoms, self.alphabet, self.max_states)


class PoolFirst:
    """Try the short-trace pool, then the complete search `inner`."""

    def __init__(self, pool, inner):
        self.pool, self.inner = pool, inner

    def __call__(self, phi, psi):
        witness = self.pool.find(phi, psi) if self.pool is not None else None
        return witness if witness is not None else self.inner(phi, psi)
