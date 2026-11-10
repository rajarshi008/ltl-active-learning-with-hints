"""Finite traces and fast formula evaluation on them.

A trace is a tuple of valuations; a valuation is the sorted tuple of the atoms
true at that position. ``()`` is the empty trace.

Semantics (LTLf, as in the paper): X is strong (false at the last position);
past the end of a trace, atoms, X, F and U are false, and negated atoms and G
are true. So the empty trace satisfies G(a) and !a, but not F(a).

`holds` evaluates one formula on one trace. `PackedSample` evaluates a formula
on many traces at once with integer bit operations, and `filter_consistent`
uses it to keep the formulas that agree with a labelled sample.
"""
from itertools import combinations


def valuation_alphabet(atoms):
    """All valuations over `atoms`, smallest first: ((), ('a',), ('b',), ('a', 'b'))."""
    atoms = tuple(dict.fromkeys(atoms))
    return tuple(tuple(c) for n in range(len(atoms) + 1) for c in combinations(atoms, n))


def normalize_trace(trace, atoms):
    """Return `trace` as a tuple of sorted valuation tuples; check the atoms are known."""
    allowed, out = set(atoms), []
    for step in trace:
        if isinstance(step, str):
            raise TypeError("a valuation must be a collection of atoms, not a string")
        values = tuple(sorted(set(step)))
        if not set(values) <= allowed:
            raise ValueError(f"unknown atom in {step!r}")
        out.append(values)
    return tuple(out)


def random_trace(rng, alphabet, length_range):
    """A uniformly random trace: length uniform in `length_range`, letters uniform."""
    return tuple(rng.choice(alphabet) for _ in range(rng.randint(*length_range)))


# ---------------------------------------------------------------------------
# One formula, one trace
# ---------------------------------------------------------------------------

def holds(formula, trace, memo=None):
    """Whether `formula` holds on `trace` (at position 0).

    memo: optional dict shared between calls on the *same* trace, so shared
    subformulas are evaluated once.
    """
    return _values(formula, trace, {} if memo is None else memo)[0]


def _values(formula, trace, memo):
    """Truth values at positions 0..n (position n is just past the end)."""
    entry = memo.get(id(formula))
    if entry is not None and entry[0] is formula:
        return entry[1]
    n, label = len(trace), formula.label
    if label == "true":
        values = [True] * (n + 1)
    elif label == "false":
        values = [False] * (n + 1)
    elif label == "!":
        values = [not v for v in _values(formula.left, trace, memo)]
    elif label in ("&", "|", "->"):
        left = _values(formula.left, trace, memo)
        right = _values(formula.right, trace, memo)
        if label == "&":
            values = [l and r for l, r in zip(left, right)]
        elif label == "|":
            values = [l or r for l, r in zip(left, right)]
        else:
            values = [(not l) or r for l, r in zip(left, right)]
    elif label == "X":
        child = _values(formula.left, trace, memo)
        values = [i + 1 < n and child[i + 1] for i in range(n)] + [False]
    elif label in ("F", "G"):
        child = _values(formula.left, trace, memo)
        values = [label == "G"] * (n + 1)
        for i in range(n - 1, -1, -1):
            values[i] = (child[i] and values[i + 1]) if label == "G" else (child[i] or values[i + 1])
    elif label == "U":
        left = _values(formula.left, trace, memo)
        right = _values(formula.right, trace, memo)
        values = [False] * (n + 1)
        for i in range(n - 1, -1, -1):
            values[i] = right[i] or (left[i] and values[i + 1])
    elif label.startswith("!"):          # negated atom
        values = [label[1:] not in step for step in trace] + [True]
    else:                                 # atom
        values = [label in step for step in trace] + [False]
    memo[id(formula)] = (formula, values)
    return values


# ---------------------------------------------------------------------------
# Many traces at once
# ---------------------------------------------------------------------------

class PackedSample:
    """Labelled traces packed into one integer, so a formula is evaluated on all of them at once.

    A trace of length n gets a block of n + 1 bits, stored in reverse (bit 0 of
    the block is the position past the end, bit n is position 0). A formula's
    truth values on all traces are then one integer: Boolean operators are
    bitwise operations, X is a shift, and U uses the carry of an addition.
    Same semantics as `holds`.
    """

    def __init__(self, labelled_traces):
        offset = ends = lasts = starts = positive = 0
        atom_bits = {}
        for trace, label in labelled_traces:
            n = len(trace)
            ends |= 1 << offset
            if n >= 1:
                lasts |= 1 << (offset + 1)
            start = 1 << (offset + n)
            starts |= start
            if label:
                positive |= start
            for i, step in enumerate(trace):
                bit = 1 << (offset + n - i)
                for atom in step:
                    atom_bits[atom] = atom_bits.get(atom, 0) | bit
            offset += n + 1
        self.full = (1 << offset) - 1
        self.inner = self.full & ~ends          # real positions
        self.next_mask = self.inner & ~lasts    # positions that have a next position
        self.starts = starts                    # position 0 of every trace
        self.positive = positive                # position 0 of positive traces
        self.atom_bits = atom_bits

    def consistent(self, formula, memo):
        """Whether `formula` gives every trace its label."""
        return self.value(formula, memo) & self.starts == self.positive

    def until(self, left, right):
        """Truth values of U(left, right) from those of its operands."""
        right &= self.inner
        run = left & ~right & self.inner
        seeds = (right << 1) & run
        return right | (run & ~(run + seeds))

    def value(self, formula, memo):
        """Truth values of `formula` at every position of every trace, as one integer."""
        entry = memo.get(id(formula))
        if entry is not None and entry[0] is formula:
            return entry[1]
        v = self.compute(formula, memo)
        memo[id(formula)] = (formula, v)
        return v

    def compute(self, formula, memo):
        """Truth values of `formula` from those of its children (children are memoized)."""
        label, full, value = formula.label, self.full, self.value
        if label == "!":
            return full ^ value(formula.left, memo)
        if label == "&":
            return value(formula.left, memo) & value(formula.right, memo)
        if label == "|":
            return value(formula.left, memo) | value(formula.right, memo)
        if label == "->":
            return (full ^ value(formula.left, memo)) | value(formula.right, memo)
        if label == "X":
            return (value(formula.left, memo) << 1) & self.next_mask
        if label == "F":
            return self.until(full, value(formula.left, memo))
        if label == "G":
            return full ^ self.until(full, full ^ value(formula.left, memo))
        if label == "U":
            return self.until(value(formula.left, memo), value(formula.right, memo))
        if label == "true":
            return full
        if label == "false":
            return 0
        if label.startswith("!"):
            return full ^ self.atom_bits.get(label[1:], 0)
        return self.atom_bits.get(label, 0)


def filter_consistent(formulas, labelled_traces):
    """The formulas that agree with every (trace, label) pair, in input order."""
    formulas, labelled_traces = list(formulas), list(labelled_traces)
    if not formulas or not labelled_traces:
        return formulas
    sample, memo = PackedSample(labelled_traces), {}
    return [f for f in formulas if sample.consistent(f, memo)]


def consistency_check(labelled_traces):
    """A predicate formula -> bool that does not store the formula's own values.

    Used while enumerating the largest size, where most formulas are thrown away.
    """
    sample, memo = PackedSample(labelled_traces), {}
    compute, starts, positive = sample.compute, sample.starts, sample.positive
    return lambda formula: compute(formula, memo) & starts == positive
