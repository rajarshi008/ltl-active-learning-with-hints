"""Bottom-up enumeration of LTLf formulas by size and temporal depth.

Follows the grammar-based enumeration of Roy et al. (IJCAI 2025): formulas of
size n are built from smaller ones with X F G U & | ->, starting from the atoms
and their negations. Redundant formulas are skipped:

  - pair check: two operands that are the same formula, or one the negation of
    the other (compared by `formula.canonical`), are not combined;
  - temporal simplification: F(F f), G(G f), U(X f, X g), &(G f, G g),
    &(X f, X g), |(F f, F g) and |(X f, X g) are skipped, since a smaller
    formula says the same over finite traces.
"""
from .formula import Formula, canonical, negate

TEMPORAL_UNARY = ("X", "F", "G")
TEMPORAL_BINARY = ("U",)
BOOLEAN_BINARY = ("&", "|", "->")


def _pair_redundant(f1, f2):
    """True if combining f1 and f2 with a binary operator is redundant."""
    if f1.size == 1 and f2.size == 1:
        a, b = f1.label, f2.label
        if a == b or a == "!" + b or b == "!" + a:
            return True
    c1, c2 = canonical(f1), canonical(f2)
    return c1 == c2 or c1 == negate(c2)


def _simplifiable(f):
    """True if `f` has one of the temporal-simplification forms listed above."""
    label, left, right = f.label, f.left, f.right
    if label in ("F", "G"):
        return left.label == label
    if label == "U":
        return left.label == "X" and right.label == "X"
    if label == "&":
        return left.label == right.label and left.label in ("G", "X")
    if label == "|":
        return left.label == right.label and left.label in ("F", "X")
    return False


class Enumerator:
    """All formulas up to `max_size` nodes and temporal depth `max_depth`, one size at a time.

    layers[(depth, size)] is the set of formulas of exactly that depth and size.
    Call next_size() to build the next size; built[(depth, size)] counts the
    formulas generated for a layer, including those the `keep` filter dropped.
    """

    def __init__(self, atoms, max_size, max_depth):
        self.max_size, self.max_depth = max_size, max_depth
        self.layers = {(d, s): set() for d in range(max_depth + 1) for s in range(max_size + 1)}
        for atom in atoms:
            self.layers[(0, 1)].update({Formula(atom), Formula("!" + atom)})
        self.size = 1
        self.built = {(0, 1): len(self.layers[(0, 1)])}

    def formulas_of_size(self, size):
        """All stored formulas of exactly this size (any depth), in no particular order."""
        return [f for d in range(self.max_depth + 1) for f in self.layers.get((d, size), ())]

    def count_of_size(self, size):
        """How many formulas of this size were generated (stored or dropped)."""
        return sum(self.built.get((d, size), 0) for d in range(self.max_depth + 1))

    def next_size(self, keep=None):
        """Build all formulas of size self.size + 1.

        keep: optional predicate; formulas failing it are counted but not stored.
        Use it only for the last size, since stored formulas are the building
        blocks of larger ones.
        """
        size = self.size + 1
        for depth in range(self.max_depth + 1):
            layer, dropped = set(), set()
            for formula in self._candidates(depth):
                if keep is None or keep(formula):
                    layer.add(formula)
                else:
                    dropped.add((formula.label, id(formula.left), id(formula.right)))
            self.layers[(depth, size)] = layer
            self.built[(depth, size)] = len(layer) + len(dropped)
        self.size = size

    def iter_next_size(self, roots=None):
        """Yield each distinct formula of size self.size + 1 whose top operator is in `roots`.

        Nothing is stored and self.size does not change. roots=None yields all.
        Used for the largest size when hints restrict the search.
        """
        for depth in range(self.max_depth + 1):
            seen = set()
            for formula in self._candidates(depth, roots):
                key = (formula.label, id(formula.left), id(formula.right))
                if key not in seen:
                    seen.add(key)
                    yield formula

    def _candidates(self, depth, roots=None):
        """Formulas of size self.size + 1 and temporal depth `depth` (may repeat)."""
        n, layers = self.size, self.layers

        def wanted(op):
            return roots is None or op in roots

        if depth > 0:
            for op in TEMPORAL_UNARY:
                if wanted(op):
                    for child in layers[(depth - 1, n)]:
                        f = Formula(op, child)
                        if not _simplifiable(f):
                            yield f
            if wanted("U"):
                for size in range(1, n):
                    for d in range(depth):
                        for f1 in layers[(d, size)]:
                            for f2 in layers[(depth - 1, n - size)]:
                                if _pair_redundant(f1, f2):
                                    continue
                                g1, g2 = Formula("U", f1, f2), Formula("U", f2, f1)
                                if _simplifiable(g1) or _simplifiable(g2):
                                    continue
                                yield g1
                                yield g2
        ops = [op for op in BOOLEAN_BINARY if wanted(op)]
        if not ops:
            return
        for size in range(1, n):
            for d in range(depth + 1):
                for f1 in layers[(d, size)]:
                    for f2 in layers[(depth, n - size)]:
                        if _pair_redundant(f1, f2):
                            continue
                        for op in ops:
                            f = Formula(op, f1, f2)
                            if not _simplifiable(f):
                                yield f
