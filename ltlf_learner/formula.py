"""LTLf formulas: the syntax tree, a parser for prefix syntax, and printing.

Formulas are written in prefix syntax, for example ``G(->(a,F(b)))``:

    atoms        a, b, door_1, ...   (lower-case names)
    constants    true, false
    unary        !(f)  X(f)  F(f)  G(f)
    binary       &(f,g)  |(f,g)  ->(f,g)  U(f,g)

A negated atom ``!(a)`` is stored as a single leaf with label ``'!a'``, so it has
size 1, as in the paper. Formulas are immutable and compared structurally.
"""

UNARY = ("!", "X", "F", "G")
BINARY = ("&", "|", "->", "U")
TEMPORAL = ("X", "F", "G", "U")


class Formula:
    """An LTLf syntax tree node: `label` plus up to two children.

    Leaves are atoms ('a'), negated atoms ('!a') or constants ('true', 'false').
    `size` counts syntax-tree nodes (a negated atom counts as one node).
    """

    __slots__ = ("label", "left", "right", "size", "_hash", "_text", "_canon")

    def __init__(self, label, left=None, right=None):
        self.label, self.left, self.right = label, left, right
        self.size = 1 + (left.size if left is not None else 0) + (right.size if right is not None else 0)
        self._hash = None
        self._text = None
        self._canon = None

    # -- comparison ------------------------------------------------------------
    def __hash__(self):
        if self._hash is None:
            self._hash = hash((self.label, self.left, self.right))
        return self._hash

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Formula) or hash(self) != hash(other):
            return False
        return (self.label == other.label and self.left == other.left
                and self.right == other.right)

    # -- printing --------------------------------------------------------------
    def __str__(self):
        """Prefix text, e.g. G(->(a,F(b))). Negated atoms print as !a."""
        if self._text is None:
            if self.left is None:
                self._text = self.label
            elif self.right is None:
                self._text = f"{self.label}({self.left})"
            else:
                self._text = f"{self.label}({self.left},{self.right})"
        return self._text

    __repr__ = __str__

    def infix(self):
        """Infix text for ltlf2dfa/MONA, e.g. (G (a -> (F b)))."""
        if self.left is None:
            return self.label
        if self.right is None:
            return f"({self.label} {self.left.infix()})"
        return f"({self.left.infix()} {self.label} {self.right.infix()})"

    # -- measures --------------------------------------------------------------
    def temporal_depth(self):
        """Largest number of temporal operators on a root-to-leaf path."""
        if self.left is None:
            return 0
        below = max(self.left.temporal_depth(),
                    self.right.temporal_depth() if self.right is not None else 0)
        return below + (self.label in TEMPORAL)

    def atoms(self):
        """The atoms occurring in the formula, as a sorted tuple."""
        if self.left is None:
            name = self.label.lstrip("!")
            return () if name in ("true", "false") else (name,)
        found = set(self.left.atoms()) | set(self.right.atoms() if self.right else ())
        return tuple(sorted(found))


def parse(text):
    """Parse prefix syntax into a Formula; raises ValueError on bad input.

    Accepts both !(a) and !a for a negated atom; both give the leaf '!a'.
    """
    tokens = _tokenize(text)
    formula, position = _parse(tokens, 0)
    if position != len(tokens):
        raise ValueError(f"unexpected text after formula in {text!r}")
    return formula


def _tokenize(text):
    tokens, i = [], 0
    while i < len(text):
        c = text[i]
        if c.isspace():
            i += 1
        elif text.startswith("->", i):
            tokens.append("->")
            i += 2
        elif c in "()!&|,XFGU":
            tokens.append(c)
            i += 1
        elif c.isalpha() and c.islower():
            j = i
            while j < len(text) and (text[j].isalnum() or text[j] == "_"):
                j += 1
            tokens.append(text[i:j])
            i = j
        else:
            raise ValueError(f"unexpected character {c!r} in {text!r}")
    return tokens


def _expect(tokens, position, token):
    if position >= len(tokens) or tokens[position] != token:
        raise ValueError(f"expected {token!r} at token {position} of {' '.join(tokens)!r}")
    return position + 1


def _parse(tokens, position):
    if position >= len(tokens):
        raise ValueError("formula ends too early")
    token = tokens[position]
    if token in UNARY:
        if token == "!" and position + 1 < len(tokens) and tokens[position + 1] not in "(":
            child, position = _parse(tokens, position + 1)     # !a without brackets
        else:
            position = _expect(tokens, position + 1, "(")
            child, position = _parse(tokens, position)
            position = _expect(tokens, position, ")")
        if token == "!" and child.left is None and child.label not in ("true", "false"):
            label = child.label[1:] if child.label.startswith("!") else "!" + child.label
            return Formula(label), position                     # negated atom as one leaf
        return Formula(token, child), position
    if token in BINARY:
        position = _expect(tokens, position + 1, "(")
        left, position = _parse(tokens, position)
        position = _expect(tokens, position, ",")
        right, position = _parse(tokens, position)
        position = _expect(tokens, position, ")")
        return Formula(token, left, right), position
    if token in ("(", ")", ","):
        raise ValueError(f"unexpected {token!r}")
    return Formula(token), position + 1


def canonical(formula):
    """A hashable normal form used to spot syntactic duplicates during enumeration.

    Equal canonical forms mean equal formulas up to: the order and repetition of
    & and | operands, nested & / |, -> written as |(!l, r), double negation,
    F(F(f)) = F(f) and G(G(f)) = G(f). Negations are kept as ('!', ...).
    The result is cached on the formula.
    """
    if formula._canon is None:
        formula._canon = _canonical(formula)
    return formula._canon


def _canonical(formula):
    label = formula.label
    if formula.left is None:
        if label.startswith("!"):
            return ("!", ("ap", label[1:]))
        return ("ap", label)
    if label == "!":
        return negate(canonical(formula.left))
    if label in ("&", "|", "->"):
        op = "&" if label == "&" else "|"
        left = canonical(formula.left)
        if label == "->":
            left = negate(left)
        parts = set()
        for part in (left, canonical(formula.right)):
            parts.update(part[1] if part[0] == op else (part,))
        if len(parts) == 1:
            return next(iter(parts))
        return (op, frozenset(parts))
    child = canonical(formula.left)
    if label in ("F", "G") and child[0] == label:
        return child
    if label == "U":
        return ("U", child, canonical(formula.right))
    return (label, child)


def negate(form):
    """Canonical form of the negation of a canonical form (removes double negation)."""
    return form[1] if form[0] == "!" else ("!", form)
