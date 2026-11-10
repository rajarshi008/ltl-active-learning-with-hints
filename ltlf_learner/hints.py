"""Natural-language hints (Section V-D).

Level 1: the teacher answers yes to question types (Table I of the paper). Each
question type stands for a few specification patterns ("formula classes"),
written as templates whose holes are Boolean formulas.
Level 2: the teacher states a requirement h with target |= h, translated into
an LTLf formula (here chosen by a simulated teacher from fixed templates).

Hints are used in two ways:
    ranking   rivals are sorted (stably) by their distance to the hinted classes;
    pruning   the class schedule becomes levels: first the formulas in the
              hinted classes, then formulas with a matching top operator, then
              all formulas; a Level-2 hint adds a first level of formulas that
              reject every short trace h rejects.
"""
from functools import lru_cache
from itertools import permutations

from .formula import Formula, parse
from .progression import Progressor, distinguish_by_progression

# Question types and the formula classes they stand for.
QUESTION_CLASSES = {
    "undesired_behavior": ("avoidance", "invariance", "mutual_exclusion"),
    "missing_behavior": ("eventuality", "response", "strict_response"),
    "ordering": ("until", "sequence", "strict_sequence", "response",
                 "strict_response", "immediate_response"),
    "completion": ("eventuality", "response", "strict_response", "stabilization"),
    "repetition": ("recurrence",),
}

# Class templates: None is a Boolean formula; (label, child, ...) a fixed node.
# SEQUENCE is S ::= F(B) | F(B & S); STRICT_SEQUENCE is S ::= F(B) | F(B & X(S)).
CLASS_TEMPLATES = {
    "boolean": None,
    "invariance": ("G", None),
    "avoidance": ("G", ("!", None)),
    "eventuality": ("F", None),
    "response": ("G", ("|", ("!", None), ("F", None))),
    "strict_response": ("G", ("|", ("!", None), ("X", ("F", None)))),
    "immediate_response": ("G", ("|", ("!", None), ("X", None))),
    "recurrence": ("G", ("F", None)),
    "stabilization": ("F", ("G", None)),
    "until": ("U", None, None),
    "mutual_exclusion": ("G", ("!", ("&", None, None))),
    "sequence": "SEQUENCE",
    "strict_sequence": "STRICT_SEQUENCE",
}

_TEMPORAL = frozenset("XFGU")


# ---------------------------------------------------------------------------
# Level 1: class membership and distance
# ---------------------------------------------------------------------------

def classes_of(question_types):
    """The formula classes of the given question types, without repeats, in order."""
    return tuple(dict.fromkeys(c for q in question_types for c in QUESTION_CLASSES[q]))


def class_distances(formula):
    """For each class, the fewest node edits that turn `formula` into an instance of it.

    0 means the formula is an instance. -> is read as |(!l, r) and a negated
    atom as !(atom), so that the templates apply.
    """
    nodes = []

    def add(label, left=-1, right=-1):
        nodes.append((label, left, right))
        return len(nodes) - 1

    def snapshot(f):
        if f is None:
            return -1
        left, right = snapshot(f.left), snapshot(f.right)
        if f.label == "->":
            return add("|", add("!", left), right)
        if f.left is None and f.label.startswith("!"):
            return add("!", add(f.label[1:]))
        return add(f.label, left, right)

    root = snapshot(formula)

    @lru_cache(None)
    def size(i):
        return 0 if i == -1 else 1 + size(nodes[i][1]) + size(nodes[i][2])

    @lru_cache(None)
    def boolean_distance(i):
        if i == -1:
            return 1
        label, left, right = nodes[i]
        return (int(label in _TEMPORAL) + (boolean_distance(left) if left != -1 else 0)
                + (boolean_distance(right) if right != -1 else 0))

    @lru_cache(None)
    def match(i, template):
        if template is None:
            return boolean_distance(i)
        if template in ("SEQUENCE", "STRICT_SEQUENCE"):
            if i == -1:
                return 2
            rest = ("X", template) if template == "STRICT_SEQUENCE" else template
            return min(match(i, ("F", None)), match(i, ("F", ("&", None, rest))))
        label, *children = template
        if i == -1:
            return 1 + sum(match(-1, c) for c in children)
        actual, left, right = nodes[i]
        cost = int(actual != label)
        for position, source in enumerate((left, right)):
            cost += match(source, children[position]) if position < len(children) else size(source)
        return cost

    return {name: match(root, template) for name, template in CLASS_TEMPLATES.items()}


def matching_classes(formula):
    """The classes `formula` is an instance of."""
    return [name for name, d in class_distances(formula).items() if d == 0]


def _is_boolean(f):
    return f is None or (f.label not in _TEMPORAL and _is_boolean(f.left) and _is_boolean(f.right))


def _boolean_template(t):
    if t is None:
        return True
    if isinstance(t, str):
        return False
    return t[0] not in _TEMPORAL and all(_boolean_template(c) for c in t[1:])


def _simplify_template(t):
    """Replace each temporal-free part of a template by a Boolean hole (None)."""
    if _boolean_template(t):
        return None
    if isinstance(t, str):
        return t
    return (t[0],) + tuple(_simplify_template(c) for c in t[1:])


def _match(f, t):
    """Syntactic instance check, up to the order of & / | operands and -> as |(!l, r)."""
    if t is None:
        return _is_boolean(f)
    if f is None:
        return False
    if t in ("SEQUENCE", "STRICT_SEQUENCE"):
        if f.label != "F":
            return False
        body = f.left
        if _is_boolean(body):
            return True
        if body.label != "&":
            return False
        for boolean_part, rest in ((body.left, body.right), (body.right, body.left)):
            if t == "STRICT_SEQUENCE":
                if rest.label != "X":
                    continue
                rest = rest.left
            if _is_boolean(boolean_part) and _match(rest, t):
                return True
        return False
    label, *children = t
    if label in ("|", "&") and len(children) == 2:
        if f.label == label:
            return any(_match(l, children[0]) and _match(r, children[1])
                       for l, r in ((f.left, f.right), (f.right, f.left)))
        if label == "|" and f.label == "->":
            return ((children[0] is None and _is_boolean(f.left) and _match(f.right, children[1]))
                    or (children[1] is None and _is_boolean(f.left) and _is_boolean(f.right)))
        return False
    if f.label != label:
        return False
    return all(_match((f.left, f.right)[i], c) for i, c in enumerate(children))


def in_classes(formula, classes):
    """Whether `formula` is an instance of one of the named classes."""
    return any(_match(formula, _simplify_template(CLASS_TEMPLATES[c])) for c in classes)


def root_operators(classes):
    """Top operators that instances of the named classes can have."""
    roots = set()
    for c in classes:
        t = CLASS_TEMPLATES[c]
        roots.update(("&", "|", "->") if t is None else ("F",) if isinstance(t, str) else (t[0],))
    return tuple(sorted(roots))


class HintGuidance:
    """Level-1 hints as seen by the rival orders: distance(formula) to the hinted classes."""

    def __init__(self, classes=()):
        self.classes = tuple(classes)

    def distance(self, formula):
        if not self.classes:
            return 0
        distances = class_distances(formula)
        return min(distances[c] for c in self.classes)


def search_levels(classes):
    """Hint pruning levels: (name, root operators, classes) from narrowest to all formulas."""
    if not classes:
        return [("full", None, None)]
    roots = root_operators(classes)
    return [("template", roots, tuple(classes)), ("root", roots, None), ("full", None, None)]


# ---------------------------------------------------------------------------
# Simulated teacher: Level-1 answers
# ---------------------------------------------------------------------------

def teacher_level1(target):
    """Question types a simulated teacher answers yes for `target`.

    Yes for every question type whose classes contain the target; none if the
    target is in no class (the teacher answers unsure).
    """
    matches = set(matching_classes(target))
    return [q for q, classes in QUESTION_CLASSES.items() if matches & set(classes)]


def wrong_level1(target):
    """An incorrect Level-1 answer: a question type whose classes do not contain the target."""
    matches = set(matching_classes(target))
    for q in ("repetition", "undesired_behavior", "completion", "ordering"):
        if not matches & set(QUESTION_CLASSES[q]):
            return [q]
    return None


# ---------------------------------------------------------------------------
# Level 2: requirements
# ---------------------------------------------------------------------------

# (question type, answer text, formula) with p, q two distinct atoms.
LEVEL2_TEMPLATES = (
    ("undesired_behavior", "{p} should not happen", "G(!({p}))"),
    ("undesired_behavior", "{p} should not happen when {q}", "G(->({q},!({p})))"),
    ("missing_behavior", "{p} should happen", "F({p})"),
    ("missing_behavior", "{p} should happen after {q}", "G(->({q},F({p})))"),
    ("missing_behavior", "{p} should happen right after {q}", "G(->({q},X({p})))"),
    ("ordering", "{p} should happen before {q}", "|(U(!({q}),{p}),G(!({q})))"),
    ("completion", "after {q}, the task is complete when {p}", "G(->({q},F({p})))"),
    ("repetition", "{p} should happen repeatedly", "G(F({p}))"),
    ("repetition", "{p} should happen only once", "G(->({p},X(G(!({p})))))"),
)


def level2_filter(pool, requirements):
    """Predicate for the Level-2 level: does a formula reject every pool trace some h rejects?

    pool: a ShortTracePool. This checks formula |= h on the short traces only,
    so a formula that really implies every h is never dropped.
    """
    sample, memo = pool.sample, pool.memo
    allowed = sample.starts
    for h in requirements:
        allowed &= sample.value(h, memo)
    forbidden = sample.starts & ~allowed
    return lambda formula: not (sample.value(formula, memo) & forbidden)


def level2_requirements(atoms):
    """All template instances over `atoms`: (question type, answer text, formula), no repeats."""
    seen, out = set(), []
    for question, answer, pattern in LEVEL2_TEMPLATES:
        pairs = permutations(atoms, 2) if "{q}" in pattern else ((p, p) for p in atoms)
        for p, q in pairs:
            text = pattern.format(p=p, q=q)
            if text not in seen:
                seen.add(text)
                out.append((question, answer.format(p=p, q=q), parse(text)))
    return out


def implies(first, second, alphabet, progressor=None):
    """Whether every finite trace satisfying `first` satisfies `second` (exact)."""
    both = Formula("&", first, Formula("!", second))
    return distinguish_by_progression(both, Formula("false"), alphabet,
                                      progressor=progressor) is None


def simulated_level2(target, atoms, alphabet, pool, mode="correct"):
    """The Level-2 requirement a simulated teacher states for `target`, or None.

    mode 'correct': the strongest template instance h (fewest pool traces
        accepted) with target |= h that is neither valid nor equivalent to the target.
    mode 'wrong': an h the target does not imply, violated by the target on the
        fewest pool traces (a plausible mistake).
    pool: the PackedSample of a ShortTracePool, used to rank by strength.
    """
    progressor, memo, starts = Progressor(), {}, pool.starts
    true = Formula("true")
    target_value = pool.value(target, memo)
    scored = []
    for _, _, h in level2_requirements(atoms):
        if implies(true, h, alphabet, progressor):
            continue
        h_value = pool.value(h, memo)
        accepted = (h_value & starts).bit_count()
        target_implies_h = implies(target, h, alphabet, progressor)
        if mode == "wrong":
            if not target_implies_h:
                violations = (target_value & ~h_value & starts).bit_count()
                scored.append(((violations, accepted, str(h)), h))
        elif target_implies_h and not implies(h, target, alphabet, progressor):
            scored.append(((accepted, str(h)), h))
    return min(scored, key=lambda item: item[0])[1] if scored else None
