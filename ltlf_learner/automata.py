"""Automata-based Distinguish (optional): LTLf -> DFA with ltlf2dfa/MONA, then a product search.

Needs the `ltlf2dfa` Python package and the `mona` binary on PATH. The learner
uses this only with distinguish="automata"; everything else works without it.
Automata are not cached: every comparison compiles both formulas again.
"""
from collections import deque
from contextlib import redirect_stdout
from functools import lru_cache
from io import StringIO


class DFA:
    """A complete DFA over valuations: start, accepting states, and step(state, valuation)."""

    def __init__(self, start, accepting, edges, atoms):
        self.start, self.accepting = start, frozenset(accepting)
        self.edges, self.atoms = edges, tuple(atoms)   # edges: state -> [(guard, target)]
        self._cache = {}

    def step(self, state, valuation):
        key = (state, valuation)
        if key not in self._cache:
            true = set(valuation)
            self._cache[key] = next(target for guard, target in self.edges[state]
                                    if _evaluate(guard, true))
        return self._cache[key]

    def accepts(self, trace):
        state = self.start
        for valuation in trace:
            state = self.step(state, valuation)
        return state in self.accepting


@lru_cache(maxsize=1)
def _parser():
    """One ltlf2dfa parser per process (building one reads its grammar file)."""
    import warnings

    from ltlf2dfa.parser.ltlf import LTLfParser

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return LTLfParser()


def compile_formula(formula, atoms):
    """The DFA of `formula` (a Formula), computed by MONA through ltlf2dfa."""
    with redirect_stdout(StringIO()):
        dot = _parser()(formula.infix()).to_dfa()
    return _read_dot(dot, atoms)


def _read_dot(dot, atoms):
    """Parse the DOT text written by ltlf2dfa into a DFA."""
    start, accepting, edges = None, set(), {}
    for line in dot.splitlines():
        line = line.strip()
        if "doublecircle" in line:
            accepting.update(s.strip() for s in line.split("]", 1)[1].split(";") if s.strip())
        elif line.startswith("init ->"):
            start = line.split("->")[1].strip(" ;")
        elif "->" in line and "label=" in line:
            source, rest = line.split("->", 1)
            target = rest.split("[")[0].strip()
            guard = rest.split('label="', 1)[1].split('"', 1)[0]
            edges.setdefault(source.strip(), []).append((_parse_guard(guard), target))
    return DFA(start, accepting, edges, atoms)


def _parse_guard(text):
    """Parse a MONA edge guard such as '~a & b | c' into a small tree."""
    tokens = text.replace("(", " ( ").replace(")", " ) ").replace("~", " ~ ") \
                 .replace("&", " & ").replace("|", " | ").split()

    def disjunction(i):
        node, i = conjunction(i)
        while i < len(tokens) and tokens[i] == "|":
            right, i = conjunction(i + 1)
            node = ("|", node, right)
        return node, i

    def conjunction(i):
        node, i = negation(i)
        while i < len(tokens) and tokens[i] == "&":
            right, i = negation(i + 1)
            node = ("&", node, right)
        return node, i

    def negation(i):
        if tokens[i] == "~":
            child, i = negation(i + 1)
            return ("~", child), i
        if tokens[i] == "(":
            node, i = disjunction(i + 1)
            return node, i + 1
        return ("atom", tokens[i]), i + 1

    return disjunction(0)[0]


def _evaluate(guard, true_atoms):
    kind = guard[0]
    if kind == "atom":
        name = guard[1]
        return name == "true" or (name != "false" and name in true_atoms)
    if kind == "~":
        return not _evaluate(guard[1], true_atoms)
    if kind == "&":
        return _evaluate(guard[1], true_atoms) and _evaluate(guard[2], true_atoms)
    return _evaluate(guard[1], true_atoms) or _evaluate(guard[2], true_atoms)


def distinguish_by_automata(first, second, atoms, alphabet, max_product_states=100000):
    """Shortest trace on which the formulas disagree, first in alphabet order, or None.

    Breadth-first search over pairs of DFA states; same answer as
    progression.distinguish_by_progression. Raises RuntimeError when more than
    `max_product_states` pairs would be stored.
    """
    a, b = compile_formula(first, atoms), compile_formula(second, atoms)
    start = (a.start, b.start)
    queue, seen = deque([(start, ())]), {start}
    while queue:
        (s1, s2), prefix = queue.popleft()
        if (s1 in a.accepting) != (s2 in b.accepting):
            return prefix
        for valuation in alphabet:
            pair = (a.step(s1, valuation), b.step(s2, valuation))
            if pair not in seen:
                if len(seen) >= max_product_states:
                    raise RuntimeError("product-state budget exceeded")
                seen.add(pair)
                queue.append((pair, prefix + (valuation,)))
    return None
