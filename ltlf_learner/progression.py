"""LTLf formula progression: decide equivalence and find distinguishing traces without automata.

Reading the first valuation s of a trace turns a formula phi into prog(phi, s),
the formula the rest of the trace must satisfy; a formula holds on the empty
trace when its end value is true. Progressed formulas are kept in a normal form
(negation normal form, with & / | as a set of clauses), so only finitely many
arise, and a breadth-first search over pairs of them finds a shortest trace on
which two formulas disagree (Section V-C and Appendix B of the paper).

Same semantics as traces.holds: X is strong, past the end atoms, X, F and U are
false and negated atoms and G are true; the empty trace is allowed.
"""
from collections import deque


FALSE, TRUE = 0, 1


class Progressor:
    """Formula table, progression and empty-trace acceptance for one alphabet.

    Keeps every node it has built and caches prog(node, valuation), so later
    comparisons in the same run reuse earlier work. Not thread-safe. The cache
    grows with the formulas compared; create one per learning run.

    Node kinds (the tuple stored for an id):
      ('ap', a), ('nap', a)       atom a / its negation
      ('X', c), ('WX', c)         strong / weak next
      ('F', c), ('G', c)
      ('U', l, r), ('R', l, r)    until / release
      ('and', frozenset), ('or', frozenset)
    """

    def __init__(self):
        self.nodes = [('false',), ('true',)]
        self.index = {('false',): FALSE, ('true',): TRUE}
        self.empty = [False, True]       # accepts_empty per id
        self.prog_cache = {}             # (id, valuation) -> id
        self.translated = {}             # (formula, positive) -> id
        self.non_empty = self._node(('F', TRUE))   # "the rest of the trace is non-empty"
        self.at_end = self._node(('G', FALSE))     # "the rest of the trace is empty"

    # -- building nodes -----------------------------------------------------

    def _node(self, key):
        """Id of the node `key`, creating it (and its empty-trace value) if new."""
        node = self.index.get(key)
        if node is not None:
            return node
        node = len(self.nodes)
        self.nodes.append(key)
        self.index[key] = node
        kind = key[0]
        if kind in ('nap', 'WX', 'G', 'R'):
            value = True
        elif kind == 'and':
            value = all(self.empty[c] for c in key[1])
        elif kind == 'or':
            value = any(self.empty[c] for c in key[1])
        else:  # ap, X, F, U
            value = False
        self.empty.append(value)
        return node

    def conj(self, operands):
        """Id of the conjunction of the given ids, in disjunctive normal form (see _clauses)."""
        result = {frozenset()}
        for operand in operands:
            if operand == TRUE:
                continue
            result = {a | b for a in result for b in self._clauses(operand)}
            result = {c for c in result if not self._contradictory(c)}
            if not result:
                return FALSE
        return self._from_clauses(result)

    def disj(self, operands):
        """Id of the disjunction of the given ids, in disjunctive normal form (see _clauses)."""
        result = set()
        for operand in operands:
            if operand == TRUE:
                return TRUE
            result.update(self._clauses(operand))
        return self._from_clauses(result)

    def _clauses(self, node):
        """The node as a disjunction of clauses; a clause is a frozenset of basic ids.

        Basic ids are everything except and/or/true/false. Every and/or node is
        built by _from_clauses, so an 'or' only holds clauses and an 'and' only
        basic ids. Keeping states in this form (with absorption) bounds the number
        of distinct states by the subformulas of the compared formulas.
        """
        if node == TRUE:
            return (frozenset(),)
        if node == FALSE:
            return ()
        key = self.nodes[node]
        if key[0] == 'or':
            return tuple(self.nodes[c][1] if self.nodes[c][0] == 'and' else frozenset((c,))
                         for c in key[1])
        if key[0] == 'and':
            return (key[1],)
        return (frozenset((node,)),)

    def _contradictory(self, clause):
        """True if the clause holds an atom and its negation."""
        seen = {}
        for node in clause:
            key = self.nodes[node]
            if key[0] in ('ap', 'nap'):
                if seen.setdefault(key[1], key[0]) != key[0]:
                    return True
        return False

    def _from_clauses(self, clauses):
        """Id of the disjunction of `clauses` (a set of frozensets of basic ids).

        Clauses that contain another clause are dropped (absorption: c | (c & d) = c).
        The result depends only on the set of clauses, not on iteration order.
        """
        if not clauses:
            return FALSE
        if frozenset() in clauses:
            return TRUE
        kept = []
        for clause in sorted(clauses, key=len):
            if not any(k <= clause for k in kept):
                kept.append(clause)
        # p | !p is true.
        units = {self.nodes[next(iter(c))] for c in kept if len(c) == 1}
        if any(key[0] == 'ap' and ('nap', key[1]) in units for key in units):
            return TRUE
        ids = [next(iter(c)) if len(c) == 1 else self._node(('and', c)) for c in kept]
        if len(ids) == 1:
            return ids[0]
        return self._node(('or', frozenset(ids)))

    # -- from Formula ---------------------------------------------------------

    def translate(self, formula, positive=True):
        """Id of `formula` (a Formula), or of its negation when positive is False.

        Pushes negations down to the atoms (negation normal form). Labels are the
        ones holds() understands: atoms, leaves '!a', 'true', 'false', the
        operators ! & | -> X F G U.
        """
        key = (formula, positive)
        node = self.translated.get(key)
        if node is not None:
            return node
        label = formula.label
        t = self.translate
        if label == 'true':
            node = TRUE if positive else FALSE
        elif label == 'false':
            node = FALSE if positive else TRUE
        elif label == '!':
            node = t(formula.left, not positive)
        elif label == '&':
            parts = [t(formula.left, positive), t(formula.right, positive)]
            node = self.conj(parts) if positive else self.disj(parts)
        elif label == '|':
            parts = [t(formula.left, positive), t(formula.right, positive)]
            node = self.disj(parts) if positive else self.conj(parts)
        elif label == '->':
            parts = [t(formula.left, not positive), t(formula.right, positive)]
            node = self.disj(parts) if positive else self.conj(parts)
        elif label == 'X':
            node = self._node(('X' if positive else 'WX', t(formula.left, positive)))
        elif label == 'F':
            node = self._node(('F' if positive else 'G', t(formula.left, positive)))
        elif label == 'G':
            node = self._node(('G' if positive else 'F', t(formula.left, positive)))
        elif label == 'U':
            node = self._node(('U' if positive else 'R',
                               t(formula.left, positive), t(formula.right, positive)))
        elif label.startswith('!'):  # negated-atom leaf such as '!a'
            node = self._node(('nap' if positive else 'ap', label[1:]))
        else:  # atom
            node = self._node(('ap' if positive else 'nap', label))
        self.translated[key] = node
        return node

    # -- progression --------------------------------------------------------

    def accepts_empty(self, node):
        """True iff the formula with this id holds on the empty trace."""
        return self.empty[node]

    def prog(self, node, valuation):
        """Id of the formula the rest of a trace must satisfy after reading `valuation` first.

        `valuation` is a collection of the atoms true at that step (a sorted tuple
        in the learner). Cached per (id, valuation).
        """
        if node <= TRUE:
            return node
        cache_key = (node, valuation)
        result = self.prog_cache.get(cache_key)
        if result is not None:
            return result
        key = self.nodes[node]
        kind = key[0]
        if kind == 'ap':
            result = TRUE if key[1] in valuation else FALSE
        elif kind == 'nap':
            result = FALSE if key[1] in valuation else TRUE
        elif kind == 'and':
            result = self.conj([self.prog(c, valuation) for c in key[1]])
        elif kind == 'or':
            result = self.disj([self.prog(c, valuation) for c in key[1]])
        elif kind == 'X':
            child = key[1]
            # The rest must be non-empty; that is implied when child fails on empty.
            result = self.conj([child, self.non_empty]) if self.empty[child] else child
        elif kind == 'WX':
            child = key[1]
            result = child if self.empty[child] else self.disj([child, self.at_end])
        elif kind == 'F':
            result = self.disj([self.prog(key[1], valuation), node])
        elif kind == 'G':
            result = self.conj([self.prog(key[1], valuation), node])
        elif kind == 'U':
            left, right = key[1], key[2]
            result = self.disj([self.prog(right, valuation),
                                self.conj([self.prog(left, valuation), node])])
        else:  # 'R'
            left, right = key[1], key[2]
            result = self.conj([self.prog(right, valuation),
                                self.disj([self.prog(left, valuation), node])])
        self.prog_cache[cache_key] = result
        return result

    def holds(self, formula, trace):
        """Whether the Formula holds on `trace`, by progressing through it (reference use)."""
        node = self.translate(formula)
        for valuation in trace:
            node = self.prog(node, valuation)
        return self.accepts_empty(node)


def distinguish_by_progression(first, second, alphabet, *,
                                        max_product_states=100000, progressor=None):
    """Shortest trace on which two Formulas disagree, first in alphabet order, or None.

    A breadth-first search over pairs of progressed formulas that tries the
    valuations in `alphabet` order.
    A pair of identical formulas is not expanded (they can never disagree),
    which does not change the first witness found.

    Returns None iff the formulas are equivalent over finite traces (including
    the empty one). Raises RuntimeError when more than `max_product_states`
    pairs would be stored. progressor: a Progressor to reuse across calls
    (one per run), so progressed formulas are shared between comparisons.
    """
    p = progressor if progressor is not None else Progressor()
    start = (p.translate(first), p.translate(second))
    if start[0] == start[1]:
        return None
    empty = p.empty
    queue = deque([(start, ())])
    seen = {start}
    while queue:
        (s1, s2), prefix = queue.popleft()
        if empty[s1] != empty[s2]:
            return prefix
        for symbol in alphabet:
            pair = (p.prog(s1, symbol), p.prog(s2, symbol))
            if pair[0] == pair[1] or pair in seen:
                continue
            if len(seen) >= max_product_states:
                raise RuntimeError("equivalence product-state budget exceeded")
            seen.add(pair)
            queue.append((pair, prefix + (symbol,)))
    return None
