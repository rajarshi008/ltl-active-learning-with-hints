"""VersionSpace (Section V-A), by enumeration: the formulas of the class that fit every label.

Interface used by the learning loop (learner.py):
    stages(state, teacher)   generator; each step adds the consistent formulas of
                             the next part of the class schedule and, if there is
                             no hypothesis yet, takes the first (smallest) one
    update(state, teacher)   drop formulas that contradict new labels
    has_rivals(state)        is there a formula besides the hypothesis?
    rivals(state)            the other formulas, in version-space order
    remove(state, formula)   drop a formula equivalent to the hypothesis
    exhausted()              has the version space's own budget run out?
    built                    number of formulas generated so far

state.candidates is the version space, kept in order of size, then text, so the
first candidate is a smallest consistent formula.
"""
import gc
from contextlib import contextmanager

from .enumerate import Enumerator
from .hints import in_classes
from .traces import consistency_check, filter_consistent


class EnumeratedVersionSpace:
    """Version space over all formulas up to `max_size` nodes and temporal depth `max_depth`.

    levels: None for the default schedule (size 1, 2, ..., max_size), or hint
        levels (name, root operators, classes[, extra predicate]) from
        hints.search_levels. Each level is searched by size; the search widens
        to the next level only while no hypothesis survives.
    After a run, `levels_searched` lists the levels used.
    """

    def __init__(self, atoms, max_size, max_depth, levels=None):
        self.enumerator = Enumerator(atoms, max_size, max_depth)
        self.max_size, self.levels = max_size, levels
        self.built = 0
        self.levels_searched = []

    # -- the class schedule ---------------------------------------------------
    def stages(self, state, teacher):
        if self.levels is None:
            yield from self._by_size(state, teacher)
        else:
            yield from self._by_level(state, teacher)

    def _by_size(self, state, teacher):
        enum = self.enumerator
        for size in range(1, self.max_size + 1):
            with _gc_paused():
                if size > 1:
                    # The largest size builds nothing larger, so inconsistent
                    # formulas are dropped while it is generated.
                    last = size == self.max_size
                    enum.next_size(keep=consistency_check(teacher.labels.items()) if last else None)
                self.built += enum.count_of_size(size)
                _add(state, teacher, enum.formulas_of_size(size))
            yield size

    def _by_level(self, state, teacher):
        enum, counted = self.enumerator, set()
        for level in self.levels:
            name, roots, classes = level[:3]
            extra = level[3] if len(level) > 3 else None
            self.levels_searched.append(name)
            for size in range(1, self.max_size + 1):
                with _gc_paused():
                    formulas = self._level_formulas(enum, teacher, size, roots, classes, counted)
                    if extra is not None:
                        formulas = [f for f in filter_consistent(formulas, teacher.labels.items())
                                    if extra(f)]
                    if state.candidates:   # keep formulas found in an earlier level once
                        known = set(state.candidates)
                        formulas = [f for f in formulas if f not in known]
                    _add(state, teacher, formulas)
                yield (name, size)
            if state.current is not None:
                return

    def _level_formulas(self, enum, teacher, size, roots, classes, counted):
        """Formulas of `size` in the level. The largest size is built afresh per level."""
        def in_level(f):
            return ((roots is None or f.label in roots)
                    and (classes is None or in_classes(f, classes)))
        if size < self.max_size or size == 1:
            while enum.size < size:
                enum.next_size()
            if size not in counted:
                counted.add(size)
                self.built += enum.count_of_size(size)
            return [f for f in enum.formulas_of_size(size) if in_level(f)]
        while enum.size < size - 1:
            enum.next_size()
        keep, out = consistency_check(teacher.labels.items()), []
        for f in enum.iter_next_size(roots):
            self.built += 1
            if keep(f) and in_level(f):
                out.append(f)
        return out

    # -- updates ----------------------------------------------------------------
    def update(self, state, teacher):
        _drop_inconsistent(state, teacher)

    def has_rivals(self, state):
        return len(state.candidates) > 1

    def rivals(self, state):
        return [f for f in state.candidates if f != state.current]

    def remove(self, state, formula):
        state.candidates.remove(formula)

    def exhausted(self):
        return False


def _add(state, teacher, formulas):
    """Update old candidates with new labels, then add the new formulas that fit all labels.

    New formulas are sorted by text, so candidates stay in order of size, then text.
    """
    new_labels = list(teacher.labels.items())[state.checked:]
    state.candidates = filter_consistent(state.candidates, new_labels)
    state.candidates.extend(sorted(filter_consistent(formulas, teacher.labels.items()), key=str))
    state.checked = len(teacher.labels)
    if state.current is None and state.candidates:
        state.current = state.candidates[0]


def _drop_inconsistent(state, teacher):
    """Remove candidates that contradict labels added since the last check.

    If the hypothesis was removed, the first remaining candidate takes over.
    """
    new_labels = list(teacher.labels.items())[state.checked:]
    state.candidates = filter_consistent(state.candidates, new_labels)
    state.checked = len(teacher.labels)
    if state.current not in state.candidates:
        state.current = state.candidates[0] if state.candidates else None


@contextmanager
def _gc_paused():
    """Pause the cyclic garbage collector while millions of formulas are built (speed only)."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()
