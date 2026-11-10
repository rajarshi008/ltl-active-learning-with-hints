"""The simulated teacher: answers membership queries for a target formula.

The learner never reads the target; it sees only the labels (teacher.labels is
the sample S) and, when enabled, the teacher's hints.
"""
from .traces import holds, normalize_trace


class Teacher:
    """Labels traces with `target`. `labels` maps each trace to its label, in the order asked."""

    def __init__(self, target, atoms):
        self.target, self.atoms = target, tuple(atoms)
        self.labels = {}

    @property
    def num_queries(self):
        """Distinct traces labelled so far (asking the same trace again is free)."""
        return len(self.labels)

    def membership_query(self, trace):
        """Whether `trace` satisfies the target; the answer is recorded."""
        trace = normalize_trace(trace, self.atoms)
        if trace not in self.labels:
            self.labels[trace] = holds(self.target, trace)
        return self.labels[trace]
