"""Active learning of LTLf formulas through distinguishing queries.

    from ltlf_learner import learn
    result = learn("G(->(a,F(b)))", atoms=("a", "b"), max_size=6)
    print(result.hypothesis, result.queries)
"""
from .formula import Formula, parse
from .learner import Result, learn
from .traces import holds

__all__ = ["Formula", "parse", "learn", "Result", "holds"]
