"""Certify (Section V-E): a PAC test of the final hypothesis.

The hypothesis is frozen and tested on P fresh traces drawn i.i.d. from a
distribution D. If all P tests agree with the teacher, then with probability at
least 1 - delta the error err_D is at most ln(1/delta)/P (Theorem 4).

Distributions: free traces have a geometric length (P(k) = (1-p)^(k-1) p, k >= 1)
and uniform letters; environment traces have a geometric number of labels and
uniformly random actions. Neither draws the empty trace.
"""
import math

from .traces import holds


def tests_needed(epsilon, delta):
    """P = ceil(ln(1/delta) / epsilon) tests certify error at most epsilon."""
    if not (0 < epsilon < 1 and 0 < delta < 1):
        raise ValueError("epsilon and delta must lie in (0, 1)")
    return math.ceil(math.log(1 / delta) / epsilon)


def error_bound(num_tests, delta):
    """The bound ln(1/delta) / P certified by P agreeing tests."""
    return math.log(1 / delta) / num_tests


def geometric_length(rng, p):
    k = 1
    while rng.random() >= p:
        k += 1
    return k


def free_trace(rng, alphabet, p=0.2):
    """A free trace with geometric length (mean 1/p) and uniform letters."""
    return tuple(rng.choice(alphabet) for _ in range(geometric_length(rng, p)))


def environment_trace(rng, env, p=0.1):
    """A feasible trace with a geometric number of labels and random actions."""
    state, trace = env.start, [env.label(env.start)]
    for _ in range(geometric_length(rng, p) - 1):
        state = env.transition(state, rng.choice(env.actions))
        trace.append(env.label(state))
    return tuple(trace)


def certify(hypothesis, teacher, sampler, num_tests):
    """Test `hypothesis` on up to `num_tests` traces from sampler(); stop at the first disagreement.

    Each test is a membership query. Returns a dict with 'passed', 'tests_used'
    and 'counterexample' (None when every test agreed).
    """
    for used in range(1, num_tests + 1):
        trace = sampler()
        if holds(hypothesis, trace) != teacher.membership_query(trace):
            return {"passed": False, "tests_used": used, "counterexample": trace}
    return {"passed": True, "tests_used": num_tests, "counterexample": None}
