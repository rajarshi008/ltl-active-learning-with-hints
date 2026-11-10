"""Algorithm 1 of the paper (active learning by distinguishing queries) and its entry point `learn`.

    S <- Initialize                                    line 1
    for each step of the class schedule:               VersionSpace     (version_space*.py)
        add the consistent formulas; pick a smallest hypothesis
        AskHint (optional, at most once)               hints.py
        repeat
            drop formulas that contradict new labels
            for the first K rivals in order:           NextRival        (rivals.py)
                w <- Distinguish(hypothesis, rival)    Distinguish      (distinguish.py)
                none: remove the rival (equivalent)
                else: ask the teacher about w, keep the formula that is right; next round
        until a round asks nothing
    Certify (optional)                                 certify.py

`learn` builds the procedures from simple options and runs the loop.
"""
import math
import random
import time
from dataclasses import dataclass, field

from .certify import certify, environment_trace, free_trace
from .distinguish import UNKNOWN, Automata, PoolFirst, Progression, ShortTracePool
from .environments import EnvironmentPool, EnvironmentSearch
from .formula import parse
from .hints import (HintGuidance, classes_of, level2_filter, matching_classes, search_levels,
                    teacher_level1)
from .progression import Progressor, distinguish_by_progression
from .rivals import RivalOrder
from .teacher import Teacher
from .traces import holds, random_trace, valuation_alphabet
from .version_space import EnumeratedVersionSpace


@dataclass
class LearnerState:
    """What the learner knows during a run (the sample itself is teacher.labels)."""
    candidates: list = field(default_factory=list)   # the version space (enumerated)
    current: object = None                            # the hypothesis
    checked: int = 0                                  # labels the candidates were checked against
    comparisons: int = 0                              # comparisons counted against the budget
    equivalence_checks: int = 0                       # rivals found equivalent (exhaustive rule)
    guidance: HintGuidance = field(default_factory=HintGuidance)
    budget_exhausted: bool = False
    initial_queries: int = 0                          # labels from Initialize


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def active_learn(teacher, initial_traces, version_space, order, distinguish, k=None,
                 ask_hint=None, max_comparisons=math.inf, state=None):
    """Run Algorithm 1 and return the final LearnerState.

    k: shortlist size K (top-K rule), or None for the exhaustive rule.
    ask_hint: optional callable(state) for AskHint (line 7).
    """
    state = state or LearnerState()
    for trace in initial_traces:
        teacher.membership_query(trace)
    state.initial_queries = teacher.num_queries
    for _ in version_space.stages(state, teacher):
        if state.current is None:
            continue
        if ask_hint is not None:
            ask_hint(state)
        while state.comparisons < max_comparisons:
            version_space.update(state, teacher)
            if state.current is None or not version_space.has_rivals(state):
                break
            if not _round(state, teacher, version_space, order, distinguish, k, max_comparisons):
                break
        if state.comparisons >= max_comparisons or version_space.exhausted():
            state.budget_exhausted = True
            break
    return state


def _round(state, teacher, version_space, order, distinguish, k, max_comparisons):
    """One refinement round: compare the hypothesis with rivals until one gives a query.

    Top-K rule (k given): only the first k rivals, and every comparison counts
    against the budget. Exhaustive rule (k None): all rivals; comparisons that
    find an equivalent rival count as equivalence checks instead.
    Returns True if a query was asked.
    """
    exhaustive = k is None
    rivals = order(state, version_space)
    if not exhaustive:
        rivals = list(rivals)[:max(1, k)]
    for rival in rivals:
        if state.comparisons >= max_comparisons:
            break
        if not exhaustive:
            state.comparisons += 1
        witness = distinguish(state.current, rival)
        if witness is None or witness == UNKNOWN:
            if witness is None:
                version_space.remove(state, rival)      # equivalent to the hypothesis
            state.equivalence_checks += exhaustive
            continue
        if exhaustive:
            state.comparisons += 1
        if witness in teacher.labels:                   # cannot happen: both fit every label
            continue
        label = teacher.membership_query(witness)
        if holds(state.current, witness) != label:
            state.current = rival                       # the rival is right on the new label
        return True
    return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@dataclass
class Result:
    """The outcome of one learning run."""
    hypothesis: object          # the final hypothesis (a Formula), or None
    status: str                 # 'completed', 'budget_exhausted' or 'no_consistent_formula'
    queries: int                # membership queries, including the initial sample
    initial_queries: int
    comparisons: int
    equivalence_checks: int
    candidates: int             # formulas generated by the version space
    seconds: float
    levels: list                # hint levels searched (empty without pruning)
    hint_classes: tuple         # Level-1 classes used
    equivalent: object = None   # final check against the target: True, False or None (unknown)
    witness: object = None      # a trace on which hypothesis and target disagree, if any
    certification: object = None
    target_text: str = ""

    @property
    def exact(self):
        """Whether the hypothesis is the target formula itself (same text)."""
        return self.hypothesis is not None and str(self.hypothesis) == self.target_text

    def summary(self):
        return (f"hypothesis {self.hypothesis}  equivalent={self.equivalent}  "
                f"queries={self.queries}  comparisons={self.comparisons}  "
                f"candidates={self.candidates}  {self.seconds:.2f}s")


def learn(target, atoms, max_size, max_depth=2, *, rule="exhaustive", k=3, order="enumeration",
          distinguish="progression", version_space="enumeration", level1=None, level1_mode="rank",
          level2=None, environment=None, sample_size=10, length_range=None, initial_traces=None,
          seed=0, max_comparisons=100, max_states=100000, max_steps=30, certify_tests=0,
          sat_time_limit=600.0, sat_max_calls=200000, evaluate=True):
    """Learn an LTLf formula for `target` from membership queries (and optional hints).

    target: the teacher's formula (a Formula or prefix text). The learner only
        sees the teacher's answers.
    atoms: the propositions. max_size, max_depth: the searched class L_{n,d}.
    rule: 'exhaustive' (compare with all rivals) or 'top_k' (only the first k).
    order: 'enumeration', 'semantic', 'semantic_far' or 'split'.
    distinguish: 'progression' or 'automata' (needs ltlf2dfa and MONA).
    version_space: 'enumeration' or 'sat' (needs z3; exhaustive rule,
        enumeration order, free traces and no hints only).
    level1: None, 'teacher' (a simulated teacher answers), or a list of question
        types answered yes (e.g. an incorrect hint).
    level1_mode: 'rank' (reorder rivals) or 'prune' (restrict the class schedule).
    level2: optional list of requirements h (Formula or text) with target |= h.
    environment: an environments.Environment; queries are then feasible traces.
    sample_size, length_range: the initial random traces (or rollouts);
        initial_traces replaces them by a given list.
    max_comparisons: the comparison budget. max_states, max_steps: limits of one
        Distinguish search (max_steps is the environment horizon).
    certify_tests: number of Certify tests (0: no certification).
    evaluate: compare the final hypothesis with the target (never used for learning).
    """
    begin = time.perf_counter()
    target = parse(target) if isinstance(target, str) else target
    atoms = tuple(atoms)
    alphabet = valuation_alphabet(atoms)
    rng = random.Random(seed)
    teacher = Teacher(target, atoms)
    if rule not in ("exhaustive", "top_k"):
        raise ValueError("rule must be 'exhaustive' or 'top_k'")

    # Initialize, pool of short traces, and Distinguish.
    if environment is not None:
        env = environment.project(atoms, seed=seed)
        length_range = length_range or (1, 10)
        initial = initial_traces or env.rollouts(sample_size, length_range)
        pool = EnvironmentPool(env, max_steps, max_states)
        search = EnvironmentSearch(env, Progressor(), max_steps, max_states)
        if distinguish != "progression":
            raise ValueError("environments use distinguish='progression'")
    else:
        env = None
        length_range = length_range or (1, 6)
        initial = initial_traces or [random_trace(rng, alphabet, length_range) for _ in range(sample_size)]
        pool = ShortTracePool(alphabet, max_product_states=max_states)
        if distinguish == "progression":
            search = Progression(alphabet, max_states)
        elif distinguish == "automata":
            search = Automata(atoms, alphabet, max_states)
        else:
            raise ValueError("distinguish must be 'progression' or 'automata'")
    distinguisher = PoolFirst(pool, search)

    # Hints.
    state = LearnerState()
    ask_hint, levels = None, None
    given = teacher_level1(target) if level1 == "teacher" else list(level1 or [])
    requirements = [parse(h) if isinstance(h, str) else h for h in (level2 or [])]
    if level1_mode == "prune" or requirements:
        classes = classes_of(given) if level1_mode == "prune" else ()
        state.guidance = HintGuidance(classes)
        levels = search_levels(classes) if level1_mode == "prune" else [("full", None, None)]
        if requirements:
            levels = [("level2", None, None, level2_filter(pool, requirements))] + levels
    elif level1 == "teacher":
        ask_hint = _teacher_ranking_hint(target, given)
    elif given:
        state.guidance = HintGuidance(classes_of(given))

    # VersionSpace and NextRival.
    if version_space == "sat":
        if rule != "exhaustive" or order != "enumeration" or env is not None or level1 or level2:
            raise ValueError("the SAT version space supports the exhaustive rule, enumeration "
                             "order, free traces and no hints")
        from .version_space_sat import SatVersionSpace, SolverBudget
        space = SatVersionSpace(atoms, max_size, max_depth,
                                SolverBudget(sat_max_calls, sat_time_limit, begin))
    elif version_space == "enumeration":
        space = EnumeratedVersionSpace(atoms, max_size, max_depth, levels)
    else:
        raise ValueError("version_space must be 'enumeration' or 'sat'")
    probe = None
    if env is not None and order in ("semantic", "semantic_far"):
        probe = env.fork((seed or 0) * 1000003 + 29).rollouts(200, length_range)
    rival_order = RivalOrder(order, alphabet, length_range, seed, pool, probe_traces=probe)

    state = active_learn(teacher, initial, space, rival_order, distinguisher,
                         k=k if rule == "top_k" else None, ask_hint=ask_hint,
                         max_comparisons=max_comparisons, state=state)
    if version_space == "sat":
        space.close()

    # Certify.
    certification = None
    if certify_tests and state.current is not None:
        cert_rng = random.Random(f"certify-{seed}")
        sampler = ((lambda: environment_trace(cert_rng, env)) if env is not None
                   else (lambda: free_trace(cert_rng, alphabet)))
        certification = certify(state.current, teacher, sampler, certify_tests)

    result = Result(
        hypothesis=state.current,
        status=("no_consistent_formula" if state.current is None
                else "budget_exhausted" if state.budget_exhausted else "completed"),
        queries=teacher.num_queries, initial_queries=state.initial_queries,
        comparisons=state.comparisons, equivalence_checks=state.equivalence_checks,
        candidates=space.built, seconds=time.perf_counter() - begin,
        levels=list(getattr(space, "levels_searched", [])),
        hint_classes=state.guidance.classes, certification=certification,
        target_text=str(target))
    if evaluate and state.current is not None:
        if env is not None:
            check = EnvironmentSearch(env, Progressor(), max_steps, 10 * max_states)
            result.witness = check(state.current, target, unbounded=True)
            result.equivalent = None if result.witness == UNKNOWN else result.witness is None
        else:
            result.witness = distinguish_by_progression(state.current, target, alphabet,
                                                        max_product_states=10 * max_states)
            result.equivalent = result.witness is None
    return result


def _teacher_ranking_hint(target, answer):
    """AskHint for ranking: the simulated teacher answers once, when the hypothesis
    is not in every class the target is in (so the hint has something to correct)."""
    target_classes = set(matching_classes(target))

    def ask(state):
        if state.guidance.classes or target_classes <= set(matching_classes(state.current)):
            return
        state.guidance = HintGuidance(classes_of(answer))
    return ask
