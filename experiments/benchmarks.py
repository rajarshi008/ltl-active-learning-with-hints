"""The paper's benchmarks (Table II) and its hint settings (RQ3).

Each target is searched in L_{n,d} with n = target size + 1 and d = 2
(d = 3 for the two free targets of temporal depth 3).
"""
import warnings

from ltlf_learner import parse
from ltlf_learner.distinguish import ShortTracePool
from ltlf_learner.environments import EnvironmentPool, minigrid, office_world, taxi
from ltlf_learner.hints import simulated_level2, wrong_level1
from ltlf_learner.traces import valuation_alphabet

SEEDS = (41, 42, 43, 44, 45)
ABC = ("a", "b", "c")

# (target, max_size, max_depth) over atoms a, b, c; queries are arbitrary traces.
FREE = [
    ("F(a)", 3, 2), ("G(a)", 3, 2), ("F(G(a))", 4, 2), ("G(F(a))", 4, 2), ("U(a,b)", 4, 2),
    ("->(F(a),F(b))", 6, 2), ("G(->(a,F(b)))", 6, 2), ("G(->(a,X(b)))", 6, 2),
    ("F(&(a,F(b)))", 6, 2), ("U(a,U(b,c))", 6, 2), ("&(F(a),G(!(b)))", 6, 2),
    ("F(&(a,X(b)))", 6, 2), ("G(->(a,U(b,c)))", 7, 2), ("->(a,X(U(b,c)))", 7, 2),
    ("U(!(a),&(b,F(c)))", 7, 2), ("G(|(!(a),X(G(!(b)))))", 7, 3), ("G(->(&(a,b),X(c)))", 8, 2),
    ("&(G(F(a)),G(F(b)))", 8, 2), ("G(->(a,X(U(a,b))))", 8, 3), ("F(&(a,F(&(b,c))))", 8, 2),
]

# (environment, atoms, target, max_size, max_depth); queries are feasible traces.
FGN, EGN, GP, GKO = ("f", "g", "n"), ("e", "g", "n"), ("g", "p"), ("g", "k", "o")
ENVIRONMENT = [
    ("office", FGN, "G(!(n))", 3, 2), ("office", FGN, "U(!(n),f)", 4, 2),
    ("office", FGN, "F(&(f,F(g)))", 6, 2), ("office", FGN, "&(F(f),G(!(n)))", 6, 2),
    ("office", FGN, "U(!(n),&(f,F(g)))", 7, 2), ("office", EGN, "F(&(e,F(g)))", 6, 2),
    ("office", ABC, "F(&(a,F(b)))", 6, 2),
    ("taxi", GP, "F(g)", 3, 2), ("taxi", GP, "F(&(p,F(g)))", 6, 2), ("taxi", GP, "G(->(p,F(g)))", 6, 2),
    ("doorkey5", GKO, "F(g)", 3, 2), ("doorkey5", GKO, "U(!(o),k)", 4, 2),
    ("doorkey5", GKO, "F(&(o,F(g)))", 6, 2), ("doorkey5", GKO, "G(->(k,F(o)))", 6, 2),
    ("doorkey6", GKO, "F(g)", 3, 2), ("doorkey6", GKO, "F(&(k,F(g)))", 6, 2),
]

_ENVIRONMENTS = {}


def environment(name):
    """The benchmark environment called `name` (built once per process)."""
    if name not in _ENVIRONMENTS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ENVIRONMENTS[name] = {
                "office": office_world,
                "taxi": lambda: taxi(0),
                "doorkey5": lambda: minigrid("MiniGrid-DoorKey-5x5-v0", 0),
                "doorkey6": lambda: minigrid("MiniGrid-DoorKey-6x6-v0", 0),
            }[name]()
    return _ENVIRONMENTS[name]


# Hint settings of RQ3 (Table VI). All use the enumeration order of rivals.
HINT_SETTINGS = ("none", "l1_rank", "l1_prune", "l2", "l1_prune_l2",
                 "wrong_l1_rank", "wrong_l1_prune", "wrong_l2")


def hint_options(setting, target, atoms, env_name=None):
    """learn() options for a hint setting, or None when the target gets no such hint."""
    if setting == "none":
        return {}
    if setting == "l1_rank":
        return {"level1": "teacher"}
    if setting == "l1_prune":
        return {"level1": "teacher", "level1_mode": "prune"}
    if setting in ("wrong_l1_rank", "wrong_l1_prune"):
        wrong = wrong_level1(parse(target))
        if wrong is None:
            return None
        return {"level1": wrong, "level1_mode": "prune" if setting == "wrong_l1_prune" else "rank"}
    # Level-2 settings: a requirement chosen by the simulated teacher.
    alphabet = valuation_alphabet(atoms)
    pool = (ShortTracePool(alphabet) if env_name is None
            else EnvironmentPool(environment(env_name).project(atoms)))
    h = simulated_level2(parse(target), atoms, alphabet, pool.sample,
                         "wrong" if setting == "wrong_l2" else "correct")
    if h is None:
        return None
    options = {"level2": [h]}
    if setting == "l1_prune_l2":
        options.update(level1="teacher", level1_mode="prune")
    return options
