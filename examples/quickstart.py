"""A few small runs of the learner. Run from the repository root:

    python examples/quickstart.py
"""
from ltlf_learner import learn
from ltlf_learner.environments import office_world

# 1. Free traces: the teacher labels any trace over the atoms a, b.
result = learn("G(->(a,F(b)))", atoms=("a", "b"), max_size=6)
print("free traces:      ", result.summary())

# 2. Natural-language hints: the teacher also answers a Level-1 question, and
#    the hinted pattern classes restrict the search.
result = learn("G(->(a,F(b)))", atoms=("a", "b"), max_size=6,
               level1="teacher", level1_mode="prune")
print("with Level-1 hint:", result.summary())

# 3. A Level-2 hint: a requirement the target implies.
result = learn("G(->(a,X(b)))", atoms=("a", "b"), max_size=6, level2=["G(->(a,F(b)))"])
print("with Level-2 hint:", result.summary())

# 4. An environment: queries are feasible traces of the Office world.
result = learn("F(&(f,F(g)))", atoms=("f", "g", "n"), max_size=6, environment=office_world())
print("Office world:     ", result.summary())

# 5. Certify the final hypothesis with 50 fresh random traces.
result = learn("F(&(a,F(b)))", atoms=("a", "b"), max_size=6, certify_tests=50)
print("certified:        ", result.certification)
