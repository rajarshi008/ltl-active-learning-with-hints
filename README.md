# ltlf-learner

Active learning of LTLf formulas through distinguishing queries.

A teacher knows a target formula and answers two kinds of questions:

- **membership queries**: is this trace acceptable?
- **natural-language queries** (optional): what kind of problem is there (Level 1), or which requirement holds (Level 2)?

The learner keeps every formula of a bounded class that fits the answers so far (the *version space*). It compares its current hypothesis with other formulas (*rivals*) and asks the teacher about a trace on which they disagree. When no remaining formula can be told apart from the hypothesis, it stops.

The paper, with all appendices, is in [`paper/active-learning-ltlf-with-hints.pdf`](paper/active-learning-ltlf-with-hints.pdf).

## Install

```bash
pip install -e .                 # the core learner needs nothing else
pip install -e ".[all]"          # optional: Z3 (SAT), ltlf2dfa (MONA), Gymnasium and MiniGrid
```

The core needs Python 3.10 or newer. The automata-based comparison also needs the [`mona`](https://www.brics.dk/mona/) binary on your PATH.

## Use

```python
from ltlf_learner import learn

result = learn("G(->(a,F(b)))", atoms=("a", "b"), max_size=6)
print(result.hypothesis, result.queries, result.equivalent)
# G(->(a,F(b))) 15 True
```

Formulas use prefix syntax: `F(a)`, `G(->(a,F(b)))`, `U(!(a),b)`, `&(F(a),G(!(b)))`.

The main options of `learn`:

| Option | Values | Meaning |
|---|---|---|
| `max_size`, `max_depth` | integers | the searched class: formulas with at most this many nodes and this temporal depth |
| `rule` | `"exhaustive"` (default), `"top_k"` | compare with all rivals, or only the first `k` (default 3) |
| `order` | `"enumeration"`, `"semantic"`, `"semantic_far"`, `"split"` | which rivals are compared first |
| `distinguish` | `"progression"` (default), `"automata"` | how a distinguishing trace is found |
| `version_space` | `"enumeration"` (default), `"sat"` | list all formulas, or ask a SAT solver for them |
| `level1` | `None`, `"teacher"`, list of question types | Level-1 hints |
| `level1_mode` | `"rank"`, `"prune"` | hints reorder the rivals, or restrict the search |
| `level2` | list of formulas | Level-2 hints: requirements the target implies |
| `environment` | an `Environment` | ask only about traces the environment can produce |
| `certify_tests` | integer | test the final hypothesis on this many fresh traces |
| `seed` | integer | random seed; seeded runs are reproducible |

More examples are in [`examples/quickstart.py`](examples/quickstart.py).

## Code layout

One file per procedure of the paper's algorithm:

```text
ltlf_learner/
  learner.py            Algorithm 1 and learn()
  version_space.py      VersionSpace by enumeration (with the hint levels)
  version_space_sat.py  VersionSpace by a SAT solver (Z3)
  rivals.py             NextRival: the order of rivals
  distinguish.py        Distinguish: short-trace pool, then a complete search
  progression.py        LTLf formula progression
  automata.py           LTLf to DFA with MONA (optional)
  hints.py              Level-1 and Level-2 hints
  certify.py            Certify: PAC test of the final hypothesis
  environments.py       Office world, Taxi, MiniGrid DoorKey
  teacher.py            the simulated teacher
  formula.py            formulas, parsing, printing
  enumerate.py          bottom-up formula enumeration
  traces.py             traces and fast evaluation
experiments/            the paper's experiments
tests/                  unit tests
```

## Reproduce the paper's experiments

```bash
python -m experiments.run_paper                  # RQ1-RQ3: 4,100 runs, results/paper.json
python -m experiments.tables results/paper.json  # Tables III, IV and VI

python -m experiments.run_implementations        # Table V: progression, MONA and SAT
python -m experiments.tables results/implementations.json
```

Add `--quick` to either run for a small version that finishes in seconds.

## Tests

```bash
python -m unittest discover -s tests -t .
```

Tests that need Z3 or MONA are skipped when these are not installed.
