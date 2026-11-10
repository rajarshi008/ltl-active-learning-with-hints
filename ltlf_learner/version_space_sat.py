"""VersionSpace by a SAT solver (Section V-A, "SAT-based construction"), solved with Z3.

The formulas of size n that fit every label are the models of one incremental
solver per size. The encoding follows Neider and Gavran (FMCAD 2018): a formula
of size n has rows 1..n, each with one label, and operator rows choose their
children among earlier rows. Every row except the root has exactly one parent,
so formulas are syntax trees and n is their size; temporal depth is bounded by
extra variables. A formula has one encoding per order of its rows, so a
formula that was already handled is blocked again, encoding by encoding.

The version space is never listed: rivals are the solvers' models, smallest
size first, and removing an equivalent rival adds a blocking clause. This is
the version space of previous SAT-based learners, used here for comparison.
Needs z3-solver.
"""
import itertools
import time

import z3

from .formula import Formula


class VarPool:
    """Integer ids for named variables: id(key) gives 1, 2, ... in order of first use.

    Keys are any hashable values (tuples tagged by their encoding). The same key
    always gives the same id. Used by FormulaEncoding, SymbolicTrace and
    _fixed_formula to name CNF variables; SizeSolver owns one pool per solver.
    """

    def __init__(self):
        self._ids = {}

    def id(self, key):
        """The id of `key`, allocating the next free one on first use."""
        found = self._ids.get(key)
        if found is None:
            found = self._ids[key] = len(self._ids) + 1
        return found


class Z3Solver:
    """An incremental Z3 solver over integer CNF literals (v > 0: variable v true, -v: false).

    add_clause(literals), solve(assumptions) -> bool and get_model() -> the
    true variables of the last model, as positive ints. Variable v is the Z3
    Boolean constant "v<v>", created on first use. Clauses persist across solve()
    calls, so new labels and blocking clauses only add constraints; assumptions
    hold for one call only.

    Clauses are built and models are read with Z3's low-level API (Z3_mk_or,
    Z3_solver_assert, Z3_model_get_const_interp) on cached literal terms: the
    high-level wrappers re-check every argument and create one Python object per
    term, which made clause building and model reading take most of a run's time.
    """

    def __init__(self):
        # QF_FD selects Z3's incremental SAT engine. The default z3.Solver() also
        # answers correctly, but its time per call grew with the number of blocking
        # clauses (49 s against 7 s for the 26,752 models of size 5, two atoms).
        # Each solver gets its own Z3 context, so a run does not depend on
        # earlier runs in the same process (they would change Z3's model order).
        self._ctx = z3.Context()
        self.solver = z3.SolverFor("QF_FD", ctx=self._ctx)
        self._lits = {}
        self._var_of_decl = {}   # Z3 ast id of a variable's declaration -> its integer

    def _literal(self, lit):
        """The Z3 term of one integer literal (cached, together with its negation)."""
        term = self._lits.get(lit)
        if term is None:
            v = abs(lit)
            var = z3.Bool(f"v{v}", self._ctx)
            self._lits[v], self._lits[-v] = var, z3.Not(var, self._ctx)
            self._var_of_decl[var.decl().get_id()] = v
            term = self._lits[lit]
        return term

    def add_clause(self, literals):
        """Add the disjunction of `literals` (an empty clause makes the solver unsatisfiable)."""
        ref = self._ctx.ref()
        if len(literals) == 1:
            clause = self._literal(literals[0])
        elif not literals:
            clause = z3.BoolVal(False, self._ctx)
        else:
            args = (z3.Ast * len(literals))(*(self._literal(l).as_ast() for l in literals))
            # Wrapping the new term in a BoolRef holds a reference to it until the solver owns it.
            clause = z3.BoolRef(z3.Z3_mk_or(ref, len(literals), args), self._ctx)
        z3.Z3_solver_assert(ref, self.solver.solver, clause.as_ast())

    def solve(self, assumptions=()):
        """True if the clauses and the assumption literals are satisfiable together."""
        result = self.solver.check(*[self._literal(l) for l in assumptions])
        if result == z3.unknown:
            raise RuntimeError(f"Z3 returned unknown: {self.solver.reason_unknown()}")
        return result == z3.sat

    def get_model(self):
        """The variables true in the last model (variables Z3 leaves unassigned count as false)."""
        ref = self._ctx.ref()
        model = z3.Z3_solver_get_model(ref, self.solver.solver)
        z3.Z3_model_inc_ref(ref, model)
        try:
            true = []
            for i in range(z3.Z3_model_get_num_consts(ref, model)):
                decl = z3.Z3_model_get_const_decl(ref, model, i)
                value = z3.Z3_model_get_const_interp(ref, model, decl)
                if z3.Z3_get_bool_value(ref, value) == z3.Z3_L_TRUE:
                    true.append(self._var_of_decl[
                        z3.Z3_get_ast_id(ref, z3.Z3_func_decl_to_ast(ref, decl))])
            return true
        finally:
            z3.Z3_model_dec_ref(ref, model)

    def delete(self):
        """Release the solver."""
        self.solver = None
        self._lits = {}


UNARY = ("X", "F", "G")


BINARY = ("U", "&", "|", "->")


TEMPORAL = ("X", "F", "G", "U")


def _symbols(atoms):
    """Row labels as (kind, name, arity): atoms, negated atoms, then operators, in a fixed order.

    kind is 'ap' (atom), 'nap' (negated atom: one leaf of size 1) or 'op'. There
    is no general negation operator. FormulaEncoding.x[i, s] indexes this list.
    """
    leaves = [("ap", a, 0) for a in atoms] + [("nap", a, 0) for a in atoms]
    return leaves + [("op", o, 1) for o in UNARY] + [("op", o, 2) for o in BINARY]


class FormulaEncoding:
    """CNF for 'a syntax tree with exactly `size` nodes and temporal depth <= `depth`'.

    Variables (as in Neider and Gavran), rows i = 1..size, root = row `size`:
        x[i, s]   row i has label s (an index into _symbols), exactly one per row
        l[i, j]   the left (or only) child of row i is row j < i
        r[i, j]   the right child of row i is row j < i
        le[i, k]  row i has temporal depth <= k
    Differences from their encoding: every row except the root has exactly one
    parent (a tree, not a DAG); leaves are atoms and negated atoms, both of size
    1; finite-trace semantics with an end position (they use lasso traces); and
    the temporal depth is bounded through le. As in theirs, rows are not
    ordered, so a formula has one encoding per order of its rows.

    Variables live in `pool` (a VarPool) and clauses are passed to `emit`
    (a solver's add_clause). Labelled traces are added by `add_trace`.
    """

    def __init__(self, atoms, size, depth, pool, emit, tag):
        self.atoms, self.n, self.d = tuple(atoms), size, depth
        self.pool, self.emit, self.tag = pool, emit, tag
        self.symbols = _symbols(self.atoms)
        self.trace_count = 0
        v = lambda *k: pool.id((tag,) + k)
        n = size
        self.x = {(i, s): v("x", i, s) for i in range(1, n + 1) for s in range(len(self.symbols))}
        self.l = {(i, j): v("l", i, j) for i in range(2, n + 1) for j in range(1, i)}
        self.r = {(i, j): v("r", i, j) for i in range(2, n + 1) for j in range(1, i)}
        self.le = {(i, k): v("le", i, k) for i in range(1, n + 1) for k in range(0, depth + 1)}
        self._structure()

    def _at_most_one(self, lits):
        for a, b in itertools.combinations(lits, 2):
            self.emit([-a, -b])

    def _exactly_one(self, lits):
        self.emit(list(lits))
        self._at_most_one(lits)

    def _structure(self):
        """Clauses for: one label per row, children of the right arity, a tree, depth <= d."""
        n, d, x, l, r, emit = self.n, self.d, self.x, self.l, self.r, self.emit
        symbols = self.symbols
        ops = [s for s, (_, _, arity) in enumerate(symbols) if arity >= 1]
        binary = [s for s, (_, _, arity) in enumerate(symbols) if arity == 2]
        for i in range(1, n + 1):
            self._exactly_one([x[i, s] for s in range(len(symbols))])
        # Row 1 is a leaf (theirs: 'first operator a variable').
        for s in ops:
            emit([-x[1, s]])
        # Operators have exactly one left child, binary ones also exactly one
        # right child; leaves have none, unary operators no right child.
        for i in range(2, n + 1):
            lefts = [l[i, j] for j in range(1, i)]
            rights = [r[i, j] for j in range(1, i)]
            for s in ops:
                emit([-x[i, s]] + lefts)
            for s in binary:
                emit([-x[i, s]] + rights)
            self._at_most_one(lefts)
            self._at_most_one(rights)
            for lit in lefts:
                emit([-lit] + [x[i, s] for s in ops])
            for lit in rights:
                emit([-lit] + [x[i, s] for s in binary])
        # Tree (D1): every row except the root is the child of exactly one row.
        for j in range(1, n):
            self._exactly_one([l[i, j] for i in range(j + 1, n + 1)]
                              + [r[i, j] for i in range(j + 1, n + 1)])
        # Temporal depth (D5): le[i, k] means "depth of row i is at most k"; root <= d.
        # Bounds pass from a row to its chosen children (one less below X F G U).
        le = self.le
        emit([le[n, d]])
        for i in range(2, n + 1):
            for s in ops:
                _, o, arity = symbols[s]
                temporal = o in TEMPORAL
                if temporal:
                    emit([-x[i, s], -le[i, 0]])
                for k in range(0, d + 1):
                    kc = k - 1 if temporal else k
                    if kc < 0:
                        continue
                    for j in range(1, i):
                        emit([-x[i, s], -l[i, j], -le[i, k], le[j, kc]])
                        if arity == 2:
                            emit([-x[i, s], -r[i, j], -le[i, k], le[j, kc]])

    # -- semantics ---------------------------------------------------------------

    def _define(self, out, op, args, guard):
        """Clauses for guard -> (out <-> op(args)), op in and/or/imp/const."""
        g = [-l for l in guard]
        if op == "const":
            self.emit(g + ([out] if args else [-out]))
        elif op == "eq":
            a, = args
            self.emit(g + [-out, a]); self.emit(g + [out, -a])
        elif op == "and":
            for a in args:
                self.emit(g + [-out, a])
            self.emit(g + [out] + [-a for a in args])
        elif op == "or":
            self.emit(g + [-out] + list(args))
            for a in args:
                self.emit(g + [out, -a])
        elif op == "imp":
            a, b = args
            self.emit(g + [-out, -a, b]); self.emit(g + [out, a]); self.emit(g + [out, -b])

    def _aux_and(self, a, b, key):
        z = self.pool.id((self.tag,) + key)
        self._define(z, "and", [a, b], [])
        return z

    def _semantics(self, y, length, letter, alive):
        """Add clauses fixing y[(i, p)] for p = 0..length (position `length` is past the end).

        letter(p, atom) -> a literal or a bool (concrete traces); alive(p) -> a literal
        or a bool. A dead position takes the end-of-trace values: atoms, X, F, U
        false; negated atoms, G true. Operator rules are guarded by the row's
        label and its chosen children, one rule per choice (D7, as in Neider and
        Gavran); X F G U unfold one step (D4).
        """
        n, x, l, r = self.n, self.x, self.l, self.r
        for i in range(1, n + 1):
            for s, (kind, name, arity) in enumerate(self.symbols):
                if arity >= 1 and i == 1:  # row 1 is a leaf
                    continue
                xi = x[i, s]
                for p in range(0, length + 1):
                    out = y[i, p]
                    if kind in ("ap", "nap") or name in TEMPORAL:
                        dead_value = {"ap": False, "nap": True, "X": False, "F": False,
                                      "U": False, "G": True}[name if kind == "op" else kind]
                        al = alive(p)
                        if al is False:
                            self._define(out, "const", dead_value, [xi]); continue
                        if al is not True:
                            self._define(out, "const", dead_value, [xi, -al])
                        live = [xi] if al is True else [xi, al]
                    else:
                        live = [xi]
                    if kind in ("ap", "nap"):
                        lit = letter(p, name)
                        if isinstance(lit, bool):
                            self._define(out, "const", lit == (kind == "ap"), live)
                        else:
                            self._define(out, "eq", [lit if kind == "ap" else -lit], live)
                    elif name == "X":
                        nxt = alive(p + 1) if p + 1 <= length else False
                        if nxt is False:
                            self._define(out, "const", False, live)
                            continue
                        for j in range(1, i):
                            if nxt is True:
                                self._define(out, "eq", [y[j, p + 1]], live + [l[i, j]])
                            else:
                                self._define(out, "and", [nxt, y[j, p + 1]], live + [l[i, j]])
                    elif name in ("F", "G"):
                        op = "or" if name == "F" else "and"
                        for j in range(1, i):
                            self._define(out, op, [y[j, p], y[i, p + 1]], live + [l[i, j]])
                    elif name == "U":
                        # a U b at p: b at p, or a at p and (a U b) at p+1.
                        for j in range(1, i):
                            z = self._aux_and(y[j, p], y[i, p + 1],
                                              ("u", self.trace_count, i, j, p))
                            for k in range(1, i):
                                self._define(out, "or", [y[k, p], z],
                                             live + [l[i, j], r[i, k]])
                    else:  # & | -> at every position
                        op = {"&": "and", "|": "or", "->": "imp"}[name]
                        for j in range(1, i):
                            for k in range(1, i):
                                self._define(out, op, [y[j, p], y[k, p]],
                                             live + [l[i, j], r[i, k]])

    def add_trace(self, trace, label):
        """Constrain the formula to give `label` on the concrete normalized `trace`."""
        self.trace_count += 1
        m = len(trace)
        y = {(i, p): self.pool.id((self.tag, "y", self.trace_count, i, p))
             for i in range(1, self.n + 1) for p in range(0, m + 1)}
        self._semantics(y, m, lambda p, a: p < m and a in trace[p],
                        lambda p: p < m)
        self.emit([y[self.n, 0]] if label else [-y[self.n, 0]])


    def decode(self, model):
        """The Formula of a satisfying assignment, and its true x/l/r variables.

        The second value identifies this encoding (one order of the rows); pass
        it to blocking_clause. Other orders of the same formula stay possible.
        """
        true = {v for v in model if v > 0}
        chosen, label, left, right = [], {}, {}, {}
        for i in range(1, self.n + 1):
            s = next(s for s in range(len(self.symbols)) if self.x[i, s] in true)
            label[i] = self.symbols[s]
            chosen.append(self.x[i, s])
            arity = label[i][2]
            if arity >= 1:
                left[i] = next(j for j in range(1, i) if self.l[i, j] in true)
                chosen.append(self.l[i, left[i]])
            if arity == 2:
                right[i] = next(j for j in range(1, i) if self.r[i, j] in true)
                chosen.append(self.r[i, right[i]])

        def build(i):
            kind, name, arity = label[i]
            if kind == "ap":
                return Formula(name)
            if kind == "nap":
                return Formula("!" + name)
            if arity == 1:
                return Formula(name, build(left[i]))
            return Formula(name, build(left[i]), build(right[i]))

        return build(self.n), chosen

    def blocking_clause(self, chosen):
        """Excludes this encoding (blocks its x/l/r assignment, as Neider and Gavran do)."""
        return [-v for v in chosen]

class SizeSolver:
    """One incremental SAT solver for formulas of exactly one size (step 2 of a solver run).

    Wraps a FormulaEncoding (tree encoding with x/l/r rows).
    Used by version_space_sat.py. Propositional semantics.
    """

    def __init__(self, atoms, size, depth, tag):
        self.pool = VarPool()
        self.solver = Z3Solver()
        self.enc = FormulaEncoding(atoms, size, depth, self.pool, self.solver.add_clause,
                                   tag)

    def add_label(self, trace, label):
        self.enc.add_trace(trace, label)

    def next_formula(self, assumptions=()):
        """A consistent formula of this size and its encoding (for block), or None."""
        if not self.solver.solve(assumptions=list(assumptions)):
            return None
        return self.enc.decode(self.solver.get_model())

    def block(self, chosen):
        """Exclude one encoding returned by next_formula (other row orders stay)."""
        self.solver.add_clause(self.enc.blocking_clause(chosen))

    def delete(self):
        self.solver.delete()


class SolverBudget:
    """Counts solver calls and time; exhausted when `max_calls` or `time_limit` (seconds since `begin`) is reached.

    found: number of formulas decoded from models,
        not counting repeated encodings.
    duplicates: models that repeated a formula already handled and were skipped
        (SatVersionSpace.rivals); each also used one solver call.
    """

    def __init__(self, max_calls, time_limit, begin):
        self.max_calls, self.time_limit, self.begin = max_calls, time_limit, begin
        self.calls, self.seconds, self.found = 0, 0.0, 0
        self.duplicates = 0
        self.exhausted = False

    def out(self):
        """True (and remembered) when the budget is used up; checked before each solve."""
        if (self.calls >= self.max_calls
                or time.perf_counter() - self.begin >= self.time_limit):
            self.exhausted = True
        return self.exhausted

    def solve(self, fn):
        """Call fn() (one solver call) and account for it."""
        start = time.perf_counter()
        result = fn()
        self.calls += 1
        self.seconds += time.perf_counter() - start
        return result


class SatVersionSpace:
    """One persistent solver per size 1..max_size; rivals are its models.

    stages(): one stage. It adds the initial labels and selects a smallest
    consistent formula as the hypothesis (blocking its encoding).
    rivals(): the solvers' models, smallest size first; after a query the next
    round starts again at size 1. remove(): blocks the last model returned.
    update(): adds new labels to every solver. A model whose formula is the
    hypothesis or a removed rival (another encoding of it) is blocked and
    skipped, as Neider and Gavran skip repeated formulas. This is safe: such a
    formula is still consistent only if it is equivalent to the current
    hypothesis (equivalent formulas get the same labels).
    """

    def __init__(self, atoms, max_size, max_depth, budget):
        self.budget = budget
        self.solvers = [SizeSolver(atoms, n, max_depth, ("s", n))
                        for n in range(1, max_size + 1)]
        self.fed = 0
        self.resume = 0     # first round continues at the hypothesis's size
        self.last = None    # (solver, encoding) of the last rival returned
        self.seen = set()   # str of the hypotheses and removed rivals

    @property
    def built(self):
        """Formulas decoded from the solvers (repeated encodings not counted)."""
        return self.budget.found

    def stages(self, state, teacher):
        """One stage: add the labels, take the smallest consistent formula as hypothesis.

        The hypothesis's encoding is blocked and its formula added to `seen`.
        """
        self.update(state, teacher)
        for index, solver in enumerate(self.solvers):
            if self.budget.out():
                break
            found = self.budget.solve(solver.next_formula)
            if found is not None:
                formula, chosen = found
                self.budget.found += 1
                state.current = formula
                self.seen.add(str(formula))
                solver.block(chosen)
                self.resume = index
                break
        yield 1

    def update(self, state, teacher):
        for trace, label in list(teacher.labels.items())[self.fed:]:
            for solver in self.solvers:
                solver.add_label(trace, label)
        self.fed = teacher.num_queries

    def has_rivals(self, state):
        return True  # unknown without solving; a round without a model asks nothing

    def rivals(self, state):
        """Yield consistent formulas (solver models), smallest size first, skipping repeats.

        Used as the rival order of SAT runs. Each model costs one solver call; a model whose formula is in `seen` or is
        the hypothesis is blocked and counted in budget.duplicates instead.
        """
        start, self.resume = self.resume, 0
        for solver in self.solvers[start:]:
            while True:
                if self.budget.out():
                    return
                found = self.budget.solve(solver.next_formula)
                if found is None:
                    break
                formula, chosen = found
                key = str(formula)
                if key in self.seen or key == str(state.current):
                    solver.block(chosen)  # another encoding of a formula already handled
                    self.budget.duplicates += 1
                    continue
                self.budget.found += 1
                self.last = (solver, chosen)
                yield formula

    def remove(self, state, formula):
        """Block the last rival's encoding and remember the formula (its other encodings are skipped)."""
        solver, chosen = self.last
        solver.block(chosen)
        self.seen.add(str(formula))

    def exhausted(self):
        return self.budget.exhausted

    def close(self):
        for solver in self.solvers:
            solver.delete()


