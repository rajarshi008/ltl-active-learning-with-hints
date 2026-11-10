"""Environments: queries restricted to feasible traces (Section VII, Appendix A).

An environment is a finite deterministic labelled transition system with a
fixed start state. A feasible trace is the start state's label followed by the
label after each action, so it is never empty. Labels are valuations (sorted
tuples of the true atoms). Two formulas are equivalent relative to the
environment when they agree on every feasible trace.

Benchmarks (as in the paper):
    office_world()                     Office world of Toro Icarte et al.
    taxi(seed)                         Gymnasium Taxi-v4, deterministic
    minigrid("MiniGrid-DoorKey-5x5-v0", seed)
Taxi and MiniGrid need gymnasium (and minigrid); ending states are made absorbing.
"""
import copy
import random
from collections import deque

from .distinguish import UNKNOWN
from .traces import PackedSample


class Environment:
    """A deterministic labelled transition system with random rollouts.

    start: start state. actions: tuple of action names.
    table: state -> tuple of successor states, one per action.
    labels: state -> set of true propositions.
    atoms: if given, labels keep only these atoms (`project` makes such a copy).
    """

    def __init__(self, start, actions, table, labels, atoms=None, seed=None, name=None):
        self.start, self.actions, self.table, self.name = start, tuple(actions), table, name
        self._index = {a: i for i, a in enumerate(self.actions)}
        known = sorted({p for value in labels.values() for p in value})
        keep = set(known if atoms is None else atoms)
        if keep - set(known):
            raise ValueError(f"unknown atoms {sorted(keep - set(known))}; choose from {known}")
        self.raw_labels = labels
        self.atoms = tuple(sorted(keep))
        self._labels = {s: tuple(sorted(set(v) & keep)) for s, v in labels.items()}
        self.rng = random.Random(seed)

    @property
    def num_states(self):
        return len(self.table)

    def project(self, atoms, seed=None):
        """A copy whose labels show only `atoms` (shares the transition table)."""
        return Environment(self.start, self.actions, self.table, self.raw_labels, atoms, seed, self.name)

    def fork(self, seed=None):
        """A copy with its own random generator."""
        other = copy.copy(self)
        other.rng = random.Random(seed)
        return other

    def label(self, state):
        return self._labels.get(state, ())

    def transition(self, state, action):
        return self.table[state][self._index[action]]

    def rollouts(self, count, length_range):
        """`count` rollouts with uniformly random actions; lengths (in labels) in `length_range`."""
        low, high = length_range
        traces = []
        for _ in range(count):
            state, trace = self.start, [self.label(self.start)]
            for _ in range(self.rng.randint(low - 1, high - 1)):
                state = self.transition(state, self.rng.choice(self.actions))
                trace.append(self.label(state))
            traces.append(tuple(trace))
        return traces

    def is_feasible(self, trace):
        """Whether some action sequence produces `trace`."""
        if not trace or trace[0] != self.label(self.start):
            return False
        states = {self.start}
        for valuation in trace[1:]:
            states = {self.transition(s, a) for s in states for a in self.actions
                      if self.label(self.transition(s, a)) == valuation}
            if not states:
                return False
        return True


# ---------------------------------------------------------------------------
# Distinguish inside an environment
# ---------------------------------------------------------------------------

class EnvironmentSearch:
    """Shortest feasible trace on which two formulas disagree (a distinguisher).

    Breadth-first over (environment state, progressed formula, progressed
    formula), trying actions in order. Returns UNKNOWN when it stops at the
    horizon `max_steps` (actions) or after `max_states` triples.
    """

    def __init__(self, env, progressor, max_steps=30, max_states=100000):
        self.env, self.p, self.max_steps, self.max_states = env, progressor, max_steps, max_states

    def __call__(self, first, second, unbounded=False):
        env, p = self.env, self.p
        horizon = None if unbounded else self.max_steps
        start_label = env.label(env.start)
        root = (env.start, p.prog(p.translate(first), start_label),
                p.prog(p.translate(second), start_label))
        parents, queue, cut = {root: None}, deque([(root, 0)]), False
        while queue:
            node, depth = queue.popleft()
            state, s1, s2 = node
            if p.empty[s1] != p.empty[s2]:
                labels = []
                while node is not None:
                    labels.append(env.label(node[0]))
                    node = parents[node]
                return tuple(reversed(labels))
            if s1 == s2:
                continue
            for action in env.actions:
                successor = env.transition(state, action)
                label = env.label(successor)
                nxt = (successor, p.prog(s1, label), p.prog(s2, label))
                if nxt in parents:
                    continue
                if horizon is not None and depth >= horizon:
                    cut = True
                    continue
                if len(parents) >= self.max_states:
                    return UNKNOWN
                parents[nxt] = node
                queue.append((nxt, depth + 1))
        return UNKNOWN if cut else None


class EnvironmentPool:
    """Short feasible traces in breadth-first order; same interface as distinguish.ShortTracePool.

    Holds the label traces of all action sequences of up to `depth` actions, with
    depth the largest value giving at most `max_traces` (state, trace) pairs and
    not exceeding the search horizon.
    """

    def __init__(self, env, max_steps=30, max_states=100000, max_traces=5000):
        start = (env.start, (env.label(env.start),))
        levels, seen, totals = [[start]], {start}, [1]
        while totals[-1] <= max_traces and len(levels) <= max_steps + 1:
            layer = []
            for state, trace in levels[-1]:
                for action in env.actions:
                    successor = env.transition(state, action)
                    node = (successor, trace + (env.label(successor),))
                    if node not in seen:
                        seen.add(node)
                        layer.append(node)
            if not layer:
                break
            levels.append(layer)
            totals.append(totals[-1] + len(layer))
        depth = 0
        while (depth + 1 < len(levels) and totals[depth + 1] <= max_traces
               and depth + 1 <= max_steps
               and (depth + 2 >= len(levels) or totals[depth + 2] < max_states)):
            depth += 1
        self.depth = depth
        traces, known = [], set()
        for layer in levels[:depth + 1]:
            for _, trace in layer:
                if trace not in known:
                    known.add(trace)
                    traces.append(trace)
        self.trace_at, offset = {}, 0
        for trace in traces:
            self.trace_at[offset + len(trace)] = trace
            offset += len(trace) + 1
        self.sample, self.memo = PackedSample([(t, False) for t in traces]), {}

    def find(self, first, second):
        s = self.sample
        differ = (s.value(first, self.memo) ^ s.value(second, self.memo)) & s.starts
        return self.trace_at[(differ & -differ).bit_length() - 1] if differ else None


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def _explore(start, actions, step):
    """Breadth-first closure of step(state, action) from `start`, as a transition table."""
    table, queue, seen = {}, deque([start]), {start}
    while queue:
        state = queue.popleft()
        table[state] = tuple(step(state, a) for a in actions)
        for successor in table[state]:
            if successor not in seen:
                seen.add(successor)
                queue.append(successor)
    return table


OFFICE_OBJECTS = {
    (1, 1): "a", (1, 7): "b", (10, 7): "c", (10, 1): "d",
    (7, 4): "e",                                         # mail
    (8, 2): "f", (3, 6): "f",                            # coffee
    (4, 4): "g",                                         # office
    (4, 1): "n", (7, 1): "n", (4, 7): "n", (7, 7): "n", (1, 4): "n", (10, 4): "n",  # plants
}


def office_world(atoms=None, seed=None):
    """The Office world: 12x9 cells in 3x3 rooms, start (2, 1); moves into walls stay put.

    Atoms: a, b, c, d (corners), e (mail), f (coffee), g (office), n (plants).
    """
    blocked = set()
    for x in range(12):
        for y in (0, 3, 6):
            blocked |= {(x, y, "down"), (x, y + 2, "up")}
    for y in range(9):
        for x in (0, 3, 6, 9):
            blocked |= {(x, y, "left"), (x + 2, y, "right")}
    for y in (1, 7):
        for x in (2, 5, 8):
            blocked -= {(x, y, "right"), (x + 1, y, "left")}
    for x in (1, 4, 7, 10):
        blocked -= {(x, 5, "up"), (x, 6, "down")}
    for x in (1, 10):
        blocked -= {(x, 2, "up"), (x, 3, "down")}
    moves = {"up": (0, 1), "right": (1, 0), "down": (0, -1), "left": (-1, 0)}

    def step(state, action):
        if (*state, action) in blocked:
            return state
        return state[0] + moves[action][0], state[1] + moves[action][1]

    actions = ("up", "right", "down", "left")
    table = _explore((2, 1), actions, step)
    labels = {cell: {p} for cell, p in OFFICE_OBJECTS.items()}
    return Environment((2, 1), actions, table, labels, atoms, seed, "office")


TAXI_ACTIONS = ("south", "north", "east", "west", "pickup", "dropoff")


def taxi(reset_seed=0, atoms=None, seed=None):
    """Gymnasium Taxi-v4 without rain or fickle passenger; reset_seed fixes the layout.

    Atoms: p (passenger in the taxi), g (passenger delivered). Delivery is absorbing.
    """
    import gymnasium as gym

    env = gym.make("Taxi-v4", is_rainy=False, fickle_passenger=False)
    start, _ = env.reset(seed=reset_seed)
    model = env.unwrapped

    def step(state, action):
        _, _, passenger, destination = model.decode(state)
        if passenger == destination:
            return state
        (_, successor, _, _), = model.P[state][TAXI_ACTIONS.index(action)]
        return int(successor)

    table = _explore(int(start), TAXI_ACTIONS, step)
    labels = {}
    for state in table:
        _, _, passenger, destination = model.decode(state)
        labels[state] = {"p"} if passenger == 4 else {"g"} if passenger == destination else set()
    env.close()
    return Environment(int(start), TAXI_ACTIONS, table, labels, atoms, seed, "taxi")


MINIGRID_ACTIONS = ("left", "right", "forward", "pickup", "toggle")


def minigrid(env_id="MiniGrid-DoorKey-5x5-v0", reset_seed=0, atoms=None, seed=None, max_states=50000):
    """A MiniGrid layout as a finite environment; reaching the goal is absorbing.

    Atoms: k (carrying a key), o (a door is open), g (on the goal).
    """
    import gymnasium as gym
    import minigrid as _  # noqa: F401  (registers the environments)

    env = gym.make(env_id)
    env.reset(seed=reset_seed)
    base = env.unwrapped
    action_ids = {name: getattr(base.actions, name) for name in MINIGRID_ACTIONS}

    def key(e):
        carrying = None if e.carrying is None else (e.carrying.type, e.carrying.color)
        return (tuple(int(v) for v in e.agent_pos), int(e.agent_dir), carrying, e.grid.encode().tobytes())

    def props(e):
        out = set()
        if e.carrying is not None and e.carrying.type == "key":
            out.add("k")
        if any(c is not None and c.type == "door" and c.is_open for c in e.grid.grid):
            out.add("o")
        here = e.grid.get(*e.agent_pos)
        if here is not None and here.type == "goal":
            out.add("g")
        return out

    start = key(base)
    copies, labels, table, queue = {start: copy.deepcopy(base)}, {}, {}, deque([start])
    while queue:
        state = queue.popleft()
        current = copies[state]
        labels[state] = props(current)
        successors = []
        for name in MINIGRID_ACTIONS:
            if "g" in labels[state]:
                successors.append(state)
                continue
            nxt = copy.deepcopy(current)
            nxt.step_count = 0
            nxt.step(action_ids[name])
            nkey = key(nxt)
            if nkey not in copies:
                if len(copies) >= max_states:
                    raise RuntimeError(f"{env_id} has more than {max_states} states")
                copies[nkey] = nxt
                queue.append(nkey)
            successors.append(nkey)
        table[state] = tuple(successors)
    env.close()
    return Environment(start, MINIGRID_ACTIONS, table, labels, atoms, seed, env_id)
