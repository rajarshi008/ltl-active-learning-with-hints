"""Run the experiments of RQ1-RQ3 and save one JSON row per run.

    python -m experiments.run_paper                    # all 4,100 runs
    python -m experiments.run_paper --quick            # targets up to size 5, seed 41 only
    python -m experiments.run_paper --only orders      # RQ1/RQ2 only ('orders') or RQ3 only ('hints')

Configurations:
    orders  4 rival orders x 2 stopping rules, no hints        (Tables III and IV)
    hints   8 hint settings x 2 stopping rules, enumeration    (Table VI)
Then print the tables with `python -m experiments.tables results/paper.json`.
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

from experiments.benchmarks import ABC, ENVIRONMENT, FREE, HINT_SETTINGS, SEEDS, environment, hint_options

ORDERS = ("enumeration", "semantic", "semantic_far", "split")
RULES = ("top_k", "exhaustive")


def tasks(only=None, quick=False):
    """All runs as (group, setting, rule, env, atoms, target, max_size, max_depth, seed)."""
    benches = [(None, ABC, *t) for t in FREE] + list(ENVIRONMENT)
    seeds = SEEDS[:1] if quick else SEEDS
    settings = []
    if only in (None, "orders"):
        settings += [("orders", o) for o in ORDERS]
    if only in (None, "hints"):
        settings += [("hints", h) for h in HINT_SETTINGS]
    out = []
    for env, atoms, target, max_size, depth in benches:
        if quick and max_size > 6:
            continue
        for seed in seeds:
            for group, setting in settings:
                for rule in RULES:
                    out.append((group, setting, rule, env, atoms, target, max_size, depth, seed))
    out.sort(key=lambda t: -t[6])   # slowest first
    return out


def run_one(task):
    """One learning run; returns a JSON row, or None when the target gets no such hint."""
    from ltlf_learner import learn
    group, setting, rule, env, atoms, target, max_size, depth, seed = task
    if group == "orders":
        options = {"order": setting}
    else:
        options = hint_options(setting, target, atoms, env)
        if options is None:
            return None
    if env is not None:
        options["environment"] = environment(env)
    r = learn(target, atoms, max_size, depth, rule=rule, seed=seed, **options)
    return {"group": group, "setting": setting, "rule": rule, "bench": "env" if env else "free",
            "env": env, "target": target, "size": max_size - 1, "seed": seed,
            "hypothesis": str(r.hypothesis), "equivalent": r.equivalent, "exact": r.exact,
            "queries": r.queries, "comparisons": r.comparisons,
            "equivalence_checks": r.equivalence_checks, "candidates": r.candidates,
            "seconds": r.seconds, "status": r.status, "levels": r.levels}


def main():
    parser = argparse.ArgumentParser(description="Run the paper's experiments (RQ1-RQ3).")
    parser.add_argument("--only", choices=("orders", "hints"))
    parser.add_argument("--quick", action="store_true", help="small targets, one seed")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--out", default="results/paper.json")
    args = parser.parse_args()
    work = tasks(args.only, args.quick)
    rows, start = [], time.perf_counter()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(run_one, work), 1):
            if row is not None:
                rows.append(row)
            if i % 200 == 0 or i == len(work):
                print(f"{i}/{len(work)} runs, {time.perf_counter() - start:.0f}s", flush=True)
    rows.sort(key=lambda r: (r["group"], r["setting"], r["rule"], r["bench"], r["target"], r["seed"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=1))
    print(f"wrote {len(rows)} runs to {args.out}")


if __name__ == "__main__":
    main()
