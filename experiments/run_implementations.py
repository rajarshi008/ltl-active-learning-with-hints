"""Compare three implementations on the free targets (Table V): time and peak memory per run.

    python -m experiments.run_implementations              # 20 targets x 5 seeds x 3 implementations
    python -m experiments.run_implementations --quick      # targets up to size 5, seed 41
    python -m experiments.run_implementations --scale      # size-7 targets with n = 9 and 10 (slow)

    progression  enumerated version space, progression-based Distinguish (the default)
    automata     enumerated version space, MONA-based Distinguish (needs ltlf2dfa + mona)
    sat          SAT-based version space (Z3), progression-based Distinguish

All runs use the exhaustive rule and enumeration order. Each run has its own
process, so peak memory is per run (this adds about 0.7 s of start-up time).
"""
import argparse
import json
import multiprocessing as mp
import os
import resource
import sys
from pathlib import Path

from experiments.benchmarks import ABC, FREE, SEEDS

IMPLEMENTATIONS = {
    "progression": {},
    "automata": {"distinguish": "automata"},
    "sat": {"version_space": "sat"},
}


def run_one(task):
    from ltlf_learner import learn, parse
    name, target, max_size, depth, seed, time_limit = task
    r = learn(target, ABC, max_size, depth, seed=seed, sat_time_limit=time_limit,
              **IMPLEMENTATIONS[name])
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / 2**20 if sys.platform == "darwin" else peak / 2**10   # bytes on macOS, KiB on Linux
    return {"implementation": name, "target": target, "size": parse(target).size,
            "max_size": max_size, "seed": seed, "hypothesis": str(r.hypothesis),
            "equivalent": r.equivalent, "exact": r.exact, "queries": r.queries,
            "candidates": r.candidates, "status": r.status, "seconds": r.seconds,
            "peak_mb": peak_mb}


def main():
    parser = argparse.ArgumentParser(description="Compare implementations (Table V).")
    parser.add_argument("--scale", action="store_true", help="size-7 targets with n = 9, 10")
    parser.add_argument("--quick", action="store_true", help="targets up to size 5, seed 41")
    parser.add_argument("--implementations", nargs="*", default=list(IMPLEMENTATIONS))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--time-limit", type=float, default=600.0, help="SAT time limit (s)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.scale:
        size7 = [(t, d) for t, n, d in FREE if n == 8]
        # SAT: all four targets, seeds 41-42, n = 9 and 10. Enumeration needs
        # about 3.5 GB at n = 9, so it runs once (first target, seed 41).
        work = [("sat", t, n, d, seed, args.time_limit) for n in (9, 10) for t, d in size7
                for seed in SEEDS[:2] if "sat" in args.implementations]
        work += [(name, size7[0][0], 9, size7[0][1], SEEDS[0], args.time_limit)
                 for name in args.implementations if name != "sat"]
        out = args.out or "results/implementations_scale.json"
    else:
        targets = [b for b in FREE if not args.quick or b[1] <= 6]
        seeds = SEEDS[:1] if args.quick else SEEDS
        work = [(name, t, n, d, seed, args.time_limit) for name in args.implementations
                for t, n, d in targets for seed in seeds]
        out = args.out or "results/implementations.json"
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        rows = list(pool.imap_unordered(run_one, work))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rows, indent=1))
    print(f"wrote {len(rows)} runs to {out}")


if __name__ == "__main__":
    main()
