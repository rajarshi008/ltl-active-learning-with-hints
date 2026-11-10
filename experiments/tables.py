"""Print the paper's tables from saved runs.

    python -m experiments.tables results/paper.json             # Tables III, IV and VI
    python -m experiments.tables results/implementations.json   # Table V
"""
import json
import sys
from collections import defaultdict
from statistics import fmean


def mean(values):
    values = [v for v in values if v is not None]
    return fmean(values) if values else float("nan")


def groups(rows, *keys):
    out = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def identification(rows):
    """Table III: success, exact and queries by order of rivals and stopping rule."""
    g = groups([r for r in rows if r["group"] == "orders"], "setting", "rule", "bench")
    print("Table III: identification and cost (no hints)")
    print(f"{'order':14s} {'bench':5s} | {'top-3: succ exact MQ':22s} | {'exhaustive: succ exact MQ Eq':30s}")
    for order in ("enumeration", "semantic", "semantic_far", "split"):
        for bench in ("free", "env"):
            cells = []
            for rule in ("top_k", "exhaustive"):
                runs = g.get((order, rule, bench), [])
                succ = sum(r["equivalent"] is True for r in runs)
                text = (f"{succ:3d}/{len(runs):<3d} {sum(r['exact'] for r in runs):3d} "
                        f"{mean([r['queries'] for r in runs]):5.1f}")
                if rule == "exhaustive":
                    text += f" {mean([r['equivalence_checks'] for r in runs]):6.1f}"
                cells.append(text)
            print(f"{order:14s} {bench:5s} | {cells[0]:22s} | {cells[1]}")
    print()


def cost_by_size(rows):
    """Table IV: cost by target size (exhaustive rule, enumeration order)."""
    runs = [r for r in rows if r["group"] == "orders" and r["setting"] == "enumeration"
            and r["rule"] == "exhaustive"]
    g = groups(runs, "bench", "size")
    print("Table IV: cost by target size (exhaustive, enumeration order)")
    print(f"{'bench':5s} {'size':>4s} {'N':>3s} {'MQ':>6s} {'min-max':>8s} {'time':>6s} {'max':>6s} {'cand(k)':>9s}")
    for (bench, size), rs in sorted(g.items()):
        q = [r["queries"] for r in rs]
        t = [r["seconds"] for r in rs]
        print(f"{bench:5s} {size:4d} {len(rs):3d} {mean(q):6.1f} {min(q):3d}-{max(q):<4d} "
              f"{mean(t):6.2f} {max(t):6.2f} {mean([r['candidates'] for r in rs]) / 1000:9.1f}")
    print()


def hints(rows):
    """Table VI: effect of hints (enumeration order)."""
    g = groups([r for r in rows if r["group"] == "hints"], "setting", "rule", "bench")
    print("Table VI: effect of hints")
    print(f"{'setting':16s} {'bench':5s} | {'succ top-3':>10s} {'succ exh':>9s} | exhaustive: MQ  cand(k)  time")
    for setting in ("none", "l1_rank", "l1_prune", "l2", "l1_prune_l2",
                    "wrong_l1_rank", "wrong_l1_prune", "wrong_l2"):
        for bench in ("free", "env"):
            top, exh = g.get((setting, "top_k", bench), []), g.get((setting, "exhaustive", bench), [])
            if not exh:
                continue
            succ = lambda rs: f"{sum(r['equivalent'] is True for r in rs)}/{len(rs)}"
            print(f"{setting:16s} {bench:5s} | {succ(top):>10s} {succ(exh):>9s} | "
                  f"{mean([r['queries'] for r in exh]):13.1f} {mean([r['candidates'] for r in exh]) / 1000:8.0f} "
                  f"{mean([r['seconds'] for r in exh]):6.2f}")
    print("(Level-2 rows cover only the targets that receive a Level-2 hint.)\n")


def implementations(rows):
    """Table V: time and peak memory by implementation and target size."""
    g = groups(rows, "implementation", "max_size", "size")
    names = ("progression", "automata", "sat")
    keys = sorted({(r["size"], r["max_size"]) for r in rows})
    print("Table V: implementations (exhaustive rule, enumeration order, free targets)")
    print(f"{'size':>4s} {'n':>3s} " + " ".join(f"{n + ' time  MB':>20s}" for n in names))
    for size, n in keys:
        cells = []
        for name in names:
            rs = g.get((name, n, size), [])
            done = [r for r in rs if r["status"] == "completed"]
            if not rs:
                cells.append(f"{'--':>20s}")
            elif len(done) < len(rs):
                cells.append(f"{f'{len(done)}/{len(rs)} done':>13s} {mean([r['peak_mb'] for r in rs]):6.0f}")
            else:
                cells.append(f"{mean([r['seconds'] for r in rs]):13.2f} {mean([r['peak_mb'] for r in rs]):6.0f}")
        print(f"{size:4d} {n:3d} " + " ".join(cells))
    for name in names:
        rs = [r for r in rows if r["implementation"] == name and r["max_size"] == r["size"] + 1]
        if rs:
            print(f"{name}: exact {sum(r['exact'] for r in rs)}/{len(rs)}, "
                  f"equivalent {sum(r['equivalent'] is True for r in rs)}/{len(rs)}, "
                  f"MQ {mean([r['queries'] for r in rs]):.1f}")
    print()


def main():
    rows = json.loads(open(sys.argv[1] if len(sys.argv) > 1 else "results/paper.json").read())
    if rows and "implementation" in rows[0]:
        implementations(rows)
    else:
        identification(rows)
        cost_by_size(rows)
        hints(rows)


if __name__ == "__main__":
    main()
