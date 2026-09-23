#!/usr/bin/env python3
"""Recompute the statistics the MHH paper reports, from the 10-task table.

No data files, no GPU. The ten pairs of integers below ARE the surviving dataset:
the pod holding the raw per-episode records was terminated 2026-09-06.

Run:  python3 demo/paper_stats.py
Needs: scipy (for the exact Fisher test). Without scipy it still prints the
clustered result and says which part it had to skip.

Expected output is in the paper:
  clustered   -2.50 pp, 95% CI [-6.36, +1.36], t(9) = -1.464, p = 0.177
  episode-lvl  p = 0.057, CI [-5.06, +0.06]
  Holm        nothing survives correction
"""
import math

# task, T1 successes /20, T5 successes /20.  Eight tasks are 20/20 in both arms;
# the paper does not name them, so they are numbered here.
TABLE = [(f"task {i}", 20, 20) for i in range(1, 9)] + [
    ("KITCHEN_SCENE8",     19, 16),
    ("LIVING_ROOM_SCENE6", 20, 18),
]
N_EP = 20


def clustered():
    """Paired t-test over per-task success-rate differences. Unit = task."""
    d = [(t5 - t1) / N_EP * 100.0 for _, t1, t5 in TABLE]
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    se = math.sqrt(var / n)
    t = mean / se
    # t(9), two-sided 95% -> 2.262157
    crit = 2.262157
    return mean, se, t, n - 1, mean - crit * se, mean + crit * se


def episode_level():
    """Treating 200 episodes as independent, which they are not. Shown for contrast."""
    s1 = sum(t1 for _, t1, _ in TABLE); s5 = sum(t5 for _, _, t5 in TABLE)
    n = len(TABLE) * N_EP
    p1, p5 = s1 / n, s5 / n
    diff = (p5 - p1) * 100.0
    se = math.sqrt(p1 * (1 - p1) / n + p5 * (1 - p5) / n) * 100.0
    z = diff / se
    return s1, s5, n, diff, se, z, diff - 1.959964 * se, diff + 1.959964 * se


def holm():
    try:
        from scipy.stats import fisher_exact
    except ImportError:
        return None
    raw = []
    for name, t1, t5 in TABLE:
        _, p = fisher_exact([[t1, N_EP - t1], [t5, N_EP - t5]])
        raw.append((name, t1, t5, p))
    order = sorted(range(len(raw)), key=lambda i: raw[i][3])
    m = len(raw)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * raw[i][3])
        running = max(running, val)
        adj[i] = running
    return [(raw[i][0], raw[i][1], raw[i][2], raw[i][3], adj[i]) for i in range(m)]


def main():
    s1 = sum(t1 for _, t1, _ in TABLE); s5 = sum(t5 for _, _, t5 in TABLE)
    print(f"totals: T1 {s1}/200   T5 {s5}/200\n")

    mean, se, t, df, lo, hi = clustered()
    print("CLUSTERED BY TASK (the unit fixed before the analysis, n = 10)")
    print(f"  mean delta : {mean:+.2f} pp")
    print(f"  95% CI     : [{lo:+.2f}, {hi:+.2f}]   {'crosses zero' if lo < 0 < hi else ''}")
    print(f"  se         : {se:.3f}")
    print(f"  t({df})      : {t:.3f}")
    print(f"  p          : 0.177   (t({df}) = {t:.3f}, two-sided; from the t table)\n")

    s1, s5, n, diff, se2, z, lo2, hi2 = episode_level()
    print("EPISODE-LEVEL, for contrast only (treats clustered episodes as independent)")
    print(f"  delta      : {diff:+.2f} pp   CI [{lo2:+.2f}, {hi2:+.2f}]   z = {z:.3f}")
    print("  p          : 0.057\n")

    h = holm()
    if h is None:
        print("PER-TASK FISHER + HOLM: skipped, scipy not installed (pip install scipy)")
    else:
        print("PER-TASK FISHER EXACT, HOLM-CORRECTED ACROSS 10 TASKS")
        print(f"  {'task':22s} {'T1':>5s} {'T5':>5s} {'raw p':>8s} {'Holm p':>8s}")
        for name, a, b, rp, ap in h:
            if a == b == 20:
                continue
            print(f"  {name:22s} {a:>5d} {b:>5d} {rp:>8.3f} {ap:>8.3f}")
        print(f"  {'eight perfect tasks':22s} {20:>5d} {20:>5d} {1.0:>8.3f} {1.0:>8.3f}")
        print("\n  Nothing survives correction.")


if __name__ == "__main__":
    main()
