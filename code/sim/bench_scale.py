"""Runtime benchmark for omics-scale ANCHOR (single CPU process).

Lipidome-like design: n participants with 2-20 logged nights (median about 14), two anchors, m outcomes.
Times (seconds) for: the per-outcome loop (fit_anchor called m times), the vectorised fit (fit_anchor_many),
and a 200-draw participant bootstrap of all m outcomes (bootstrap_many). Writes results/bench_scale.csv.
"""
import os, sys, csv, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import anchor as A

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results")
SU = np.array([[1.9585, -1.6261], [-1.6261, 1.9356]]); SX = np.array([[0.9626, -0.4944], [-0.4944, 0.7072]])


def data(rng, n, m):
    k = np.clip(rng.poisson(14, n), 2, 20); X = rng.multivariate_normal([0, 0], SX, n)
    pid = np.repeat(np.arange(n), k); W = X[pid] + rng.multivariate_normal([0, 0], SU, k.sum())
    Y = X @ rng.normal(0, 0.05, (2, m)) + rng.normal(0, 1, (n, m)); C = rng.normal(size=(n, 3))
    return pid, W, Y, C


def main():
    rng = np.random.default_rng(7); rows = []
    for n, m in [(5608, 170), (5608, 1000), (5608, 10000), (100000, 1000)]:
        pid, W, Y, C = data(rng, n, m)
        t0 = time.perf_counter(); _, Wb, k, Su, _ = A.person_summaries(pid, W); t_sum = time.perf_counter() - t0
        reps = min(m, 50)
        t0 = time.perf_counter()
        for j in range(reps):
            A.fit_anchor(Wb, k, Su, Y[:, j], C)
        t_loop = (time.perf_counter() - t0) * m / reps
        t0 = time.perf_counter(); A.fit_anchor_many(Wb, k, Su, Y, C); t_many = time.perf_counter() - t0
        B = 200 if n <= 10000 else 50
        t0 = time.perf_counter(); A.bootstrap_many(pid, W, Y, C, B=B); t_boot = (time.perf_counter() - t0) * 200 / B
        rows.append(dict(n=n, m=m, nights=int(k.sum()), summaries_s=round(t_sum, 3), loop_s=round(t_loop, 2),
                         vectorised_s=round(t_many, 3), bootstrap200_s=round(t_boot, 1), speedup=round(t_loop / t_many, 1)))
        print(rows[-1], flush=True)
    with open(os.path.join(OUT, "bench_scale.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
