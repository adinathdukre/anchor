"""Simulation study for ANCHOR, calibrated to HPP-like timing data.

Calibration (from the HPP meal-timing audit): within-person nightly SD 0.84 h (clock M)
and 0.87 h (gap D); single-night ICC 0.33 (M) and 0.26 (D); person-level corr(M, D)
about -0.72; 2-3 nights per person (70% / 30%); outcome SD 0.113.

Usage: python sim_study.py [reps] [boot]
Writes results/sim_results.csv
"""
import os
import sys
import itertools
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "anchor"))
import anchor as A  # noqa: E402

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
BOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 100
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(OUT, exist_ok=True)

SW = np.array([0.84, 0.87])
ICC = np.array([0.33, 0.26])
SB = np.sqrt(ICC / (1 - ICC)) * SW
SDY = 0.113


def simulate(rng, n, beta, rho_b, rho_w, gamma_c=0.3):
    Sx = np.array([[SB[0] ** 2, rho_b * SB[0] * SB[1]], [rho_b * SB[0] * SB[1], SB[1] ** 2]])
    Su = np.array([[SW[0] ** 2, rho_w * SW[0] * SW[1]], [rho_w * SW[0] * SW[1], SW[1] ** 2]])
    # covariate (e.g. standardised age) correlated with the clock habit
    c = rng.standard_normal(n)
    X = rng.multivariate_normal([0, 0], Sx, n)
    X[:, 0] += 0.2 * SB[0] * c
    X += np.array([20.8, 3.0])
    k = rng.choice([2, 3], n, p=[0.7, 0.3])
    pid = np.repeat(np.arange(n), k)
    Wn = X[pid] + rng.multivariate_normal([0, 0], Su, k.sum())
    sig = X @ beta + gamma_c * 0.02 * c
    y = sig + rng.normal(0, np.sqrt(max(SDY**2 - sig.var(), 1e-6)), n)
    return pid, Wn, X, y, c[:, None], Sx, Su, k


def ci_cover(est, se, truth):
    return (est - 1.96 * se <= truth) & (truth <= est + 1.96 * se)


def run():
    rng = np.random.default_rng(2027)
    truths = {
        "gap_only": np.array([0.0, -0.01]),
        "clock_only": np.array([0.01, 0.0]),
        "null": np.array([0.0, 0.0]),
    }
    grid = list(itertools.product(truths.items(), [2011, 8000], [-0.72], [-0.3, -0.6, -0.9]))
    rows = []
    for (tname, beta), n, rho_b, rho_w in grid:
        acc = {m: [] for m in ["oracle", "naive", "anchor"]}
        lam = None
        for r in range(REPS):
            pid, Wn, X, y, C, Sx, Su, k = simulate(rng, n, beta, rho_b, rho_w)
            ids, Wbar, kk, Suh, _ = A.person_summaries(pid, Wn)
            if lam is None:
                lam = A.attribution_matrix(Sx, Su, kk)
            b_o, V_o = A.fit_naive(X, y, C)
            b_n, V_n = A.fit_naive(Wbar, y, C)
            b_a = A.fit_anchor(Wbar, kk, Suh, y, C)
            draws = A.person_bootstrap(pid, Wn, y, C, B=BOOT, seed=r)
            V_a = np.cov(draws.T)
            for m, b, V in [("oracle", b_o, V_o), ("naive", b_n, V_n), ("anchor", b_a, V_a)]:
                se = np.sqrt(np.diag(V))
                t = A.anchor_tests(b, V)
                acc[m].append(
                    [*b, *ci_cover(b, se, beta), t["H_gap_clockcoef"][2] < 0.05, t["H_clock_gapcoef"][2] < 0.05]
                )
        for m, v in acc.items():
            v = np.asarray(v, float)
            rows.append(
                dict(
                    truth=tname, n=n, rho_w=rho_w, method=m,
                    bias_M=v[:, 0].mean() - beta[0], bias_D=v[:, 1].mean() - beta[1],
                    rmse_M=np.sqrt(((v[:, 0] - beta[0]) ** 2).mean()),
                    rmse_D=np.sqrt(((v[:, 1] - beta[1]) ** 2).mean()),
                    cover_M=v[:, 2].mean(), cover_D=v[:, 3].mean(),
                    reject_clockcoef0=v[:, 4].mean(), reject_gapcoef0=v[:, 5].mean(),
                    lam11=lam[0, 0], lam12=lam[0, 1], lam21=lam[1, 0], lam22=lam[1, 1],
                )
            )
        print(tname, n, rho_w, "done", flush=True)
    import csv

    with open(os.path.join(OUT, "sim_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    run()
