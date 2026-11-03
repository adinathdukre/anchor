"""Simulations calibrated to the REAL HPP error structure (estimated in the TRE):
Sigma_U = [[1.9585,-1.6261],[-1.6261,1.9356]], Sigma_X = [[0.9626,-0.4944],[-0.4944,0.7072]],
k: 1144 people x 2 nights, 867 x 3 nights. Outcome SD 0.113 (liver attenuation).

Part A (analytic): false cross-anchor signal of the naive Wald test vs n.
Part B (Monte Carlo): bias, coverage, false-signal rate, and decision accuracy of the
pre-registered reading rule for naive vs ANCHOR, at the real n and at 4x n.
Writes results/sim_real.csv and results/sim_real_theory.csv
"""
import os, sys, csv
import numpy as np
from math import erf, sqrt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "anchor"))
import anchor as A

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results")
SU = np.array([[1.9585, -1.6261], [-1.6261, 1.9356]])
SX = np.array([[0.9626, -0.4944], [-0.4944, 0.7072]])
PK = np.array([1144, 867]) / 2011
SDY = 0.113
Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))


def theory():
    k = np.repeat([2, 3], [1144, 867])
    lam = A.attribution_matrix(SX, SU, k)
    Sw = SX + SU * np.mean(1 / k)
    rows = []
    for truth, beta in [("clock_only", np.array([0.008, 0])), ("gap_only", np.array([0, -0.008]))]:
        bn = lam @ beta
        # residual variance of naive regression; asymptotic var of naive coefs = s2 * Sw^-1 / n
        s2 = SDY**2 - beta @ SX @ beta + beta @ SX @ beta - bn @ Sw @ bn
        for n in [500, 1000, 2011, 4000, 8000, 16000, 32000, 64000]:
            se = np.sqrt(np.diag(s2 * np.linalg.inv(Sw) / n))
            j = 1 if truth == "clock_only" else 0  # the coefficient that should be zero
            z = bn[j] / se[j]
            rej = 1 - Phi(1.96 - z) + Phi(-1.96 - z)
            rows.append(dict(truth=truth, n=n, naive_null_coef=bn[j], se=se[j], false_signal_rate=rej))
    with open(os.path.join(OUT, "sim_real_theory.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("Lambda", lam.round(3).tolist())
    for r in rows: print(r["truth"], r["n"], f"naive null-coef {r['naive_null_coef']:+.5f}  false-signal {r['false_signal_rate']:.3f}")


def simulate(rng, n, beta):
    k = rng.choice([2, 3], n, p=PK)
    X = rng.multivariate_normal([20.8, 3.0], SX, n)
    pid = np.repeat(np.arange(n), k)
    Wn = X[pid] + rng.multivariate_normal([0, 0], SU, k.sum())
    c = rng.standard_normal(n)[:, None]
    sig = X @ beta + 0.02 * c[:, 0]
    y = sig + rng.normal(0, np.sqrt(SDY**2 - sig.var()), n)
    return pid, Wn, X, y, c


def rule(lo, hi):
    """Pre-registered reading rule on the joint model: clear = CI excludes 0."""
    cm, cd = (lo[0] > 0) | (hi[0] < 0), (lo[1] > 0) | (hi[1] < 0)
    return "clock" if cm and not cd else "gap" if cd and not cm else "both" if cm and cd else "neither"


def mc(reps=300, boot=150):
    rng = np.random.default_rng(42)
    rows = []
    for truth, beta in [("clock_only", np.array([0.008, 0])), ("gap_only", np.array([0, -0.008])), ("null", np.zeros(2))]:
        for n in [2011, 8044]:
            acc = {"oracle": [], "naive": [], "anchor": []}
            for r in range(reps):
                pid, Wn, X, y, C = simulate(rng, n, beta)
                _, Wb, k, Su, _ = A.person_summaries(pid, Wn)
                for m in acc:
                    if m == "anchor":
                        b = A.fit_anchor(Wb, k, Su, y, C)
                        d = A.person_bootstrap(pid, Wn, y, C, B=boot, seed=r)
                        lo, hi = np.percentile(d, [2.5, 97.5], 0)
                    else:
                        b, V = A.fit_naive(X if m == "oracle" else Wb, y, C)
                        se = np.sqrt(np.diag(V)); lo, hi = b - 1.96 * se, b + 1.96 * se
                    acc[m].append((b, lo, hi, rule(lo, hi)))
            want = {"clock_only": "clock", "gap_only": "gap", "null": "neither"}[truth]
            for m, v in acc.items():
                B_ = np.array([x[0] for x in v]); LO = np.array([x[1] for x in v]); HI = np.array([x[2] for x in v])
                dec = [x[3] for x in v]
                zero = 1 if truth == "clock_only" else 0
                rows.append(dict(truth=truth, n=n, method=m,
                                 bias_M=B_[:, 0].mean() - beta[0], bias_D=B_[:, 1].mean() - beta[1],
                                 rmse_M=np.sqrt(((B_[:, 0] - beta[0]) ** 2).mean()), rmse_D=np.sqrt(((B_[:, 1] - beta[1]) ** 2).mean()),
                                 cover_M=((LO[:, 0] <= beta[0]) & (beta[0] <= HI[:, 0])).mean(),
                                 cover_D=((LO[:, 1] <= beta[1]) & (beta[1] <= HI[:, 1])).mean(),
                                 false_cross_signal=((LO[:, zero] > 0) | (HI[:, zero] < 0)).mean() if truth != "null" else np.nan,
                                 correct_decision=np.mean([d == want for d in dec]),
                                 wrong_anchor=np.mean([d in ("clock", "gap") and d != want for d in dec])))
                print(rows[-1], flush=True)
    with open(os.path.join(OUT, "sim_real.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    theory()
    if len(sys.argv) > 1:
        mc(int(sys.argv[1]), int(sys.argv[2]))
