"""Stress tests of ANCHOR outside its assumptions (clock-only truth, HPP calibration, n = 8,044).

Scenarios
  classical      reference: i.i.d. Gaussian nightly error (A2 holds)
  heavy_tails    nightly error multivariate t with 3 df, same covariance
  heterosc       error SD grows with the clock habit (|z| of X_M), same average covariance
  habit_dep      non-classical: nights regress towards the population mean, U = -0.2 (X - mu) + classical
  ar1            nightly errors autocorrelated across consecutive nights, AR(1) with rho = 0.3, same marginal covariance
  three_anchors  p = 3: clock M, gap D = S - M and sleep duration Q = Wake - S, synthetic covariances
Reports bias x 1e3, 95% coverage (participant bootstrap) and false-anchor rate for naive and ANCHOR.
Writes results/sim_stress.csv. Usage: python sim_stress.py [reps] [boot]
"""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "anchor"))
import anchor as A

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results")
SU2 = np.array([[1.9585, -1.6261], [-1.6261, 1.9356]]); SX2 = np.array([[0.9626, -0.4944], [-0.4944, 0.7072]])
PK = np.array([1144, 867]) / 2011; SDY = 0.113; N = 8044
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
BOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 100

# p = 3: event-level covariances of (meal m, sleep s, wake w), anchors M = m, D = s - m, Q = w - s
def _cov(sd, r):
    R = np.array([[1, r[0], r[1]], [r[0], 1, r[2]], [r[1], r[2], 1]]); return np.outer(sd, sd) * R
Bm = np.array([[1, 0, 0], [-1, 1, 0], [0, -1, 1]])
SX3 = Bm @ _cov([1.0, 0.8, 0.8], [0.5, 0.3, 0.6]) @ Bm.T
SU3 = Bm @ _cov([1.4, 0.9, 0.9], [0.3, 0.1, 0.4]) @ Bm.T


def simulate(rng, scen):
    p3 = scen == "three_anchors"; SX, SU = (SX3, SU3) if p3 else (SX2, SU2); p = SX.shape[0]
    beta = np.zeros(p); beta[0] = 0.008
    k = rng.choice([2, 3], N, p=PK); X = rng.multivariate_normal(np.zeros(p), SX, N)
    pid = np.repeat(np.arange(N), k); m = k.sum()
    U = rng.multivariate_normal(np.zeros(p), SU, m)
    if scen == "heavy_tails":
        U = U / np.sqrt(rng.chisquare(3, m) / 3)[:, None] * np.sqrt(1 / 3)  # t3 rescaled to covariance SU
    elif scen == "heterosc":
        z = X[pid, 0] / np.sqrt(SX[0, 0]); U = U * np.sqrt((0.5 + z ** 2) / 1.5)[:, None]
    elif scen == "habit_dep":
        U = U - 0.2 * X[pid]
    elif scen == "ar1":
        first = np.r_[True, pid[1:] != pid[:-1]]
        for i in range(1, m):
            if not first[i]:
                U[i] = 0.3 * U[i - 1] + np.sqrt(1 - 0.3 ** 2) * U[i]
    Wn = X[pid] + U
    c = rng.standard_normal(N); sig = X @ beta + 0.02 * c
    y = sig + rng.normal(0, np.sqrt(SDY ** 2 - sig.var()), N)
    return pid, Wn, y, c[:, None], beta


def boot_ci(pid, Wn, y, C, rng):
    rows = np.split(np.arange(len(pid)), np.flatnonzero(np.diff(pid)) + 1); out = []
    for _ in range(BOOT):
        pick = rng.integers(0, len(rows), len(rows)); r = np.concatenate([rows[i] for i in pick])
        npid = np.repeat(np.arange(len(pick)), [len(rows[i]) for i in pick])
        _, Wb, kb, Sub, _ = A.person_summaries(npid, Wn[r]); out.append(A.fit_anchor(Wb, kb, Sub, y[pick], C[pick]))
    return np.percentile(np.array(out), [2.5, 97.5], axis=0)


def main():
    rng = np.random.default_rng(2027); rows = []
    scens = sys.argv[3].split(",") if len(sys.argv) > 3 else ["classical", "heavy_tails", "heterosc", "habit_dep", "ar1", "three_anchors"]
    for scen in scens:
        acc = {"naive": [], "ANCHOR": []}
        for _ in range(REPS):
            pid, Wn, y, C, beta = simulate(rng, scen)
            _, Wb, k, Su, _ = A.person_summaries(pid, Wn)
            b, V = A.fit_naive(Wb, y, C); se = np.sqrt(np.diag(V)); acc["naive"].append((b, b - 1.96 * se, b + 1.96 * se))
            b = A.fit_anchor(Wb, k, Su, y, C); lo, hi = boot_ci(pid, Wn, y, C, rng); acc["ANCHOR"].append((b, lo, hi))
        for mth, v in acc.items():
            B_ = np.array([x[0] for x in v]); LO = np.array([x[1] for x in v]); HI = np.array([x[2] for x in v])
            sig = (LO > 0) | (HI < 0); cov = ((LO <= beta) & (beta <= HI)).mean(0)
            rows.append(dict(scenario=scen, method=mth, bias_M=1e3 * (B_[:, 0].mean() - beta[0]),
                             bias_other=1e3 * np.abs(B_[:, 1:].mean(0)).max(), cover_M=cov[0], cover_other=cov[1:].min(),
                             false_other=sig[:, 1:].any(1).mean()))
            print({k_: (round(float(x), 3) if not isinstance(x, str) else x) for k_, x in rows[-1].items()}, flush=True)
    path = os.path.join(OUT, "sim_stress.csv"); old = list(csv.DictReader(open(path))) if os.path.exists(path) else []
    old = [r for r in old if r["scenario"] not in scens] + rows
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(old)


if __name__ == "__main__":
    main()
