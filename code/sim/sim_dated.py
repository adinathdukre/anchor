"""Dated-night extension of ANCHOR under autocorrelated nightly error (clock-only truth, HPP calibration, n = 8,044).

Scenarios (nightly error Cov(U_ij, U_il) = rho(|t_j - t_l|) Sigma_U, rho(d) = phi**d):
  iid_spread    phi = 0, nights on distinct random days of a 14-day window
  ar_spread     phi = 0.5 per day, nights on distinct random days of a 14-day window
  ar_consec     phi = 0.3 per day, consecutive nights (the AR(1) row of Table A2)
Methods: naive (HC3), ANCHOR (i.i.d. nights), ANCHOR-dated (phi and Sigma_U from the within-person semivariogram);
ANCHOR methods use a participant bootstrap that re-estimates everything. Writes results/sim_dated.csv.
Usage: python sim_dated.py [reps] [boot]
"""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "anchor"))
import anchor as A

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results")
SU = np.array([[1.9585, -1.6261], [-1.6261, 1.9356]]); SX = np.array([[0.9626, -0.4944], [-0.4944, 0.7072]])
PK = np.array([1144, 867]) / 2011; SDY = 0.113; N = 8044; BETA = np.array([0.008, 0.0])
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
BOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 150
LU = np.linalg.cholesky(SU)


def simulate(rng, scen):
    phi = {"iid_spread": 0.0, "ar_spread": 0.5, "ar_consec": 0.3}[scen]
    k = rng.choice([2, 3], N, p=PK); X = rng.multivariate_normal([20.8, 3.0], SX, N)
    pid, W, T = [], [], []
    for i in range(N):
        if scen == "ar_consec":
            tt = np.arange(k[i], dtype=float) + rng.integers(0, 12)
        else:
            tt = np.sort(rng.choice(14, k[i], replace=False)).astype(float)
        R = phi ** np.abs(tt[:, None] - tt[None, :]) if phi > 0 else np.eye(k[i])
        U = np.linalg.cholesky(R) @ rng.standard_normal((k[i], 2)) @ LU.T
        pid += [i] * k[i]; W.append(X[i] + U); T += list(tt)
    pid = np.array(pid); W = np.vstack(W); T = np.array(T)
    c = rng.standard_normal(N); sig = X @ BETA + 0.02 * c
    y = sig + rng.normal(0, np.sqrt(SDY ** 2 - sig.var()), N)
    return pid, W, T, y, c[:, None]


def boot(pid, W, T, y, C, rng, dated):
    rows = np.split(np.arange(len(pid)), np.flatnonzero(np.diff(pid)) + 1); out = []
    for _ in range(BOOT):
        pick = rng.integers(0, len(rows), len(rows)); r = np.concatenate([rows[i] for i in pick])
        q = np.repeat(np.arange(len(pick)), [len(rows[i]) for i in pick])
        if dated:
            out.append(A.fit_anchor_dated(q, W[r], T[r], y[pick], C[pick])[0])
        else:
            _, Wb, k, Su, _ = A.person_summaries(q, W[r]); out.append(A.fit_anchor(Wb, k, Su, y[pick], C[pick]))
    return np.percentile(np.array(out), [2.5, 97.5], axis=0)


def main():
    rng = np.random.default_rng(51); rows = []
    scens = sys.argv[3].split(",") if len(sys.argv) > 3 else ["iid_spread", "ar_spread", "ar_consec"]
    for scen in scens:
        acc = {"naive": [], "ANCHOR": [], "ANCHOR-dated": []}; phis = []
        for _ in range(REPS):
            pid, W, T, y, C = simulate(rng, scen)
            _, Wb, k, Su, _ = A.person_summaries(pid, W)
            b, V = A.fit_naive(Wb, y, C); se = np.sqrt(np.diag(V)); acc["naive"].append((b, b - 1.96 * se, b + 1.96 * se))
            b = A.fit_anchor(Wb, k, Su, y, C); lo, hi = boot(pid, W, T, y, C, rng, False); acc["ANCHOR"].append((b, lo, hi))
            b, phi = A.fit_anchor_dated(pid, W, T, y, C); lo, hi = boot(pid, W, T, y, C, rng, True); acc["ANCHOR-dated"].append((b, lo, hi)); phis.append(phi)
        for m, v in acc.items():
            B_ = np.array([x[0] for x in v]); LO = np.array([x[1] for x in v]); HI = np.array([x[2] for x in v])
            sig = (LO > 0) | (HI < 0)
            rows.append(dict(scenario=scen, method=m, reps=len(v), bias_M=1e3 * (B_[:, 0].mean() - BETA[0]), bias_D=1e3 * B_[:, 1].mean(),
                             cover_M=((LO[:, 0] <= BETA[0]) & (BETA[0] <= HI[:, 0])).mean(), cover_D=((LO[:, 1] <= 0) & (0 <= HI[:, 1])).mean(),
                             false_gap=sig[:, 1].mean(), phi_hat=np.mean(phis) if m == "ANCHOR-dated" else np.nan))
            print({kk: (round(float(x), 3) if not isinstance(x, str) else x) for kk, x in rows[-1].items()}, flush=True)
    with open(os.path.join(OUT, f"sim_dated_{'_'.join(scens)}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
