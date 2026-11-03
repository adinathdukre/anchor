"""Additional baselines at HPP calibration (reviewer-requested comparators).

Methods for a one-shot outcome:
  oracle      OLS on true habits
  naive       OLS on person means (HC3)
  pooled      OLS of Y_i on each night's W_ij (Y repeated per night), cluster-robust SE by person
  univ-RC     standard univariate regression calibration / random-intercept BLUP: each anchor shrunk
              separately by its own reliability, ignoring the error and habit cross-covariances
              (participant bootstrap)
  ANCHOR      multivariate calibration with unequal k and reliability floor (participant bootstrap)
Writes results/sim_baselines.csv. Usage: python sim_baselines.py [reps] [boot]
"""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "anchor"))
import anchor as A

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results")
SU = np.array([[1.9585, -1.6261], [-1.6261, 1.9356]]); SX = np.array([[0.9626, -0.4944], [-0.4944, 0.7072]])
PK = np.array([1144, 867]) / 2011; SDY = 0.113
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
BOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 100


def simulate(rng, n, beta):
    k = rng.choice([2, 3], n, p=PK); X = rng.multivariate_normal([20.8, 3.0], SX, n)
    pid = np.repeat(np.arange(n), k); Wn = X[pid] + rng.multivariate_normal([0, 0], SU, k.sum())
    c = rng.standard_normal(n)[:, None]; sig = X @ beta + 0.02 * c[:, 0]
    y = sig + rng.normal(0, np.sqrt(SDY ** 2 - sig.var()), n)
    return pid, Wn, X, y, c


def pooled(pid, Wn, y, C):
    Yn, Cn = y[pid], C[pid]
    Xd = np.column_stack([np.ones(len(Yn)), Wn, Cn]); b = np.linalg.lstsq(Xd, Yn, rcond=None)[0]; r = Yn - Xd @ b
    XtXi = np.linalg.inv(Xd.T @ Xd); G = np.zeros((Xd.shape[1], Xd.shape[1]))
    s = np.zeros((pid.max() + 1, Xd.shape[1])); np.add.at(s, pid, Xd * r[:, None]); G = s.T @ s
    V = XtXi @ G @ XtXi
    return b[1:3], np.sqrt(np.diag(V))[1:3]


def univ_rc(Wb, k, Su, y, C):
    mu = Wb.mean(0); Xh = np.empty_like(Wb)
    for j in range(2):
        sx2 = max(Wb[:, j].var() - Su[j, j] * np.mean(1 / k), 1e-6)
        lam = sx2 / (sx2 + Su[j, j] / k); Xh[:, j] = mu[j] + lam * (Wb[:, j] - mu[j])
    return A.fit_naive(Xh, y, C, robust=False)[0]


def boot(pid, Wn, y, C, B, rng, f):
    ids = np.unique(pid); rows = [np.flatnonzero(pid == i) for i in ids]; out = []
    for _ in range(B):
        pick = rng.integers(0, len(ids), len(ids)); r = np.concatenate([rows[i] for i in pick])
        np_ = np.repeat(np.arange(len(pick)), [len(rows[i]) for i in pick])
        _, Wb, kb, Sub, _ = A.person_summaries(np_, Wn[r]); out.append(f(Wb, kb, Sub, y[pick], C[pick]))
    return np.percentile(np.array(out), [2.5, 97.5], axis=0)


def main():
    rng = np.random.default_rng(99); rows = []
    for tname, beta in [("clock_only", np.array([0.008, 0])), ("gap_only", np.array([0, -0.008])), ("null", np.zeros(2))]:
        for n in [2011, 8044]:
            acc = {m: [] for m in ["oracle", "naive", "pooled", "univ-RC", "ANCHOR"]}
            for r in range(REPS):
                pid, Wn, X, y, C = simulate(rng, n, beta)
                _, Wb, k, Su, _ = A.person_summaries(pid, Wn)
                for m in acc:
                    if m in ("oracle", "naive"):
                        b, V = A.fit_naive(X if m == "oracle" else Wb, y, C); se = np.sqrt(np.diag(V)); lo, hi = b - 1.96 * se, b + 1.96 * se
                    elif m == "pooled":
                        b, se = pooled(pid, Wn, y, C); lo, hi = b - 1.96 * se, b + 1.96 * se
                    elif m == "univ-RC":
                        b = univ_rc(Wb, k, Su, y, C); lo, hi = boot(pid, Wn, y, C, BOOT, rng, univ_rc)
                    else:
                        b = A.fit_anchor(Wb, k, Su, y, C); lo, hi = boot(pid, Wn, y, C, BOOT, rng, lambda W, kk, S, yy, CC: A.fit_anchor(W, kk, S, yy, CC))
                    acc[m].append((b, lo, hi))
            want = {"clock_only": "clock", "gap_only": "gap", "null": "neither"}[tname]
            for m, v in acc.items():
                B_ = np.array([x[0] for x in v]); LO = np.array([x[1] for x in v]); HI = np.array([x[2] for x in v])
                sig = (LO > 0) | (HI < 0)
                dec = np.where(sig[:, 0] & ~sig[:, 1], "clock", np.where(sig[:, 1] & ~sig[:, 0], "gap", np.where(sig.all(1), "both", "neither")))
                zero = 1 if tname == "clock_only" else 0
                rows.append(dict(truth=tname, n=n, method=m,
                                 bias_M=B_[:, 0].mean() - beta[0], bias_D=B_[:, 1].mean() - beta[1],
                                 rmse_M=np.sqrt(((B_[:, 0] - beta[0]) ** 2).mean()), rmse_D=np.sqrt(((B_[:, 1] - beta[1]) ** 2).mean()),
                                 cover_M=((LO[:, 0] <= beta[0]) & (beta[0] <= HI[:, 0])).mean(), cover_D=((LO[:, 1] <= beta[1]) & (beta[1] <= HI[:, 1])).mean(),
                                 false_cross=sig[:, zero].mean() if tname != "null" else np.nan, correct=(dec == want).mean()))
                print({k: (round(float(x), 4) if isinstance(x, (float, np.floating)) else x) for k, x in rows[-1].items()}, flush=True)
    with open(os.path.join(OUT, "sim_baselines.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
