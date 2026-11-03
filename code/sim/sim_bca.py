"""Percentile versus BCa participant-bootstrap intervals for ANCHOR at the liver sample size.
Global null and clock-only truths, n = 2,011, HPP calibration. BCa uses a leave-one-participant-out jackknife
for the acceleration. Writes results/sim_bca.csv. Usage: python sim_bca.py [reps] [boot]
"""
import os, sys, csv
import numpy as np
from math import erf, sqrt
from statistics import NormalDist
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "anchor"))
import anchor as A
from sim_baselines import simulate

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results")
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
BOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 500
ND = NormalDist()


def summaries(pid, Wn):
    _, Wb, k, Su, df = A.person_summaries(pid, Wn)
    inv = np.searchsorted(np.unique(pid), pid)
    dev = Wn - Wb[inv]; ss = np.zeros((len(k), 2, 2)); np.add.at(ss, inv, np.einsum("ij,ik->ijk", dev, dev))
    return Wb, k, ss


def fit(Wb, k, ss, y, C):
    Su = ss.sum(0) / (k - 1).sum()
    return A.fit_anchor(Wb, k, Su, y, C)


def intervals(Wb, k, ss, y, C, rng):
    est = fit(Wb, k, ss, y, C); n = len(k)
    bs = np.array([fit(Wb[p], k[p], ss[p], y[p], C[p]) for p in (rng.integers(0, n, n) for _ in range(BOOT))])
    jk = np.array([fit(np.delete(Wb, i, 0), np.delete(k, i), np.delete(ss, i, 0), np.delete(y, i), np.delete(C, i, 0)) for i in range(n)])
    pct = np.percentile(bs, [2.5, 97.5], axis=0); bca = np.empty_like(pct)
    for j in range(2):
        z0 = ND.inv_cdf(min(max((bs[:, j] < est[j]).mean(), 1e-4), 1 - 1e-4))
        d = jk[:, j].mean() - jk[:, j]; acc = (d ** 3).sum() / (6 * (d ** 2).sum() ** 1.5)
        q = [ND.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z))) for z in (ND.inv_cdf(0.025), ND.inv_cdf(0.975))]
        bca[:, j] = np.percentile(bs[:, j], [100 * q[0], 100 * q[1]])
    return pct, bca


def main():
    rng = np.random.default_rng(41); rows = []
    for name, beta in [("null", np.zeros(2)), ("clock_only", np.array([0.008, 0.0]))]:
        cov = {"percentile": [], "BCa": []}
        for r in range(REPS):
            pid, Wn, X, y, C = simulate(rng, 2011, beta)
            Wb, k, ss = summaries(pid, Wn)
            pct, bca = intervals(Wb, k, ss, y, C, rng)
            for m, iv in (("percentile", pct), ("BCa", bca)):
                cov[m].append((iv[0] <= beta) & (beta <= iv[1]))
        for m, v in cov.items():
            v = np.array(v); se = np.sqrt(v.mean(0) * (1 - v.mean(0)) / len(v))
            rows.append(dict(truth=name, method=m, reps=len(v), cover_M=v[:, 0].mean(), cover_D=v[:, 1].mean(), mcse_M=se[0], mcse_D=se[1]))
            print(rows[-1], flush=True)
    with open(os.path.join(OUT, "sim_bca.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
