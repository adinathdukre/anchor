"""Nightly-outcome simulation (real HPP error structure, CGM-like n and noise).
Compares naive person-mean, one-shot ANCHOR (misapplied), within FE (acute), nightly ANCHOR (habitual)."""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "anchor"))
import anchor as A
SU = np.array([[1.9585, -1.6261], [-1.6261, 1.9356]]); SX = np.array([[0.9626, -0.4944], [-0.4944, 0.7072]])
rng = np.random.default_rng(7); rows = []
scen = {"h=a (gap)": ([0, -2.5], [0, -2.5]), "h>a (gap)": ([0, -5], [0, -2.5]),
        "habit clock, acute gap": ([1, 0], [0, -2.5]), "acute only (gap)": ([0, 0], [0, -2.5])}
for name, (bh, ba) in scen.items():
    bh, ba = np.array(bh, float), np.array(ba, float); acc = {m: [] for m in ["naive", "oneshot", "within", "nightly"]}
    for r in range(500):
        n = 1250; k = rng.choice([2, 3], n, p=[.7, .3]); pid = np.repeat(np.arange(n), k)
        X = rng.multivariate_normal([20.8, 3], SX, n); U = rng.multivariate_normal([0, 0], SU, k.sum())
        yn = X[pid] @ bh + U @ ba + rng.normal(0, 15, k.sum())
        res = A.fit_anchor_nightly(pid, X[pid] + U, yn)
        uid, Wb, kk, Su, _ = A.person_summaries(pid, X[pid] + U); yb = np.bincount(pid, yn) / kk
        acc["naive"].append(res["s"]); acc["oneshot"].append(A.fit_anchor(Wb, kk, Su, yb, r_min=0))
        acc["within"].append(res["b_a"]); acc["nightly"].append(res["b_h"])
    for m, v in acc.items():
        v = np.array(v); tgt = ba if m == "within" else bh
        rows.append(dict(scenario=name, method=m, target="acute" if m == "within" else "habitual",
                         est_M=v[:, 0].mean(), est_D=v[:, 1].mean(), true_M=tgt[0], true_D=tgt[1],
                         bias_M=v[:, 0].mean() - tgt[0], bias_D=v[:, 1].mean() - tgt[1],
                         rmse_M=np.sqrt(((v[:, 0] - tgt[0]) ** 2).mean()), rmse_D=np.sqrt(((v[:, 1] - tgt[1]) ** 2).mean())))
        print(name.ljust(24), m.ljust(8), "est %+.2f %+.2f  true %+.2f %+.2f" % (v[:, 0].mean(), v[:, 1].mean(), *tgt))
with open(os.path.join(os.path.dirname(__file__), "..", "..", "results", "sim_nightly.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
