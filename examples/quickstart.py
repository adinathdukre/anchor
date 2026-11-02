"""ANCHOR on simulated data: only the clock matters, and the gap should come out null.

Covariances below are illustrative values chosen for this example. Run: python examples/quickstart.py
"""
import numpy as np
import anchor

rng = np.random.default_rng(1)
n = 50000
Sx = np.array([[1.0, -0.5], [-0.5, 0.8]])    # true habits (clock, gap)
Su = np.array([[2.0, -1.7], [-1.7, 2.0]])    # night-to-night error, steeper than the habits

X = rng.multivariate_normal([0, 0], Sx, n)
nights = rng.integers(2, 4, n)               # 2 or 3 nights per person
pid = np.repeat(np.arange(n), nights)
W = X[pid] + rng.multivariate_normal([0, 0], Su, nights.sum())
y = 0.2 * X[:, 0] + rng.normal(0, 1, n)      # true clock effect 0.2, true gap effect 0

anchor.leak_check(pid, W)
ids, Wbar, k, Su_hat, _ = anchor.person_summaries(pid, W)
naive, _ = anchor.fit_naive(Wbar, y)
corrected = anchor.fit_anchor(Wbar, k, Su_hat, y)
print(f"true    clock +0.200  gap +0.000")
print(f"naive   clock {naive[0]:+.3f}  gap {naive[1]:+.3f}")
print(f"ANCHOR  clock {corrected[0]:+.3f}  gap {corrected[1]:+.3f}")
