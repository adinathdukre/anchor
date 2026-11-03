"""Head-to-head with a jointly fitted multilevel latent-covariate model (Ludtke et al. 2008 style).

The latent-covariate model (LCM) is fitted by full maximum likelihood under normality: within-person
deviations identify Sigma_U, and person means and the outcome are jointly normal given k_i,
  (Wbar_i, Y_i) ~ N((mu, a + b'mu + g c_i), [[S_X + S_U/k_i, S_X b], [b'S_X, b'S_X b + s2]]),
with Wald intervals from the observed information. ANCHOR uses calibration with a 150-draw participant
bootstrap. HPP calibration, clock-only truth. Writes results/sim_latent.csv.
Usage: python sim_latent.py [reps] [boot]
"""
import os, sys, csv
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "anchor"))
import anchor as A
from sim_baselines import simulate, boot

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results")
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
BOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 150


def chol(v):
    return np.array([[np.exp(v[0]), 0.0], [v[1], np.exp(v[2])]])


def unpack(th):
    mu = th[0:2]; Lx = chol(th[2:5]); Lu = chol(th[5:8])
    return mu, Lx @ Lx.T, Lu @ Lu.T, th[8], th[9:11], th[11], np.exp(2 * th[12])


def negll(th, Wb, k, y, c, Sw_within, df):
    mu, Sx, Su, a, b, g, s2 = unpack(th)
    ll = -0.5 * df * np.log(np.linalg.det(Su)) - 0.5 * np.trace(np.linalg.solve(Su, Sw_within))
    for kk in np.unique(k):
        m = k == kk
        V = np.zeros((3, 3)); V[:2, :2] = Sx + Su / kk; V[:2, 2] = V[2, :2] = Sx @ b; V[2, 2] = b @ Sx @ b + s2
        R = np.column_stack([Wb[m] - mu, y[m] - a - b @ mu - g * c[m]])
        Vi = np.linalg.inv(V); sign, ld = np.linalg.slogdet(V)
        ll += -0.5 * m.sum() * ld - 0.5 * np.einsum("ij,jk,ik->", R, Vi, R)
    return -ll


def fit_lcm(pid, Wn, y, c):
    _, Wb, k, Su, df = A.person_summaries(pid, Wn)
    Sw_within = Su * df
    Sx0 = A._nearest_pd(np.cov(Wb.T) - Su * np.mean(1 / k))
    b0 = np.linalg.lstsq(np.column_stack([np.ones(len(y)), Wb, c]), y, rcond=None)[0]
    Lx = np.linalg.cholesky(Sx0); Lu = np.linalg.cholesky(Su)
    th0 = np.r_[Wb.mean(0), np.log(Lx[0, 0]), Lx[1, 0], np.log(Lx[1, 1]), np.log(Lu[0, 0]), Lu[1, 0], np.log(Lu[1, 1]),
                b0[0], b0[1:3], b0[3], np.log(y.std())]
    f = lambda t: negll(t, Wb, k, y, c, Sw_within, df)
    r = minimize(f, th0, method="L-BFGS-B")
    th = r.x; h = 1e-4 * np.maximum(1, np.abs(th)); p = len(th); H = np.zeros((p, p))
    for i in range(p):
        for j in range(i, p):
            e_i = np.zeros(p); e_i[i] = h[i]; e_j = np.zeros(p); e_j[j] = h[j]
            H[i, j] = H[j, i] = (f(th + e_i + e_j) - f(th + e_i - e_j) - f(th - e_i + e_j) + f(th - e_i - e_j)) / (4 * h[i] * h[j])
    V = np.linalg.pinv(H)
    return th[9:11], np.sqrt(np.clip(np.diag(V)[9:11], 0, None))


def main():
    rng = np.random.default_rng(31); rows = []; beta = np.array([0.008, 0.0])
    for n in [2011, 8044]:
        acc = {"LCM (ML)": [], "ANCHOR": []}
        for r in range(REPS):
            pid, Wn, X, y, C = simulate(rng, n, beta)
            b, se = fit_lcm(pid, Wn, y, C[:, 0]); acc["LCM (ML)"].append((b, b - 1.96 * se, b + 1.96 * se))
            _, Wb, k, Su, _ = A.person_summaries(pid, Wn)
            ba = A.fit_anchor(Wb, k, Su, y, C); lo, hi = boot(pid, Wn, y, C, BOOT, rng, lambda W, kk, S, yy, CC: A.fit_anchor(W, kk, S, yy, CC))
            acc["ANCHOR"].append((ba, lo, hi))
        for m, v in acc.items():
            B_ = np.array([x[0] for x in v]); LO = np.array([x[1] for x in v]); HI = np.array([x[2] for x in v])
            sig = (LO > 0) | (HI < 0)
            rows.append(dict(n=n, method=m, reps=len(v), bias_M=B_[:, 0].mean() - beta[0], bias_D=B_[:, 1].mean() - beta[1],
                             rmse_M=np.sqrt(((B_[:, 0] - beta[0]) ** 2).mean()), rmse_D=np.sqrt(((B_[:, 1] - beta[1]) ** 2).mean()),
                             cover_M=((LO[:, 0] <= beta[0]) & (beta[0] <= HI[:, 0])).mean(), cover_D=((LO[:, 1] <= 0) & (0 <= HI[:, 1])).mean(),
                             false_gap=sig[:, 1].mean(), corr_MD=np.corrcoef(np.array([x[0][0] for x in acc["ANCHOR"]]), np.array([x[0][0] for x in acc["LCM (ML)"]]))[0, 1]))
            print({kk: (round(float(x), 4) if not isinstance(x, str) else x) for kk, x in rows[-1].items()}, flush=True)
    with open(os.path.join(OUT, "sim_latent.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
