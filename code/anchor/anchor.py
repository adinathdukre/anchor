"""ANCHOR: attribution of anchored timing exposures measured over few nights.

Exposures are defined on different time anchors from two primitive nightly event
times, meal end M and sleep onset S (clock = M, gap D = S - M). Person-level habits
are estimated by averaging k_i noisy nights, which (i) attenuates and (ii) mixes the
coefficients of correlated anchors. This module provides

  * attribution_matrix      closed-form plim of the naive estimator: beta_naive = Lambda beta
  * fit_naive               OLS on person means (the standard approach)
  * fit_anchor              covariate-aware, unequal-k multivariate regression calibration
  * person_bootstrap        nonparametric person-level bootstrap for any fitter
  * anchor_tests            Wald tests for H_clock (gap coef = 0), H_gap (clock coef = 0)
                            and the fixed-sleep contrast (clock - gap)

All functions take nightly data in "long" form so that the within-person error
covariance Sigma_U can be estimated from night-to-night deviations.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# data helpers


def person_summaries(pid, Wn):
    """Group nightly exposures Wn (n_nights x p) by person id.

    Returns unique ids, person means (n x p), nights per person k (n,), and the
    pooled within-person covariance Sigma_U (p x p) with its degrees of freedom.
    """
    pid = np.asarray(pid)
    Wn = np.asarray(Wn, float)
    ids, inv, k = np.unique(pid, return_inverse=True, return_counts=True)
    p = Wn.shape[1]
    sums = np.zeros((len(ids), p))
    np.add.at(sums, inv, Wn)
    Wbar = sums / k[:, None]
    dev = Wn - Wbar[inv]
    df = int((k - 1).sum())
    Su = dev.T @ dev / df
    return ids, Wbar, k, Su, df


def _design(X, C):
    cols = [np.ones(len(X)), X] + ([C] if C is not None else [])
    return np.column_stack(cols)


def _ols(A, y):
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def _hc3(A, y, coef):
    r = y - A @ coef
    AtA_inv = np.linalg.inv(A.T @ A)
    h = np.einsum("ij,jk,ik->i", A, AtA_inv, A)
    w = (r / (1 - h)) ** 2
    return AtA_inv @ (A.T * w) @ A @ AtA_inv


# ---------------------------------------------------------------------------
# theory


def attribution_matrix(Sx, Su, k):
    """plim of naive person-mean OLS coefficients: beta_naive = Lambda @ beta.

    Sx : covariance of true habits given covariates (p x p)
    Su : within-person (night-level) error covariance (p x p)
    k  : array of nights per person (heterogeneous allowed)

    With unequal k the naive estimator targets (Sx + E[Su/k])^-1 Sx.
    """
    k = np.atleast_1d(np.asarray(k, float))
    Sw = Sx + Su * np.mean(1.0 / k)
    return np.linalg.solve(Sw, Sx)


# ---------------------------------------------------------------------------
# estimators


def fit_naive(Wbar, y, C=None, robust=True):
    A = _design(Wbar, C)
    coef = _ols(A, y)
    p = Wbar.shape[1]
    V = _hc3(A, y, coef) if robust else None
    return coef[1 : 1 + p], (V[1 : 1 + p, 1 : 1 + p] if V is not None else None)


def calibrate(Wbar, k, Su, C=None, r_min=0.1):
    """E[X | Wbar, C] under the classical additive nightly error model.

    Uses the joint covariance of (Wbar_i, C_i), whose upper-left block is
    Sx + Su/k_i, so each person is shrunk according to their own k_i.
    Sx and Cov(X, C) are moment-estimated: Sx = Cov(Wbar) - Su * mean(1/k).
    """
    n, p = Wbar.shape
    mu_w = Wbar.mean(0)
    Sx = np.cov(Wbar.T) - Su * np.mean(1.0 / k)
    Sx = reliability_floor(_nearest_pd(Sx), Su, k, r_min)
    if C is None:
        out = np.empty_like(Wbar)
        for kk in np.unique(k):
            m = k == kk
            G = Sx @ np.linalg.inv(Sx + Su / kk)
            out[m] = mu_w + (Wbar[m] - mu_w) @ G.T
        return out, Sx
    C = np.asarray(C, float)
    mu_c = C.mean(0)
    Sxc = ((Wbar - mu_w).T @ (C - mu_c)) / (n - 1)  # Cov(X,C) = Cov(Wbar,C)
    Sc = np.cov(C.T).reshape(C.shape[1], C.shape[1])
    Zc = np.hstack([Wbar - mu_w, C - mu_c])
    out = np.empty_like(Wbar)
    covXZ = np.hstack([Sx, Sxc])
    for kk in np.unique(k):
        m = k == kk
        VZ = np.block([[Sx + Su / kk, Sxc], [Sxc.T, Sc]])
        G = covXZ @ np.linalg.pinv(VZ)
        out[m] = mu_w + Zc[m] @ G.T
    return out, Sx


def fit_anchor(Wbar, k, Su, y, C=None, r_min=0.1):
    """ANCHOR estimator: regression calibration (with reliability floor) then OLS."""
    Xhat, _ = calibrate(Wbar, k, Su, C, r_min)
    A = _design(Xhat, C)
    coef = _ols(A, y)
    p = Wbar.shape[1]
    return coef[1 : 1 + p]


def _nearest_pd(S, eps=1e-8):
    w, V = np.linalg.eigh((S + S.T) / 2)
    return V @ np.diag(np.maximum(w, eps)) @ V.T


def reliability_floor(Sx, Su, k, r_min=0.1):
    """Floor the reliability of the person-mean along every direction.

    In coordinates where the mean error covariance Su*mean(1/k) is the identity,
    Sx has eigenvalues nu; the reliability along each eigen-direction is
    nu/(nu+1). Directions with reliability < r_min are raised to r_min, which
    bounds the calibration gain (noise amplification) by 1/r_min. r_min=0
    recovers plain regression calibration.
    """
    if r_min <= 0:
        return Sx
    Se = Su * np.mean(1.0 / np.asarray(k, float))
    L = np.linalg.cholesky(_nearest_pd(Se))
    Li = np.linalg.inv(L)
    T = Li @ Sx @ Li.T
    w, V = np.linalg.eigh((T + T.T) / 2)
    nu_min = r_min / (1 - r_min)
    T2 = V @ np.diag(np.maximum(w, nu_min)) @ V.T
    return L @ T2 @ L.T


# ---------------------------------------------------------------------------
# inference


def person_bootstrap(pid, Wn, y_by_id, C_by_id, B=500, seed=0, method="anchor"):
    """Person-level bootstrap. Resamples whole people (all their nights).

    y_by_id, C_by_id are aligned to the sorted unique ids returned by
    person_summaries. Returns a (B x p) array of coefficient draws.
    """
    rng = np.random.default_rng(seed)
    pid = np.asarray(pid)
    ids, inv = np.unique(pid, return_inverse=True)
    rows_of = [np.flatnonzero(inv == i) for i in range(len(ids))]
    draws = []
    for _ in range(B):
        pick = rng.integers(0, len(ids), len(ids))
        rows = np.concatenate([rows_of[i] for i in pick])
        new_pid = np.repeat(np.arange(len(pick)), [len(rows_of[i]) for i in pick])
        _, Wb, kb, Sub, _ = person_summaries(new_pid, Wn[rows])
        yb = y_by_id[pick]
        Cb = None if C_by_id is None else C_by_id[pick]
        if method == "anchor":
            draws.append(fit_anchor(Wb, kb, Sub, yb, Cb))
        else:
            draws.append(fit_naive(Wb, yb, Cb, robust=False)[0])
    return np.asarray(draws)


def anchor_tests(beta, V):
    """Wald tests in the (clock M, gap D) basis.

    H_clock : gap coefficient = 0 (outcome depends on clock time only)
    H_gap   : clock coefficient = 0 (outcome depends on the gap only)
    fixed_sleep contrast : meal 1 h later with sleep onset fixed = b_M - b_D
    """
    from math import erf, sqrt

    def p2(z):
        return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))

    bM, bD = beta
    c = np.array([1.0, -1.0])
    est_c = c @ beta
    se_c = float(np.sqrt(c @ V @ c))
    return {
        "H_gap_clockcoef": (bM, np.sqrt(V[0, 0]), p2(bM / np.sqrt(V[0, 0]))),
        "H_clock_gapcoef": (bD, np.sqrt(V[1, 1]), p2(bD / np.sqrt(V[1, 1]))),
        "fixed_sleep_contrast": (est_c, se_c, p2(est_c / se_c)),
    }


# ---------------------------------------------------------------------------
# repeated (nightly) outcomes: habitual vs acute decomposition


def lambda_given_C(Wbar, k, Su, C=None):
    """Attribution matrix using Sigma_X|C (habit covariance net of covariates)."""
    Wr = Wbar - Wbar.mean(0)
    if C is not None:
        Cd = np.column_stack([np.ones(len(C)), C])
        Wr = Wbar - Cd @ np.linalg.lstsq(Cd, Wbar, rcond=None)[0]
    Sx = _nearest_pd(np.cov(Wr.T) - Su * np.mean(1.0 / k))
    return attribution_matrix(Sx, Su, k)


def fit_anchor_nightly(pid, Wn, yn, C=None, Cn=None):
    """Habitual (between) and acute (within) effects for an outcome measured every night.

    Model: y_ij = a + b_h' X_i + b_a' (W_ij - X_i) + g' C + e_ij.
    Naive person-mean slope s = Lambda b_h + (I - Lambda) b_a, so
        b_h = Lambda^-1 (s - (I - Lambda) b_a),  b_a from within-person FE.
    Returns dict with s (naive between), b_a (within), b_h (habitual), b_h - b_a.
    C  : person-level covariates aligned to sorted unique ids
    Cn : night-level covariates (n_nights x q) for the within model
    """
    uid, Wbar, k, Su, _ = person_summaries(pid, Wn)
    inv = np.searchsorted(uid, pid)
    ybar = np.bincount(inv, yn) / k
    Cb = C
    if Cn is not None:
        Cn_bar = np.column_stack([np.bincount(inv, Cn[:, j]) / k for j in range(Cn.shape[1])])
        Cb = Cn_bar if C is None else np.column_stack([C, Cn_bar])
    s = fit_naive(Wbar, ybar, Cb, robust=False)[0]
    Xd = Wn - Wbar[inv]
    if Cn is not None:
        Xd = np.column_stack([Xd, Cn - Cn_bar[inv]])
    yd = yn - ybar[inv]
    ba = np.linalg.lstsq(Xd, yd, rcond=None)[0][:2]
    L = lambda_given_C(Wbar, k, Su, Cb)
    bh = np.linalg.solve(L, s - (np.eye(2) - L) @ ba)
    return {"s": s, "b_a": ba, "b_h": bh, "diff": bh - ba, "Lambda": L}


# ---------------------------------------------------------------------------
# dated nights: night-to-night correlation


def _within_pairs(pid, Wn, t):
    """All within-person night pairs: day gaps and half outer products of differences."""
    pid = np.asarray(pid); Wn = np.asarray(Wn, float); t = np.asarray(t, float)
    o = np.lexsort((t, pid)); pid, Wn, t = pid[o], Wn[o], t[o]
    starts = np.r_[0, np.flatnonzero(pid[1:] != pid[:-1]) + 1, len(pid)]
    a, b = [], []
    for s, e in zip(starts[:-1], starts[1:]):
        for i in range(s, e):
            for j in range(i + 1, e):
                a.append(i); b.append(j)
    a, b = np.array(a, int), np.array(b, int)
    d = Wn[a] - Wn[b]
    return np.abs(t[b] - t[a]), 0.5 * np.einsum("ij,ik->ijk", d, d)


def night_correlation(pid, Wn, t, phi_grid=np.linspace(0.0, 0.95, 96)):
    """Fit rho(lag) = phi**lag (lag in days) from the within-person semivariogram.

    For two nights of one person, E[(W_j - W_l)(W_j - W_l)^T]/2 = (1 - rho(|t_j - t_l|)) Sigma_U,
    which does not involve the habit. For each phi on the grid Sigma_U is the least-squares sill;
    phi minimises the residual sum of squares. phi = 0 recovers i.i.d. nights.
    """
    lag, G = _within_pairs(pid, Wn, t)
    best = None
    for phi in phi_grid:
        w = 1.0 - phi ** lag
        Su = np.einsum("i,ijk->jk", w, G) / (w @ w)
        rss = ((G - w[:, None, None] * Su) ** 2).sum()
        if best is None or rss < best[0]:
            best = (rss, phi, Su)
    return best[1], best[2]


def effective_k(pid, t, phi):
    """Per-person effective number of nights: 1 / [(1/k^2) sum_{j,l} phi^|t_j - t_l|], sorted by person id."""
    pid = np.asarray(pid); t = np.asarray(t, float)
    out = []
    for u in np.unique(pid):
        tt = t[pid == u]; R = phi ** np.abs(tt[:, None] - tt[None, :])
        out.append(len(tt) ** 2 / R.sum())
    return np.array(out)


def fit_anchor_dated(pid, Wn, t, y, C=None, r_min=0.1):
    """ANCHOR for dated nights: error covariance of a person mean is Sigma_U / k_eff,i."""
    phi, Su = night_correlation(pid, Wn, t)
    _, Wbar, k, _, _ = person_summaries(pid, Wn)
    keff = np.round(effective_k(pid, t, phi), 3)
    return fit_anchor(Wbar, keff, Su, y, C, r_min), phi


# ---------------------------------------------------------------------------
# many outcomes (omics scale) and a one-line diagnostic


def fit_anchor_many(Wbar, k, Su, Y, C=None, r_min=0.1):
    """ANCHOR for many outcomes at once.

    The calibration does not depend on the outcome, so habits are predicted once and all
    outcomes (columns of Y, n x m) are solved in a single least-squares call.
    Returns the (p x m) matrix of habitual effects.
    """
    Xhat, _ = calibrate(Wbar, k, Su, C, r_min)
    A = _design(Xhat, C)
    coef, *_ = np.linalg.lstsq(A, np.asarray(Y, float), rcond=None)
    return coef[1 : 1 + Wbar.shape[1]]


def bootstrap_many(pid, Wn, Y_by_id, C_by_id=None, B=200, seed=0, r_min=0.1):
    """Participant bootstrap for many outcomes.

    Person-level summaries (mean, number of nights, within-person sum of squares) are computed
    once; each draw resamples people, re-estimates Sigma_U and the calibration from the
    resampled summaries and solves all outcomes together. Returns a (B x p x m) array.
    """
    rng = np.random.default_rng(seed)
    pid = np.asarray(pid); Wn = np.asarray(Wn, float)
    ids, inv, k = np.unique(pid, return_inverse=True, return_counts=True)
    p = Wn.shape[1]
    Wbar = np.zeros((len(ids), p)); np.add.at(Wbar, inv, Wn); Wbar /= k[:, None]
    dev = Wn - Wbar[inv]
    SS = np.zeros((len(ids), p, p)); np.add.at(SS, inv, np.einsum("ij,ik->ijk", dev, dev))
    Y = np.asarray(Y_by_id, float); Y = Y[:, None] if Y.ndim == 1 else Y
    out = np.empty((B, p, Y.shape[1]))
    for b in range(B):
        pick = rng.integers(0, len(ids), len(ids))
        Su = SS[pick].sum(0) / (k[pick] - 1).sum()
        Cb = None if C_by_id is None else np.asarray(C_by_id)[pick]
        out[b] = fit_anchor_many(Wbar[pick], k[pick], Su, Y[pick], Cb, r_min)
    return out


def leak_check(pid, Wn, names=("clock", "gap"), verbose=True):
    """Answer 'do my anchors leak?' from the nightly table alone, before any outcome is fitted.

    Returns the attribution matrix, between- and within-person slopes of the second anchor on the
    first, and the largest leakage (off-diagonal of Lambda).
    """
    _, Wbar, k, Su, _ = person_summaries(pid, Wn)
    Sx = _nearest_pd(np.cov(Wbar.T) - Su * np.mean(1.0 / k))
    L = attribution_matrix(Sx, Su, k)
    s_x, s_u = Sx[1, 0] / Sx[0, 0], Su[1, 0] / Su[0, 0]
    leak = np.abs(L - np.diag(np.diag(L))).max()
    res = {"Lambda": L, "slope_between": s_x, "slope_within": s_u, "max_leakage": leak,
           "reliability": np.diag(Sx) / (np.diag(Sx) + np.diag(Su))}
    if verbose:
        verdict = "LEAKS" if leak > 0.05 else "little leakage"
        print(f"{verdict}: {100 * L[1, 0]:.0f}% of a true {names[0]} effect appears as {names[1]} "
              f"(slopes of {names[1]} on {names[0]}: between {s_x:+.2f}, within {s_u:+.2f}); "
              f"single-night reliabilities {np.round(res['reliability'], 2).tolist()}")
    return res
