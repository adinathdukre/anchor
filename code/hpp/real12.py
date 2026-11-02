"""Real-data run 12: plasmode spike-in on the REAL HPP nightly anchors (liver cohort) — aggregate output only.
The nightly (M, D) table, its unequal k and its non-Gaussian night-to-night deviations are kept as observed.
For each replicate, participants are resampled to size n, a true habit X_i is drawn from its conditional
distribution given the observed person mean (Gaussian working model), and a synthetic outcome with a
clock-only effect (beta_M = 0.008/h, outcome SD 0.113, real covariates) is generated. Naive person-mean
regression and ANCHOR are then fitted exactly as in the main analysis. Reports false gap-signal rates."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/anchor_work/anchor"))
import anchor as A
D = os.path.expanduser("~/studies/meal-timing-liver-fat/audit_20260918")
W = os.path.expanduser("~/anchor_work"); log = open(f"{W}/r12_full.txt", "w")
def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()
rng = np.random.default_rng(12)
P = pd.read_parquet(f"{D}/model_complete_cases.parquet").set_index("participant_id")
N = pd.read_parquet(f"{D}/paired_nights.parquet")
N = N[N.participant_id.isin(P.index)].dropna(subset=["M50", "D50"])
kc = N.calories_kcal; r = (4 * N.carbohydrate_g + 9 * N.lipid_g + 4 * N.protein_g) / kc
N = N[(kc >= 500) & (kc <= 5000) & (r >= .5) & (r <= 1.5)]
c = N.participant_id.value_counts(); N = N[N.participant_id.isin(c[c >= 2].index)].sort_values("participant_id")
pid = N.participant_id.to_numpy(); Wn = N[["M50", "D50"]].to_numpy(float)
uid, Wbar, k, Su, _ = A.person_summaries(pid, Wn)
Q = P.loc[uid]
C = pd.concat([Q[["age", "age2", "sex", "energy", "fat", "protein"]].astype(float),
               pd.get_dummies(Q["alcohol_current_frequency"].astype(str), drop_first=True, dtype=float)], axis=1).to_numpy(float)
ok = ~np.isnan(C).any(1); keep = set(uid[ok])
N = N[N.participant_id.isin(keep)]; pid = N.participant_id.to_numpy(); Wn = N[["M50", "D50"]].to_numpy(float)
uid, Wbar, k, Su, _ = A.person_summaries(pid, Wn); C = C[ok]
Sx = A._nearest_pd(np.cov(Wbar.T) - Su * np.mean(1 / k)); mu = Wbar.mean(0)
out("people", len(uid), "nights", len(pid), "Lambda", A.attribution_matrix(Sx, Su, k).round(3).tolist())
dev = Wn - Wbar[np.searchsorted(uid, pid)]
out("nightly deviation kurtosis (M, D):", np.round(((dev / dev.std(0)) ** 4).mean(0) - 3, 2).tolist())
rows = [np.flatnonzero(pid == u) for u in uid]
# conditional draw of the true habit given each person's mean (depends only on k)
post = {}
for kk in np.unique(k):
    G = Sx @ np.linalg.inv(Sx + Su / kk); post[kk] = (G, np.linalg.cholesky(A._nearest_pd(Sx - G @ Sx)))
age_z = (C[:, 0] - C[:, 0].mean()) / C[:, 0].std()
BETA = np.array([0.008, 0.0]); SDY = 0.113


def draw(n):
    pick = rng.integers(0, len(uid), n); r_ = np.concatenate([rows[i] for i in pick])
    npid = np.repeat(np.arange(n), [len(rows[i]) for i in pick])
    Wn_ = Wn[r_]; _, Wb, kb, Sub, _ = A.person_summaries(npid, Wn_)
    X = np.empty_like(Wb)
    for kk, (G, L) in post.items():
        m = kb == kk; X[m] = mu + (Wb[m] - mu) @ G.T + rng.standard_normal((m.sum(), 2)) @ L.T
    sig = X @ BETA + 0.02 * age_z[pick]
    y = sig + rng.normal(0, np.sqrt(SDY ** 2 - sig.var()), n)
    return npid, Wn_, Wb, kb, Sub, y, C[pick]


def anchor_ci(npid, Wn_, y, Cc, B):
    rr = np.split(np.arange(len(npid)), np.flatnonzero(np.diff(npid)) + 1); dr = []
    for _ in range(B):
        p2 = rng.integers(0, len(rr), len(rr)); r2 = np.concatenate([rr[i] for i in p2])
        q = np.repeat(np.arange(len(p2)), [len(rr[i]) for i in p2])
        _, Wb2, k2, Su2, _ = A.person_summaries(q, Wn_[r2]); dr.append(A.fit_anchor(Wb2, k2, Su2, y[p2], Cc[p2]))
    return np.percentile(np.array(dr), [2.5, 97.5], 0)


out("n,method,reps,false_gap_rate,clock_detect_rate,mean_gap_coef_x1e3")
for n, reps_naive, reps_anchor, B in [(2011, 1000, 300, 150), (8044, 1000, 300, 150), (32176, 1000, 100, 100), (128704, 400, 0, 0)]:
    acc = {"naive": [], "ANCHOR": []}
    for rep in range(reps_naive):
        npid, Wn_, Wb, kb, Sub, y, Cc = draw(n)
        b, V = A.fit_naive(Wb, y, Cc); se = np.sqrt(np.diag(V))
        acc["naive"].append((b, b - 1.96 * se, b + 1.96 * se))
        if rep < reps_anchor:
            ba = A.fit_anchor(Wb, kb, Sub, y, Cc); lo, hi = anchor_ci(npid, Wn_, y, Cc, B); acc["ANCHOR"].append((ba, lo, hi))
    for m, v in acc.items():
        if not v: continue
        Bc = np.array([x[0] for x in v]); LO = np.array([x[1] for x in v]); HI = np.array([x[2] for x in v])
        sig = (LO > 0) | (HI < 0)
        out(f"{n},{m},{len(v)},{sig[:, 1].mean():.3f},{sig[:, 0].mean():.3f},{1e3 * Bc[:, 1].mean():+.3f}")
log.close()
