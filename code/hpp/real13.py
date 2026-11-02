"""Real-data run 13: checks of assumption A2 (i.i.d. nightly error independent of k), aggregate output only.
A. Semivariogram of nightly anchors by day lag: under i.i.d. nights, E[(W_j - W_l)^2]/2 = Sigma_U for every lag.
B. Sigma_U and Lambda from non-adjacent night pairs only (lag >= 4 days) and after removing weekend effects.
C. Selection: do person means differ between participants with 2 and 3 qualifying nights?"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/anchor_work/anchor"))
import anchor as A
D = os.path.expanduser("~/studies/meal-timing-liver-fat/audit_20260918")
W = os.path.expanduser("~/anchor_work"); log = open(f"{W}/r13_full.txt", "w")
def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()
P = pd.read_parquet(f"{D}/model_complete_cases.parquet").set_index("participant_id")
N = pd.read_parquet(f"{D}/paired_nights.parquet")
out("night columns:", list(N.columns))
N = N[N.participant_id.isin(P.index)].dropna(subset=["M50", "D50"])
kc = N.calories_kcal; r = (4 * N.carbohydrate_g + 9 * N.lipid_g + 4 * N.protein_g) / kc
N = N[(kc >= 500) & (kc <= 5000) & (r >= .5) & (r <= 1.5)]
c = N.participant_id.value_counts(); N = N[N.participant_id.isin(c[c >= 2].index)].copy()
dcols = [x for x in N.columns if pd.api.types.is_datetime64_any_dtype(N[x])] + [x for x in N.columns if any(t in x.lower() for t in ("date", "night", "day")) and not pd.api.types.is_datetime64_any_dtype(N[x])]
dcol = None
for x in dcols:
    t = pd.to_datetime(N[x], errors="coerce")
    if t.notna().mean() > 0.95 and t.dt.normalize().nunique() > 30:
        t = t.dt.tz_localize(None) if t.dt.tz is not None else t
        dcol = x; N["_d"] = (t - pd.Timedelta(hours=12)) if x == "onset" else t; break  # night of a sleep onset = date of (onset - 12 h)
out("date column used:", dcol)
pid = N.participant_id.to_numpy(); Wn = N[["M50", "D50"]].to_numpy(float)
uid, Wbar, k, Su, _ = A.person_summaries(pid, Wn)
Sx = A._nearest_pd(np.cov(Wbar.T) - Su * np.mean(1 / k))
out("base: people", len(uid), "nights", len(pid), "SigmaU", Su.round(3).tolist(), "Lambda", A.attribution_matrix(Sx, Su, k).round(3).tolist())
if dcol is not None:
    N = N.sort_values(["participant_id", "_d"])
    rows = []
    for _, g in N.groupby("participant_id"):
        w = g[["M50", "D50"]].to_numpy(float); d = g["_d"].dt.normalize().to_numpy()
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                lag = abs((d[b] - d[a]) / np.timedelta64(1, "D"))
                rows.append((lag, *(w[a] - w[b])))
    R = pd.DataFrame(rows, columns=["lag", "dM", "dD"])
    R["bin"] = pd.cut(R.lag, [0, 1.5, 3.5, 7.5, 14.5, 1e4], labels=["1", "2-3", "4-7", "8-14", ">14"])
    out("A. semivariogram (half mean squared difference) by lag bin [pairs, M, D, cov MD]:")
    for b, g in R.groupby("bin", observed=True):
        out(f"   lag {b}: {len(g)}  {0.5 * (g.dM ** 2).mean():.3f}  {0.5 * (g.dD ** 2).mean():.3f}  {0.5 * (g.dM * g.dD).mean():.3f}")
    out("   lag 0 pairs (same date):", int((R.lag < 0.5).sum()))
    far = R[R.lag >= 3.5]
    Su_far = np.array([[0.5 * (far.dM ** 2).mean(), 0.5 * (far.dM * far.dD).mean()], [0.5 * (far.dM * far.dD).mean(), 0.5 * (far.dD ** 2).mean()]])
    Sx_far = A._nearest_pd(np.cov(Wbar.T) - Su_far * np.mean(1 / k))
    out("B1. SigmaU from lag>=4 pairs", Su_far.round(3).tolist(), "Lambda", A.attribution_matrix(Sx_far, Su_far, k).round(3).tolist(), "pairs", len(far))
    Wn_s = N[["M50", "D50"]].to_numpy(float); pid_s = N.participant_id.to_numpy()
    wk_s = N["_d"].dt.dayofweek.isin([3, 4]).astype(float).to_numpy()  # Thu/Fri evenings, before the Israeli weekend
    out("   share of Thu/Fri evenings:", round(wk_s.mean(), 3))
    # within-person weekend effect, then remove it from nights
    dev = Wn_s - pd.DataFrame(Wn_s).groupby(pid_s).transform("mean").to_numpy()
    wdev = wk_s - pd.Series(wk_s).groupby(pid_s).transform("mean").to_numpy()
    g_wk = (wdev @ dev) / (wdev @ wdev)
    out("B2. within-person weekend shift (h) M, D:", np.round(g_wk, 3).tolist())
    _, Wb2, k2, Su2, _ = A.person_summaries(pid_s, Wn_s - np.outer(wk_s, g_wk))
    Sx2 = A._nearest_pd(np.cov(Wb2.T) - Su2 * np.mean(1 / k2))
    out("    SigmaU weekend-adjusted", Su2.round(3).tolist(), "Lambda", A.attribution_matrix(Sx2, Su2, k2).round(3).tolist())
    # D. correlation of within-person deviations: adjacent (1 day) vs non-adjacent pairs, against the i.i.d.
    #    benchmark -1/(k-1) implied by deviations from a k-night mean
    dv = Wn_s - pd.DataFrame(Wn_s).groupby(pid_s).transform("mean").to_numpy()
    kk_s = pd.Series(pid_s).map(pd.Series(pid_s).value_counts()).to_numpy()
    rows2 = []
    for p_, g in pd.DataFrame({"p": pid_s, "d": N["_d"].dt.normalize().to_numpy(), "m": dv[:, 0], "dd": dv[:, 1], "k": kk_s}).groupby("p"):
        v = g.to_numpy()
        for a_ in range(len(v)):
            for b_ in range(a_ + 1, len(v)):
                lag = abs((v[b_][1] - v[a_][1]) / np.timedelta64(1, "D"))
                rows2.append((lag, v[a_][2] * v[b_][2], v[a_][2] ** 2 + v[b_][2] ** 2, v[a_][3] * v[b_][3], v[a_][3] ** 2 + v[b_][3] ** 2, v[a_][4]))
    Q2 = pd.DataFrame(rows2, columns=["lag", "pm", "sm", "pd", "sd", "k"])
    for nm, sel in (("adjacent (1 day)", Q2.lag < 1.5), ("non-adjacent (>=2 days)", Q2.lag >= 1.5)):
        g = Q2[sel]
        out(f"D. deviation correlation {nm}: pairs {len(g)}  M {2 * g.pm.sum() / g.sm.sum():.3f}  D {2 * g.pd.sum() / g.sd.sum():.3f}  iid benchmark {np.mean(-1 / (g.k - 1)):.3f}")
m2, m3 = Wbar[k == 2], Wbar[k == 3]
for j, nm in enumerate(["M", "D"]):
    t = (m3[:, j].mean() - m2[:, j].mean()) / np.sqrt(m3[:, j].var() / len(m3) + m2[:, j].var() / len(m2))
    out(f"C. person mean {nm}: k=2 {m2[:, j].mean():.3f}  k=3 {m3[:, j].mean():.3f}  t={t:.2f}")
# E. does the number of qualifying nights depend on participant characteristics?
Q = P.loc[uid]
X = pd.DataFrame({"age": pd.to_numeric(Q["age"], errors="coerce"), "sex": pd.to_numeric(Q["sex"], errors="coerce"),
                  "M_mean": Wbar[:, 0], "D_mean": Wbar[:, 1]}, index=uid)
bcol = [c for c in P.columns if c.lower() in ("bmi", "body_mass_index")]
if bcol: X["bmi"] = pd.to_numeric(Q[bcol[0]], errors="coerce")
ok = X.notna().all(1).to_numpy(); Z = (X[ok] - X[ok].mean()) / X[ok].std(); yk = (k[ok] == 3).astype(float)
A1 = np.column_stack([np.ones(ok.sum()), Z.to_numpy()]); bk = np.linalg.lstsq(A1, yk, rcond=None)[0]
r_ = yk - A1 @ bk; XtXi = np.linalg.inv(A1.T @ A1); se = np.sqrt(np.diag(XtXi @ (A1.T * r_ ** 2) @ A1 @ XtXi))
out("E. linear probability of k=3 (per SD; HC0 SE):", {c: f"{bk[i + 1]:+.3f} ({se[i + 1]:.3f})" for i, c in enumerate(Z.columns)}, "n", int(ok.sum()))
log.close()
