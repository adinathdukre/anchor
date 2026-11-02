"""Real-data run 15: dated-night ANCHOR on the liver cohort, aggregate output only.
Estimates the night-to-night correlation rho(lag) = phi**lag (days) and the error covariance Sigma_U from the
within-person semivariogram of the dated nights, the per-person effective number of nights, the corrected
attribution matrix, and the liver estimate under the estimated rho, with a 1,000-draw participant bootstrap
that re-estimates phi in every draw. Compares with i.i.d. ANCHOR."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/anchor_work/anchor"))
import anchor as A
D = os.path.expanduser("~/studies/meal-timing-liver-fat/audit_20260918")
W = os.path.expanduser("~/anchor_work"); log = open(f"{W}/r15_full.txt", "w")
def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()
rng = np.random.default_rng(15)
P = pd.read_parquet(f"{D}/model_complete_cases.parquet").set_index("participant_id")
N = pd.read_parquet(f"{D}/paired_nights.parquet")
N = N[N.participant_id.isin(P.index)].dropna(subset=["M50", "D50"])
kc = N.calories_kcal; r = (4 * N.carbohydrate_g + 9 * N.lipid_g + 4 * N.protein_g) / kc
N = N[(kc >= 500) & (kc <= 5000) & (r >= .5) & (r <= 1.5)]
c = N.participant_id.value_counts(); N = N[N.participant_id.isin(c[c >= 2].index)].copy()
dcol = None
for x in [x for x in N.columns if pd.api.types.is_datetime64_any_dtype(N[x])] + [x for x in N.columns if any(t in x.lower() for t in ("date", "night", "day"))]:
    t = pd.to_datetime(N[x], errors="coerce")
    if t.notna().mean() > 0.95 and t.dt.normalize().nunique() > 30:
        t = t.dt.tz_localize(None) if t.dt.tz is not None else t
        t = (t - pd.Timedelta(hours=12)) if x == "onset" else t  # night of a sleep onset = date of (onset - 12 h)
        dcol = x; N["_t"] = (t.dt.normalize() - pd.Timestamp("2018-01-01")).dt.days.astype(float); break
out("date column:", dcol)
if dcol is None:
    log.close(); sys.exit()
Q = P.copy()
C_all = pd.concat([Q[["age", "age2", "sex", "energy", "fat", "protein"]].astype(float),
                   pd.get_dummies(Q["alcohol_current_frequency"].astype(str), drop_first=True, dtype=float)], axis=1)
y_all = Q["attenuation"].astype(float)
okp = C_all.notna().all(1) & y_all.notna()
N = N[N.participant_id.isin(okp[okp].index)].sort_values(["participant_id", "_t"])
pid = N.participant_id.to_numpy(); Wn = N[["M50", "D50"]].to_numpy(float); T = N["_t"].to_numpy()
uid, Wbar, k, Su_iid, _ = A.person_summaries(pid, Wn)
C = C_all.loc[uid].to_numpy(float); y = y_all.loc[uid].to_numpy(float)
phi, Su = A.night_correlation(pid, Wn, T)
keff = A.effective_k(pid, T, phi)
lag = np.concatenate([np.abs(np.diff(T[pid == u])) for u in uid])
out(f"people {len(uid)} nights {len(pid)}; adjacent-night day gaps median {np.median(lag):.0f}, IQR {np.percentile(lag, 25):.0f}-{np.percentile(lag, 75):.0f}")
out(f"phi (per day) = {phi:.3f}; implied rho at 1, 2, 7 days = {phi:.3f}, {phi ** 2:.3f}, {phi ** 7:.3f}")
out("SigmaU iid", Su_iid.round(3).tolist(), " SigmaU sill", Su.round(3).tolist(), f" mean k {k.mean():.2f} mean k_eff {keff.mean():.2f}")
Sx_iid = A._nearest_pd(np.cov(Wbar.T) - Su_iid * np.mean(1 / k)); Sx_d = A._nearest_pd(np.cov(Wbar.T) - Su * np.mean(1 / keff))
out("Lambda iid", A.attribution_matrix(Sx_iid, Su_iid, k).round(3).tolist(), " Lambda dated", A.attribution_matrix(Sx_d, Su, keff).round(3).tolist())
b_iid = A.fit_anchor(Wbar, k, Su_iid, y, C); b_d, _ = A.fit_anchor_dated(pid, Wn, T, y, C)
rows = np.split(np.arange(len(pid)), np.flatnonzero(pid[1:] != pid[:-1]) + 1); dr = []; ph = []
for _ in range(int(sys.argv[1]) if len(sys.argv) > 1 else 1000):
    pick = rng.integers(0, len(rows), len(rows)); rr = np.concatenate([rows[i] for i in pick])
    q = np.repeat(np.arange(len(pick)), [len(rows[i]) for i in pick])
    bb, pp = A.fit_anchor_dated(q, Wn[rr], T[rr], y[pick], C[pick]); dr.append(bb); ph.append(pp)
dr = np.array(dr); lo, hi = np.percentile(dr, [2.5, 97.5], 0)
out(f"liver ANCHOR iid: clock {b_iid[0]:+.5f}  gap {b_iid[1]:+.5f}")
out(f"liver ANCHOR dated: clock {b_d[0]:+.5f} ({lo[0]:+.5f},{hi[0]:+.5f})  gap {b_d[1]:+.5f} ({lo[1]:+.5f},{hi[1]:+.5f}); "
    f"clock p {2 * min((dr[:, 0] > 0).mean(), (dr[:, 0] < 0).mean()):.3f}; phi 95% CI {np.percentile(ph, 2.5):.2f}-{np.percentile(ph, 97.5):.2f}")
log.close()
