"""Real-data run 16 (aggregate output only), for the sixth review.
A. Liver: ANCHOR (i.i.d. and dated) with the night-count indicator added to the covariates, so habits may
   differ by k; 500-draw participant bootstrap each.
B. Early-sleep glucose: habitual vs acute decomposition with night-count indicators in the covariates.
C. Night-to-night decay fitted separately for M and D (checks the shared-decay assumption)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/anchor_work/anchor"))
import anchor as A
D = os.path.expanduser("~/studies/meal-timing-liver-fat/audit_20260918")
W = os.path.expanduser("~/anchor_work"); log = open(f"{W}/r16_full.txt", "w")
def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()
rng = np.random.default_rng(16)
P = pd.read_parquet(f"{D}/model_complete_cases.parquet").set_index("participant_id")
def ci(v):
    lo, hi = np.percentile(v, [2.5, 97.5]); return f"({lo:+.5f},{hi:+.5f}) p={2 * min((v > 0).mean(), (v < 0).mean()):.3f}"

# ---------- A. liver
N = pd.read_parquet(f"{D}/paired_nights.parquet")
N = N[N.participant_id.isin(P.index)].dropna(subset=["M50", "D50", "onset"])
kc = N.calories_kcal; r = (4 * N.carbohydrate_g + 9 * N.lipid_g + 4 * N.protein_g) / kc
N = N[(kc >= 500) & (kc <= 5000) & (r >= .5) & (r <= 1.5)]
t = pd.to_datetime(N["onset"]); t = t.dt.tz_localize(None) if t.dt.tz is not None else t
N = N.assign(_t=((t - pd.Timedelta(hours=12)).dt.normalize() - pd.Timestamp("2018-01-01")).dt.days.astype(float))
Cl = pd.concat([P[["age", "age2", "sex", "energy", "fat", "protein"]].astype(float),
                pd.get_dummies(P["alcohol_current_frequency"].astype(str), drop_first=True, dtype=float)], axis=1)
yl = P["attenuation"].astype(float); okp = Cl.notna().all(1) & yl.notna()
c = N.participant_id.value_counts(); N = N[N.participant_id.isin(c[c >= 2].index) & N.participant_id.isin(okp[okp].index)].sort_values(["participant_id", "_t"])
pid = N.participant_id.to_numpy(); Wn = N[["M50", "D50"]].to_numpy(float); T = N["_t"].to_numpy()
uid, Wbar, k, Su, _ = A.person_summaries(pid, Wn)
C0 = Cl.loc[uid].to_numpy(float); y = yl.loc[uid].to_numpy(float)
Ck = np.column_stack([C0, (k == 3).astype(float)])
out(f"A. liver: people {len(uid)}, share k=3 {np.mean(k == 3):.3f}")
b0 = A.fit_anchor(Wbar, k, Su, y, C0); bk = A.fit_anchor(Wbar, k, Su, y, Ck)
bdk, phik = A.fit_anchor_dated(pid, Wn, T, y, Ck)
rows = np.split(np.arange(len(pid)), np.flatnonzero(pid[1:] != pid[:-1]) + 1); dk, dd = [], []
for _ in range(500):
    pick = rng.integers(0, len(rows), len(rows)); rr = np.concatenate([rows[i] for i in pick])
    q = np.repeat(np.arange(len(pick)), [len(rows[i]) for i in pick])
    _, Wb2, k2, Su2, _ = A.person_summaries(q, Wn[rr]); Ck2 = np.column_stack([C0[pick], (k2 == 3).astype(float)])
    dk.append(A.fit_anchor(Wb2, k2, Su2, y[pick], Ck2)); dd.append(A.fit_anchor_dated(q, Wn[rr], T[rr], y[pick], Ck2)[0])
dk, dd = np.array(dk), np.array(dd)
out(f"   ANCHOR (no k term):        clock {b0[0]:+.5f}  gap {b0[1]:+.5f}")
out(f"   ANCHOR + k indicator:      clock {bk[0]:+.5f} {ci(dk[:, 0])}  gap {bk[1]:+.5f} {ci(dk[:, 1])}")
out(f"   dated ANCHOR + k indicator: clock {bdk[0]:+.5f} {ci(dd[:, 0])}  gap {bdk[1]:+.5f} {ci(dd[:, 1])}  phi {phik:.3f}")

# ---------- C. separate decay for M and D
for j, nm in enumerate(["M", "D"]):
    ph, s1 = A.night_correlation(pid, Wn[:, [j]], T)
    out(f"C. decay fitted for {nm} alone: phi {ph:.3f}, sill {s1[0, 0]:.3f}")
ph2, _ = A.night_correlation(pid, Wn, T); out(f"   shared phi (both anchors): {ph2:.3f}")

# ---------- B. glucose with night-count indicators
G = pd.read_parquet(f"{D}/cgm_qualified_nights.parquet")
G = G[G.participant_id.isin(P.index)].dropna(subset=["M50", "D50", "glucose_3h"])
c = G.participant_id.value_counts(); G = G[G.participant_id.isin(c[c >= 2].index)].sort_values("participant_id").reset_index(drop=True)
gp = G.participant_id.to_numpy(); Gw = G[["M50", "D50"]].to_numpy(float); gy = G["glucose_3h"].to_numpy(float)
Gn = G[["calories_kcal", "fat_pct", "protein_pct"]].to_numpy(float)
gu = np.unique(gp); Q = P.loc[gu]
Cg0 = pd.concat([Q[["age", "age2", "sex"]].astype(float), pd.get_dummies(Q["alcohol_current_frequency"].astype(str), drop_first=True, dtype=float)], axis=1).to_numpy(float)
def kdum(pp):
    kk = np.bincount(np.searchsorted(np.unique(pp), pp)); return np.column_stack([(kk == 3), (kk >= 4)]).astype(float)
grows = [np.flatnonzero(gp == u) for u in gu]
def fitg(pp, rr, ci_, withk):
    Cg = np.column_stack([Cg0[ci_], kdum(pp)]) if withk else Cg0[ci_]
    return A.fit_anchor_nightly(pp, Gw[rr], gy[rr], Cg, Gn[rr])
for withk in (False, True):
    e = fitg(gp, np.arange(len(gp)), np.arange(len(gu)), withk); dr = []
    for _ in range(300):
        pick = rng.integers(0, len(gu), len(gu)); rr = np.concatenate([grows[i] for i in pick])
        pp = np.repeat(np.arange(len(pick)), [len(grows[i]) for i in pick]); rres = fitg(pp, rr, pick, withk)
        dr.append(np.concatenate([rres["b_h"], rres["b_a"]]))
    dr = np.array(dr)
    out(f"B. glucose {'+ k indicators' if withk else '(no k term)  '}: habitual clock {e['b_h'][0]:+.2f} {ci(dr[:, 0])}  habitual gap {e['b_h'][1]:+.2f} {ci(dr[:, 1])}  acute gap {e['b_a'][1]:+.2f}")
log.close()
