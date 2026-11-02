"""Real-data run 11: follow-up visit (02_00_visit) replication — aggregate output only.
1. Error structure at follow-up vs baseline (Sigma_U, Sigma_X, ICC, Lambda) -> transportability.
2. Liver: naive vs ANCHOR at follow-up (1000-draw person bootstrap); ANCHOR with baseline Sigma_U (transported).
3. Change-on-change for people with both visits: delta liver ~ delta habits, naive vs calibrated
   (error of a difference of means has covariance Sigma_U (1/k1 + 1/k2)). Removes time-invariant confounders."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/anchor_work/anchor"))
import anchor as A
D = os.path.expanduser("~/studies/meal-timing-liver-fat/audit_20260918")
rng = np.random.default_rng(11)
def nut_ok(N):
    kc = N.calories_kcal; r = (4 * N.carbohydrate_g + 9 * N.lipid_g + 4 * N.protein_g) / kc
    return N[(kc >= 500) & (kc <= 5000) & (r >= .5) & (r <= 1.5)]
def nights(path, ids=None):
    N = pd.read_parquet(path).dropna(subset=["M50", "D50"]); N = nut_ok(N)
    if ids is not None: N = N[N.participant_id.isin(ids)]
    c = N.participant_id.value_counts(); N = N[N.participant_id.isin(c[c >= 2].index)].sort_values("participant_id")
    return N
def structure(N, lab):
    pid = N.participant_id.to_numpy(); Wn = N[["M50", "D50"]].to_numpy(float)
    uid, Wb, k, Su, _ = A.person_summaries(pid, Wn)
    Sx = A._nearest_pd(np.cov(Wb.T) - Su * np.mean(1 / k)); lam = A.attribution_matrix(Sx, Su, k)
    icc = np.diag(Sx) / (np.diag(Sx) + np.diag(Su))
    print(f"{lab}: people {len(uid)} nights {len(pid)} ICC {icc.round(3)} within-corr {Su[0,1]/np.sqrt(Su[0,0]*Su[1,1]):.3f} habit-corr {Sx[0,1]/np.sqrt(Sx[0,0]*Sx[1,1]):.3f} Lambda {lam.round(3).tolist()}")
    return pid, Wn, uid, Wb, k, Su
# ---- 1. structure
Pb = pd.read_parquet(f"{D}/model_complete_cases.parquet").set_index("participant_id")
Nb = nights(f"{D}/paired_nights.parquet", set(Pb.index)); _, _, uidb, Wbb, kb, Sub = structure(Nb, "baseline")
Pf = pd.read_parquet(f"{D}/followup_02/alcohol_resolved_cohort.parquet").drop_duplicates("participant_id").set_index("participant_id")
print("follow-up cohort cols:", [c for c in Pf.columns if c in ("age", "age2", "sex", "energy", "fat", "protein", "fat_pct", "protein_pct", "alcohol_current_frequency", "age_at_research_stage", "calories_kcal")])
Nf = nights(f"{D}/followup_02/paired_nights.parquet", set(Pf.index)); pidf, Wnf, uidf, Wbf, kf, Suf = structure(Nf, "follow-up")
print("Sigma_U baseline", Sub.round(3).tolist(), "follow-up", Suf.round(3).tolist())
# ---- 2. liver at follow-up
Q = Pf.loc[uidf].copy()
age = pd.to_numeric(Q.get("age", Q.get("age_at_research_stage")), errors="coerce")
energy = Q["energy"] if "energy" in Q else Nf.groupby("participant_id").calories_kcal.mean().loc[uidf]
fat = Q["fat"] if "fat" in Q else Nf.groupby("participant_id").fat_pct.mean().loc[uidf] if "fat_pct" in Nf else Q["fat_pct"]
prot = Q["protein"] if "protein" in Q else Nf.groupby("participant_id").protein_pct.mean().loc[uidf] if "protein_pct" in Nf else Q["protein_pct"]
Cdf = pd.concat([pd.DataFrame({"age": age, "age2": age ** 2, "sex": pd.to_numeric(Q.sex, errors="coerce"), "energy": energy, "fat": fat, "protein": prot}, index=uidf),
                 pd.get_dummies(Q["alcohol_current_frequency"].astype(str), drop_first=True, dtype=float)], axis=1)
y = pd.to_numeric(Q.attenuation, errors="coerce")
ok = (Cdf.notna().all(1) & y.notna()).to_numpy(); ids = uidf[ok]
m = np.isin(pidf, ids); pid2, Wn2 = pidf[m], Wnf[m]; _, Wb2, k2, Su2, _ = A.person_summaries(pid2, Wn2)
C = Cdf.loc[ids].to_numpy(float); yv = y.loc[ids].to_numpy(float)
bn, V = A.fit_naive(Wb2, yv, C); se = np.sqrt(np.diag(V))
ba = A.fit_anchor(Wb2, k2, Su2, yv, C); dr = A.person_bootstrap(pid2, Wn2, yv, C, B=1000, seed=3); lo, hi = np.percentile(dr, [2.5, 97.5], 0)
bt = A.fit_anchor(Wb2, k2, Sub, yv, C)
print(f"liver follow-up n={len(ids)} (mean att {yv.mean():.3f} SD {yv.std():.3f})")
print(f"  naive  clock {bn[0]:+.5f} ({bn[0]-1.96*se[0]:+.5f},{bn[0]+1.96*se[0]:+.5f}) gap {bn[1]:+.5f} ({bn[1]-1.96*se[1]:+.5f},{bn[1]+1.96*se[1]:+.5f})")
print(f"  ANCHOR clock {ba[0]:+.5f} ({lo[0]:+.5f},{hi[0]:+.5f}) gap {ba[1]:+.5f} ({lo[1]:+.5f},{hi[1]:+.5f})  p_clock={2*min((dr[:,0]>0).mean(),(dr[:,0]<0).mean()):.3f}")
print(f"  ANCHOR with transported baseline Sigma_U: clock {bt[0]:+.5f} gap {bt[1]:+.5f}")
# ---- 3. change-on-change (both visits)
both = np.intersect1d(uidb, ids)
ib = pd.Index(uidb).get_indexer(both); i2 = pd.Index(ids).get_indexer(both)
dW = Wb2[i2] - Wbb[ib]; dk = 1 / kb[ib] + 1 / k2[i2]
dy = yv[i2] - Pb.loc[both, "attenuation"].to_numpy(float)
dE = C[i2][:, 3] - Pb.loc[both, "energy"].to_numpy(float)
Su_pool = (Sub + Su2) / 2
def change_fit(dW, dk, dy, dE):
    Cc = np.column_stack([dE])
    bn = A.fit_naive(dW, dy, Cc, robust=False)[0]
    Sd = A._nearest_pd(np.cov(dW.T) - Su_pool * np.mean(dk)); mu = dW.mean(0)
    Xh = np.array([mu + Sd @ np.linalg.inv(Sd + Su_pool * d) @ (w - mu) for w, d in zip(dW, dk)])
    ba = np.linalg.lstsq(np.column_stack([np.ones(len(dy)), Xh, Cc]), dy, rcond=None)[0][1:3]
    return bn, ba
bn_c, ba_c = change_fit(dW, dk, dy, dE)
B = []
for _ in range(1000):
    s = rng.integers(0, len(both), len(both)); B.append(np.concatenate(change_fit(dW[s], dk[s], dy[s], dE[s])))
B = np.array(B); q = np.percentile(B, [2.5, 97.5], 0)
rel = np.diag(A._nearest_pd(np.cov(dW.T) - Su_pool * np.mean(dk))) / np.diag(np.cov(dW.T))
print(f"change-on-change n={len(both)} (both visits); reliability of change in M, D: {rel.round(3)}; corr(dM,dD) {np.corrcoef(dW.T)[0,1]:.3f}")
print(f"  naive  d-clock {bn_c[0]:+.5f} ({q[0,0]:+.5f},{q[1,0]:+.5f}) d-gap {bn_c[1]:+.5f} ({q[0,1]:+.5f},{q[1,1]:+.5f})")
print(f"  ANCHOR d-clock {ba_c[0]:+.5f} ({q[0,2]:+.5f},{q[1,2]:+.5f}) d-gap {ba_c[1]:+.5f} ({q[0,3]:+.5f},{q[1,3]:+.5f})")
