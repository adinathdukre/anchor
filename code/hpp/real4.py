"""Real-data run 4 (aggregate output only):
A. CGM acute (within-person) vs habitual (calibrated between-person) decomposition, joint bootstrap.
B. Liver ANCHOR robustness to extra adjustment sets (BMI, VAT, sleep hours) and >=3 nights.
C. Molecular scans: serum lipidomics (>=50% observed, median-imputed) and Nightingale NMR.
D. List follow-up folder (meal-aligned windows)."""
import sys, os, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/anchor_work/anchor"))
import anchor as A
from math import erf
D = os.path.expanduser("~/studies/meal-timing-liver-fat/audit_20260918")
P = pd.read_parquet(f"{D}/model_complete_cases.parquet").set_index("participant_id")
rng = np.random.default_rng(3)
pz = lambda z: 2 * (1 - 0.5 * (1 + erf(abs(z) / np.sqrt(2))))

def nights_for(ids):
    N = pd.read_parquet(f"{D}/paired_nights.parquet")
    N = N[N.participant_id.isin(ids)].dropna(subset=["M50", "D50"])
    kcal = N.calories_kcal; r = (4 * N.carbohydrate_g + 9 * N.lipid_g + 4 * N.protein_g) / kcal
    N = N[(kcal >= 500) & (kcal <= 5000) & (r >= .5) & (r <= 1.5)]
    c = N.participant_id.value_counts(); return N[N.participant_id.isin(c[c >= 2].index)]

def cov_block(Q, extra=()):
    cols = ["age", "age2", "sex", "energy", "fat", "protein"] + list(extra)
    return pd.concat([Q[cols].astype(float), pd.get_dummies(Q["alcohol_current_frequency"].astype(str), drop_first=True, dtype=float)], axis=1)

def boot_pairs(pid, Wn, uid, f, B):
    rows_of = {u: np.flatnonzero(pid == u) for u in uid}; out = []
    for _ in range(B):
        pick = rng.choice(uid, len(uid)); rows = np.concatenate([rows_of[u] for u in pick])
        npid = np.repeat(np.arange(len(pick)), [len(rows_of[u]) for u in pick])
        out.append(f(pick, npid, rows))
    return np.array(out)

print("=== A. CGM acute vs habitual ===")
G = pd.read_parquet(f"{D}/cgm_qualified_nights.parquet")
G = G[G.participant_id.isin(P.index)].dropna(subset=["M50", "D50", "glucose_3h"])
c = G.participant_id.value_counts(); G = G[G.participant_id.isin(c[c >= 2].index)].reset_index(drop=True)
gp = G.participant_id.to_numpy(); GW = G[["M50", "D50"]].to_numpy(float)
nut = ["calories_kcal", "fat_pct", "protein_pct"]
def cgm_fit(sub_pid, sub):
    uid, Wb, k, Su, _ = A.person_summaries(sub_pid, sub[["M50", "D50"]].to_numpy(float))
    g = sub.assign(pp=sub_pid)
    yb = g.groupby("pp").glucose_3h.mean().loc[uid].to_numpy(float)
    Q = P.loc[g.groupby("pp").participant_id.first().loc[uid]]
    C = pd.concat([Q[["age", "age2", "sex"]].astype(float).reset_index(drop=True),
                   g.groupby("pp")[nut].mean().loc[uid].reset_index(drop=True),
                   pd.get_dummies(Q["alcohol_current_frequency"].astype(str), drop_first=True, dtype=float).reset_index(drop=True)], axis=1).to_numpy(float)
    bh = A.fit_anchor(Wb, k, Su, yb, C); bn = A.fit_naive(Wb, yb, C, robust=False)[0]
    dm = lambda col: g[col] - g.groupby("pp")[col].transform("mean")
    X = np.column_stack([dm(v) for v in ["M50", "D50"] + nut]); yw = dm("glucose_3h").to_numpy(float)
    bw = np.linalg.lstsq(X, yw, rcond=None)[0][:2]
    return np.concatenate([bw, bn, bh])
est = cgm_fit(gp, G)
uidG = np.unique(gp)
dr = boot_pairs(gp, GW, uidG, lambda pick, npid, rows: cgm_fit(npid, G.iloc[rows].reset_index(drop=True)), 200)
lab = ["within clock", "within gap", "naive-between clock", "naive-between gap", "ANCHOR-between clock", "ANCHOR-between gap"]
for i, l in enumerate(lab):
    lo, hi = np.percentile(dr[:, i], [2.5, 97.5]); print(f"{l:22s} {est[i]:+.3f} ({lo:+.3f},{hi:+.3f})")
for nm, j in [("naive", 2), ("ANCHOR", 4)]:
    for a, w in [("clock", 0), ("gap", 1)]:
        d = dr[:, j + w] - dr[:, w]; e = est[j + w] - est[w]; lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"habitual-minus-acute {nm:6s} {a:5s} {e:+.3f} ({lo:+.3f},{hi:+.3f}) p={pz(e / d.std()):.3f}")

print("=== B. liver robustness ===")
N = nights_for(set(P.index)); pid = N.participant_id.to_numpy(); Wn = N[["M50", "D50"]].to_numpy(float)
for name, extra, mink in [("core", [], 2), ("+BMI", ["bmi"], 2), ("+VAT", ["total_scan_vat_area"], 2), ("+sleep hours", ["sleep_hours"], 2), (">=3 nights", [], 3)]:
    Q0 = P.copy()
    if "sleep_hours" in extra:
        Q0["sleep_hours"] = N.groupby("participant_id").sleep_hours.mean()
    Cdf = cov_block(Q0, extra); ok = Cdf.notna().all(1) & Q0.attenuation.notna()
    cnt = N.participant_id.value_counts(); ok &= Q0.index.isin(cnt[cnt >= mink].index)
    m = np.isin(pid, Q0.index[ok]); uid, Wb, k, Su, _ = A.person_summaries(pid[m], Wn[m])
    y = Q0.loc[uid, "attenuation"].to_numpy(float); C = Cdf.loc[uid].to_numpy(float)
    bn, V = A.fit_naive(Wb, y, C); se = np.sqrt(np.diag(V)); ba = A.fit_anchor(Wb, k, Su, y, C)
    d = A.person_bootstrap(pid[m], Wn[m], y, C, B=200, seed=9); lo, hi = np.percentile(d, [2.5, 97.5], 0)
    print(f"{name:13s} n={len(uid):4d} naive clock {bn[0]:+.4f} ({bn[0]-1.96*se[0]:+.4f},{bn[0]+1.96*se[0]:+.4f}) gap {bn[1]:+.4f} | ANCHOR clock {ba[0]:+.4f} ({lo[0]:+.4f},{hi[0]:+.4f}) gap {ba[1]:+.4f} ({lo[1]:+.4f},{hi[1]:+.4f})")

print("=== C. molecular scans ===")
from pheno_utils import PhenoLoader
def scan(ds, B=200):
    L = PhenoLoader(ds).dfs[ds].reset_index()
    if "research_stage" in L: L = L[L.research_stage == "00_00_visit"]
    L = L[L.participant_id.isin(P.index)].drop_duplicates("participant_id").set_index("participant_id")
    cols = [c for c in L.columns if pd.api.types.is_numeric_dtype(L[c]) and c not in ("array_index",)]
    Y = L[cols].astype(float); Y = Y.loc[:, (Y.notna().mean() >= .5) & (Y.std() > 0)]
    Y = np.log(Y.clip(lower=Y[Y > 0].min() / 2)) if (Y.min() >= 0).all() else Y
    Y = Y.fillna(Y.median()); Y = (Y - Y.mean()) / Y.std()
    N2 = nights_for(set(Y.index)); p2 = N2.participant_id.to_numpy(); W2 = N2[["M50", "D50"]].to_numpy(float)
    uid, Wb, k, Su, _ = A.person_summaries(p2, W2); C = cov_block(P.loc[uid]).to_numpy(float); Ym = Y.loc[uid].to_numpy(float)
    coef = lambda X, C, Ym: np.linalg.lstsq(np.column_stack([np.ones(len(X)), X, C]), Ym, rcond=None)[0][1:3]
    bn = coef(Wb, C, Ym); ba = coef(A.calibrate(Wb, k, Su, C)[0], C, Ym)
    rows_of = {u: np.flatnonzero(p2 == u) for u in uid}; pos = {u: i for i, u in enumerate(uid)}; dn, da = [], []
    for _ in range(B):
        pick = rng.choice(uid, len(uid)); rows = np.concatenate([rows_of[u] for u in pick])
        npid = np.repeat(np.arange(len(pick)), [len(rows_of[u]) for u in pick]); _, Wbb, kb, Sub, _ = A.person_summaries(npid, W2[rows])
        ii = [pos[u] for u in pick]; dn.append(coef(Wbb, C[ii], Ym[ii])); da.append(coef(A.calibrate(Wbb, kb, Sub, C[ii])[0], C[ii], Ym[ii]))
    dn, da = np.array(dn), np.array(da)
    def bh(p):
        o = np.argsort(p); q = p[o] * len(p) / (np.arange(len(p)) + 1); q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty_like(q); out[o] = np.minimum(q, 1); return out
    print(f"{ds}: n={len(uid)} features={Y.shape[1]}")
    for meth, e, d in [("naive", bn, dn), ("ANCHOR", ba, da)]:
        for j, a in enumerate(["clock", "gap"]):
            p = np.array([pz(z) for z in e[j] / d[:, j].std(0)]); q = bh(p); top = np.argsort(p)[:5]
            print(f"  {meth:6s} {a:5s} FDR<.05 {int((q<.05).sum()):3d} FDR<.10 {int((q<.1).sum()):3d} top", [(Y.columns[t][:20], round(e[j, t], 3), f"{q[t]:.2g}") for t in top])
for ds in ["serum_lipidomics", "nightingale_metabolomics"]:
    try: scan(ds)
    except Exception as e: print(ds, "ERR", repr(e)[:300])

print("=== D. follow-up folder ===")
for f in sorted(glob.glob(f"{D}/followup_2026*/*"))[:60]: print(" ", os.path.basename(f))
