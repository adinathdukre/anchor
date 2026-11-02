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

print("=== C. molecular scans ===")
from pheno_utils import PhenoLoader
def scan(ds, B=200):
    L = PhenoLoader(ds).dfs[ds].reset_index()
    if "research_stage" in L: L = L[L.research_stage == "00_00_visit"]
    L = L[L.participant_id.isin(P.index)].drop_duplicates("participant_id").set_index("participant_id")
    cols = [c for c in L.columns if pd.api.types.is_numeric_dtype(L[c]) and c not in ("array_index",)]
    Y = L[cols].astype(float); Y = Y.loc[:, (Y.notna().mean() >= .5) & (Y.std() > 0)]
    Y = np.log(Y.clip(lower=Y[Y > 0].min() / 2, axis=1)) if (Y.min() >= 0).all() else Y
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


print("=== E. matched windows ===")
Wm = pd.read_parquet(f"{D}/followup_20261001/matched_window_nights.parquet")
print(Wm.shape, list(Wm.columns))
print(open(f"{D}/followup_20261001/WINDOW_METHODS.txt").read()[:1500])
