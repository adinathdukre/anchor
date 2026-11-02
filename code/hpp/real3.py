"""Real-data run 3: anchor-resolved serum lipidome scan, naive vs ANCHOR.
Shared person bootstrap (calibration re-estimated per draw) applied to all lipids at once.
Aggregate output only (counts of FDR hits, top lipids, attribution shifts)."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/anchor_work/anchor"))
import anchor as A
from pheno_utils import PhenoLoader

B = int(sys.argv[1]) if len(sys.argv) > 1 else 300
D = os.path.expanduser("~/studies/meal-timing-liver-fat/audit_20260918")
P = pd.read_parquet(f"{D}/model_complete_cases.parquet").set_index("participant_id")
L = PhenoLoader("serum_lipidomics").dfs["serum_lipidomics"].reset_index()
print("lipidomics index cols:", [c for c in L.columns if c in ("participant_id", "cohort", "research_stage", "array_index")])
if "research_stage" in L:
    print("stages:", L.research_stage.value_counts().head(5).to_dict())
    L = L[L.research_stage == "00_00_visit"]
L = L[L.participant_id.isin(P.index)].drop_duplicates("participant_id").set_index("participant_id")
lip = [c for c in L.columns if c not in ("collection_date", "collection_timestamp", "timezone", "cohort", "research_stage", "array_index")
       and pd.api.types.is_numeric_dtype(L[c])]
Y = L[lip].astype(float)
Y = Y.loc[:, Y.notna().mean() > 0.9]
Y = np.log(Y.clip(lower=Y[Y > 0].min().min() / 2))
Y = (Y - Y.mean()) / Y.std()
print("lipids", Y.shape)

N = pd.read_parquet(f"{D}/paired_nights.parquet")
N = N[N.participant_id.isin(Y.index)].dropna(subset=["M50", "D50"])
kcal = N.calories_kcal; ratio = (4 * N.carbohydrate_g + 9 * N.lipid_g + 4 * N.protein_g) / kcal
N = N[(kcal >= 500) & (kcal <= 5000) & (ratio >= .5) & (ratio <= 1.5)]
cnt = N.participant_id.value_counts(); N = N[N.participant_id.isin(cnt[cnt >= 2].index)]
pid = N.participant_id.to_numpy(); Wn = N[["M50", "D50"]].to_numpy(float)
uid, Wbar, k, Su, _ = A.person_summaries(pid, Wn)
Q = P.loc[uid]
C = pd.concat([Q[["age", "age2", "sex", "energy", "fat", "protein"]].astype(float),
               pd.get_dummies(Q["alcohol_current_frequency"].astype(str), drop_first=True, dtype=float)], axis=1).to_numpy(float)
Ym = Y.loc[uid].fillna(0).to_numpy(float)
print("n people", len(uid))

def coefs(X, C, Ym):
    Ad = np.column_stack([np.ones(len(X)), X, C])
    return np.linalg.lstsq(Ad, Ym, rcond=None)[0][1:3]  # 2 x lipids

def anchor_coefs(Wbar, k, Su, C, Ym):
    Xh, _ = A.calibrate(Wbar, k, Su, C)
    return coefs(Xh, C, Ym)

bn = coefs(Wbar, C, Ym); ba = anchor_coefs(Wbar, k, Su, C, Ym)
rng = np.random.default_rng(5)
ids_all = np.unique(pid); rows_of = {u: np.flatnonzero(pid == u) for u in ids_all}
pos = {u: i for i, u in enumerate(uid)}
dn, da = [], []
for b in range(B):
    pick = rng.choice(uid, len(uid))
    rows = np.concatenate([rows_of[u] for u in pick])
    npid = np.repeat(np.arange(len(pick)), [len(rows_of[u]) for u in pick])
    _, Wb, kb, Sub, _ = A.person_summaries(npid, Wn[rows])
    ii = [pos[u] for u in pick]
    dn.append(coefs(Wb, C[ii], Ym[ii])); da.append(anchor_coefs(Wb, kb, Sub, C[ii], Ym[ii]))
dn, da = np.array(dn), np.array(da)

def pvals(est, draws):
    se = draws.std(0); z = est / se
    from math import erf
    return np.vectorize(lambda t: 2 * (1 - 0.5 * (1 + erf(abs(t) / np.sqrt(2)))))(z), se

def bh(p):
    o = np.argsort(p); q = p[o] * len(p) / (np.arange(len(p)) + 1); q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q); out[o] = np.minimum(q, 1); return out

names = np.array(Y.columns)
for meth, est, dr in [("naive", bn, dn), ("ANCHOR", ba, da)]:
    p, se = pvals(est, dr)
    for j, lab in enumerate(["clock", "gap"]):
        q = bh(p[j]); hit = q < 0.05
        top = np.argsort(p[j])[:6]
        print(f"{meth:6s} {lab:5s} FDR<0.05 hits: {hit.sum():3d}  top:", [(names[t][:22], round(est[j, t], 3), f"{q[t]:.2g}") for t in top])
# attribution shift: lipids where naive and ANCHOR disagree on which anchor dominates
pn, _ = pvals(bn, dn); pa, _ = pvals(ba, da)
qn = np.vstack([bh(pn[0]), bh(pn[1])]); qa = np.vstack([bh(pa[0]), bh(pa[1])])
dom_n = np.where(qn.min(0) < .05, np.argmin(qn, 0), -1); dom_a = np.where(qa.min(0) < .05, np.argmin(qa, 0), -1)
print("dominant anchor (clock=0, gap=1, none=-1) cross-tab naive x ANCHOR:")
print(pd.crosstab(dom_n, dom_a, rownames=["naive"], colnames=["ANCHOR"]))
print("median |ANCHOR|/|naive| ratio clock, gap:", np.round(np.median(np.abs(ba) / np.maximum(np.abs(bn), 1e-9), 1), 2))
