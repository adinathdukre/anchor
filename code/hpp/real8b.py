"""Molecular anchor attribution at the blood draw (aggregate output only).
Per lipid: z(log lipid) ~ T (draw clock h) + F (h since last >=50 kcal occasion) + age + age2 + sex [+ habitual evening meal H]
Reparameterisation: L = T - F (clock of last meal). Reports FDR hits per anchor and which anchor dominates."""
import os
import numpy as np, pandas as pd
from pheno_utils import PhenoLoader
from math import erf
A = pd.read_parquet(os.path.expanduser("~/anchor_work/draw_anchors.parquet"))
A = A[A.F.between(0.25, 24)]
pl = PhenoLoader("serum_lipidomics"); Ld = pl.dfs["serum_lipidomics"].reset_index()
Ld = Ld[Ld.research_stage == "00_00_visit"].drop_duplicates("participant_id").set_index("participant_id")
ag = pl.dfs["age_sex"].reset_index(); ag = (ag[ag.research_stage == "00_00_visit"] if "research_stage" in ag else ag).drop_duplicates("participant_id").set_index("participant_id")
print("age_sex cols", list(ag.columns)[:8])
agecol, sexcol = "age", "sex"
ids = A.participant_id[A.participant_id.isin(Ld.index) & A.participant_id.isin(ag.index)]
A = A.set_index("participant_id").loc[ids]
feat = [c for c in Ld.columns if pd.api.types.is_numeric_dtype(Ld[c]) and c not in ("array_index",)]
Y = Ld.loc[ids, feat].astype(float); Y = Y.loc[:, (Y.notna().mean() >= .5) & (Y.std() > 0)]
Y = np.log(Y.clip(lower=Y[Y > 0].min() / 2, axis=1)); Y = Y.fillna(Y.median()); Y = (Y - Y.mean()) / Y.std()
age = pd.to_numeric(ag.loc[ids, agecol], errors="coerce"); sex = pd.to_numeric(ag.loc[ids, sexcol], errors="coerce")
base = pd.DataFrame({"age": age, "age2": age ** 2, "sex": sex}, index=ids)
ok = base.notna().all(1).to_numpy()
print("n", ok.sum(), "features", Y.shape[1], "T median", round(A["T"].median(), 2), "F median", round(A.F.median(), 2), "corr(T,F)", round(A[["T", "F"]].corr().iloc[0, 1], 3))
pz = np.vectorize(lambda z: 2 * (1 - 0.5 * (1 + erf(abs(z) / np.sqrt(2)))))
def bh(p):
    o = np.argsort(p); q = p[o] * len(p) / (np.arange(len(p)) + 1); q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty_like(q); out[o] = np.minimum(q, 1); return out
def fit(X, Ym):
    X1 = np.column_stack([np.ones(len(X)), X]); B = np.linalg.lstsq(X1, Ym, rcond=None)[0]; R = Ym - X1 @ B
    XtXi = np.linalg.inv(X1.T @ X1); h = np.einsum("ij,jk,ik->i", X1, XtXi, X1)
    W = (R / (1 - h)[:, None]) ** 2
    se = np.sqrt(np.einsum("jk,ik,il,lj->ji", XtXi, X1, X1, XtXi).T @ W) if False else None
    V = np.array([XtXi @ (X1.T * W[:, f]) @ X1 @ XtXi for f in range(Ym.shape[1])])  # HC3 per feature
    return B, np.sqrt(np.array([np.diag(v) for v in V])).T
Ym = Y.to_numpy(float)[ok]; Bz = base.to_numpy(float)[ok]
for name, cols in [("T + F (draw clock, fasting)", ["T", "F"]), ("L + F (meal clock, fasting)", ["L", "F"]), ("T alone", ["T"]), ("F alone", ["F"])]:
    X = np.column_stack([A[cols].to_numpy(float)[ok], Bz]); B, SE = fit(X, Ym)
    msg = []
    for j, c in enumerate(cols):
        q = bh(pz(B[1 + j] / SE[1 + j])); top = np.argsort(q)[:4]
        msg.append(f"{c}: FDR<.05 {int((q < .05).sum())}/{len(q)} top {[(Y.columns[t][:16], round(B[1+j,t],3)) for t in top]}")
    print(f"[{name}]"); [print("   ", m) for m in msg]
# dominant anchor in joint T+F model
X = np.column_stack([A[["T", "F"]].to_numpy(float)[ok], Bz]); B, SE = fit(X, Ym)
qT, qF = bh(pz(B[1] / SE[1])), bh(pz(B[2] / SE[2]))
print("joint T+F: T-only sig", int(((qT < .05) & (qF >= .05)).sum()), "F-only sig", int(((qF < .05) & (qT >= .05)).sum()), "both", int(((qF < .05) & (qT < .05)).sum()))
