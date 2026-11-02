"""Real-data run 9 (aggregate output only): molecular arm, strengthened.
A. Habitual evening-meal clock (H, ~14 logged evenings/person) -> lipidome, naive vs ANCHOR-calibrated (unequal k).
B. Joint habit + draw state: lipid ~ H_cal + T (draw clock) + F (h since last meal) + covariates.
C. Robustness of draw anchors: +BMI, +weekday/month, log(F), F>=2h subset.
D. Lipid-class enrichment of T- and F-associated species (Fisher exact).
Full results -> ~/anchor_work/r9_full.txt ; concise summary printed."""
import os, re
import numpy as np, pandas as pd
from math import erf
from scipy.stats import fisher_exact
from pheno_utils import PhenoLoader
W = os.path.expanduser("~/anchor_work"); log = open(f"{W}/r9_full.txt", "w")
def out(*a):
    s = " ".join(str(x) for x in a); print(s); log.write(s + "\n")
pz = np.vectorize(lambda z: 2 * (1 - 0.5 * (1 + erf(abs(z) / np.sqrt(2)))))
def bh(p):
    o = np.argsort(p); q = p[o] * len(p) / (np.arange(len(p)) + 1); q = np.minimum.accumulate(q[::-1])[::-1]; r = np.empty_like(q); r[o] = np.minimum(q, 1); return r
def ols_hc3(X, Y):
    X1 = np.column_stack([np.ones(len(X)), X]); B = np.linalg.lstsq(X1, Y, rcond=None)[0]; R = Y - X1 @ B
    XtXi = np.linalg.inv(X1.T @ X1); h = np.einsum("ij,jk,ik->i", X1, XtXi, X1); Wt = (R / (1 - h)[:, None]) ** 2
    M = XtXi @ X1.T; SE = np.sqrt(np.einsum("ki,if,ki->kf", M, Wt, M))
    return B, SE

anc = pd.read_parquet(f"{W}/draw_anchors.parquet").set_index("participant_id")
pl = PhenoLoader("serum_lipidomics"); Ld = pl.dfs["serum_lipidomics"].reset_index()
Ld = Ld[Ld.research_stage == "00_00_visit"].drop_duplicates("participant_id").set_index("participant_id")
draw = pd.to_datetime(Ld.collection_timestamp, utc=True).dt.tz_convert("Asia/Jerusalem")
ag = pl.dfs["age_sex"].reset_index(); ag = ag[ag.research_stage == "00_00_visit"].drop_duplicates("participant_id").set_index("participant_id")
try:
    an = PhenoLoader("anthropometrics"); ad = an.dfs[list(an.dfs)[0]].reset_index()
    ad = ad[ad.research_stage == "00_00_visit"].drop_duplicates("participant_id").set_index("participant_id")
    bcol = [c for c in ad.columns if c.lower() in ("bmi", "body_mass_index")] or [c for c in ad.columns if "bmi" in c.lower()]
    bmi = pd.to_numeric(ad[bcol[0]], errors="coerce"); out("BMI column:", bcol[0])
except Exception as e:
    bmi = pd.Series(dtype=float); out("BMI unavailable", repr(e)[:120])
feat = [c for c in Ld.columns if pd.api.types.is_numeric_dtype(Ld[c]) and c != "array_index"]
def lipids(ids):
    Y = Ld.loc[ids, feat].astype(float); Y = Y.loc[:, (Y.notna().mean() >= .5) & (Y.std() > 0)]
    Y = np.log(Y.clip(lower=Y[Y > 0].min() / 2, axis=1)); Y = Y.fillna(Y.median()); return (Y - Y.mean()) / Y.std()
def base(ids, extra=()):
    a = pd.to_numeric(ag.loc[ids, "age"], errors="coerce"); s = pd.to_numeric(ag.loc[ids, "sex"], errors="coerce")
    B = pd.DataFrame({"age": a, "age2": a ** 2, "sex": s}, index=ids)
    if "bmi" in extra: B["bmi"] = bmi.reindex(ids)
    if "calendar" in extra:
        d = draw.reindex(ids); B = pd.concat([B, pd.get_dummies(d.dt.dayofweek.astype(str), prefix="dow", drop_first=True, dtype=float),
                                             pd.get_dummies(d.dt.month.astype(str), prefix="m", drop_first=True, dtype=float)], axis=1)
    return B
def cls(name):
    n = name.lower(); m = re.match(r"(lysopc|lysope|tag|dag|pc|pe|pi|pg|ps|sm|cer|hexcer|ce|fa|acca|mag|lpc|lpe)", n)
    return m.group(1) if m else "other"

# ---------- A. habitual evening meal -> lipidome (naive vs ANCHOR, unequal k)
ida = anc.index[(anc.H_k >= 2) & anc.H_mean.notna()].intersection(Ld.index).intersection(ag.index)
H = anc.loc[ida]
su2 = np.nansum((H.H_k - 1) * H.H_var) / np.nansum(H.H_k - 1)          # pooled within-person (night) variance
sx2 = np.var(H.H_mean) - su2 * np.mean(1 / H.H_k)
lam_i = sx2 / (sx2 + su2 / H.H_k)                                       # person-specific reliability
Hcal = H.H_mean.mean() + lam_i * (H.H_mean - H.H_mean.mean())
out(f"A. habitual evening meal: n={len(ida)} median k={H.H_k.median():.0f} night var {su2:.3f} habit var {sx2:.3f} single-night ICC {sx2/(sx2+su2):.3f} mean-reliability median {lam_i.median():.3f}")
Ya = lipids(ida); Ba = base(ida); okA = Ba.notna().all(1).to_numpy()
for nm, x in [("naive H", H.H_mean), ("ANCHOR H", Hcal)]:
    X = np.column_stack([x.to_numpy(float), Ba.to_numpy(float)])[okA]; B, SE = ols_hc3(X, Ya.to_numpy(float)[okA])
    q = bh(pz(B[1] / SE[1])); top = np.argsort(q)[:5]
    out(f"   {nm:9s} FDR<.05 {int((q<.05).sum())}/{len(q)} FDR<.10 {int((q<.10).sum())}  top", [(Ya.columns[t][:16], round(B[1, t], 3), f"{q[t]:.2g}") for t in top])

# ---------- B. joint habit + draw state
idb = ida.intersection(anc.index[anc.F.between(0.25, 24)])
Yb = lipids(idb); Bb = base(idb); okB = Bb.notna().all(1).to_numpy()
X = np.column_stack([Hcal.loc[idb].to_numpy(float), anc.loc[idb, ["T", "F"]].to_numpy(float), Bb.to_numpy(float)])[okB]
B, SE = ols_hc3(X, Yb.to_numpy(float)[okB]); qs = [bh(pz(B[j] / SE[j])) for j in (1, 2, 3)]
out(f"B. joint H_cal + T + F: n={okB.sum()} hits H {int((qs[0]<.05).sum())}  T {int((qs[1]<.05).sum())}  F {int((qs[2]<.05).sum())} of {Yb.shape[1]}")

# ---------- C. robustness of draw anchors
idc = anc.index[anc.F.between(0.25, 24)].intersection(Ld.index).intersection(ag.index)
Yc = lipids(idc)
def tf(ids, extra=(), Ffun=lambda f: f, label=""):
    Bc = base(ids, extra); ok = Bc.notna().all(1).to_numpy()
    X = np.column_stack([anc.loc[ids, "T"].to_numpy(float), Ffun(anc.loc[ids, "F"].to_numpy(float)), Bc.to_numpy(float)])[ok]
    Bt, St = ols_hc3(X, Yc.loc[ids].to_numpy(float)[ok]); qT, qF = bh(pz(Bt[1] / St[1])), bh(pz(Bt[2] / St[2]))
    out(f"C. {label:22s} n={ok.sum():5d} T hits {int((qT<.05).sum()):3d}  F hits {int((qF<.05).sum()):3d}")
    return qT, qF, Bt
qT0, qF0, B0 = tf(idc, label="core")
tf(idc, ("bmi",), label="+BMI")
tf(idc, ("calendar",), label="+weekday+month")
tf(idc, ("bmi", "calendar"), label="+BMI+calendar")
tf(idc, (), np.log, label="log(F)")
tf(anc.loc[idc][anc.loc[idc].F >= 2].index, label="F>=2h subset")

# ---------- D. lipid-class enrichment
C = pd.Series([cls(c) for c in Yc.columns], index=Yc.columns)
out("D. lipid classes:", C.value_counts().to_dict())
for lab, q, b in [("T", qT0, B0[1]), ("F", qF0, B0[2])]:
    for k in C.value_counts().index:
        inC = (C == k).to_numpy(); sig = q < .05
        if inC.sum() < 3: continue
        tab = [[int((sig & inC).sum()), int((~sig & inC).sum())], [int((sig & ~inC).sum()), int((~sig & ~inC).sum())]]
        orr, p = fisher_exact(tab)
        if tab[0][0] > 0: out(f"   {lab} class {k:7s} hits {tab[0][0]:2d}/{inC.sum():2d}  mean effect {b[inC].mean():+.3f}  OR {orr:.2f} p={p:.2g}")
log.close()
