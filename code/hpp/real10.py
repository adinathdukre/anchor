"""Real-data run 10: genetic chronotype instrument (213 Jones-2019 morningness loci) — aggregate output only.
A. Instrument strength for habitual evening-meal clock H (repeated logged evenings).
B. Which anchor does genetics move? GRS -> person-mean M, D, S (liver nights cohort).
C. One-sample MR (2SLS, GRS instrument) of habitual meal clock on 170 serum lipids;
   triangulation: naive OLS (attenuated) vs ANCHOR (error-corrected) vs IV (error- and confounding-robust).
D. MR of schedule on liver attenuation.
Classical error in the exposure does not bias IV, so IV is an external check on ANCHOR's correction."""
import os
import numpy as np, pandas as pd
from math import erf
from pheno_utils import PhenoLoader
W = os.path.expanduser("~/anchor_work"); D = os.path.expanduser("~/studies/meal-timing-liver-fat/audit_20260918")
log = open(f"{W}/r10_full.txt", "w")
def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n")
pz = np.vectorize(lambda z: 2 * (1 - 0.5 * (1 + erf(abs(z) / np.sqrt(2)))))
def bh(p):
    o = np.argsort(p); q = p[o] * len(p) / (np.arange(len(p)) + 1); q = np.minimum.accumulate(q[::-1])[::-1]; r = np.empty_like(q); r[o] = np.minimum(q, 1); return r
def ols(X, Y):
    X1 = np.column_stack([np.ones(len(X)), X]); B = np.linalg.lstsq(X1, Y, rcond=None)[0]; R = Y - X1 @ B
    XtXi = np.linalg.inv(X1.T @ X1); h = np.einsum("ij,jk,ik->i", X1, XtXi, X1)
    Y2 = Y if Y.ndim == 2 else Y[:, None]; R2 = R if R.ndim == 2 else R[:, None]
    Wt = (R2 / (1 - h)[:, None]) ** 2; M = XtXi @ X1.T; SE = np.sqrt(np.einsum("ki,if,ki->kf", M, Wt, M))
    return B, (SE if Y.ndim == 2 else SE[:, 0])

# ---- genotype score
snps = pd.read_csv(f"{W}/hpp/data/chronotype_jones2019_gwascat.tsv", sep="\t")
G = pd.read_parquet(f"{W}/grs_dosages.parquet")
w = snps.set_index("rsid").loc[G.columns, "log_or_morning"].to_numpy(float)
Gd = G.to_numpy(float); keep = np.isfinite(Gd).mean(0) > 0.95
Gd = np.where(np.isfinite(Gd), Gd, np.nanmean(Gd, 0))[:, keep]; w = w[keep]
grs = pd.Series(Gd @ w, index=G.index); grs = (grs - grs.mean()) / grs.std()
out(f"GRS: {len(grs)} people, {keep.sum()} variants (morning-increasing, standardised)")
pl = PhenoLoader("human_genetics"); pca = pl.dfs["pca"].reset_index().drop_duplicates("participant_id").set_index("participant_id")
pcs = pca[[c for c in pca.columns if pd.api.types.is_numeric_dtype(pca[c]) and c not in ("array_index",)]].iloc[:, :10]
ls = PhenoLoader("serum_lipidomics"); ag = ls.dfs["age_sex"].reset_index(); ag = ag[ag.research_stage == "00_00_visit"].drop_duplicates("participant_id").set_index("participant_id")
def covs(ids):
    a = pd.to_numeric(ag.reindex(ids)["age"], errors="coerce"); s = pd.to_numeric(ag.reindex(ids)["sex"], errors="coerce")
    return pd.concat([pd.DataFrame({"age": a, "age2": a ** 2, "sex": s}, index=ids), pcs.reindex(ids)], axis=1)

# ---- A. habitual evening meal
anc = pd.read_parquet(f"{W}/draw_anchors.parquet").set_index("participant_id")
H = anc[(anc.H_k >= 2) & anc.H_mean.notna()]
su2 = np.nansum((H.H_k - 1) * H.H_var) / np.nansum(H.H_k - 1); sx2 = np.var(H.H_mean) - su2 * np.mean(1 / H.H_k)
lam_i = sx2 / (sx2 + su2 / H.H_k); Hcal = H.H_mean.mean() + lam_i * (H.H_mean - H.H_mean.mean())
ida = H.index.intersection(grs.index); C = covs(ida); ok = C.notna().all(1).to_numpy()
ida = ida[ok]; C = C.loc[ida].to_numpy(float); g = grs.loc[ida].to_numpy(float); h = H.H_mean.loc[ida].to_numpy(float)
B, SE = ols(np.column_stack([g, C]), h); F = (B[1] / SE[1]) ** 2
r2 = np.corrcoef(g - np.column_stack([np.ones(len(C)), C]) @ np.linalg.lstsq(np.column_stack([np.ones(len(C)), C]), g, rcond=None)[0],
                 h - np.column_stack([np.ones(len(C)), C]) @ np.linalg.lstsq(np.column_stack([np.ones(len(C)), C]), h, rcond=None)[0])[0, 1] ** 2
out(f"A. first stage, habitual evening meal clock (n={len(ida)}): {B[1]*60:+.2f} min per SD morning-GRS (SE {SE[1]*60:.2f}), F={F:.1f}, partial R2={r2:.4f}")
# robustness of the instrument: drop core clock-gene loci
clock = snps.gene.fillna("").str.contains("PER|CRY|ARNTL|RGS16|FBXL13|VIP|RASD1")
out(f"   clock-gene loci in score: {int(clock.sum())}")

# ---- B. which anchor does genetics move (liver nights cohort)
P = pd.read_parquet(f"{D}/model_complete_cases.parquet").set_index("participant_id")
idb = P.index.intersection(grs.index); Cb = covs(idb); okb = Cb.notna().all(1).to_numpy(); idb = idb[okb]
for nm, col in [("meal clock M", "M50"), ("gap D", "D50")]:
    y = P.loc[idb, col].to_numpy(float)
    Bb, Sb = ols(np.column_stack([grs.loc[idb].to_numpy(float), Cb.loc[idb].to_numpy(float)]), y)
    out(f"B. GRS -> {nm} (n={len(idb)}): {Bb[1]*60:+.2f} min per SD (SE {Sb[1]*60:.2f}) p={pz(Bb[1]/Sb[1]):.2g}")
y = (P.loc[idb, "M50"] + P.loc[idb, "D50"]).to_numpy(float)
Bb, Sb = ols(np.column_stack([grs.loc[idb].to_numpy(float), Cb.loc[idb].to_numpy(float)]), y)
out(f"B. GRS -> sleep onset S=M+D (n={len(idb)}): {Bb[1]*60:+.2f} min per SD (SE {Sb[1]*60:.2f}) p={pz(Bb[1]/Sb[1]):.2g}")

# ---- C. MR on lipids
Ld = ls.dfs["serum_lipidomics"].reset_index(); Ld = Ld[Ld.research_stage == "00_00_visit"].drop_duplicates("participant_id").set_index("participant_id")
idc = ida.intersection(Ld.index)
feat = [c for c in Ld.columns if pd.api.types.is_numeric_dtype(Ld[c]) and c != "array_index"]
Y = Ld.loc[idc, feat].astype(float); Y = Y.loc[:, (Y.notna().mean() >= .5) & (Y.std() > 0)]
Y = np.log(Y.clip(lower=Y[Y > 0].min() / 2, axis=1)); Y = Y.fillna(Y.median()); Y = (Y - Y.mean()) / Y.std(); Ym = Y.to_numpy(float)
pos = pd.Index(ida).get_indexer(idc); Cc = C[pos]; gc = g[pos]; hc = h[pos]; hcal = Hcal.loc[idc].to_numpy(float)
Bn, Sn = ols(np.column_stack([hc, Cc]), Ym); Ba, Sa = ols(np.column_stack([hcal, Cc]), Ym)
Bfs, Sfs = ols(np.column_stack([gc, Cc]), hc); Brf, Srf = ols(np.column_stack([gc, Cc]), Ym)
iv = Brf[1] / Bfs[1]; iv_se = np.abs(Srf[1] / Bfs[1]) * np.sqrt(1 + (Brf[1] / Srf[1]) ** 2 * (Sfs[1] / Bfs[1]) ** 2)
qn, qa, qiv, qrf = bh(pz(Bn[1] / Sn[1])), bh(pz(Ba[1] / Sa[1])), bh(pz(iv / iv_se)), bh(pz(Brf[1] / Srf[1]))
out(f"C. lipids n={len(idc)}, {Y.shape[1]} species, first-stage F={(Bfs[1]/Sfs[1])**2:.1f}")
out(f"   FDR<.05: naive {int((qn<.05).sum())}, ANCHOR {int((qa<.05).sum())}, MR (reduced form GRS->lipid) {int((qrf<.05).sum())}, MR IV {int((qiv<.05).sum())}")
hit = qa < .05
for nm, b in [("naive", Bn[1]), ("ANCHOR", Ba[1])]:
    sl = np.polyfit(b[hit], iv[hit], 1)[0] if hit.sum() > 2 else np.nan
    out(f"   ANCHOR-hit lipids (n={hit.sum()}): sign agreement IV vs {nm} {np.mean(np.sign(iv[hit]) == np.sign(b[hit])):.2f}; slope IV~{nm} {sl:.2f}; median |IV|/|{nm}| {np.median(np.abs(iv[hit])/np.abs(b[hit])):.2f}")
tw = np.sum(((iv - Ba[1]) / iv_se) ** 2 * hit); tn = np.sum(((iv - Bn[1]) / iv_se) ** 2 * hit)
out(f"   heterogeneity sum((IV-est)/SE_IV)^2 over hits: ANCHOR {tw:.1f} vs naive {tn:.1f} (df {hit.sum()})")
top = np.argsort(qa)[:6]
for t in top: out(f"   {Y.columns[t][:20]:20s} naive {Bn[1,t]:+.3f} ANCHOR {Ba[1,t]:+.3f} IV {iv[t]:+.3f} ({iv[t]-1.96*iv_se[t]:+.3f},{iv[t]+1.96*iv_se[t]:+.3f}) GRS->lipid p={pz(Brf[1,t]/Srf[1,t]):.2g}")

# ---- D. liver
yl = P.loc[idb, "attenuation"].to_numpy(float); Cl = Cb.loc[idb].to_numpy(float); gl = grs.loc[idb].to_numpy(float)
Bfs, Sfs = ols(np.column_stack([gl, Cl]), P.loc[idb, "M50"].to_numpy(float)); Brf, Srf = ols(np.column_stack([gl, Cl]), yl)
out(f"D. liver (n={len(idb)}): GRS->attenuation {Brf[1]:+.5f} (SE {Srf[1]:.5f}) p={pz(Brf[1]/Srf[1]):.2g}; Wald ratio per h meal clock {Brf[1]/Bfs[1]:+.4f} (first-stage F {(Bfs[1]/Sfs[1])**2:.1f})")
log.close()
