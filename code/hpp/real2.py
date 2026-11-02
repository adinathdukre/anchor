"""Real-data run 2: CGM (sleep-window glucose) naive vs ANCHOR vs within-person,
plus availability probe for serum metabolomics. Aggregate output only."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/anchor_work/anchor"))
import anchor as A

D = os.path.expanduser("~/studies/meal-timing-liver-fat/audit_20260918")
P = pd.read_parquet(f"{D}/model_complete_cases.parquet").set_index("participant_id")
G = pd.read_parquet(f"{D}/cgm_qualified_nights.parquet")
G = G[G.participant_id.isin(P.index)].dropna(subset=["M50", "D50", "glucose_3h"])
cnt = G.participant_id.value_counts(); G = G[G.participant_id.isin(cnt[cnt >= 2].index)]
print("CGM nights", len(G), "people", G.participant_id.nunique(), "k", cnt[cnt >= 2].value_counts().sort_index().to_dict())

pid = G.participant_id.to_numpy(); Wn = G[["M50", "D50"]].to_numpy(float)
uid, Wbar, k, Su, _ = A.person_summaries(pid, Wn)
yb = G.groupby("participant_id").glucose_3h.mean().loc[uid].to_numpy(float)
Q = P.loc[uid]
C = pd.concat([Q[["age", "age2", "sex"]].astype(float),
               G.groupby("participant_id")[["calories_kcal", "fat_pct", "protein_pct"]].mean().loc[uid],
               pd.get_dummies(Q["alcohol_current_frequency"].astype(str), drop_first=True, dtype=float)], axis=1).to_numpy(float)
Sx = A._nearest_pd(np.cov(Wbar.T) - Su * np.mean(1 / k))
print("CGM subset: within corr", round(Su[0, 1] / np.sqrt(Su[0, 0] * Su[1, 1]), 3), "ICC", (np.diag(Sx) / (np.diag(Sx) + np.diag(Su))).round(3),
      "Lambda", A.attribution_matrix(Sx, Su, k).round(3).tolist())

def show(n, b, lo, hi):
    print(f"{n:30s} clock {b[0]:+.3f} ({lo[0]:+.3f},{hi[0]:+.3f})  gap {b[1]:+.3f} ({lo[1]:+.3f},{hi[1]:+.3f})")

b, V = A.fit_naive(Wbar, yb, C); se = np.sqrt(np.diag(V)); show("between naive (person means)", b, b - 1.96 * se, b + 1.96 * se)
ba = A.fit_anchor(Wbar, k, Su, yb, C)
dr = A.person_bootstrap(pid, Wn, yb, C, B=300, seed=11); lo, hi = np.percentile(dr, [2.5, 97.5], 0)
show("between ANCHOR", ba, lo, hi)

# within-person fixed effects on nights (exact nightly exposures, no averaging error)
g = G.copy()
for c in ["M50", "D50", "glucose_3h", "calories_kcal", "fat_pct", "protein_pct"]:
    g[c + "_w"] = g[c] - g.groupby("participant_id")[c].transform("mean")
X = g[["M50_w", "D50_w", "calories_kcal_w", "fat_pct_w", "protein_pct_w"]].to_numpy(float); yv = g["glucose_3h_w"].to_numpy(float)
bw = np.linalg.lstsq(X, yv, rcond=None)[0]
r = yv - X @ bw; XtXi = np.linalg.inv(X.T @ X)
meat = sum(np.outer(X[m].T @ r[m], X[m].T @ r[m]) for m in [g.participant_id.to_numpy() == u for u in uid])
Gn, Nn, Pp = len(uid), len(yv), X.shape[1]
Vw = XtXi @ meat @ XtXi * Gn / (Gn - 1) * (Nn - 1) / (Nn - Gn - Pp)
sw = np.sqrt(np.diag(Vw))[:2]
show("within-person FE (nights)", bw[:2], bw[:2] - 1.96 * sw, bw[:2] + 1.96 * sw)

# metabolomics availability probe
try:
    from pheno_utils import PhenoLoader
    for ds in ["nightingale_metabolomics", "serum_lipidomics"]:
        try:
            pl = PhenoLoader(ds)
            df = pl.dfs[list(pl.dfs)[0]] if hasattr(pl, "dfs") else None
            print(ds, "tables", list(pl.dfs) if hasattr(pl, "dfs") else "?")
            if df is not None:
                ix = df.reset_index()
                print("  shape", df.shape, "overlap with 2011:", ix["participant_id"].isin(P.index).sum() if "participant_id" in ix else "?")
                print("  first cols", list(df.columns)[:12])
        except Exception as e:
            print(ds, "ERR", repr(e)[:200])
except Exception as e:
    print("pheno_utils ERR", repr(e)[:200])
