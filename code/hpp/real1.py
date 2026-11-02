"""Real-data run 1 (inside HPP TRE): error structure, naive replication, ANCHOR.
Prints aggregate numbers only."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/anchor_work/anchor"))
import anchor as A

D = os.path.expanduser("~/studies/meal-timing-liver-fat/audit_20260918")
P = pd.read_parquet(f"{D}/model_complete_cases.parquet")
N = pd.read_parquet(f"{D}/paired_nights.parquet")
ids = set(P.participant_id)
N = N[N.participant_id.isin(ids)].dropna(subset=["M50", "D50"])
# nutrition rule from report: 500-5000 kcal, macro-energy ratio 0.5-1.5
kcal = N.calories_kcal
ratio = (4 * N.carbohydrate_g + 9 * N.lipid_g + 4 * N.protein_g) / kcal
N = N[(kcal >= 500) & (kcal <= 5000) & (ratio >= 0.5) & (ratio <= 1.5)]
cnt = N.participant_id.value_counts()
N = N[N.participant_id.isin(cnt[cnt >= 2].index)]
print("nights", len(N), "people", N.participant_id.nunique(), "k dist", cnt[cnt >= 2].value_counts().sort_index().to_dict())

pid = N.participant_id.to_numpy()
Wn = N[["M50", "D50"]].to_numpy(float)
uid, Wbar, k, Su, df = A.person_summaries(pid, Wn)
P = P.set_index("participant_id").loc[uid]
agree = [np.corrcoef(Wbar[:, j], P[c].to_numpy(float))[0, 1] for j, c in enumerate(["M50", "D50"])]
print("agreement of recomputed person means with saved M50/D50 (r):", np.round(agree, 4))

Sx = A._nearest_pd(np.cov(Wbar.T) - Su * np.mean(1 / k))
icc = np.diag(Sx) / (np.diag(Sx) + np.diag(Su))
print("Sigma_U", Su.round(4).tolist(), "within corr", round(Su[0, 1] / np.sqrt(Su[0, 0] * Su[1, 1]), 3))
print("Sigma_X", Sx.round(4).tolist(), "true-habit corr", round(Sx[0, 1] / np.sqrt(Sx[0, 0] * Sx[1, 1]), 3))
print("observed person-mean corr", round(np.corrcoef(Wbar.T)[0, 1], 3), "single-night ICC", icc.round(3))
lam = A.attribution_matrix(Sx, Su, k)
print("Lambda (naive = Lambda @ true)", lam.round(3).tolist())

y = P["attenuation"].to_numpy(float)
cov = P[["age", "age2", "sex", "energy", "fat", "protein"]].astype(float)
alc = pd.get_dummies(P["alcohol_current_frequency"].astype(str), drop_first=True, dtype=float)
C = pd.concat([cov, alc], axis=1).to_numpy(float)
ok = ~np.isnan(C).any(1) & ~np.isnan(y)
print("complete", ok.sum())

def show(name, b, se):
    print(f"{name:28s} clock {b[0]:+.5f} ({b[0]-1.96*se[0]:+.5f},{b[0]+1.96*se[0]:+.5f})  gap {b[1]:+.5f} ({b[1]-1.96*se[1]:+.5f},{b[1]+1.96*se[1]:+.5f})")

Ws = P[["M50", "D50"]].to_numpy(float)[ok]
b, V = A.fit_naive(Ws, y[ok], C[ok]); show("naive saved means (report)", b, np.sqrt(np.diag(V)))
b, V = A.fit_naive(Wbar[ok], y[ok], C[ok]); show("naive recomputed means", b, np.sqrt(np.diag(V)))
okid = set(uid[ok]); m = np.isin(pid, list(okid))
ba = A.fit_anchor(Wbar[ok], k[ok], Su, y[ok], C[ok])
dr = A.person_bootstrap(pid[m], Wn[m], y[ok], C[ok], B=int(sys.argv[1]) if len(sys.argv) > 1 else 300, seed=7)
se = dr.std(0); lo, hi = np.percentile(dr, [2.5, 97.5], axis=0)
print(f"{'ANCHOR':28s} clock {ba[0]:+.5f} ({lo[0]:+.5f},{hi[0]:+.5f})  gap {ba[1]:+.5f} ({lo[1]:+.5f},{hi[1]:+.5f})")
c = dr[:, 0] - dr[:, 1]
print("ANCHOR fixed-sleep contrast (later meal, same sleep):", round(ba[0] - ba[1], 5), np.percentile(c, [2.5, 97.5]).round(5))
print("ANCHOR clock-coef-is-zero bootstrap p:", round(2 * min((dr[:, 0] > 0).mean(), (dr[:, 0] < 0).mean()), 3),
      " gap-coef-is-zero p:", round(2 * min((dr[:, 1] > 0).mean(), (dr[:, 1] < 0).mean()), 3))
