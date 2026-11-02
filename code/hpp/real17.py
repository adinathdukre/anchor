"""Real-data run 17: negative-control outcome (adult height), liver cohort, aggregate output only.
Meal clock and gap cannot change adult height, so naive and ANCHOR estimates should be null; an association would
flag confounding (e.g. by sex, ethnicity or socioeconomic status) of the kind that could also drive the liver result.
Same covariates as the liver analysis (age, age^2, sex, energy, fat and protein shares, alcohol frequency); 500-draw
participant bootstrap for ANCHOR, HC3 for naive."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/anchor_work/anchor"))
import anchor as A
from pheno_utils import PhenoLoader
D = os.path.expanduser("~/studies/meal-timing-liver-fat/audit_20260918")
W = os.path.expanduser("~/anchor_work"); log = open(f"{W}/r17_full.txt", "w")
def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()
rng = np.random.default_rng(17)
P = pd.read_parquet(f"{D}/model_complete_cases.parquet").set_index("participant_id")
an = PhenoLoader("anthropometrics"); ad = an.dfs[list(an.dfs)[0]].reset_index()
ad = ad[ad.research_stage == "00_00_visit"].drop_duplicates("participant_id").set_index("participant_id")
hcol = [c for c in ad.columns if c.lower() in ("height", "height_cm", "standing_height")] or [c for c in ad.columns if "height" in c.lower() and "sitting" not in c.lower()]
out("height column:", hcol[:3])
height = pd.to_numeric(ad[hcol[0]], errors="coerce")
N = pd.read_parquet(f"{D}/paired_nights.parquet")
N = N[N.participant_id.isin(P.index)].dropna(subset=["M50", "D50"])
kc = N.calories_kcal; r = (4 * N.carbohydrate_g + 9 * N.lipid_g + 4 * N.protein_g) / kc
N = N[(kc >= 500) & (kc <= 5000) & (r >= .5) & (r <= 1.5)]
Cl = pd.concat([P[["age", "age2", "sex", "energy", "fat", "protein"]].astype(float),
                pd.get_dummies(P["alcohol_current_frequency"].astype(str), drop_first=True, dtype=float)], axis=1)
ok = Cl.notna().all(1) & height.reindex(P.index).notna()
c = N.participant_id.value_counts(); N = N[N.participant_id.isin(c[c >= 2].index) & N.participant_id.isin(ok[ok].index)].sort_values("participant_id")
pid = N.participant_id.to_numpy(); Wn = N[["M50", "D50"]].to_numpy(float)
uid, Wbar, k, Su, _ = A.person_summaries(pid, Wn)
C = Cl.loc[uid].to_numpy(float); y = height.reindex(uid).to_numpy(float); y = (y - y.mean()) / y.std()
out(f"people {len(uid)}; height mean {height.reindex(uid).mean():.1f}")
bn, V = A.fit_naive(Wbar, y, C); se = np.sqrt(np.diag(V))
ba = A.fit_anchor(Wbar, k, Su, y, C); dr = A.person_bootstrap(pid, Wn, y, C, B=500, seed=17)
lo, hi = np.percentile(dr, [2.5, 97.5], 0); p = [2 * min((dr[:, j] > 0).mean(), (dr[:, j] < 0).mean()) for j in (0, 1)]
out(f"naive  (SD of height per hour): clock {bn[0]:+.4f} ({bn[0]-1.96*se[0]:+.4f},{bn[0]+1.96*se[0]:+.4f})  gap {bn[1]:+.4f} ({bn[1]-1.96*se[1]:+.4f},{bn[1]+1.96*se[1]:+.4f})")
out(f"ANCHOR (SD of height per hour): clock {ba[0]:+.4f} ({lo[0]:+.4f},{hi[0]:+.4f}) p={p[0]:.3f}  gap {ba[1]:+.4f} ({lo[1]:+.4f},{hi[1]:+.4f}) p={p[1]:.3f}")
log.close()
