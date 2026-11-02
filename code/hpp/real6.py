"""Real-data run 6: anchor-alignment test on matched windows (aggregate only).
Sleep-aligned vs meal-aligned 3h glucose on identical nights; within-person, naive-between, ANCHOR-between; joint bootstrap."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/anchor_work/anchor"))
import anchor as A
D = os.path.expanduser("~/studies/meal-timing-liver-fat/audit_20260918")
P = pd.read_parquet(f"{D}/model_complete_cases.parquet").set_index("participant_id")
G = pd.read_parquet(f"{D}/followup_20261001/matched_window_nights.parquet")
G = G[G.participant_id.isin(P.index)].dropna(subset=["M50", "D50", "glucose_3h", "meal_glucose_3h"])
c = G.participant_id.value_counts(); G = G[G.participant_id.isin(c[c >= 2].index)].reset_index(drop=True)
G["diff"] = G.glucose_3h - G.meal_glucose_3h
print("matched nights", len(G), "people", G.participant_id.nunique())
nut = ["calories_kcal", "fat_pct", "protein_pct"]; outs = ["glucose_3h", "meal_glucose_3h", "diff"]
rng = np.random.default_rng(21)
def fit(pp, g):
    uid, Wb, k, Su, _ = A.person_summaries(pp, g[["M50", "D50"]].to_numpy(float)); g = g.assign(pp=pp)
    Q = P.loc[g.groupby("pp").participant_id.first().loc[uid]]
    C = pd.concat([Q[["age", "age2", "sex"]].astype(float).reset_index(drop=True), g.groupby("pp")[nut].mean().loc[uid].reset_index(drop=True),
                   pd.get_dummies(Q["alcohol_current_frequency"].astype(str), drop_first=True, dtype=float).reset_index(drop=True)], axis=1).to_numpy(float)
    Yb = g.groupby("pp")[outs].mean().loc[uid].to_numpy(float)
    Xh, _ = A.calibrate(Wb, k, Su, C)
    co = lambda X: np.linalg.lstsq(np.column_stack([np.ones(len(X)), X, C]), Yb, rcond=None)[0][1:3]
    dm = lambda col: (g[col] - g.groupby("pp")[col].transform("mean")).to_numpy(float)
    Xw = np.column_stack([dm(v) for v in ["M50", "D50"] + nut]); Yw = np.column_stack([dm(o) for o in outs])
    bw = np.linalg.lstsq(Xw, Yw, rcond=None)[0][:2]
    return np.stack([bw, co(Wb), co(Xh)])  # 3 methods x 2 exposures x 3 outcomes
pid = G.participant_id.to_numpy(); est = fit(pid, G)
uid = np.unique(pid); rows_of = {u: np.flatnonzero(pid == u) for u in uid}; dr = []
for _ in range(200):
    pick = rng.choice(uid, len(uid)); rows = np.concatenate([rows_of[u] for u in pick])
    dr.append(fit(np.repeat(np.arange(len(pick)), [len(rows_of[u]) for u in pick]), G.iloc[rows].reset_index(drop=True)))
dr = np.array(dr)
for mi, m in enumerate(["within", "naive-between", "ANCHOR-between"]):
    for ei, e in enumerate(["clock", "gap"]):
        s = "  ".join(f"{o}: {est[mi,ei,oi]:+.2f} ({np.percentile(dr[:,mi,ei,oi],2.5):+.2f},{np.percentile(dr[:,mi,ei,oi],97.5):+.2f})" for oi, o in enumerate(outs))
        print(f"{m:15s} {e:5s} {s}")
# fixed-sleep contrast (meal +1h, same sleep) = b_clock - b_gap
for mi, m in enumerate(["within", "naive-between", "ANCHOR-between"]):
    v = est[mi, 0] - est[mi, 1]; d = dr[:, mi, 0] - dr[:, mi, 1]
    print(f"{m:15s} later-meal-same-sleep " + "  ".join(f"{o}: {v[oi]:+.2f} ({np.percentile(d[:,oi],2.5):+.2f},{np.percentile(d[:,oi],97.5):+.2f})" for oi, o in enumerate(outs)))
