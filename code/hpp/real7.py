"""Real-data run 7: nightly-outcome ANCHOR (habitual vs acute) for CGM sleep window,
meal window and their difference on matched nights; joint person bootstrap. Aggregate only."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/anchor_work/anchor"))
import anchor as A
D = os.path.expanduser("~/studies/meal-timing-liver-fat/audit_20260918")
P = pd.read_parquet(f"{D}/model_complete_cases.parquet").set_index("participant_id")
rng = np.random.default_rng(31)
def prep(G, outs):
    G = G[G.participant_id.isin(P.index)].dropna(subset=["M50", "D50"] + outs)
    c = G.participant_id.value_counts(); G = G[G.participant_id.isin(c[c >= 2].index)].sort_values("participant_id").reset_index(drop=True)
    return G
def run(G, y, B=int(os.environ.get("NBOOT", 300))):
    pid = G.participant_id.to_numpy(); Wn = G[["M50", "D50"]].to_numpy(float); yn = G[y].to_numpy(float)
    Cn = G[["calories_kcal", "fat_pct", "protein_pct"]].to_numpy(float)
    uid = np.unique(pid); Q = P.loc[uid]
    C = pd.concat([Q[["age", "age2", "sex"]].astype(float), pd.get_dummies(Q["alcohol_current_frequency"].astype(str), drop_first=True, dtype=float)], axis=1).to_numpy(float)
    f = lambda pp, rows, ci: A.fit_anchor_nightly(pp, Wn[rows], yn[rows], C[ci], Cn[rows])
    est = f(pid, np.arange(len(pid)), np.arange(len(uid)))
    rows_of = [np.flatnonzero(pid == u) for u in uid]; dr = []
    for _ in range(B):
        pick = rng.integers(0, len(uid), len(uid)); rows = np.concatenate([rows_of[i] for i in pick])
        pp = np.repeat(np.arange(len(pick)), [len(rows_of[i]) for i in pick]); r = f(pp, rows, pick)
        dr.append(np.concatenate([r["s"], r["b_a"], r["b_h"], r["diff"]]))
    dr = np.array(dr); e = np.concatenate([est["s"], est["b_a"], est["b_h"], est["diff"]])
    print(f"--- {y}: nights {len(pid)} people {len(uid)} Lambda {est['Lambda'].round(3).tolist()}")
    for i, lab in enumerate(["naive-between", "within(acute)", "habitual(ANCHOR)", "habitual-minus-acute"]):
        for j, a in enumerate(["clock", "gap"]):
            v = dr[:, 2 * i + j]; lo, hi = np.percentile(v, [2.5, 97.5]); p = 2 * min((v > 0).mean(), (v < 0).mean())
            print(f"  {lab:20s} {a:5s} {e[2*i+j]:+.2f} ({lo:+.2f},{hi:+.2f}) p={p:.3f}")
G1 = prep(pd.read_parquet(f"{D}/cgm_qualified_nights.parquet"), ["glucose_3h"])
run(G1, "glucose_3h")
G2 = prep(pd.read_parquet(f"{D}/followup_20261001/matched_window_nights.parquet"), ["glucose_3h", "meal_glucose_3h"])
G2["sleep_minus_meal"] = G2.glucose_3h - G2.meal_glucose_3h
for y in ["glucose_3h", "meal_glucose_3h", "sleep_minus_meal"]: run(G2, y)
