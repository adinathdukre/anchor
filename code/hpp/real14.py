"""Real-data run 14: real night-order test of A2 in the lipidomics cohort (meal clock, ~14 logged evenings each).
Aggregate output only.
A. Semivariogram of evening meal clock by calendar lag (i.i.d. nights => flat).
B. Real-order spike-in: per participant (>= 10 evenings), the 'observed' habit is a block of 3 evenings and the
   'true' habit is the mean of all other evenings. A synthetic outcome Y = beta * true + noise is generated, and the
   naive and ANCHOR (p = 1, unequal-k calibration with Sigma_U from the block) slopes are compared with beta for
   blocks of 3 CONSECUTIVE logged evenings versus 3 evenings at random positions. If consecutive evenings are
   autocorrelated, Sigma_U from the block is too small and ANCHOR under-corrects in the consecutive design."""
import os, glob
import numpy as np, pandas as pd, pyarrow.parquet as pq
W = os.path.expanduser("~/anchor_work"); log = open(f"{W}/r14_full.txt", "w")
def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()
rng = np.random.default_rng(14)
anc = pd.read_parquet(f"{W}/draw_anchors.parquet")
ids = set(anc.participant_id[anc.H_k >= 10])
f = glob.glob(os.path.expanduser("~/studies/hpp_datasets/diet_logging/diet_logging_events.parquet"))[0]
cols = [c for c in ["participant_id", "collection_timestamp", "local_timestamp", "calories_kcal"] if c in pq.read_schema(f).names]
E = pq.read_table(f, columns=cols).to_pandas(); E = E.reset_index() if "participant_id" not in E.columns else E
E = E[E.participant_id.isin(ids)]
tcol = "local_timestamp" if "local_timestamp" in E else "collection_timestamp"
t = pd.to_datetime(E[tcol], errors="coerce")
if getattr(t.dt, "tz", None) is not None:
    t = t.dt.tz_convert("Asia/Jerusalem").dt.tz_localize(None)
E = E.assign(t=t).dropna(subset=["t"]).sort_values(["participant_id", "t"])
series = {}
for pid, g in E.groupby("participant_id"):
    tt = g.t.values; kc = g.calories_kcal.fillna(0).values; occ = []; start = end = None; k = 0.0
    for a, b in zip(tt, kc):
        if start is None or (a - start) > np.timedelta64(30, "m"):
            if start is not None and k >= 50: occ.append(end)
            start, end, k = a, a, 0.0
        end = a; k += b
    if start is not None and k >= 50: occ.append(end)
    if not occ: continue
    oc = pd.Series(pd.to_datetime(occ)); night = (oc - pd.Timedelta(hours=4)).dt.normalize()
    ev = oc.groupby(night).max(); h = (ev - ev.index).dt.total_seconds() / 3600
    h = h[(h >= 16) & (h <= 28)]
    if len(h) >= 10: series[pid] = h
out("participants with >= 10 evenings:", len(series), "median evenings:", int(np.median([len(s) for s in series.values()])))
# A. semivariogram by calendar lag
rows = []
for s in series.values():
    d = s.index.values; v = s.values
    for a in range(len(v)):
        for b in range(a + 1, len(v)):
            rows.append(((d[b] - d[a]) / np.timedelta64(1, "D"), v[a] - v[b]))
R = pd.DataFrame(rows, columns=["lag", "dv"])
R["bin"] = pd.cut(R.lag, [0, 1.5, 2.5, 3.5, 7.5, 14.5, 28.5, 1e4], labels=["1", "2", "3", "4-7", "8-14", "15-28", ">28"])
out("A. semivariogram of evening meal clock (h^2) by lag in days [pairs, gamma]:")
for b, g in R.groupby("bin", observed=True):
    out(f"   lag {b}: {len(g)}  {0.5 * (g.dv ** 2).mean():.3f}")
ssw = sum(((s.values - s.values.mean()) ** 2).sum() for s in series.values()); dfw = sum(len(s) - 1 for s in series.values())
out("   pooled within-person variance (all evenings):", round(ssw / dfw, 3))
# B. real-order spike-in
pids = list(series); BETA = 0.10
def run(mode, reps=500):
    est = {"naive": [], "ANCHOR": []}
    for _ in range(reps):
        obs, tru, wss, dfs = [], [], 0.0, 0
        for p in pids:
            v = series[p].values; n = len(v)
            if mode == "consecutive":
                s0 = rng.integers(0, n - 2); idx = np.arange(s0, s0 + 3)
            else:
                idx = np.sort(rng.choice(n, 3, replace=False))
            blk = v[idx]; rest = np.delete(v, idx)
            obs.append(blk.mean()); tru.append(rest.mean()); wss += ((blk - blk.mean()) ** 2).sum(); dfs += 2
        obs, tru = np.array(obs), np.array(tru)
        y = BETA * (tru - tru.mean()) + rng.normal(0, 1, len(tru))
        su2 = wss / dfs; sx2 = max(obs.var() - su2 / 3, 1e-6); lam = sx2 / (sx2 + su2 / 3)
        bn = np.polyfit(obs, y, 1)[0]; est["naive"].append(bn); est["ANCHOR"].append(bn / lam)
    return {m: np.array(v) for m, v in est.items()}, lam
for mode in ("consecutive", "random"):
    est, lam = run(mode)
    for m, v in est.items():
        out(f"B. {mode:11s} {m:6s}: mean slope / beta = {v.mean() / BETA:.3f} (MC SE {v.std() / np.sqrt(len(v)) / BETA:.3f}); last-rep reliability {lam:.3f}")
log.close()
