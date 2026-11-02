"""Build blood-draw anchors per participant (saved INSIDE the TRE; aggregate summaries printed).
T  = clock hour of serum draw (local)
L  = clock time of last >=50 kcal eating occasion before the draw (30-min occasions from first item)
F  = T - L fasting hours (0 < F <= 24)
E  = clock time of the last >=50 kcal occasion on the evening before the draw (night -1, 'acute' meal timing)
Ek = same quantity on every other logged evening (habit nights), unwrapped so 00:00-04:00 -> 24-28
"""
import os, glob
import numpy as np, pandas as pd, pyarrow.parquet as pq
from pheno_utils import PhenoLoader
OUT = os.path.expanduser("~/anchor_work/draw_anchors.parquet")
cand = glob.glob(os.path.expanduser("~/studies/hpp_datasets/diet_logging/diet_logging_events.parquet"))
print("diet events file:", cand)
sch = pq.read_schema(cand[0]); print("schema:", sch.names)
cols = [c for c in ["participant_id", "collection_timestamp", "local_timestamp", "calories_kcal", "research_stage"] if c in sch.names]
E = pq.read_table(cand[0], columns=cols).to_pandas()
E = E.reset_index() if "participant_id" not in E.columns else E
print("events rows", len(E), "people", E.participant_id.nunique())
tcol = "local_timestamp" if "local_timestamp" in E else "collection_timestamp"
t = pd.to_datetime(E[tcol], errors="coerce")
if getattr(t.dt, "tz", None) is not None:
    t = t.dt.tz_convert("Asia/Jerusalem").dt.tz_localize(None)
E = E.assign(t=t).dropna(subset=["t"]).sort_values(["participant_id", "t"])

def occasions(g):
    """30-min occasions measured from the first item (no chaining); returns end time and kcal."""
    out = []; start = None; end = None; kc = 0.0
    for tt, k in zip(g.t.values, g.calories_kcal.fillna(0).values):
        if start is None or (tt - start) > np.timedelta64(30, "m"):
            if start is not None: out.append((end, kc))
            start, end, kc = tt, tt, 0.0
        end = tt; kc += k
    if start is not None: out.append((end, kc))
    o = pd.DataFrame(out, columns=["end", "kcal"]); return o[o.kcal >= 50]

L = PhenoLoader("serum_lipidomics").dfs["serum_lipidomics"].reset_index()
L = L[L.research_stage == "00_00_visit"][["participant_id", "collection_timestamp"]]
L["draw"] = pd.to_datetime(L.collection_timestamp, utc=True).dt.tz_convert("Asia/Jerusalem").dt.tz_localize(None)
L = L[L.participant_id.isin(set(E.participant_id))]
rows = []
for pid, g in E[E.participant_id.isin(set(L.participant_id))].groupby("participant_id"):
    draw = L.loc[L.participant_id == pid, "draw"].iloc[0]
    oc = occasions(g)
    if oc.empty: continue
    before = oc[oc.end < draw]
    Lt = before.end.max() if len(before) else pd.NaT
    F = (draw - Lt) / pd.Timedelta(hours=1) if pd.notna(Lt) else np.nan
    # evening last occasion per 'night': assign occasion to night = date of (end - 4h)
    oc = oc.assign(night=(oc.end - pd.Timedelta(hours=4)).dt.normalize())
    ev = oc.groupby("night").end.max()
    evh = (ev - ev.index).dt.total_seconds() / 3600 + 0.0  # hours since midnight of night date (04:00 next day -> 28)
    prev = (draw - pd.Timedelta(hours=4)).normalize() - pd.Timedelta(days=1)
    acute = evh.get(prev, np.nan)
    habit = evh.drop(prev, errors="ignore")
    habit = habit[(habit >= 16) & (habit <= 28)]
    rows.append(dict(participant_id=pid, T=draw.hour + draw.minute / 60, L=(Lt.hour + Lt.minute / 60) if pd.notna(Lt) else np.nan,
                     F=F, E=acute if (16 <= acute <= 28) else np.nan, H_mean=habit.mean() if len(habit) else np.nan,
                     H_k=len(habit), H_var=habit.var() if len(habit) > 1 else np.nan))
A = pd.DataFrame(rows); A.to_parquet(OUT)
print("people with anchors", len(A))
ok = A.F.between(0, 24, inclusive="right")
print("fasting F (h) quantiles", np.nanpercentile(A.F[ok], [5, 25, 50, 75, 95]).round(2), "share F<=24h", round(ok.mean(), 3))
print("draw T quantiles", np.nanpercentile(A["T"], [5, 50, 95]).round(2), "corr(T,F)", round(A.loc[ok, ["T", "F"]].corr().iloc[0, 1], 3))
print("acute evening E available", A.E.notna().sum(), "habit nights k: median", A.H_k.median(), "q", np.nanpercentile(A.H_k, [10, 90]))
print("evening-meal ICC approx (between var / total):",
      round(float(np.nanvar(A.H_mean) / (np.nanvar(A.H_mean) + np.nanmean(A.H_var))), 3))
print("corr(E, habit mean)", round(A[["E", "H_mean"]].corr().iloc[0, 1], 3))
