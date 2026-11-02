"""Feasibility probe for blood-draw anchors (aggregate output only)."""
import os, glob, re
import numpy as np, pandas as pd
from pheno_utils import PhenoLoader
D = os.path.expanduser("~/studies/meal-timing-liver-fat/audit_20260918")
for ds in ["serum_lipidomics", "nightingale_metabolomics"]:
    L = PhenoLoader(ds).dfs[ds].reset_index()
    L = L[L.research_stage == "00_00_visit"] if "research_stage" in L else L
    ts = pd.to_datetime(L["collection_timestamp"], errors="coerce", utc=True).dt.tz_convert("Asia/Jerusalem")
    h = ts.dt.hour + ts.dt.minute / 60
    print(ds, "baseline rows", len(L), "people", L.participant_id.nunique(), "timestamp non-null", ts.notna().sum())
    print("  draw hour quantiles", np.nanpercentile(h, [5, 25, 50, 75, 95]).round(2), "sd", round(np.nanstd(h), 2))
    print("  exact-midnight share (date-only timestamps?)", round(float(((ts.dt.hour == 0) & (ts.dt.minute == 0)).mean()), 3))
print("--- how the audit pipeline loads diet events:")
for f in sorted(glob.glob(f"{D}/0[2-4]_*.py")) + [f"{D}/39_meal_event_schema.py"]:
    s = open(f).read()
    for line in s.splitlines():
        if re.search(r"diet_logging|PhenoLoader|load_bulk|events|parquet", line) and len(line) < 160:
            print(f"  {os.path.basename(f)}: {line.strip()}")
    print("  ...")
pl = PhenoLoader("diet_logging")
print("diet_logging tables", list(pl.dfs), {k: v.shape for k, v in pl.dfs.items()})
print("diet_logging columns", list(pl.dfs["diet_logging"].columns)[:20])
