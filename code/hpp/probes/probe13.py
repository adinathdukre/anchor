"""Follow-up visit structure probe (aggregate only)."""
import os, glob, json
import pandas as pd
D = os.path.expanduser("~/studies/meal-timing-liver-fat/audit_20260918")
print("followup_02:", sorted(os.path.basename(p) for p in glob.glob(f"{D}/followup_02/*"))[:40])
for f in ["validation_followup_results.json", "followup_protocol.json"]:
    try:
        j = json.load(open(f"{D}/{f}")); print(f, "keys:", list(j)[:25] if isinstance(j, dict) else type(j))
    except Exception as e: print(f, "ERR", repr(e)[:100])
s = open(f"{D}/21_construct_later_visit.py").read()
print("--- 21 head:"); print("\n".join(l for l in s.splitlines() if any(k in l for k in ("parquet", "research_stage", "stage", "to_parquet", "def ")))[:2500])
for p in glob.glob(f"{D}/followup_02/*.parquet") + glob.glob(f"{D}/*later*.parquet") + glob.glob(f"{D}/*followup*.parquet"):
    d = pd.read_parquet(p); print(os.path.basename(p), d.shape, list(d.columns)[:30])
    if "research_stage" in d: print("   stages", d.research_stage.value_counts().to_dict())
N = pd.read_parquet(f"{D}/paired_nights.parquet"); print("paired_nights stages", N.research_stage.value_counts().to_dict())
