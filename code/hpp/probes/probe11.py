"""Genetics overlap + VCF format probe (aggregate only; no genotypes printed)."""
import os, gzip
import pandas as pd
from pheno_utils import PhenoLoader
D = os.path.expanduser("~/studies/meal-timing-liver-fat/audit_20260918")
pl = PhenoLoader("human_genetics"); hg = pl.dfs["human_genetics"].reset_index()
print("genetics index cols", [c for c in hg.columns if c in ("participant_id", "cohort", "research_stage")], "rows", len(hg), "people", hg.participant_id.nunique())
gid = set(hg.participant_id)
liver = set(pd.read_parquet(f"{D}/model_complete_cases.parquet").participant_id)
cgm = set(pd.read_parquet(f"{D}/cgm_qualified_nights.parquet").participant_id)
anc = pd.read_parquet(os.path.expanduser("~/anchor_work/draw_anchors.parquet"))
hab = set(anc.participant_id[(anc.H_k >= 2) & anc.H_mean.notna()])
for nm, s in [("liver 2011", liver), ("cgm", cgm), ("habitual meal", hab)]:
    print(f"overlap {nm}: {len(s & gid)} of {len(s)}")
qc = [c for c in hg.columns if "qc" in c.lower() or "cover" in c.lower() or "depth" in c.lower()]
print("qc cols", qc[:10])
v = hg.gencove_vcf.dropna().iloc[0]
p = v if os.path.exists(str(v)) else os.path.expanduser("~/studies/human_genetics_bulk/" + str(v).split("human_genetics_bulk/")[-1])
print("vcf path resolvable:", os.path.exists(p))
with gzip.open(p, "rt") as f:
    n = 0; hdr = []
    for line in f:
        if line.startswith("##"):
            if any(k in line for k in ("FORMAT", "reference", "source", "contig=<ID=chr1", "contig=<ID=1")): hdr.append(line.strip()[:120])
            continue
        if line.startswith("#"): print("columns:", line.strip().split("\t")[:9]); continue
        if n == 0: print("first record FORMAT:", line.split("\t")[8], "| ID example:", line.split("\t")[2][:12])
        n += 1
        if n > 200000: break
    print("records scanned (cap 200k):", n)
for h in hdr[:12]: print("  ", h)
pca = pl.dfs["pca"]; print("pca columns", list(pca.columns)[:8])
