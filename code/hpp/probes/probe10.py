"""Genetics feasibility probe (aggregate output only: file names, shapes, overlap counts)."""
import os, glob
import pandas as pd
G = os.path.expanduser("~/studies/human_genetics_bulk")
print("human_genetics_bulk top level:")
for p in sorted(glob.glob(f"{G}/*"))[:40]:
    print("  ", os.path.basename(p), "(dir)" if os.path.isdir(p) else f"{os.path.getsize(p)/1e6:.1f} MB")
print("second level (first 60):")
for p in sorted(glob.glob(f"{G}/*/*"))[:60]:
    print("  ", os.path.relpath(p, G), "(dir)" if os.path.isdir(p) else f"{os.path.getsize(p)/1e6:.1f} MB")
for pat in ["*.bim", "*.pgen", "*.bgen", "*.vcf*", "*prs*", "*PRS*", "*score*", "*pca*", "*.fam", "*.psam"]:
    hits = glob.glob(f"{G}/**/{pat}", recursive=True)
    if hits: print(pat, len(hits), "e.g.", [os.path.relpath(h, G) for h in hits[:3]])
try:
    from pheno_utils import PhenoLoader
    import pheno_utils
    for ds in ["human_genetics", "genetics", "polygenic_risk_scores", "prs"]:
        try:
            pl = PhenoLoader(ds); print("PhenoLoader", ds, "tables", list(pl.dfs), {k: v.shape for k, v in pl.dfs.items()})
            d = pl.dfs[list(pl.dfs)[0]]; print("   columns", list(d.columns)[:40])
        except Exception as e:
            print("PhenoLoader", ds, "ERR", repr(e)[:120])
except Exception as e:
    print("pheno_utils ERR", repr(e)[:150])
# overlap with analysis cohorts
D = os.path.expanduser("~/studies/meal-timing-liver-fat/audit_20260918")
ids_liver = set(pd.read_parquet(f"{D}/model_complete_cases.parquet").participant_id)
anc = pd.read_parquet(os.path.expanduser("~/anchor_work/draw_anchors.parquet"))
ids_hab = set(anc.participant_id[(anc.H_k >= 2) & anc.H_mean.notna()])
fams = glob.glob(f"{G}/**/*.fam", recursive=True) + glob.glob(f"{G}/**/*.psam", recursive=True)
if fams:
    f = pd.read_csv(fams[0], sep=r"\s+", header=None, comment="#", dtype=str)
    gid = set(f.iloc[:, 1]) | set(f.iloc[:, 0])
    print("genotyped samples in", os.path.basename(fams[0]), len(f))
    print("example id format (masked):", list(gid)[0][:3] + "***")
    for nm, s in [("liver 2011", ids_liver), ("habitual-meal 5608", ids_hab)]:
        ss = {str(x) for x in s}
        print(f"  overlap with {nm}: direct {len(ss & gid)}; suffix-match {len({x for x in ss if any(g.endswith(x[-6:]) for g in list(gid)[:0])})}")
