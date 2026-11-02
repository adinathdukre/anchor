"""VCF format/tooling probe (aggregate only)."""
import os, glob, gzip, shutil, importlib
from pheno_utils import PhenoLoader
B = os.path.expanduser("~/studies/human_genetics_bulk/reads/vcf")
fs = sorted(glob.glob(f"{B}/*.vcf.gz")); print("vcf files", len(fs), "tbi files", len(glob.glob(f"{B}/*.tbi")), "csi", len(glob.glob(f"{B}/*.csi")))
print("sizes MB (first 5):", [round(os.path.getsize(f) / 1e6, 1) for f in fs[:5]])
for t in ["bcftools", "tabix", "plink2", "plink"]: print("tool", t, shutil.which(t))
for m in ["pysam", "cyvcf2", "pgenlib"]:
    try: importlib.import_module(m); print("module", m, "OK")
    except Exception as e: print("module", m, "missing")
hg = PhenoLoader("human_genetics").dfs["human_genetics"].reset_index()
v = str(hg.gencove_vcf.dropna().iloc[0]); print("table vcf value pattern:", v[:25] + "..." + v[-30:])
print("basename in folder:", os.path.exists(f"{B}/{os.path.basename(v)}"))
f = fs[0]; n = 0
with gzip.open(f, "rt") as fh:
    for line in fh:
        if line.startswith("##"):
            if line.startswith(("##FORMAT", "##reference", "##source", "##contig=<ID=chr1,", "##contig=<ID=1,")): print("  ", line.strip()[:110])
            continue
        if line.startswith("#"): print("  samples in file:", len(line.split("\t")) - 9); continue
        x = line.split("\t")
        if n < 2: print("  record chrom", x[0], "id", x[2][:10], "FORMAT", x[8])
        n += 1
print("  total records in one file:", n)
