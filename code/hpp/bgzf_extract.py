"""Random-access extraction of target variants from single-sample, position-sorted BGZF VCFs
without tabix/bcftools: binary search over compressed-block offsets. Output stays inside the TRE.

usage: python bgzf_extract.py test            # 3 files, timing + match rate
       python bgzf_extract.py run NPROC        # all genotyped participants in analysis cohorts
"""
import os, sys, glob, struct, zlib, time
import numpy as np, pandas as pd

VCF_DIR = os.path.expanduser("~/studies/human_genetics_bulk/reads/vcf")
SNP = os.path.expanduser("~/anchor_work/hpp/data/chronotype_jones2019_gwascat.tsv")
OUT = os.path.expanduser("~/anchor_work/grs_dosages.parquet")
CH = {str(i): i for i in range(1, 23)}; CH.update({"X": 23, "Y": 24, "MT": 25})
MAGIC = b"\x1f\x8b\x08\x04"


def read_block(fh, off):
    fh.seek(off); h = fh.read(18)
    if len(h) < 18 or h[:4] != MAGIC: return None, None
    xlen = struct.unpack("<H", h[10:12])[0]
    extra = h[12:12 + xlen] if xlen <= 6 else h[12:18] + fh.read(xlen - 6)
    bsize = None; i = 0
    while i < len(extra):
        si1, si2, slen = extra[i], extra[i + 1], struct.unpack("<H", extra[i + 2:i + 4])[0]
        if si1 == 66 and si2 == 67: bsize = struct.unpack("<H", extra[i + 4:i + 6])[0]
        i += 4 + slen
    fh.seek(off + 12 + xlen); cdata = fh.read(bsize - xlen - 19)
    return zlib.decompress(cdata, -15), off + bsize + 1


def next_block_at(fh, off, size):
    """first valid BGZF block start at or after off."""
    while off < size:
        fh.seek(off); buf = fh.read(1 << 17); j = 0
        while True:
            j = buf.find(MAGIC, j)
            if j < 0: break
            if j + 16 <= len(buf) and buf[j + 12:j + 14] == b"BC":
                cand = off + j
                try:
                    d, _ = read_block(fh, cand)
                    if d is not None: return cand
                except Exception: pass
            j += 1
        off += len(buf) - 16
    return None


def first_key(data):
    k = data.find(b"\n")
    if k < 0 or k + 1 >= len(data): return None
    line = data[k + 1:].split(b"\n", 1)[0]
    if line.startswith(b"#"): return None
    p = line.split(b"\t", 2)
    try: return (CH.get(p[0].decode().replace("chr", ""), 99), int(p[1]))
    except Exception: return None


def lookup(fh, size, targets):
    """targets: sorted list of (chr, pos). returns {(chr,pos): (ref, alt, ds)}"""
    res = {}; cache = {}
    def key_at(off):
        if off not in cache:
            b = next_block_at(fh, off, size)
            if b is None: cache[off] = (None, None)
            else:
                d, _ = read_block(fh, b); cache[off] = (b, first_key(d))
        return cache[off]
    for t in targets:
        lo, hi = 0, size
        best = 0
        for _ in range(40):
            if hi - lo < 1 << 16: break
            mid = (lo + hi) // 2; b, k = key_at(mid)
            if b is None or k is None or k > t: hi = mid
            else: lo = mid; best = b
        off = best; buf = b""; done = False; first = True
        for _ in range(64):
            d, nxt = read_block(fh, off)
            if d is None: break
            if first and off > 0: d = d[d.find(b"\n") + 1:]
            first = False
            buf += d; lines = buf.split(b"\n"); buf = lines.pop()
            for ln in lines:
                if not ln or ln[0:1] == b"#": continue
                p = ln.split(b"\t")
                if len(p) < 10: continue
                k = (CH.get(p[0].decode().replace("chr", ""), 99), int(p[1]))
                if k == t:
                    fmt = p[8].split(b":"); val = p[9].split(b":")
                    ds = float(val[fmt.index(b"DS")]) if b"DS" in fmt else np.nan
                    res[t] = (p[3].decode(), p[4].decode(), ds)
                if k > t: done = True; break
            if done: break
            off = nxt
    return res


def load_snps():
    s = pd.read_csv(SNP, sep="\t", dtype={"chr37": str})
    s["key"] = [(CH[c], int(p)) for c, p in zip(s.chr37, s.pos_grch37)]
    return s.sort_values("key").reset_index(drop=True)


def one(path, snps):
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        r = lookup(fh, size, list(snps.key))
    out = np.full(len(snps), np.nan)
    for i, (k, ea) in enumerate(zip(snps.key, snps.effect_allele)):
        if k in r:
            ref, alt, ds = r[k]
            if alt == ea: out[i] = ds
            elif ref == ea: out[i] = 2 - ds
    return out


def worker(args):
    pid, path = args
    try: return pid, one(path, load_snps())
    except Exception as e: return pid, None


if __name__ == "__main__":
    snps = load_snps()
    from pheno_utils import PhenoLoader
    hg = PhenoLoader("human_genetics").dfs["human_genetics"].reset_index()
    hg["path"] = [os.path.join(VCF_DIR, os.path.basename(str(v))) for v in hg.gencove_vcf]
    hg = hg[[os.path.exists(p) for p in hg.path]]
    with open(os.path.join(VCF_DIR, os.path.basename(hg.path.iloc[0])), "rb") as fh:
        print("BGZF:", fh.read(14)[12:14] == b"BC")
    if sys.argv[1] == "test":
        for p in hg.path.iloc[:3]:
            t0 = time.time(); d = one(p, snps)
            print(f"file {time.time()-t0:.1f}s matched {np.isfinite(d).sum()}/{len(snps)} mean dosage {np.nanmean(d):.3f}")
        print("expected mean dosage ~ 2*EAF:", round(2 * pd.to_numeric(snps.eaf, errors='coerce').mean(), 3))
    else:
        D = os.path.expanduser("~/studies/meal-timing-liver-fat/audit_20260918")
        anc = pd.read_parquet(os.path.expanduser("~/anchor_work/draw_anchors.parquet"))
        want = set(anc.participant_id) | set(pd.read_parquet(f"{D}/paired_nights.parquet").participant_id)
        todo = hg[hg.participant_id.isin(want)].drop_duplicates("participant_id")
        print("to extract:", len(todo), flush=True)
        from multiprocessing import Pool
        res = {}; t0 = time.time()
        with Pool(int(sys.argv[2])) as pool:
            for i, (pid, d) in enumerate(pool.imap_unordered(worker, zip(todo.participant_id, todo.path), chunksize=4)):
                if d is not None: res[pid] = d
                if i % 250 == 0: print(i, "done", round(time.time() - t0), "s", flush=True)
        df = pd.DataFrame.from_dict(res, orient="index", columns=snps.rsid); df.index.name = "participant_id"
        df.to_parquet(OUT); print("saved", df.shape, "match rate", round(float(np.isfinite(df.values).mean()), 3))
