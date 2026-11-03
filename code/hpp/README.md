# HPP analysis scripts

These are the scripts run inside the Human Phenotype Project (HPP) trusted research environment for the ANCHOR paper, kept as they were run. They print aggregate results only. This folder contains no participant-level data; the aggregate results these scripts printed are transcribed in [`results/hpp_aggregate_results.txt`](../../results/hpp_aggregate_results.txt).

## Running them

You need approved HPP access, and the scripts must run inside the HPP environment. They cannot run on HPP data elsewhere.

- Copy `code/anchor/anchor.py` to `~/anchor_work/anchor/` and `code/hpp/` to `~/anchor_work/hpp/`. Most scripts expect these paths; `hpp_run.py` uses the repository layout instead.
- Python packages: numpy, pandas, pyarrow and HPP's `pheno_utils`.
- Most scripts read intermediate tables from `~/studies/meal-timing-liver-fat/audit_20260918`, such as `paired_nights.parquet`, `model_complete_cases.parquet` and `cgm_qualified_nights.parquet`. These tables contain participant-level data and are not included, and neither is the pipeline that built them, so a new user must rebuild them from the cohort definition in the paper.
- Run `real8a.py` first for the blood-draw analyses. Its output is used by `real8b.py`, `real9.py`, `real10.py`, `real14.py` and `bgzf_extract.py`. Run `bgzf_extract.py` before `real10.py`.

## Scripts

| Script | Analysis |
|---|---|
| `real1.py` | Error structure, attribution matrix, liver naive and ANCHOR estimates |
| `real2.py` | Sleep-window glucose: naive, ANCHOR and within-person estimates |
| `real3.py` | Lipid scan in the subset with two or three linked nights |
| `real4.py`, `real5.py` | Glucose habitual and acute effects; liver robustness to extra adjustment |
| `real6.py`, `real7.py` | Sleep-aligned and meal-aligned 3-hour glucose windows on matched nights |
| `real8a.py` | Builds the blood-draw anchors (saved inside the environment) |
| `real8b.py` | Serum lipids: draw clock versus time since the last meal |
| `real9.py` | Habitual evening meal clock and the lipidome, alone and jointly with draw anchors |
| `bgzf_extract.py`, `real10.py` | Genotype extraction without an index; genetic chronotype score |
| `real11.py` | Follow-up visit and transport of the error structure |
| `real12.py` | Spike-in on the real nightly table |
| `real13.py`, `real14.py` | Checks of the independent-nights assumption |
| `real15.py` | Dated-night ANCHOR on the liver cohort |
| `real16.py` | Night-count adjustment and separate decay for each anchor |
| `real17.py` | Negative-control outcome (adult height) |
| `hpp_run.py` | General command-line runner for ANCHOR inside the environment |
| `probes/` | Early feasibility checks of data availability and file formats |

`data/chronotype_jones2019_gwascat.tsv` lists the morningness loci from the GWAS Catalog (accession GCST007565).
