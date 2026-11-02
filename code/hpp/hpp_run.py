"""Run ANCHOR on HPP data INSIDE the trusted research environment.

Prints and saves AGGREGATE results only (coefficients, CIs, variance components).
No participant-level data are written out.

Inputs
  --nights   nightly table (csv/parquet): one row per qualified night with
             participant id, clock time M (decimal h, after-midnight unwrapped)
             and gap D (h). Produced by the existing meal-timing audit pipeline.
  --person   person-level table: participant id, outcome column(s), covariates.
  --outcomes one or more outcome columns, or 'prefix:<str>' to scan all columns
             starting with <str> (e.g. metabolomics), with BH-FDR.

Example
  python hpp_run.py --nights nights.parquet --person person.parquet \
     --id participant_id --m M50 --d D50 \
     --outcomes liver_attenuation \
     --covars age age2 sex energy_kcal fat_pct protein_pct \
     --cat alcohol_freq --boot 500 --out results_liver.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "anchor"))
import anchor as A  # noqa: E402


def read(p):
    return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)


def bh(p):
    p = np.asarray(p)
    o = np.argsort(p)
    q = p[o] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nights", required=True)
    ap.add_argument("--person", required=True)
    ap.add_argument("--id", default="participant_id")
    ap.add_argument("--m", default="M50")
    ap.add_argument("--d", default="D50")
    ap.add_argument("--outcomes", nargs="+", required=True)
    ap.add_argument("--covars", nargs="*", default=[])
    ap.add_argument("--cat", nargs="*", default=[])
    ap.add_argument("--min_nights", type=int, default=2)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--out", default="anchor_results.json")
    a = ap.parse_args()

    nights = read(a.nights)[[a.id, a.m, a.d]].dropna()
    person = read(a.person)
    outs = a.outcomes
    if len(outs) == 1 and outs[0].startswith("prefix:"):
        pre = outs[0].split(":", 1)[1]
        outs = [c for c in person.columns if c.startswith(pre)]

    C = person[[a.id] + a.covars + a.cat].copy()
    if a.cat:
        C = pd.get_dummies(C, columns=a.cat, drop_first=True, dtype=float)
    covcols = [c for c in C.columns if c != a.id]

    results = {"n_outcomes": len(outs), "per_outcome": {}}
    for y in outs:
        df = person[[a.id, y]].merge(C, on=a.id).dropna()
        cnt = nights[a.id].value_counts()
        keep = set(cnt[cnt >= a.min_nights].index) & set(df[a.id])
        nn = nights[nights[a.id].isin(keep)].sort_values(a.id)
        df = df[df[a.id].isin(keep)].sort_values(a.id)
        pid = nn[a.id].to_numpy()
        Wn = nn[[a.m, a.d]].to_numpy(float)
        ids, Wbar, k, Su, dfree = A.person_summaries(pid, Wn)
        df = df.set_index(a.id).loc[ids]
        yv = df[y].to_numpy(float)
        Cm = df[covcols].to_numpy(float) if covcols else None

        b_n, V_n = A.fit_naive(Wbar, yv, Cm)
        b_a = A.fit_anchor(Wbar, k, Su, yv, Cm)
        draws = A.person_bootstrap(pid, Wn, yv, Cm, B=a.boot, seed=1)
        V_a = np.cov(draws.T)
        Sx = np.cov(Wbar.T) - Su * np.mean(1 / k)
        lam = A.attribution_matrix(A._nearest_pd(Sx), Su, k)
        rel = np.diag(Sx) / (np.diag(Sx) + np.diag(Su))  # single-night ICC
        res = {
            "n_people": int(len(ids)), "n_nights": int(len(pid)),
            "k_counts": {int(x): int((k == x).sum()) for x in np.unique(k)},
            "Sigma_U": Su.round(5).tolist(), "Sigma_X": Sx.round(5).tolist(),
            "single_night_ICC": rel.round(4).tolist(),
            "corr_within": float(Su[0, 1] / np.sqrt(Su[0, 0] * Su[1, 1])),
            "corr_between_true": float(Sx[0, 1] / np.sqrt(abs(Sx[0, 0] * Sx[1, 1]))),
            "Lambda": lam.round(4).tolist(),
            "naive": {"beta": b_n.tolist(), "se": np.sqrt(np.diag(V_n)).tolist(),
                      "tests": A.anchor_tests(b_n, V_n)},
            "anchor": {"beta": b_a.tolist(), "se": np.sqrt(np.diag(V_a)).tolist(),
                       "ci95_pct": np.percentile(draws, [2.5, 97.5], axis=0).T.tolist(),
                       "tests": A.anchor_tests(b_a, V_a)},
        }
        results["per_outcome"][y] = res
        print(y, "n=", len(ids), "naive", np.round(b_n, 5), "anchor", np.round(b_a, 5), flush=True)

    if len(outs) > 1:
        for meth in ["naive", "anchor"]:
            for t in ["H_gap_clockcoef", "H_clock_gapcoef", "fixed_sleep_contrast"]:
                ps = [results["per_outcome"][y][meth]["tests"][t][2] for y in outs]
                for y, q in zip(outs, bh(ps)):
                    results["per_outcome"][y][meth]["tests"][t] = list(results["per_outcome"][y][meth]["tests"][t]) + [float(q)]
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1, default=float)
    print("saved", a.out)


if __name__ == "__main__":
    main()
