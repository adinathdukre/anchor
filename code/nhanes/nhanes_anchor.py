"""Independent replication of the anchor-basis result in NHANES 2017-2018 (public data).

A blood draw has a clock time T (here the examination session: morning, afternoon or evening, coded by
the session midpoint 9, 14 or 18.5 h) and a time since the last meal F (PHAFSTHR + PHAFSTMN/60). The last
meal clock is L = T - F. For each marker measured in all sessions (HDL and total cholesterol) we report the
fasting coefficient under three anchor choices:
  F alone (the usual fasting adjustment), F holding the draw clock T fixed, and F holding the meal clock L fixed.
Lemma A.1 predicts these are three different contrasts: beta_F + beta_T Cov(T,F)/Var(F), beta_F, beta_F + beta_T.
Adjusted for age, age^2 and sex; HC3 standard errors; unweighted (survey weights ignored). Writes results/nhanes_anchor.csv.
"""
import os, sys, csv
import numpy as np, pandas as pd

H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, "..", "..", "data_external", "nhanes_2017_2018")
OUT = os.path.join(H, "..", "..", "results")
rd = lambda f: pd.read_sas(os.path.join(D, f + ".xpt"), format="xport").set_index("SEQN")
F = rd("FASTQX_J"); Dm = rd("DEMO_J")[["RIDAGEYR", "RIAGENDR"]]
df = F[["PHAFSTHR", "PHAFSTMN", "PHDSESN"]].join(Dm).join(rd("HDL_J")[["LBDHDD"]]).join(rd("TCHOL_J")[["LBXTC"]])
df = df[(df.RIDAGEYR >= 18)].dropna(subset=["PHAFSTHR", "PHAFSTMN", "PHDSESN"])
df["F"] = df.PHAFSTHR + df.PHAFSTMN / 60.0
df = df[(df.F > 0) & (df.F <= 24)]
df["T"] = df.PHDSESN.round().astype(int).map({0: 9.0, 1: 14.0, 2: 18.5})
df["L"] = df["T"] - df["F"]


def ols(y, X):
    A = np.column_stack([np.ones(len(y)), X]); b = np.linalg.lstsq(A, y, rcond=None)[0]; r = y - A @ b
    Ai = np.linalg.inv(A.T @ A); h = np.einsum("ij,jk,ik->i", A, Ai, A); V = Ai @ (A.T * (r / (1 - h)) ** 2) @ A @ Ai
    return b[1:], np.sqrt(np.diag(V))[1:]


rows = []
print(f"adults with fasting time and session: {len(df)}; corr(T, F) = {df[['T', 'F']].corr().iloc[0, 1]:.2f}; sessions {df.PHDSESN.value_counts().sort_index().to_dict()}")
for y, lab in [("LBDHDD", "HDL cholesterol"), ("LBXTC", "total cholesterol")]:
    d = df.dropna(subset=[y]); yy = np.log(d[y].to_numpy(float)); yy = (yy - yy.mean()) / yy.std()
    cov = np.column_stack([d.RIDAGEYR, d.RIDAGEYR ** 2, (d.RIAGENDR == 2).astype(float)])
    specs = {"F alone": d[["F"]], "F | T fixed": d[["F", "T"]], "F | L fixed": d[["F", "L"]]}
    for name, X in specs.items():
        b, se = ols(yy, np.column_stack([X.to_numpy(float), cov]))
        z = b[0] / se[0]
        rows.append(dict(marker=lab, n=len(d), anchor=name, beta_F_SD_per_h=round(b[0], 4), se=round(se[0], 4), z=round(z, 2)))
        print(rows[-1])
with open(os.path.join(OUT, "nhanes_anchor.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# ---- sensitivity analyses (design confounding: morning-session participants form the fasting subsample)
sens = []
def run(d, label, Tmap=None, categorical=False):
    d = d.copy()
    if Tmap is not None:
        d["T"] = d.PHDSESN.round().astype(int).map(Tmap); d["L"] = d["T"] - d["F"]
    for y, lab in [("LBDHDD", "HDL cholesterol"), ("LBXTC", "total cholesterol")]:
        e = d.dropna(subset=[y]); yy = np.log(e[y].to_numpy(float)); yy = (yy - yy.mean()) / yy.std()
        cov = np.column_stack([e.RIDAGEYR, e.RIDAGEYR ** 2, (e.RIAGENDR == 2).astype(float)])
        ses = pd.get_dummies(e.PHDSESN.round().astype(int), drop_first=True, dtype=float).to_numpy()
        specs = {"F alone": e[["F"]].to_numpy(float)}
        specs["F | T fixed"] = np.column_stack([e["F"], ses]) if categorical else e[["F", "T"]].to_numpy(float)
        if not categorical:
            specs["F | L fixed"] = e[["F", "L"]].to_numpy(float)
        for name, X in specs.items():
            if X.shape[1] > 1 and np.linalg.matrix_rank(X - X.mean(0)) < X.shape[1]:
                continue
            b, se = ols(yy, np.column_stack([X, cov]))
            sens.append(dict(analysis=label, marker=lab, n=len(e), anchor=name, beta_F=round(b[0], 4), z=round(b[0] / se[0], 2)))
            print(sens[-1])
run(df, "session categorical", categorical=True)
run(df, "clock coding 9/13/16 (compressed)", Tmap={0: 9.0, 1: 13.0, 2: 16.0})
run(df, "clock coding 8/14/20 (stretched)", Tmap={0: 8.0, 1: 14.0, 2: 20.0})
run(df[df.PHDSESN.round() > 0], "afternoon and evening sessions only")
with open(os.path.join(OUT, "nhanes_anchor_sensitivity.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(sens[0])); w.writeheader(); w.writerows(sens)
