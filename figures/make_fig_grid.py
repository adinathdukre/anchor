"""Appendix benchmark grid (Nullstrap-DE style): false gap-signal rate (top) and power for the true clock
effect (bottom) against cohort size, faceted by nights per person k, for the naive joint model, univariate
regression calibration and ANCHOR. Analytic large-sample rates at HPP calibration (validated against
simulation in Fig. 3d); points are Monte Carlo results where available (k = 2-3, results/sim_baselines.csv).
"""
import os, csv
from math import erf, sqrt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "..", "results")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8c8b86", "#e7e6e1"
GAP, ANCH, NAIVE = "#eb6834", "#1baf7a", "#8c8b86"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7,
                     "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
                     "xtick.labelsize": 6.4, "ytick.labelsize": 6.4, "axes.labelsize": 6.8, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "legend.frameon": False})
Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2))); rej = lambda z: 1 - Phi(1.96 - z) + Phi(-1.96 - z)
SX = np.array([[0.9626, -0.4944], [-0.4944, 0.7072]]); SU = np.array([[1.9585, -1.6261], [-1.6261, 1.9356]])
SDY = 0.113; BETA = np.array([0.008, 0.0])


def rates(n, ek, method):
    Sw = SX + SU * ek; L = np.linalg.solve(Sw, SX); bn = L @ BETA; s2 = SDY ** 2 - bn @ Sw @ bn
    V = s2 * np.linalg.inv(Sw) / n
    if method == "naive":
        b = bn
    elif method == "univariate":
        D = np.diag(np.diag(SX) / np.diag(Sw)); b = np.linalg.solve(D, bn); V = s2 * np.linalg.inv(D @ Sw @ D) / n
    else:
        Li = np.linalg.inv(L); b = BETA; V = Li @ V @ Li.T
    return rej(b[1] / np.sqrt(V[1, 1])), rej(b[0] / np.sqrt(V[0, 0]))


KS = [("k = 2–3 (HPP)", np.mean(1 / np.repeat([2, 3], [1144, 867]))), ("k = 7", 1 / 7), ("k = 14", 1 / 14), ("k = 30", 1 / 30)]
ns = np.logspace(np.log10(500), np.log10(3e5), 90)
M = [("univariate RC", GAP + "99", "-", 3.2), ("naive", "#4a4945", (0, (2.2, 1.6)), 1.1), ("ANCHOR", ANCH, "-", 1.6)]
fig, ax = plt.subplots(2, 4, figsize=(7.0, 3.1), sharex=True, sharey="row", gridspec_kw={"hspace": 0.25, "wspace": 0.12})
sb = list(csv.DictReader(open(os.path.join(R, "sim_baselines.csv"))))
mp = {"naive": NAIVE, "univ-RC": GAP, "ANCHOR": ANCH}
for j, (lab, ek) in enumerate(KS):
    for name, col, ls, lw in M:
        fr = np.array([rates(n, ek, name.split()[0]) for n in ns])
        ax[0, j].plot(ns, fr[:, 0], color=col, ls=ls, lw=lw, label=name, solid_capstyle="round")
        ax[1, j].plot(ns, fr[:, 1], color=col, ls=ls, lw=lw, solid_capstyle="round")
    if j == 0:
        for x in sb:
            if x["truth"] == "clock_only" and x["method"] in mp:
                ax[0, 0].scatter(float(x["n"]), float(x["false_cross"]), s=14, color=mp[x["method"]], edgecolor="white", lw=0.5, zorder=4)
    for i in (0, 1):
        a = ax[i, j]; a.set_xscale("log"); a.set_ylim(-0.02, 1.02); a.grid(axis="y", color=GRID, lw=0.5); a.set_axisbelow(True)
        a.axhline(0.05 if i == 0 else 0.8, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax[0, j].set_title(lab, fontsize=7, fontweight="bold", color=INK, pad=4)
    ax[1, j].set_xlabel("participants (n)")
ax[0, 0].set_ylabel("false gap-signal rate"); ax[1, 0].set_ylabel("power, true clock effect")
ax[0, 0].legend(loc="upper left", fontsize=6.2, handlelength=1.8)
fig.savefig(os.path.join(H, "figA1.pdf"), bbox_inches="tight", pad_inches=0.05)
fig.savefig(os.path.join(H, "figA1.png"), bbox_inches="tight", pad_inches=0.05, dpi=220)
for lab, ek in KS:
    print(lab, "naive false at n=64k:", round(rates(64000, ek, "naive")[0], 3), "| ANCHOR power at n=8k:", round(rates(8044, ek, "ANCHOR")[1], 3))
