"""Figure 3 (theory & simulation), styled after the RECOMB 2026 exemplars:
(a) failure-regime heatmap (Yelmen et al.): naive false-anchor probability over cohort size x within-night error
    correlation, with probability contours, the leakage-free line and HPP marked;
(b) leakage Lambda_21 over nights per person x error correlation;
(c) false-anchor rate vs n per estimator (Nullstrap-style), analytic curves with Monte Carlo points;
(d) theory vs simulation agreement across all simulated settings.
Uses only HPP aggregate covariances and simulation outputs in results/."""
import os, csv
import numpy as np
from math import erf, sqrt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "..", "results")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8c8b86", "#e7e6e1"
CLOCK, GAP, ANCH, NAIVE = "#2a78d6", "#eb6834", "#1baf7a", "#8c8b86"
NAIVE_LINE = "#4b4f58"   # darker grey so the naive theory line stays visible
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"], "font.size": 7,
                     "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
                     "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "axes.labelsize": 6.8, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "legend.frameon": False, "legend.fontsize": 6})
Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
rej = lambda z: 1 - Phi(1.96 - z) + Phi(-1.96 - z)

SX = np.array([[0.9626, -0.4944], [-0.4944, 0.7072]]); SUh = np.array([[1.9585, -1.6261], [-1.6261, 1.9356]])
SW = np.sqrt(np.diag(SUh)); RHO_HPP = SUh[0, 1] / (SW[0] * SW[1]); SDY = 0.113; EK = np.mean(1 / np.repeat([2, 3], [1144, 867]))
BETA = np.array([0.008, 0.0])

def SU_of(rho): return np.array([[SW[0] ** 2, rho * SW[0] * SW[1]], [rho * SW[0] * SW[1], SW[1] ** 2]])

def naive_rate(n, rho, ek=EK, beta=BETA, Sx=SX, sdy=SDY, j=1, D=None, k1=False):
    Su = SU_of(rho) if np.ndim(rho) == 0 else rho
    Se = Su * (1.0 if k1 else ek); Sw = Sx + Se; lam = np.linalg.solve(Sw, Sx); bn = lam @ beta
    s2 = sdy ** 2 - bn @ Sw @ bn
    if D is not None:  # univariate calibration: coefficient D^-1 Lambda beta, covariance s2 (D Sw D)^-1
        bn = np.linalg.solve(D, bn); V = s2 * np.linalg.inv(D @ Sw @ D) / n
    else:
        V = s2 * np.linalg.inv(Sw) / n
    return rej(bn[j] / np.sqrt(V[j, j])), bn

fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.15), gridspec_kw={"width_ratios": [1.15, 1.15, 1.0, 0.85], "wspace": 1.0})
cm = LinearSegmentedColormap.from_list("w", ["#fbfbf9", "#fde3d4", "#f6a77f", "#eb6834", "#a23b14"])

# ---- (a) failure-regime heatmap
ns = np.logspace(np.log10(500), 6, 160); rhos = np.linspace(-0.95, 0.0, 150)
Z = np.array([[naive_rate(n, r)[0] for n in ns] for r in rhos])
a = ax[0]; im = a.pcolormesh(ns, rhos, Z, cmap=cm, vmin=0, vmax=1, shading="auto", rasterized=True)
cs = a.contour(ns, rhos, Z, levels=[0.1, 0.5, 0.9], colors=[INK2], linewidths=0.6, linestyles=["--", "-", "--"])
# place contour labels inside the panel (on the lower branch, rho = -0.72) so none is clipped or overlapping
LAB_RHO = -0.72; iz = np.argmin(np.abs(rhos - LAB_RHO)); manual = []
for lev in (0.1, 0.5, 0.9):
    crossing = np.where(Z[iz] >= lev)[0]
    if len(crossing): manual.append((ns[crossing[0]], LAB_RHO))
a.clabel(cs, fmt={0.1: "0.1", 0.5: "0.5", 0.9: "0.9"}, fontsize=5.8, inline_spacing=2, manual=manual)
rho0 = SX[0, 1] / SX[0, 0] * SW[0] / SW[1]
a.axhline(rho0, color=CLOCK, lw=0.9, ls=(0, (3, 1.5))); a.text(6.5e5, rho0 + 0.035, "no leakage", color=CLOCK, fontsize=6.1, ha="right")
a.scatter([2011], [RHO_HPP], marker="*", s=60, color=INK, edgecolor="white", lw=0.5, zorder=5); a.text(2600, RHO_HPP - 0.07, "HPP", fontsize=6.1, color=INK)
a.set_xscale("log"); a.set_xlim(500, 1e6); a.set_xlabel("participants (n)"); a.set_ylabel("within-night corr. of M, D errors")
cb = fig.colorbar(im, ax=a, fraction=0.05, pad=0.03); cb.ax.tick_params(labelsize=5.4, length=1.5); cb.outline.set_visible(False)
cb.set_label("P(false gap)", fontsize=6.1, labelpad=2)

# ---- (b) leakage vs nights per person
ks = np.linspace(1, 20, 120); Lz = np.array([[np.linalg.solve(SX + SU_of(r) / kk, SX)[1, 0] for kk in ks] for r in rhos])
b = ax[1]; cmb = LinearSegmentedColormap.from_list("b", ["#fbfbf9", "#cde2fb", "#6da7ec", "#2a78d6", "#104281"])
lim = np.nanmax(np.abs(Lz)); im2 = b.pcolormesh(ks, rhos, np.abs(Lz), cmap=cmb, vmin=0, vmax=lim, shading="auto", rasterized=True)
cs2 = b.contour(ks, rhos, np.abs(Lz), levels=[0.05, 0.1, 0.2], colors=[INK2], linewidths=0.6)
b.axhline(rho0, color=CLOCK, lw=0.9, ls=(0, (3, 1.5)))
b.scatter([1 / EK], [RHO_HPP], marker="*", s=60, color=INK, edgecolor="white", lw=0.5, zorder=5); b.text(3.4, RHO_HPP - 0.035, "HPP", fontsize=6.1)
b.scatter([14], [RHO_HPP], marker="D", s=16, color=ANCH, edgecolor="white", lw=0.5, zorder=5); b.text(14, RHO_HPP + 0.05, "14 evenings", fontsize=6.1, ha="center")
b.set_xlabel("nights per person (k)"); b.set_yticklabels([])
cb2 = fig.colorbar(im2, ax=b, fraction=0.05, pad=0.03); cb2.ax.tick_params(labelsize=5.4, length=1.5); cb2.outline.set_visible(False)
cb2.set_label(r"leakage $|\Lambda_{21}|$", fontsize=6.1)

# ---- (c) false-anchor rate vs n by estimator
c = ax[2]; nn = np.logspace(np.log10(500), 5.3, 80)
Sw = SX + SUh * EK; D = np.diag(np.diag(SX) / np.diag(Sw))
# naive and univariate calibration give the same false-gap rate; draw naive solid dark and univariate dashed on top
c.plot(nn, [naive_rate(n, SUh)[0] for n in nn], color=NAIVE_LINE, lw=2.2, ls="-", solid_capstyle="round", label="naive", zorder=3)
c.plot(nn, [naive_rate(n, SUh, D=D)[0] for n in nn], color=GAP, lw=1.3, ls=(0, (3, 2)), label="univariate", zorder=4)
c.plot(nn, [0.05 for n in nn], color=ANCH, lw=1.5, label="ANCHOR", zorder=3)
sb = list(csv.DictReader(open(os.path.join(R, "sim_baselines.csv"))))
mp = {"naive": NAIVE_LINE, "pooled": "#b8b6ae", "univ-RC": GAP, "ANCHOR": ANCH}
for x in sb:
    if x["truth"] == "clock_only" and x["method"] in mp:
        c.scatter([float(x["n"])], [float(x["false_cross"])], s=16, color=mp[x["method"]], edgecolor="white", lw=0.6, zorder=5)
c.axhline(0.05, color=MUTED, lw=0.6, ls=(0, (3, 2)))
c.set_xscale("log"); c.set_ylim(0, 1); c.set_xlim(500, 2e5); c.set_xlabel("participants (n)"); c.set_ylabel("false-signal rate")
c.grid(axis="y", color=GRID, lw=0.5); c.set_axisbelow(True)
c.scatter([], [], s=16, color="#b8b6ae", label="pooled (sim.)")
c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=5.8, handlelength=1.8, columnspacing=0.8)

# ---- (d) theory vs simulation
d = ax[3]; pts = []
for x in list(csv.DictReader(open(os.path.join(R, "sim_real.csv")))) + sb:
    if x["method"] == "naive" and x["truth"] in ("clock_only", "gap_only"):
        beta = BETA if x["truth"] == "clock_only" else np.array([0, -0.008]); j = 1 if x["truth"] == "clock_only" else 0
        fx = x.get("false_cross_signal", x.get("false_cross"))
        pts.append((naive_rate(float(x["n"]), SUh, beta=beta, j=j)[0], float(fx), CLOCK if j == 0 else GAP))
d.plot([0, 0.4], [0, 0.4], color=MUTED, lw=0.7, ls=(0, (3, 2)))
for p, q, col in pts: d.scatter([p], [q], s=18, color=col, edgecolor="white", lw=0.6, zorder=3)
d.set_xlim(0, 0.4); d.set_ylim(0, 0.4)
d.set_xlabel("theory"); d.set_ylabel("simulation")
d.text(0.02, 0.38, "naive false-anchor rate", fontsize=6.1, color=INK2, va="top")
d.scatter([], [], s=14, color=GAP, label="spurious gap"); d.scatter([], [], s=14, color=CLOCK, label="spurious clock")
d.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9), fontsize=6.1, handletextpad=0.1)

for a_, L, T in zip(ax, "abcd", ["Where naive attribution fails", "Leakage shrinks with nights", "False signals by method", "Theory matches simulation"]):
    a_.set_title(f"$\\bf{{{L}}}$   {T}", loc="left", fontsize=7.2, fontweight="bold", pad=6)
fig.savefig(os.path.join(H, "fig2.pdf"), bbox_inches="tight", pad_inches=0.05)
fig.savefig(os.path.join(H, "fig2.png"), bbox_inches="tight", pad_inches=0.05, dpi=240)
print("fig2 written; leakage-free rho", round(rho0, 3), "HPP rho", round(RHO_HPP, 3))
