"""Publication figures for the ANCHOR paper.

Uses only aggregate quantities: estimated covariance matrices, simulation outputs and
HPP estimates transcribed from the trusted research environment (results/hpp_aggregate_results.txt).
No participant-level data. Run:  python figures/make_figures.py
Colour roles (validated categorical palette, CVD-safe): clock = blue, gap/fasting = orange,
ANCHOR / habitual = aqua, naive = grey, oracle = light grey.
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch
from matplotlib.lines import Line2D

H = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(H, "..", "results")

# ---------------------------------------------------------------- style
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8c8b86", "#e7e6e1", "#ffffff"
CLOCK, GAP, ANCH = "#2a78d6", "#eb6834", "#1baf7a"
NAIVE, ORACLE = "#8c8b86", "#c9c8c2"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 0,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "legend.frameon": False, "legend.fontsize": 6.6, "figure.dpi": 300, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42, "text.color": INK,
})

def panel(ax, letter, title, x=None):
    ax._ptitle = (letter, title)

def _place_titles(fig):
    fig.canvas.draw()
    axs = [a for a in fig.axes if hasattr(a, "_ptitle")]
    if not axs: return
    top = max(a.get_position().y1 for a in axs)
    for a in axs:
        x0 = a.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).x0
        x0 = max(x0, 0.0)
        letter, title = a._ptitle
        fig.text(x0, top + 0.045, letter, fontsize=9.0, fontweight="bold", ha="left", va="bottom", color=INK)
        fig.text(x0 + 0.022, top + 0.045, title, fontsize=7.5, fontweight="bold", ha="left", va="bottom", color=INK)

def vgrid(ax, zero=True):
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
    if zero: ax.axvline(0, color=INK2, lw=0.6, zorder=1)
    ax.spines["left"].set_visible(False)

def save(fig, name):
    _place_titles(fig)
    fig.savefig(os.path.join(H, name + ".pdf"), bbox_inches="tight", pad_inches=0.06)
    fig.savefig(os.path.join(H, name + ".png"), bbox_inches="tight", pad_inches=0.06, dpi=220)
    plt.close(fig)

def rd(f): return list(csv.DictReader(open(os.path.join(R, f))))

# aggregate HPP quantities (results/hpp_aggregate_results.txt)
SU = np.array([[1.9585, -1.6261], [-1.6261, 1.9356]]); SX = np.array([[0.9626, -0.4944], [-0.4944, 0.7072]])
LAM_B = np.array([[0.660, 0.061], [0.190, 0.508]]); LAM_F = np.array([[0.673, 0.088], [0.237, 0.497]])
k = np.repeat([2, 3], [1144, 867]); SE_MEAN = SU * np.mean(1 / k)

# ====================================================================== Figure 1
from matplotlib.patches import FancyBboxPatch, Circle, Wedge, Rectangle
from matplotlib.colors import LinearSegmentedColormap
NIGHT1, NIGHT2, ANCH_D = "#fdf3e7", "#e6ecfa", "#0e7a54"
def plate(ax, x, y, z=5):
    ax.scatter([x], [y], s=70, facecolor="white", edgecolor=CLOCK, lw=1.1, zorder=z)
    ax.scatter([x], [y], s=18, facecolor="#cfe0f8", edgecolor="none", zorder=z + 0.1)
def moon(ax, x, y, bg=NIGHT2, z=5):
    ax.scatter([x], [y], s=62, facecolor="#f2b84b", edgecolor="none", zorder=z)
    ax.scatter([x + 0.09], [y + 0.06], s=50, facecolor=bg, edgecolor="none", zorder=z + 0.1)
fig = plt.figure(figsize=(7.0, 2.5))
gs = fig.add_gridspec(1, 4, width_ratios=[1.38, 0.92, 0.58, 1.30], wspace=0.5)
a = fig.add_subplot(gs[0]); sky = LinearSegmentedColormap.from_list("sky", [NIGHT1, NIGHT2])
nights = [(20.0, 23.0), (21.0, 22.5), (19.6, 24.2)]
for i, (m, s_) in enumerate(nights):
    y = 2 - i
    a.imshow(np.linspace(0, 1, 100)[None, :], extent=(18.6, 25.4, y - 0.30, y + 0.30), cmap=sky, aspect="auto", zorder=1)
    a.add_patch(FancyBboxPatch((18.6, y - 0.30), 6.8, 0.60, boxstyle="round,pad=0,rounding_size=0.18", fc="none", ec=GRID, lw=0.7, zorder=1.5))
    a.annotate("", xy=(s_ - 0.25, y - 0.05), xytext=(m + 0.28, y - 0.05), arrowprops=dict(arrowstyle="<->", color=GAP, lw=1.0, shrinkA=0, shrinkB=0), zorder=4)
    a.text((m + s_) / 2 + (0.35 if s_ - m < 2 else 0), y + 0.07, f"D = {s_ - m:.1f} h", color=GAP, ha="center", va="bottom", fontsize=6.0, fontweight="bold", zorder=6)
    plate(a, m, y); moon(a, s_, y, bg=matplotlib.colors.to_hex(sky((s_ + 0.09 - 18.6) / 6.8)))
    a.add_patch(FancyBboxPatch((17.15, y - 0.17), 1.2, 0.34, boxstyle="round,pad=0,rounding_size=0.15", fc="#f3f2ee", ec="none", zorder=1))
    a.text(17.75, y, f"night {i+1}", ha="center", va="center", fontsize=6.0, color=INK2)
a.axvline(np.mean([n[0] for n in nights]), ymin=0.05, ymax=0.83, color=CLOCK, lw=0.8, ls=(0, (2, 2)), zorder=0.5)
a.text(np.mean([n[0] for n in nights]), -0.52, "usual M", color=CLOCK, fontsize=6.0, ha="center", va="center", fontweight="bold")
a.set_xlim(17.0, 25.6); a.set_ylim(-0.75, 2.75)
a.set_xticks([19, 21, 23, 25], ["19:00", "21:00", "23:00", "01:00"]); a.set_yticks([])
a.spines["left"].set_visible(False)
a.text(21.9, 2.55, "habit $X_i$ = average of $k_i$ = 2–3 nights", color=INK2, fontsize=6.2, ha="center", va="center")
panel(a, "a", "Two anchors, few nights")
b = fig.add_subplot(gs[1])
rng0 = np.random.default_rng(3); pts = rng0.multivariate_normal([0, 0], SX, 260)
b.scatter(pts[:, 0], pts[:, 1], s=3, color=CLOCK, alpha=0.28, lw=0, zorder=1)
def ell(ax, S, color, ls, fill, lw=1.3):
    w, V = np.linalg.eigh(S); ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    ax.add_patch(Ellipse((0, 0), 2 * 1.96 * np.sqrt(w[1]), 2 * 1.96 * np.sqrt(w[0]), angle=ang, facecolor=fill, edgecolor=color, lw=lw, ls=ls, zorder=2))
    x = np.linspace(-2.6, 2.6, 2); ax.plot(x, S[0, 1] / S[0, 0] * x, color=color, lw=0.9, ls=(0, (3, 2)), zorder=3)
ell(b, SX, CLOCK, "-", CLOCK + "14"); ell(b, SE_MEAN, GAP, "--", GAP + "10")
b.set_xlim(-2.7, 2.7); b.set_ylim(-2.7, 2.7); b.set_aspect("equal")
b.set_xlabel("clock M (h, centred)"); b.set_ylabel("gap D (h, centred)", labelpad=1)
b.set_xticks([-2, 0, 2]); b.set_yticks([-2, 0, 2]); b.tick_params(axis="y", length=2.5)
b.text(1.2, 1.85, "habits\nslope −0.51", color=CLOCK, fontsize=6.2, ha="left", va="center", fontweight="bold")
b.text(-2.55, -2.5, "night error\nslope −0.83", color=GAP, fontsize=6.2, ha="left", va="bottom", fontweight="bold")
panel(b, "b", "Error ≠ habit")
cmap = LinearSegmentedColormap.from_list("b", ["#eef4fc"] + BLUES)
# Λ matrix; the leakage cell is outlined and labelled below the matrix (not inside the cell)
c = fig.add_subplot(gs[2]); c.set_xlim(-0.55, 1.55); c.set_ylim(1.95, -0.55); c.set_aspect("equal")
for (r, q), v in np.ndenumerate(LAM_B):
    fc = matplotlib.colors.to_hex(cmap(v / 0.8))
    c.add_patch(FancyBboxPatch((q - 0.45, r - 0.45), 0.9, 0.9, boxstyle="round,pad=0,rounding_size=0.12", fc=fc,
                               ec=GAP if (r, q) == (1, 0) else "none", lw=1.4, zorder=2))
    c.text(q, r, f"{v:.2f}", ha="center", va="center", fontsize=7.0, color="white" if v > 0.45 else INK, fontweight="bold" if r == q else "normal", zorder=3)
c.text(0, 1.72, "leakage", ha="center", va="center", fontsize=5.6, color=GAP, fontweight="bold", zorder=3)
c.set_xticks([0, 1], ["true\nclock", "true\ngap"], fontsize=6.2); c.set_yticks([0, 1], ["naive clock", "naive gap"], fontsize=6.2)
c.xaxis.set_ticks_position("top")
for sp in c.spines.values(): sp.set_visible(False)
c.tick_params(length=0)
panel(c, "c", "Leakage Λ")

# (d) ClusterDE-style decision glyphs: what each analysis reports when only the clock matters
from math import erf, sqrt
Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2))); rej = lambda z: 1 - Phi(1.96 - z) + Phi(-1.96 - z)
BETA = np.array([0.008, 0.0]); EK = np.mean(1 / k); Sw = SX + SU * EK; LAMd = np.linalg.solve(Sw, SX); bn = LAMd @ BETA
s2 = 0.113 ** 2 - bn @ Sw @ bn
def p_report(n, method):
    V = s2 * np.linalg.inv(Sw) / n
    if method == "naive": return [rej(bn[j] / np.sqrt(V[j, j])) for j in (0, 1)]
    Li = np.linalg.inv(LAMd); VA = Li @ V @ Li.T
    return [rej(BETA[j] / np.sqrt(VA[j, j])) for j in (0, 1)]
cols = [("truth", None), ("naive\n2k", ("naive", 2011)), ("naive\n64k", ("naive", 64000)), ("ANCHOR\n64k", ("anchor", 64000))]
d = fig.add_subplot(gs[3]); d.set_xlim(-0.55, 3.55); d.set_ylim(2.25, -1.0); d.set_aspect("equal")
truth = [True, False]
for q, (lab, spec) in enumerate(cols):
    pr = [1.0, 0.0] if spec is None else p_report(spec[1], spec[0])
    for r in range(2):
        said = pr[r] > 0.5; right = said == truth[r]
        fc = "#f3f2ee" if spec is None else ("#e3f4ec" if right else "#fbe1d6")
        d.add_patch(FancyBboxPatch((q - 0.42, r - 0.40), 0.84, 0.80, boxstyle="round,pad=0,rounding_size=0.12", fc=fc,
                                   ec=(ANCH if right else GAP) if spec is not None else "none", lw=0.9, zorder=2))
        d.text(q, r - 0.11, "✓" if said else "✗", ha="center", va="center", fontsize=10, fontfamily="DejaVu Sans",
               color=INK if spec is None else (ANCH_D if right else "#b4441a"), fontweight="bold", zorder=3)
        if spec is not None:
            pct = pr[r] if spec[0] == "naive" or r == 0 else 0.05
            d.text(q, r + 0.27, f"{100 * pct:.0f}%", ha="center", va="center", fontsize=5.2, color=INK2, zorder=3)
    d.text(q, -0.5, lab, ha="center", va="bottom", fontsize=5.8, color=ANCH_D if q == 3 else (INK if q else INK2), fontweight="bold" if q == 3 else "normal")
d.set_xticks([]); d.set_yticks([0, 1], ["clock\neffect?", "gap\neffect?"], fontsize=6.2)
for sp in d.spines.values(): sp.set_visible(False)
d.tick_params(length=0)
d.text(1.5, 1.95, "truth: clock only; % of replicates reporting an effect", ha="center", va="center", fontsize=5.3, color=INK2)
panel(d, "d", "What is concluded")

save(fig, "fig1")

# Figure 2 is produced by make_fig2_regimes.py

# ====================================================================== Figure 3
K = 1e3
liver = [("naive", (5.04, -0.21, 10.29), (0.53, -5.11, 6.17), "full"),
         ("ANCHOR", (7.96, 0.61, 15.37), (-1.73, -12.24, 7.59), "full"),
         ("ANCHOR + BMI", (6.4, 0.2, 14.9), None, "full"),
         ("ANCHOR + VAT", (2.1, -6.2, 10.3), None, "full"),
         ("ANCHOR, 2nd visit", (3.79, -7.78, 18.56), (1.73, -15.44, 21.75), "rep")]
cgm = [("naive between", (-0.85, -1.75, -0.07), (-2.58, -3.57, -1.71), "full"),
       ("one-shot correction", (-0.75, -1.96, 0.38), (-4.61, -6.74, -3.09), "bad"),
       ("acute (within person)", (-1.06, -1.70, -0.35), (-2.44, -3.14, -1.77), "full"),
       ("habitual (ANCHOR)", (-0.72, -1.96, 0.36), (-2.87, -4.63, -1.39), "full")]
win = [("sleep-aligned", (-2.46, -4.30, -0.76)), ("meal-aligned", (0.99, -0.98, 2.92)), ("difference", (-3.44, -4.58, -2.41))]
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [1.15, 1.15, 0.85], "wspace": 0.95})
BAND = "#f7f6f2"
def forest(a, rows, scale=1.0):
    n = len(rows)
    for i, (lab, c, g, kind) in enumerate(rows):
        y = n - 1 - i
        if i % 2 == 0: a.axhspan(y - 0.5, y + 0.5, color=BAND, zorder=0, lw=0)
        if "ANCHOR" in lab and kind != "rep": a.axhspan(y - 0.5, y + 0.5, color="#e8f6ef", zorder=0, lw=0)
        for est, col, mk, dy in [(c, CLOCK, "o", 0.15), (g, GAP, "s", -0.15)]:
            if est is None: continue
            e, lo, hi = [v * scale for v in est]
            hollow = kind == "bad"
            a.plot([lo, hi], [y + dy, y + dy], color=col, lw=1.6, alpha=0.45 if hollow else 1, solid_capstyle="round", zorder=2)
            a.scatter([e], [y + dy], s=30 if mk == "o" else 24, marker=mk, facecolor="white" if hollow else col, edgecolor="white" if not hollow else col, lw=1.0 if hollow else 0.7, zorder=3)
    a.set_yticks(range(n)[::-1], [r[0] for r in rows]); vgrid(a); a.set_ylim(-0.5, n - 0.5)
    for t in a.get_yticklabels():
        if "ANCHOR" in t.get_text() or "habitual" in t.get_text(): t.set_color(ANCH_D); t.set_fontweight("bold")
    if any(r[3] == "rep" for r in rows):
        a.axhspan(-0.5, 0.5, color="#eeedf7", zorder=0, lw=0); a.text(a.get_xlim()[1], 0.38, "replication", fontsize=5.6, color=INK2, ha="right", va="top", style="italic")
forest(ax[0], liver); ax[0].set_xlabel(r"liver attenuation per hour ($\times10^{-3}$ dB/cm/MHz)")
ax[0].text(16.5, 3.15, "p = 0.032", fontsize=6.0, color=ANCH_D, ha="left", va="center", fontweight="bold")
panel(ax[0], "a", "Liver (one scan, n = 2,011)")
forest(ax[1], cgm); ax[1].set_xlabel("early-sleep glucose (mg/dL per hour)")
ax[1].annotate("over-corrects", xy=(-4.61, 1.85), xytext=(-5.9, 1.42), fontsize=6.0, color=INK2, ha="center", va="top",
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
panel(ax[1], "b", "Glucose (every night, n = 1,250)")
c = ax[2]
c.axhspan(0.5, 1.5, color="#fdf1ea", zorder=0, lw=0)
for i, (lab, (e, lo, hi)) in enumerate(win):
    y = 2 - i; col = GAP if i < 2 else INK
    c.plot([lo, hi], [y, y], color=col, lw=1.6, solid_capstyle="round", zorder=2)
    c.scatter([e], [y], s=26, marker="s" if i < 2 else "D", color=col, edgecolor="white", lw=0.7, zorder=3)
c.set_yticks([2, 1, 0], [w[0] for w in win]); vgrid(c); c.set_xlabel("gap effect (mg/dL per hour)"); c.set_ylim(-0.5, 2.5)
c.text(c.get_xlim()[1], 1.38, "negative control", fontsize=5.8, color=GAP, ha="right", va="top", style="italic")
panel(c, "c", "Window alignment")
hand = [Line2D([], [], marker="o", color=CLOCK, lw=1.6, ms=4.5, mec="white", label="clock M"), Line2D([], [], marker="s", color=GAP, lw=1.6, ms=4, mec="white", label="gap D"),
        matplotlib.patches.Patch(color="#e8f6ef", label="ANCHOR")]
fig.legend(handles=hand, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.04), fontsize=6.3)
save(fig, "fig3")

# ====================================================================== Figure 4 (Nullstrap/ClusterDE-style molecular figure)
from matplotlib.patches import Circle as _C
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap as _LSC
fig, ax = plt.subplots(1, 3, figsize=(7.0, 1.8), gridspec_kw={"width_ratios": [0.95, 1.3, 0.8], "wspace": 1.25})
a = ax[0]; a.set_xlim(0, 10); a.set_ylim(0, 8); a.set_aspect("equal"); a.axis("off")
a.add_patch(_C((3.9, 4.2), 2.85, fc=CLOCK, alpha=0.20, ec=CLOCK, lw=1.2)); a.add_patch(_C((6.4, 4.2), 1.95, fc=GAP, alpha=0.22, ec=GAP, lw=1.2))
a.text(2.4, 4.2, "38", fontsize=10.0, fontweight="bold", ha="center", va="center", color=CLOCK)
a.text(5.25, 4.2, "19", fontsize=10.0, fontweight="bold", ha="center", va="center", color=INK)
a.text(7.45, 4.2, "6", fontsize=10.0, fontweight="bold", ha="center", va="center", color=GAP)
a.text(2.2, 7.4, "draw clock\nT (57)", fontsize=6.4, color=CLOCK, ha="center", fontweight="bold", linespacing=1.0)
a.text(9.0, 7.4, "since meal\nF (25)", fontsize=6.4, color=GAP, ha="center", fontweight="bold", linespacing=1.0)
a.text(5.0, 0.55, "F alone (different contrast): 16", fontsize=6.2, color=INK2, ha="center")
a.text(5.0, -0.45, "habitual meal clock (ANCHOR): 56", fontsize=6.2, color="#0e7a54", ha="center", fontweight="bold")
panel(a, "a", "Lipids by anchor")
b = ax[1]
cls_ = ["PE", "LPE", "DAG", "FA", "TAG", "PC"]; tot = np.array([5, 5, 5, 8, 73, 25])
hit = {"T": [5, 5, 5, 3, 24, 1], "F": [5, 4, 0, 3, 5, 2]}
eff = {"T": [0.060, 0.085, 0.045, 0.013, 0.024, -0.004], "F": [-0.057, -0.042, 0.0, 0.041, -0.014, 0.005]}
pv = {"T": [0.0038, 0.0038, 0.0038, 1.0, 1.0, 0.00037], "F": [4.8e-5, 0.0017, 1.0, 0.095, 0.015, 0.54]}
cmap = _LSC.from_list("div", ["#2a78d6", "#f0efec", "#e34948"]); norm = TwoSlopeNorm(vmin=-0.09, vcenter=0, vmax=0.09)
for xi, key in enumerate(["T", "F"]):
    for yi, c_ in enumerate(cls_):
        fr = hit[key][yi] / tot[yi]; sig = pv[key][yi] < 0.05
        b.scatter([xi], [len(cls_) - 1 - yi], s=20 + 420 * fr, c=[eff[key][yi]], cmap=cmap, norm=norm,
                  edgecolor=INK if sig else "#c9c8c2", lw=1.1 if sig else 0.5, zorder=3)
b.set_xticks([0, 1], ["draw\nclock T", "since\nmeal F"]); b.set_xlim(-0.6, 1.6); b.set_ylim(-0.7, len(cls_) - 0.3)
b.set_yticks(range(len(cls_))[::-1], [f"{c_} ({t})" for c_, t in zip(cls_, tot)])
b.spines["left"].set_visible(False); b.spines["bottom"].set_visible(False); b.tick_params(length=0)
b.grid(axis="y", color=GRID, lw=0.5); b.set_axisbelow(True)
sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap); cb = fig.colorbar(sm, ax=b, fraction=0.045, pad=0.02)
cb.set_label("mean effect (SD/h)", fontsize=6.2); cb.ax.tick_params(labelsize=5.2, length=1.5); cb.outline.set_visible(False)
panel(b, "b", "Class enrichment")
c = ax[2]
gen = [("meal clock M", -5.51, 2.44, CLOCK, "o"), ("sleep onset S", -3.46, 1.87, INK, "D"), ("gap D", 2.05, 2.23, GAP, "s")]
for i, (lab, e, se, col, mk) in enumerate(gen):
    yy = 2 - i; c.plot([e - 1.96 * se, e + 1.96 * se], [yy, yy], color=col, lw=1.3, solid_capstyle="round", zorder=2)
    c.scatter([e], [yy], s=20, marker=mk, color=col, zorder=3)
# p-value sits just below the meal-clock row, inside the axes (no overlap with the panel title)
c.text(-5.51, 1.72, "p = 0.024", fontsize=6.0, color=INK2, ha="center", va="top")
c.set_yticks([2, 1, 0], [g[0] for g in gen]); vgrid(c); c.set_xlabel("minutes per SD of score"); c.set_ylim(-0.5, 2.5)
c.set_xlim(-12.5, 8.0)
panel(c, "c", "Genetic chronotype")
save(fig, "fig4")
print("figures written to", H)
