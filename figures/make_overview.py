"""Graphical overview (Figure 1): cohort and devices -> nightly anchors -> ANCHOR -> findings.

Vector figure composed as SVG and printed to PDF with headless Chromium (Playwright).
Third-party artwork in figures/icons/ is public domain (CC0 1.0), from bioicons.com:
  person_cc0.svg (Marcel Tisch), liver_cc0.svg (Jan Clusmann),
  blood_tube_cc0.svg (Marcel Tisch), dna_cc0.svg (James Lloyd).
All other elements (watch, CGM sensor, phone, ultrasound probe, plots) are drawn here.
"""
import os, base64, asyncio

H = os.path.dirname(os.path.abspath(__file__))
IC = os.path.join(H, "icons")

INK, INK2, MUTED, HAIR = "#1d2026", "#4d525c", "#8b909a", "#d9dbe0"
CLOCK, GAP, ANCH, ANCH_D = "#2a78d6", "#eb6834", "#1b9e6e", "#0e7a54"
RED, LIVER, LIPID = "#c8423b", "#a8503a", "#d9a634"
BG, BG_M = "#f4f5f7", "#ecf5f0"
FONT = "Helvetica, Arial, 'Liberation Sans', sans-serif"
W, HT = 1400, 550
out = []
A = out.append


def img(name, x, y, w, h):
    d = base64.b64encode(open(os.path.join(IC, name), "rb").read()).decode()
    return (f'<image href="data:image/svg+xml;base64,{d}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
            f'height="{h:.1f}" preserveAspectRatio="xMidYMid meet"/>')


def T(x, y, s, size=15, color=INK, w="normal", anchor="middle", style="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" font-weight="{w}" '
            f'font-style="{style}" text-anchor="{anchor}" dominant-baseline="middle">{s}</text>')


def it(s, col=None):  # italic math variable
    c = f' fill="{col}"' if col else ""
    return f'<tspan font-style="italic"{c}>{s}</tspan>'


def sub(s, size=10):
    return f'<tspan font-size="{size}" dy="3.5">{s}</tspan><tspan dy="-3.5"> </tspan>'


def title(x, letter, t, s):
    return (T(x, 32, letter, 22, INK, "bold", "start") + T(x + 22, 32, t, 18, INK, "bold", "start") +
            T(x, 56, s, 13.5, MUTED, anchor="start"))


def panel(x, w, fill=BG):
    return f'<rect x="{x}" y="72" width="{w}" height="466" rx="8" fill="{fill}"/>'


def arrow(x0, x1, y=305):
    return (f'<line x1="{x0}" y1="{y}" x2="{x1-6}" y2="{y}" stroke="#a9adb5" stroke-width="1.6"/>'
            f'<path d="M{x1-8},{y-5} L{x1},{y} L{x1-8},{y+5}" fill="none" stroke="#a9adb5" stroke-width="1.6" '
            'stroke-linejoin="round" stroke-linecap="round"/>')


# ------------------------------------------------------------------ drawn device glyphs
def watch(cx, cy):
    return (f'<rect x="{cx-11}" y="{cy-30}" width="22" height="60" rx="6" fill="#8d939c"/>'
            f'<rect x="{cx-11}" y="{cy-30}" width="22" height="60" rx="6" fill="none" stroke="#6f757e" stroke-width="0.8"/>'
            f'<rect x="{cx-19}" y="{cy-20}" width="38" height="40" rx="10" fill="#3a3f47"/>'
            f'<rect x="{cx-15}" y="{cy-16}" width="30" height="32" rx="7" fill="#11151c"/>'
            f'<path d="M{cx+3},{cy-9} a8,8 0 1,0 6,11 a6.2,6.2 0 1,1 -6,-11 z" fill="#f2c14e"/>'
            f'<polyline points="{cx-11},{cy+9} {cx-7},{cy+9} {cx-5},{cy+4} {cx-2},{cy+12} {cx+1},{cy+7} {cx+11},{cy+7}" '
            'fill="none" stroke="#5aa9e6" stroke-width="1.3" stroke-linejoin="round"/>'
            f'<rect x="{cx+19}" y="{cy-5}" width="3" height="10" rx="1.2" fill="#3a3f47"/>')


def phone(cx, cy):
    s = (f'<rect x="{cx-20}" y="{cy-34}" width="40" height="68" rx="7" fill="#2b2f36"/>'
         f'<rect x="{cx-17}" y="{cy-29}" width="34" height="56" rx="3" fill="white"/>'
         f'<rect x="{cx-17}" y="{cy-29}" width="34" height="9" rx="3" fill="{CLOCK}" fill-opacity="0.85"/>')
    for k in range(4):
        y = cy - 14 + k * 9.5
        s += (f'<circle cx="{cx-11}" cy="{y}" r="2.6" fill="{GAP if k == 3 else "#c9ccd2"}"/>'
              f'<rect x="{cx-6}" y="{y-1.6}" width="{18 if k % 2 else 14}" height="3.2" rx="1.6" fill="#d4d7dc"/>')
    return s + f'<rect x="{cx-6}" y="{cy+29.5}" width="12" height="2" rx="1" fill="#6b7078"/>'


def cgm(cx, cy):
    return (f'<rect x="{cx-38}" y="{cy-4}" width="76" height="30" rx="15" fill="#e8c2a6" transform="rotate(-8 {cx} {cy+11})"/>'
            f'<circle cx="{cx-4}" cy="{cy+9}" r="13.5" fill="#f7f8fa" stroke="#9aa0a8" stroke-width="1.4"/>'
            f'<circle cx="{cx-4}" cy="{cy+9}" r="5" fill="#d5d9de" stroke="#9aa0a8" stroke-width="1"/>'
            f'<polyline points="{cx-30},{cy-22} {cx-22},{cy-20} {cx-15},{cy-30} {cx-7},{cy-16} {cx+2},{cy-26} '
            f'{cx+10},{cy-21} {cx+18},{cy-27} {cx+28},{cy-22}" fill="none" stroke="{RED}" stroke-width="2" '
            'stroke-linejoin="round" stroke-linecap="round"/>')


def ultrasound(cx, cy):
    return (img("liver_cc0.svg", cx - 34, cy - 8, 68, 48) +
            f'<path d="M{cx-6},{cy-14} L{cx-30},{cy+26} Q{cx},{cy+36} {cx+26},{cy+26} L{cx+6},{cy-14} Z" '
            f'fill="#7fb6e0" fill-opacity="0.32"/>'
            f'<rect x="{cx-9}" y="{cy-40}" width="18" height="24" rx="5" fill="#9097a1"/>'
            f'<rect x="{cx-12}" y="{cy-19}" width="24" height="7" rx="3" fill="#5f6670"/>')


# ------------------------------------------------------------------ panel a
A(title(12, "a", "Cohort and devices", "Human Phenotype Project, adults aged 40–70"))
A(panel(10, 372))
A(img("person_cc0.svg", 132.2, 106, 125.6, 380))
devices = [  # x, y (glyph centre), glyph, name, measure, body anchor, colour
    (72, 142, cgm(72, 148), "CGM", "nightly glucose", (141, 212), RED),
    (72, 284, img("blood_tube_cc0.svg", 52, 252, 40, 64), "blood draw", "170 serum lipids", (146, 254), RED),
    (72, 420, watch(72, 418), "sleep wearable", "sleep onset " + it("S"), (149, 302), INK2),
    (318, 142, img("dna_cc0.svg", 296, 108, 44, 68), "genotype", "211 loci", (205, 205), "#3f7fc0"),
    (318, 284, ultrasound(318, 282), "liver ultrasound", "attenuation", (196, 258), LIVER),
    (318, 420, phone(318, 418), "diet-logging app", "last meal " + it("M"), (239, 310), CLOCK),
]
for x, y, glyph, name, meas, (bx, by), col in devices:
    side = -1 if x < 195 else 1
    sx, sy = x - side * 44, y
    A(f'<path d="M{sx},{sy} C{(sx+bx)/2},{sy} {(sx+bx)/2},{by} {bx},{by}" fill="none" stroke="#9aa0a8" stroke-width="1.1"/>')
    A(f'<circle cx="{bx}" cy="{by}" r="4" fill="white" stroke="{col}" stroke-width="2"/>')
    A(glyph)
    A(T(x, y + 52, name, 14.5, INK, "bold"))
    A(T(x, y + 68, meas, 13.5, INK2))

A(arrow(386, 432))

# ------------------------------------------------------------------ panel b
A(title(440, "b", "Nightly anchors", "two or three nights per participant"))
A(panel(438, 274))
X0, X1 = 488, 692
hx = lambda h: X0 + (X1 - X0) * (h - 19) / 6
for h, lab in [(19, "19:00"), (21, "21:00"), (23, "23:00"), (25, "01:00")]:
    A(f'<line x1="{hx(h):.1f}" y1="96" x2="{hx(h):.1f}" y2="256" stroke="{HAIR}" stroke-width="0.9"/>')
    A(T(hx(h), 268, lab, 12.5, MUTED))
A(f'<line x1="{X0}" y1="256" x2="{X1}" y2="256" stroke="{MUTED}" stroke-width="1"/>')
nights = [(20.0, 23.0), (21.0, 22.5), (19.6, 24.2)]  # same example nights as Fig. 2a
mm = sum(n[0] for n in nights) / 3; ms = sum(n[1] for n in nights) / 3
rows = [(f"night {i+1}", m, s, False) for i, (m, s) in enumerate(nights)] + [("mean", mm, ms, True)]
for i, (lab, m, s, mean) in enumerate(rows):
    y = 118 + i * 38 + (8 if mean else 0)
    A(T(482, y, lab, 12.5, INK if mean else MUTED, "bold" if mean else "normal", "end"))
    xm, xs = hx(m), hx(s)
    A(f'<line x1="{xm}" y1="{y}" x2="{xs}" y2="{y}" stroke="{GAP}" stroke-width="{2.4 if mean else 1.6}" '
      f'stroke-opacity="{1 if mean else 0.75}"/>')
    if mean:
        A(f'<circle cx="{xm}" cy="{y}" r="6.5" fill="white" stroke="{CLOCK}" stroke-width="2.2"/>'
          f'<rect x="{xs-6}" y="{y-6}" width="12" height="12" fill="white" stroke="{INK}" stroke-width="2.2"/>')
    else:
        A(f'<circle cx="{xm}" cy="{y}" r="6" fill="{CLOCK}"/><rect x="{xs-5.5}" y="{y-5.5}" width="11" height="11" fill="{INK}"/>')
    A(T((xm + xs) / 2, y - 11, f"{s-m:.1f} h", 11.5, GAP))
A(f'<line x1="{X0}" y1="{118+2*38+19}" x2="{X1}" y2="{118+2*38+19}" stroke="{HAIR}" stroke-width="0.9" stroke-dasharray="3 3"/>')
# legend
A(f'<circle cx="452" cy="292" r="5" fill="{CLOCK}"/>' + T(462, 292, it("M", CLOCK) + " last meal (clock)", 13, INK2, anchor="start"))
A(f'<rect x="580" y="287" width="10" height="10" fill="{INK}"/>' + T(596, 292, it("S") + " sleep onset", 13, INK2, anchor="start"))
A(T(452, 312, it("D", GAP) + " = " + it("S") + " − " + it("M") + "  meal-to-sleep gap", 13, INK2, anchor="start"))
# inset: habit vs night-error covariance
ix, iy = 520, 405
A(f'<line x1="{ix-62}" y1="{iy+52}" x2="{ix+66}" y2="{iy+52}" stroke="{MUTED}" stroke-width="0.9"/>'
  f'<line x1="{ix-62}" y1="{iy+52}" x2="{ix-62}" y2="{iy-62}" stroke="{MUTED}" stroke-width="0.9"/>')
A(T(ix + 2, iy + 64, it("M"), 12.5, MUTED) + T(ix - 72, iy - 4, it("D"), 12.5, MUTED))
import numpy as np
def ell(S, col, fill, dash=""):
    # 95% ellipse of covariance S (M, D), drawn to scale; SVG y axis points down
    w, V = np.linalg.eigh(np.array(S)); k = 19.0 * np.sqrt(5.991)
    ang = np.degrees(np.arctan2(-V[1, 1], V[0, 1]))
    return (f'<ellipse cx="{ix}" cy="{iy-4}" rx="{k*np.sqrt(w[1]):.1f}" ry="{k*np.sqrt(w[0]):.1f}" '
            f'transform="rotate({ang:.1f} {ix} {iy-4})" fill="{col if fill else "none"}" fill-opacity="0.13" '
            f'stroke="{col}" stroke-width="1.5"{dash}/>')
# HPP estimates (Appendix A): habits Sigma_X and person-mean error Sigma_eps, in (M, D)
SX = [[0.963, -0.51 * 0.963], [-0.51 * 0.963, (0.51 * 0.963) ** 2 / (0.60 ** 2 * 0.963)]]
SE = [[0.839, -0.83 * 0.839], [-0.83 * 0.839, (0.83 * 0.839) ** 2 / (0.84 ** 2 * 0.839)]]
A(ell(SX, CLOCK, True))
A(ell(SE, GAP, False, ' stroke-dasharray="4 2.5"'))
A(T(ix + 74, iy - 40, "habits", 12.5, CLOCK, anchor="start") + T(ix + 74, iy - 24, "slope −0.51", 11.5, MUTED, anchor="start"))
A(T(ix + 74, iy + 4, "night error", 12.5, GAP, anchor="start") + T(ix + 74, iy + 20, "slope −0.83", 11.5, MUTED, anchor="start"))
A(T(575, 497, "Standard analysis: fit person means of " + it("M") + " and " + it("D") + " jointly", 12, INK2))
A(T(575, 515, "→ anchor effects attenuate and mix", 13, RED, "bold"))

A(arrow(716, 762))

# ------------------------------------------------------------------ panel c
A(title(770, "c", "ANCHOR", "attribution-corrected estimation"))
A(panel(768, 302, BG_M))
steps = [("Night-level error " + it("Σ") + sub("U"), "from repeated, dated nights"),
         ("Attribution matrix " + it("Λ"), "attenuation and leakage"),
         ("Calibrate habits", "for unequal nights " + it("k") + sub("i")),
         ("Habitual vs acute effects", "for outcomes measured nightly"),
         ("Basis-free contrasts", "and anchor-aligned negative controls")]
ys = [112 + i * 88 for i in range(5)]
A(f'<line x1="792" y1="{ys[0]}" x2="792" y2="{ys[-1]}" stroke="{ANCH}" stroke-width="1.4"/>')
for i, (a, b) in enumerate(steps):
    y = ys[i]
    A(f'<circle cx="792" cy="{y}" r="11" fill="{BG_M}" stroke="{ANCH}" stroke-width="1.8"/>' + T(792, y + 0.5, str(i + 1), 12.5, ANCH_D, "bold"))
    A(T(812, y - 8, a, 14.5, INK, "bold", "start"))
    A(T(812, y + 10, b, 13, INK2, anchor="start"))
# Lambda matrix
L = [[0.66, 0.06], [0.19, 0.51]]
def shade(v):
    lo, hi = (232, 240, 250), (32, 84, 160); t = min(1, 0.12 + 0.88 * v / 0.66)
    return "#%02x%02x%02x" % tuple(int(lo[k] + (hi[k] - lo[k]) * t) for k in range(3))
mx, my, c = 990, 168, 32
for k, lab in enumerate(["M", "D"]):
    A(T(mx + c / 2 + k * c, my - 10, it(lab, [CLOCK, GAP][k]), 12.5))
    A(T(mx - 11, my + c / 2 + k * c, it(lab, [CLOCK, GAP][k]), 12.5))
for r in range(2):
    for q in range(2):
        v = L[r][q]
        A(f'<rect x="{mx+q*c}" y="{my+r*c}" width="{c-1.5}" height="{c-1.5}" fill="{shade(v)}"/>')
        A(T(mx + q * c + c / 2 - 0.7, my + r * c + c / 2, f"{v:.2f}", 11.5, "white" if v > 0.3 else INK))
A(f'<rect x="{mx}" y="{my+c}" width="{c-1.5}" height="{c-1.5}" fill="none" stroke="{RED}" stroke-width="2"/>')
A(T(mx + c, my + 2 * c + 14, "leakage", 11.5, RED))
A(T(919, 520, it("Λ") + " = (" + it("Σ") + sub("X") + "+ " + it("Σ") + sub("ε") + ")" +
    '<tspan font-size="9" dy="-6">−1</tspan><tspan dy="6"> </tspan>' + it("Σ") + sub("X"), 13.5, INK2))

A(arrow(1074, 1122))

# ------------------------------------------------------------------ panel d
A(title(1132, "d", "Findings", "HPP, up to 5,608 participants"))
A(panel(1130, 260))
lip = "".join(f'<circle cx="{1170+dx}" cy="{{y}}" r="{r}" fill="{LIPID}" fill-opacity="0.85" stroke="#b07f17" stroke-width="0.9" transform="translate(0,{dy})"/>'
              for dx, dy, r in [(-9, -6, 7), (8, -8, 6), (0, 7, 8), (-12, 9, 4.5), (12, 8, 4.5)])
finds = [
    (lambda y: img("liver_cc0.svg", 1146, y - 18, 48, 36), "Liver", "marginal clock association", "(exploratory)"),
    (lambda y: f'<polyline points="1146,{y+6} 1154,{y+2} 1160,{y+10} 1167,{y-12} 1174,{y+4} 1182,{y-2} 1194,{y+3}" '
               f'fill="none" stroke="{RED}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>'
               f'<line x1="1144" y1="{y+16}" x2="1196" y2="{y+16}" stroke="{MUTED}" stroke-width="0.9"/>',
     "Glucose", "clock effect acute,", "gap effect habitual"),
    (lambda y: lip.replace("{y}", str(y)), "Lipidome", "56 of 170 lipids track the", "habitual meal clock"),
    (lambda y: img("dna_cc0.svg", 1152, y - 24, 36, 48), "Genetics", "score shifts the schedule,", "not the gap (weak score)"),
]
for i, (g, t1, d1, d2) in enumerate(finds):
    y = 126 + i * 104
    A(g(y))
    A(T(1210, y - 18, t1, 14.5, INK, "bold", "start"))
    A(T(1210, y + 1, d1, 13, INK2, anchor="start"))
    A(T(1210, y + 18, d2, 13, INK2, anchor="start"))
    if i < 3:
        A(f'<line x1="1146" y1="{y+52}" x2="1374" y2="{y+52}" stroke="{HAIR}" stroke-width="0.9"/>')

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {HT}" width="7in" height="2.75in" '
       f'font-family="{FONT}"><rect width="{W}" height="{HT}" fill="white"/>' + "".join(out) + "</svg>")
open(os.path.join(H, "fig0.svg"), "w").write(svg)


async def render():
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1400, "height": 550}, device_scale_factor=2.4)
        await pg.set_content('<html><head><style>@page{size:7in 2.75in;margin:0}html,body{margin:0}'
                             'svg{display:block}</style></head><body>' + svg + '</body></html>')
        await pg.wait_for_timeout(300)
        await pg.pdf(path=os.path.join(H, "fig0.pdf"), width="7in", height="2.75in", print_background=True,
                     margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
        await pg.set_content('<html><body style="margin:0">' +
                             svg.replace('width="7in" height="2.75in"', 'width="1400" height="550"') + '</body></html>')
        await pg.wait_for_timeout(300)
        await pg.screenshot(path=os.path.join(H, "fig0.png"), clip={"x": 0, "y": 0, "width": 1400, "height": 550})
        await b.close()

asyncio.run(render())
print("overview written")
