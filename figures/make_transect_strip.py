"""The pulse transect strip under the poster title: sample positions to scale across PR-6.

Drawn at the size it is printed on the poster (55 x 2.05 in).

    python make_transect_strip.py transect_strip.png
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

OUT = sys.argv[1] if len(sys.argv) > 1 else "transect_strip.png"
# Calibri if you have it; Carlito is a free font with the same metrics
plt.rcParams["font.family"] = ["Calibri", "Carlito", "DejaVu Sans"]

W, H = 55.0, 2.05            # inches, same as the strip in the earlier poster draft
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")

P1, P1_INK, P1_TINT = "#1F77B4", "#155A8A", "#DCE8F2"
P2, P2_INK, P2_TINT = "#FF7F0E", "#9A4306", "#FBE3CC"
WALL, MUTED, INK = "#A3201C", "#5E4A3B", "#2B1D14"

lw = 4 / 72  # 4 pt border, in inches
ax.add_patch(Rectangle((lw / 2, lw / 2), W - lw, H - lw, facecolor="#FCF7EF", edgecolor="#C8A47C", linewidth=4))

ax.text(0.4, 0.32, "WEST WALL · 0 m", fontsize=20, fontweight="bold", color=WALL, va="center", ha="left")
ax.text(W - 0.4, 0.32, "16.7 m · EAST WALL", fontsize=20, fontweight="bold", color=WALL, va="center", ha="right")
ax.text(W / 2, 0.32, "Nine samples across the 16.7 m Tunnel Dike, positions to scale · blue = pulse 1, orange = pulse 2 in every figure",
        fontsize=20, fontweight="bold", color=MUTED, va="center", ha="center")

BX, BY, BW, BH = 0.4, 0.6, W - 0.8, 0.62
s = BW / 16.7
zones = [(0, 4.0, P1, P1_TINT, P1_INK, "PULSE 1 · APHANITIC MARGIN", "left"),
         (4.0, 13.0, P2, P2_TINT, P2_INK, "PULSE 2 · PORPHYRITIC, CRYSTAL-RICH CORE", "center"),
         (13.0, 16.7, P1, P1_TINT, P1_INK, "PULSE 1 · APHANITIC MARGIN", "right")]
for a, b, c, tint, ink, lab, al in zones:
    zx, zw = BX + a * s, (b - a) * s
    ax.add_patch(Rectangle((zx, BY), zw, BH, facecolor=tint, edgecolor="none"))
    ax.add_patch(Rectangle((zx, BY), zw, 0.1, facecolor=c, edgecolor="none"))
    tx = {"left": zx + 0.25, "center": zx + zw / 2, "right": zx + zw - 0.25}[al]
    ax.text(tx, BY + 0.1 + (BH - 0.1) / 2, lab, fontsize=22, fontweight="bold", color=ink, ha=al, va="center")
ax.add_patch(Rectangle((BX - 0.1, BY), 0.1, BH, facecolor=WALL, edgecolor="none"))
ax.add_patch(Rectangle((BX + BW, BY), 0.1, BH, facecolor=WALL, edgecolor="none"))

samples = [("140", 1.6, 1), ("280", 3.3, 1), ("420", 4.7, 2), ("560", 6.3, 2), ("700", 7.3, 2),
           ("840", 8.7, 2), ("980", 10.3, 2), ("1120", 12.3, 2), ("1260", 13.7, 1)]
for sid, m, pulse in samples:
    cx = BX + m * s
    ax.plot([cx, cx], [BY + BH, BY + BH + 0.14], color=INK, linewidth=2.2, solid_capstyle="butt")
    ax.text(cx, BY + BH + 0.33, f"PR-6-{sid}", fontsize=22, fontweight="bold",
            color=P1_INK if pulse == 1 else P2_INK, ha="center", va="center")
    ax.text(cx, BY + BH + 0.63, f"{m:.1f} m", fontsize=16, color=MUTED, ha="center", va="center")

fig.savefig(OUT, dpi=300)
print("saved", OUT)
