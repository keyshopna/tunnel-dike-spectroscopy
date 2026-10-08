"""Poster Fig. 4: EDS Si-Al maps from the west wall to the east wall (maps by R. Currier).

Expects the exported map images named like PR-6-140-SiAlmap.png in one folder.

    python make_fig4_eds_maps.py path/to/eds_maps fig4_eds_maps.png
"""
import sys, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

U = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "fig4_eds_maps.png"
P1, P2 = "#1f77b4", "#ff7f0e"
samples = [("140", 1.6, 1), ("280", 3.3, 1), ("420", 4.7, 2), ("840", 8.7, 2), ("1120", 12.3, 2), ("1260", 13.7, 1)]
plt.rcParams["font.family"] = "DejaVu Sans"
fig, axes = plt.subplots(2, 3, figsize=(16, 14.2))
fig.subplots_adjust(left=0.015, right=0.985, top=0.865, bottom=0.075, wspace=0.05, hspace=0.17)
for ax, (sid, m, pulse) in zip(axes.flat, samples):
    f = glob.glob(f"{U}/*PR-6-{sid}-SiAlmap.png")[0]
    im = Image.open(f).convert("RGB")
    W = im.size[0]
    ax.imshow(im, extent=(0, 2.4, 0, 2.4))
    col = P1 if pulse == 1 else P2
    for s in ax.spines.values():
        s.set_edgecolor(col); s.set_linewidth(6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"PR-6-{sid} ({m} m)\n{'Pulse 1, aphanitic' if pulse == 1 else 'Pulse 2, porphyritic'}",
                 color=col, fontsize=24, fontweight="bold", linespacing=1.15, pad=10)
    ax.add_patch(Rectangle((0.12, 0.12), 0.5, 0.05, color="white", zorder=3))
    ax.text(0.37, 0.22, "500 µm", color="white", fontsize=22, fontweight="bold", ha="center", va="bottom", zorder=3)
fig.suptitle("Tunnel Dike textures from west wall to east wall: EDS Si–Al maps\n(distance from west wall in brackets; each map 2.4 × 2.4 mm; dike 16.7 m wide)",
             fontsize=26, y=0.985, linespacing=1.2)
fig.text(0.5, 0.02, "Pink = Al-rich (plagioclase); brown = pyroxene; black = Fe-Ti oxide;\n"
         "orange/yellow = Si-rich interstitial mesostasis. EDS maps: R. Currier.",
         ha="center", va="bottom", fontsize=20, linespacing=1.3)
fig.savefig(OUT, dpi=200)
print("saved", OUT)
