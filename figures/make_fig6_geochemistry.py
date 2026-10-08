"""Poster Fig. 6: how the two pulses differ chemically.

Input is one row per sample with majors normalized to 100% anhydrous (columns ending in *,
e.g. MgO*) and trace elements in ppm. Majors: Michigan State XRF. Traces: Middlebury ICP-MS.
PR-6-140 and PR-6-840 have no usable geochemistry.

    python make_fig6_geochemistry.py ../data/pr6_geochemistry_anhydrous.csv
"""
import sys
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

g = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "../data/pr6_geochemistry_anhydrous.csv")
g["pos"] = g.Sample.str.extract(r"(\d+)$").astype(int)
X_FROM_WEST = {140: 1.6, 280: 3.3, 420: 4.7, 560: 6.3, 700: 7.3, 840: 8.7, 980: 10.3, 1120: 12.3, 1260: 13.7}
g["x"] = g.pos.map(X_FROM_WEST)
g["pulse"] = np.where(g.pos.isin([140, 280, 1260]), 1, 2)
p1, p2 = g[g.pulse == 1], g[g.pulse == 2]

# compatible elements on the left, incompatible on the right
ORDER = ["Ni", "Co", "V", "MgO*", "TiO2*", "FeO*", "CaO*", "Sc", "Sr", "Eu", "Al2O3*", "SiO2*", "Na2O*", "Rb",
         "Ba", "K2O*", "Th", "U", "Nb", "Ta", "La", "Ce", "Pb", "Pr", "Nd", "Zr", "Hf", "Sm", "Gd", "Dy", "Y",
         "Ho", "Er", "Yb", "Lu", "P2O5*"]
ratio = [p2[c].mean() / p1[c].mean() for c in ORDER]
separated = [p2[c].min() > p1[c].max() or p2[c].max() < p1[c].min() for c in ORDER]

fig = plt.figure(figsize=(17, 5))
gs = fig.add_gridspec(1, 4)
ax = fig.add_subplot(gs[0, :2])
ax.axhspan(0.95, 1.05, color="0.85", label="±5%: ICP-MS trace precision\n(XRF majors are better than ±1%)")
ax.axhline(1, color="k", lw=0.8)
ax.scatter(range(len(ORDER)), ratio, c=["tab:red" if s else "0.5" for s in separated], edgecolor="k", s=45, zorder=3)
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels([c.rstrip("*") for c in ORDER], rotation=90)
ax.set_ylim(0.6, 1.3)
ax.set_ylabel("Pulse 2 mean ÷ pulse 1 mean")
ax.set_title(f"Pulse 2 (porphyritic, n = {len(p2)}) vs pulse 1 (aphanitic, n = {len(p1)}); "
             "red = groups completely separated", fontsize=10)
ax.text(0.03, 0.03, "← compatible (lower in pulse 2)", transform=ax.transAxes, fontsize=8)
ax.text(0.97, 0.93, "incompatible (higher in pulse 2) →", transform=ax.transAxes, fontsize=8, ha="right")
ax.legend(loc="lower right", fontsize=8)

ax = fig.add_subplot(gs[0, 2])
for pulse, color, label in [(1, "#1f77b4", "pulse 1 (aphanitic)"), (2, "#ff7f0e", "pulse 2 (porphyritic)")]:
    s = g[g.pulse == pulse]
    ax.scatter(s["MgO*"], s.V, color=color, edgecolor="k", s=60, label=label)
for _, r in g.iterrows():
    ax.annotate(f"{r.x} m", (r["MgO*"], r.V), xytext=(4, 3), textcoords="offset points", fontsize=7)
ax.set(xlabel="MgO (wt%, anhydrous)", ylabel="V (ppm)", title="Compatible elements")
ax.legend(fontsize=8, loc="lower right")

ax = fig.add_subplot(gs[0, 3])
ax.axhspan(0.95, 1.05, color="0.85")
s = g.sort_values("x")
for col, color in [("La/Yb", "tab:green"), ("Zr/Nb", "tab:purple"), ("Nb/Y", "tab:brown"), ("Th/Nb", "k")]:
    ax.plot(s.x, s[col] / s[col].mean(), "o-", color=color, label=col)
ax.set(xlabel="Distance from west wall (m)", ylabel="Ratio ÷ series mean", ylim=(0.8, 1.2),
       title="Incompatible-element ratios stay flat")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("fig6_pulse_geochemistry.png", dpi=300)
print("saved fig6_pulse_geochemistry.png")
