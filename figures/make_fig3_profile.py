"""Poster Fig. 3: crystal size, crystal shape and the 1.9 um water feature across PR-6.

Published size and shape are from Currier et al. (2026), Supplementary Table ST2.
The water feature and 2D crystal sizes come from analysis/run_pr6_analysis.py.

    python make_fig3_profile.py ../data/currier2026_ST2_plagioclase.csv ../results/tables/master_table.csv
"""
import sys
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

st2 = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "../data/currier2026_ST2_plagioclase.csv")
master = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else "../results/tables/master_table.csv")
d = st2.merge(master[["pos", "h2o_1900", "feldspar_char_len_um"]], left_on="sample_pos", right_on="pos")

WIDTH = 16.7                      # dike width (m)
P1_EDGE, P2_EDGE = 4.0, 13.0      # pulse boundaries drawn midway between samples
PULSE1 = {140, 280, 1260}         # aphanitic margins
BLUE, ORANGE = "#1f77b4", "#ff7f0e"
d["pulse"] = np.where(d.sample_pos.isin(PULSE1), 1, 2)
colors = np.where(d.pulse == 1, BLUE, ORANGE)


def exact_perm_p(values, groups):
    """Two-sided exact permutation test on ranks (rank-sum of group 1 vs its expected value)."""
    ranks = rankdata(values)
    in1 = np.asarray(groups) == 1
    n1, n = in1.sum(), len(ranks)
    expected = n1 * (n + 1) / 2
    obs = abs(ranks[in1].sum() - expected)
    sums = np.array([ranks[list(idx)].sum() for idx in combinations(range(n), n1)])
    return np.mean(np.abs(sums - expected) >= obs - 1e-9)


fig, axes = plt.subplots(3, 1, figsize=(7.5, 9.8), sharex=True)
panels = [
    ("mean_size_um", "Plagioclase size (µm)",
     "a) Crystal size: published 3D mean (circles) vs this study (squares)"),
    ("shape_I", "Shape: intermediate ÷ short axis",
     f"b) Crystal shape (ShapeCalc): margins more tabular, no overlap (p = {exact_perm_p(d.shape_I, d.pulse):.3f})"),
    ("h2o_1900", "1.9 µm water feature",
     f"c) Hyperspectral water absorption: no overlap between pulses (p = {exact_perm_p(d.h2o_1900, d.pulse):.3f})"),
]
for ax, (col, ylabel, title) in zip(axes, panels):
    ax.axvspan(0, P1_EDGE, color=BLUE, alpha=0.09, lw=0)
    ax.axvspan(P1_EDGE, P2_EDGE, color=ORANGE, alpha=0.09, lw=0)
    ax.axvspan(P2_EDGE, WIDTH, color=BLUE, alpha=0.09, lw=0)
    for wall in (0, WIDTH):
        ax.axvline(wall, color="firebrick", lw=2.5)
    ax.plot(d.x_from_west_m, d[col], "-", color="k", lw=1.2, zorder=2)
    ax.scatter(d.x_from_west_m, d[col], c=colors, s=90, edgecolor="k", zorder=3)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10)

# our 2D EBSD sizes for comparison (crystals > 100 um only)
axes[0].plot(d.x_from_west_m, d.feldspar_char_len_um, "s--", color="gray", mfc="white", ms=6,
             label="this study (2D, crystals >100 µm)", zorder=1)
axes[0].legend(loc="upper right", fontsize=8)
axes[0].text(0.15, axes[0].get_ylim()[1], "W wall", color="firebrick", fontsize=8, va="top")
for _, r in d.iterrows():
    axes[2].annotate(str(int(r.sample_pos)), (r.x_from_west_m, r.h2o_1900), xytext=(0, 7),
                     textcoords="offset points", ha="center", fontsize=7)
axes[2].set_xlabel("Distance from west wall (m)")
axes[2].set_xlim(-0.3, WIDTH + 0.3)
fig.text(0.02, 0.01, "Blue = pulse 1 (aphanitic margins); orange = pulse 2 (porphyritic core). Shading = pulse zones, "
         "boundaries drawn midway between samples\n(Currier et al. 2026). Published size and shape: Currier et al. "
         "(2026) Supplementary Table ST2. Exact permutation tests, 3 vs 6 samples.", fontsize=7)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig("fig3_size_shape_water_profile.png", dpi=300)
print("saved fig3_size_shape_water_profile.png")
