"""Poster Fig. 7: fusion mass loss as a relative volatile measure.

Mass lost when the XRF glass disks were fused (sample + flux in, glass disk out), as a
percent of sample mass. This is not a true LOI. Only disks made with the same 9:1 flux
recipe are compared (10 g and 7 g disks); the 4 g disks (PR-6-140, PR-6-840) are left out
because handling losses dominate at that size.

    python make_fig7_fusion_loss.py ../data/pr6_fusion_loss.csv
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

f = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "../data/pr6_fusion_loss.csv")
f = f[f.disk_recipe != "4 g"]
WIDTH = 16.7
colors = np.where(f.pulse == 1, "#1f77b4", "#ff7f0e")
markers = np.where(f.disk_recipe == "10 g", "o", "s")

rho_all = spearmanr(f.h2o_1900, f.fusion_loss_pct).statistic
ten = f[f.disk_recipe == "10 g"]
rho10 = spearmanr(ten.h2o_1900, ten.fusion_loss_pct).statistic

fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
for x, y, c, m in zip(f.x_from_west_m, f.fusion_loss_pct, colors, markers):
    axes[0].scatter(x, y, color=c, marker=m, s=70, edgecolor="k")
for wall in (0, WIDTH):
    axes[0].axvline(wall, color="firebrick", lw=2.5)
axes[0].set(xlabel="Distance from west wall (m)", ylabel="Fusion mass loss (% of sample mass)",
            title="Fusion mass loss across the dike")

for _, r in f.iterrows():
    axes[1].scatter(r.h2o_1900, r.fusion_loss_pct, color="#1f77b4" if r.pulse == 1 else "#ff7f0e",
                    marker="o" if r.disk_recipe == "10 g" else "s", s=70, edgecolor="k")
    axes[1].annotate(f"{r.x_from_west_m} m", (r.h2o_1900, r.fusion_loss_pct), xytext=(4, 3),
                     textcoords="offset points", fontsize=7)
axes[1].set(xlabel="1.9 µm water feature strength", ylabel="Fusion mass loss (% of sample mass)",
            title=f"ρ = {rho_all:+.2f} (all {len(f)}); {rho10:+.2f} (five 10-g disks)")
handles = [plt.Line2D([], [], marker="o", ls="", color="#1f77b4", label="pulse 1 (aphanitic)"),
           plt.Line2D([], [], marker="o", ls="", color="#ff7f0e", label="pulse 2 (porphyritic)"),
           plt.Line2D([], [], marker="o", ls="", mfc="white", mec="k", label="10 g disk"),
           plt.Line2D([], [], marker="s", ls="", mfc="white", mec="k", label="7 g disk")]
axes[1].legend(handles=handles, fontsize=8, loc="upper left")
fig.text(0.01, 0.01, "Relative volatile proxy, not a calibrated LOI. 4 g disks (PR-6-140, 840) excluded.", fontsize=7)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig("fig7_fusion_loss.png", dpi=300)
print("saved fig7_fusion_loss.png")
