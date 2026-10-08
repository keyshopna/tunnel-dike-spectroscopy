"""
PR-6 tunnel dike: hyperspectral x EBSD x XRF analysis
=====================================================
Runs the whole analysis on the real data and writes figures + tables.

Usage (from the folder containing this script, ebsdio.py and grains.py):
    python run_pr6_analysis.py "path/to/hyperspec x EBSD x XRF" "path/to/xrf_tunnel_dike_clean.xlsx"

Steps
  1. Spectra: read the 27 .sed files, clean, average the 3 spots per sample, measure spectral features
  2. EBSD: for every map, mineral proportions + grain reconstruction + crystal size statistics
  3. XRF: normalized oxides from the cleaned workbook
  4. Combine, test relationships (Spearman + shuffle test), make figures
"""
import sys, glob, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import rankdata
from ebsdio import read_ctf                     # reads Oxford .ctf files
from grains import (GROUP, fill_zero_solutions, # grain reconstruction tools
                    reconstruct, grain_table)

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("hyperspec x EBSD x XRF")
XRF_CLEAN = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("xrf_tunnel_dike_clean.xlsx")
OUT = Path("pr6_results"); (OUT/"figures").mkdir(parents=True, exist_ok=True); (OUT/"tables").mkdir(exist_ok=True)

# Maps left out, with the reason (see README)
EXCLUDE_MAPS = {"PR-6-140 EBSD Test Map": "test map: implausible apatite/quartz, poor indexing",
                "PR-6-420 Small EBSD Map": "superseded by the ReDo map (implausible 85% feldspar)",
                "PR-6-420 Large Map": "superseded by the ReDo map"}
WL = np.arange(400, 2451, 1.0)                  # common wavelength grid (nm)
# Sample positions across the dike, metres from the west wall (Currier et al. 2026, EPSL 679:119895, Fig. 1c)
X_FROM_WEST = {140: 1.6, 280: 3.3, 420: 4.7, 560: 6.3, 700: 7.3, 840: 8.7, 980: 10.3, 1120: 12.3, 1260: 13.7}
DIKE_WIDTH = 16.7                               # m
rng = np.random.default_rng(1)

# ---------------------------------------------------------------- 1. SPECTRA
def read_sed(path):
    """Return (sample position, wavelength, reflectance 0-1) for one NaturaSpec .sed file."""
    text = open(path, 'rb').read().decode('latin-1')
    comment = re.search(r'Comment:(.*)', text).group(1).strip()        # e.g. 'PR6-280-mini_2'
    pos = int(re.search(r'PR6-(\d+)', comment).group(1))                # -> 280
    rows = [l.split() for l in text.split('Wvl')[1].splitlines()[1:] if l.strip()]
    d = np.array(rows, float)
    return pos, d[:, 0], d[:, 1] / 100.0                                 # percent -> fraction

def clean(wl, r):
    keep = (wl >= WL[0]) & (wl <= WL[-1]); wl, r = wl[keep], r[keep]    # trim noisy ends
    for lo, hi in [(1000, 1010), (1790, 1810)]:                          # detector joins
        r[(wl >= lo) & (wl <= hi)] = np.nan
    ok = ~np.isnan(r)
    return np.interp(WL, wl[ok], r[ok])                                  # resample to 1-nm grid

def window(r, a, b): return r[(WL >= a) & (WL <= b)]

def band_depth(r, left, right, center, smooth=21):
    """Continuum-removed depth: straight line between the shoulder maxima, depth = 1 - lowest (smoothed) point."""
    mL = (WL >= left[0]) & (WL <= left[1]); mR = (WL >= right[0]) & (WL <= right[1])
    iL = np.nonzero(mL)[0][np.argmax(r[mL])]; iR = np.nonzero(mR)[0][np.argmax(r[mR])]
    cont = np.interp(WL, [WL[iL], WL[iR]], [r[iL], r[iR]])
    mC = (WL >= center[0]) & (WL <= center[1]); cr = r[mC] / cont[mC]
    cs = np.convolve(np.pad(cr, smooth // 2, mode='edge'), np.ones(smooth) / smooth, 'valid')
    j = np.argmin(cs); return 1 - cs[j], WL[mC][j]

def spectral_features(r):
    f = {'brightness': r.mean()}                                                    # overall albedo
    f['band1_depth'], f['band1_center'] = band_depth(r, (700, 900), (1450, 1700), (900, 1250))  # ~1 µm Fe2+ band
    f['h2o_1900'] = 1 - window(r, 1930, 1960).mean() / window(r, 1830, 1860).mean()  # 1.9 µm water feature
    f['feat2300_depth'], _ = band_depth(r, (2230, 2270), (2350, 2420), (2270, 2350), smooth=11)  # Mg/Fe-OH
    f['vis_slope'] = window(r, 780, 820).mean() / window(r, 480, 520).mean()        # red slope
    return f

spectra = {}
for f in sorted(glob.glob(str(DATA / "PR-6_Hyperspec" / "*.sed"))):
    pos, wl, r = read_sed(f); spectra.setdefault(pos, []).append(clean(wl, r))
spectra = {p: np.vstack(v) for p, v in sorted(spectra.items())}
spot_rows = [dict(pos=p, spot=k + 1, **spectral_features(r)) for p, s in spectra.items() for k, r in enumerate(s)]
spots = pd.DataFrame(spot_rows)
spec = spots.groupby('pos').mean().drop(columns='spot')
spec['spot_scatter_pct'] = {p: (100 * s.std(0, ddof=1) / s.mean(0)).mean() for p, s in spectra.items()}
spots.to_csv(OUT/"tables"/"spectral_features_by_spot.csv", index=False)

# ---------------------------------------------------------------- 2. EBSD
def csd(L_um, area_mm2, Lmin, Lmax):
    """CSD from 2D long axes: slope of ln(population density) vs size (geometric bins)."""
    L = np.asarray(L_um) / 1000
    edges = 10 ** np.arange(np.log10(Lmin / 1000), np.log10(Lmax / 1000) + 0.1, 0.1)
    c, _ = np.histogram(L, edges); mids = np.sqrt(edges[:-1] * edges[1:]); n = c / np.diff(edges) / area_mm2
    ok = c >= 3; s, i = np.polyfit(mids[ok], np.log(n[ok]), 1); return s, i

map_rows, size_rows, all_grains = [], [], []
for ctf in sorted(glob.glob(str(DATA / "PR6_EBSD" / "**" / "*.ctf"), recursive=True)):
    folder = Path(ctf).relative_to(DATA / "PR6_EBSD").parts[0]
    if 'EDS' in folder or folder in EXCLUDE_MAPS or folder in {m['folder'] for m in map_rows}: continue
    pos = int(re.search(r'PR[- ]?6[- ](\d+)', folder).group(1))
    h, phases, a = read_ctf(ctf); step = h['XStep']; kind = 'small' if step < 5 else 'large'
    P = a['Phase']; area = P.size * step ** 2 / 1e6
    frac = {'unindexed': (P == 0).mean()}
    for code, grp in GROUP.items(): frac[grp] = frac.get(grp, 0) + (P == code).mean()
    indexed = 1 - frac['unindexed']
    map_rows.append(dict(pos=pos, step=kind, folder=folder, step_um=round(step, 2), area_mm2=area,
                         unindexed_pct=100 * frac['unindexed'],
                         **{f'{g}_pct': 100 * frac.get(g, 0) / indexed for g in ['feldspar', 'augite', 'Fe-Ti oxide', 'ilmenite', 'apatite', 'quartz']}))
    lab, G = reconstruct(fill_zero_solutions(a))                  # pixels -> grains
    gt = grain_table(lab, G, step); gt['pos'] = pos; gt['step'] = kind; all_grains.append(gt)
    gt = gt[~gt.touches_edge]                                     # cut-off grains would bias sizes
    row = dict(pos=pos, step=kind)
    for m in ['feldspar', 'augite']:
        x = gt[gt.mineral == m]; lo, hi = (20, 300) if kind == 'small' else (100, 2000)
        s, _ = csd(x.long_axis_um, area, lo, hi)
        row.update({f'{m}_n_per_mm2': len(x) / area, f'{m}_median_long_um': x.long_axis_um.median(),
                    f'{m}_p90_long_um': x.long_axis_um.quantile(.9), f'{m}_csd_slope': s, f'{m}_char_len_um': -1000 / s})
    size_rows.append(row); print('EBSD done:', folder)
maps = pd.DataFrame(map_rows).sort_values(['step', 'pos']); maps.to_csv(OUT/"tables"/"ebsd_map_summary.csv", index=False)
sizes = pd.DataFrame(size_rows).sort_values(['step', 'pos']); sizes.to_csv(OUT/"tables"/"ebsd_crystal_sizes.csv", index=False)
pd.concat(all_grains).to_csv(OUT/"tables"/"ebsd_grains_all.csv.gz", index=False)

# ---------------------------------------------------------------- 3. XRF + combine
xrf = pd.read_excel(XRF_CLEAN); xrf['pos'] = xrf.Sample.str.extract(r'(\d+)$').astype(int)
large = sizes[sizes.step == 'large'].set_index('pos'); lmap = maps[maps.step == 'large'].set_index('pos')
M = (spec.join(large[['feldspar_char_len_um', 'augite_char_len_um', 'feldspar_n_per_mm2', 'feldspar_p90_long_um']])
         .join(lmap[['augite_pct', 'ilmenite_pct', 'Fe-Ti oxide_pct', 'unindexed_pct']])
         .join(xrf.set_index('pos')[['SiO2', 'TiO2', 'MgO', 'K2O']]))
M['x_from_west_m'] = pd.Series(X_FROM_WEST)
M['dist_nearest_wall_m'] = np.minimum(M.x_from_west_m, DIKE_WIDTH - M.x_from_west_m)
M['measurement_order'] = np.arange(1, len(M) + 1)          # spectra were measured 140 -> 1260 in one session
M.to_csv(OUT/"tables"/"master_table.csv")

# ---------------------------------------------------------------- 4. statistics
def shuffle_test(x, y, n=20000):
    m = ~(np.isnan(x) | np.isnan(y)); x, y = x[m], y[m]; rx, ry = rankdata(x), rankdata(y)
    obs = np.corrcoef(rx, ry)[0, 1]; sh = np.array([rng.permutation(ry) for _ in range(n)])
    sc = sh - sh.mean(1, keepdims=True); rc = rx - rx.mean()
    null = (sc @ rc) / (np.linalg.norm(sc, axis=1) * np.linalg.norm(rc))
    return obs, (np.sum(abs(null) >= abs(obs)) + 1) / (n + 1), int(m.sum())
spec_cols = ['brightness', 'band1_depth', 'band1_center', 'h2o_1900', 'feat2300_depth', 'vis_slope', 'spot_scatter_pct']
targets = ['feldspar_char_len_um', 'augite_char_len_um', 'augite_pct', 'ilmenite_pct', 'TiO2', 'MgO', 'dist_nearest_wall_m', 'measurement_order']
res = []
for s in spec_cols:
    for t in targets:
        r, p, n = shuffle_test(M[s].to_numpy(float), M[t].to_numpy(float)); res.append(dict(spectral=s, compared_with=t, spearman_rho=r, shuffle_p=p, n=n))
res = pd.DataFrame(res); res['passes_strict_p'] = res.shuffle_p < 0.05 / len(spec_cols)
res.to_csv(OUT/"tables"/"statistics.csv", index=False)
# Non-spectral quantities vs distance to the nearest wall (texture, mineralogy, chemistry)
prof = []
for col in ['feldspar_char_len_um', 'augite_char_len_um', 'feldspar_n_per_mm2', 'augite_pct', 'ilmenite_pct', 'Fe-Ti oxide_pct', 'unindexed_pct', 'MgO', 'TiO2', 'SiO2', 'K2O']:
    r, p, n = shuffle_test(M['dist_nearest_wall_m'].to_numpy(float), M[col].to_numpy(float))
    prof.append(dict(quantity=col, spearman_rho_vs_nearest_wall=r, shuffle_p=p, n=n))
prof = pd.DataFrame(prof); prof.to_csv(OUT/"tables"/"profile_statistics.csv", index=False)
print(prof.round(3).to_string(index=False))

print(res.pivot(index='spectral', columns='compared_with', values='spearman_rho').round(2).to_string())

# ---------------------------------------------------------------- 5. figures
pos = M.x_from_west_m.values                    # plot against metres from the west wall
fig, axes = plt.subplots(4, 1, figsize=(7.5, 10), sharex=True)
axes[0].plot(pos, M.feldspar_char_len_um, 'o-', label='feldspar'); axes[0].plot(pos, M.augite_char_len_um, 's-', label='augite')
axes[0].set_ylabel('CSD characteristic\nlength (µm)'); axes[0].legend(); axes[0].set_title('Tunnel Dike (PR-6): profile across the dike')
axes[1].plot(pos, M.TiO2, 'o-', color='tab:red', label='TiO₂ (XRF)'); axes[1].plot(pos, M.MgO, 's-', color='tab:purple', label='MgO (XRF)')
axes[1].set_ylabel('wt% (normalized)'); axes[1].legend()
axes[2].plot(pos, M.augite_pct, 'o-', color='tab:green'); ax2 = axes[2].twinx(); ax2.plot(pos, M.ilmenite_pct, 's-', color='k')
axes[2].set_ylabel('augite (% of indexed)', color='tab:green'); ax2.set_ylabel('ilmenite (%, black)'); axes[2].set_title('EBSD mineral proportions', fontsize=9)
axes[3].plot(pos, M.h2o_1900, 'o-', color='tab:blue'); axes[3].set_ylabel('1.9 µm H₂O\nfeature strength'); axes[3].set_title('Hyperspectral', fontsize=9)
axes[3].set_xlabel('Distance from west wall (m)')
axes[0].set_ylim(0, M.feldspar_char_len_um.max() * 1.2)
for ax in axes:
    ax.set_xlim(-0.3, DIKE_WIDTH + 0.3)
    for wall in (0, DIKE_WIDTH): ax.axvline(wall, color='firebrick', lw=2)   # dike walls
for p_, xm in X_FROM_WEST.items(): axes[0].annotate(str(p_), (xm, M.loc[p_, 'feldspar_char_len_um']), fontsize=7, xytext=(0, 6), textcoords='offset points', ha='center')
axes[0].text(0.1, axes[0].get_ylim()[1]*0.95, 'W wall', color='firebrick', fontsize=8, va='top'); axes[0].text(DIKE_WIDTH-0.1, axes[0].get_ylim()[1]*0.95, 'E wall', color='firebrick', fontsize=8, va='top', ha='right')
fig.tight_layout(); fig.savefig(OUT/"figures"/"01_dike_profile.png", dpi=300)

def stat(s, t):
    r = res[(res.spectral == s) & (res.compared_with == t)].iloc[0]
    return f"Spearman ρ = {r.spearman_rho:+.2f}, p = {r.shuffle_p:.3f} (n = {r.n})"
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, lab in [(axes[0], 'feldspar_char_len_um', 'Feldspar CSD characteristic length (µm)'),
                     (axes[1], 'TiO2', 'TiO₂ (wt%, XRF)'), (axes[2], 'dist_nearest_wall_m', 'Distance to nearest dike wall (m)')]:
    d = M[[col, 'h2o_1900']].dropna(); sc = ax.scatter(d[col], d.h2o_1900, c=M.loc[d.index, 'x_from_west_m'], cmap='viridis', vmin=0, vmax=DIKE_WIDTH, s=60, edgecolor='k')
    for p_, r_ in d.iterrows(): ax.annotate(str(p_), (r_[col], r_.h2o_1900), fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax.set_xlabel(lab); ax.set_ylabel('1.9 µm H₂O feature strength'); ax.set_title(stat('h2o_1900', col), fontsize=9)
fig.colorbar(sc, ax=axes.tolist(), label='distance from west wall (m)'); fig.savefig(OUT/"figures"/"02_h2o_vs_texture_chemistry.png", dpi=300, bbox_inches='tight')

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5)); cm = plt.cm.viridis
for i, (p_, s) in enumerate(spectra.items()):
    r = s.mean(0); axes[0].plot(WL, r * 100, color=cm(X_FROM_WEST[p_] / DIKE_WIDTH), label=f'PR-6-{p_} ({X_FROM_WEST[p_]} m)')
    m = (WL >= 1750) & (WL <= 2100); axes[1].plot(WL[m], r[m] / window(r, 1830, 1860).mean(), color=cm(X_FROM_WEST[p_] / DIKE_WIDTH))
axes[0].set(xlabel='Wavelength (nm)', ylabel='Reflectance (%)', title='Mean spectra (3 spots each)'); axes[0].legend(fontsize=7, ncol=2)
axes[1].set(xlabel='Wavelength (nm)', ylabel='Normalized reflectance', title='1.9 µm water feature (normalized at 1830–1860 nm)')
for a_, b_ in [(1830, 1860), (1930, 1960)]: axes[1].axvspan(a_, b_, color='gray', alpha=0.15)
fig.tight_layout(); fig.savefig(OUT/"figures"/"03_spectra.png", dpi=300)

from matplotlib.patches import Patch
cols = np.array([[.1,.1,.1],[.93,.82,.55],[.6,.2,.6],[.35,.35,.35],[.93,.82,.55],[.2,.65,.2],[.6,.2,.6],[.45,.25,.1],[.45,.25,.1]])
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (p_, folder) in zip(axes, [(140, 'PR-6-140 Larger EBSD Map'), (560, 'PR-6-560 Large Map'), (1260, 'PR6 1260 Large Step Map')]):
    h, ph, a = read_ctf(glob.glob(str(DATA / "PR6_EBSD" / folder / "**" / "*.ctf"), recursive=True)[0])
    ax.imshow(cols[a['Phase'][:400, :400]], interpolation='nearest'); ax.set_title(f'PR-6-{p_}: {X_FROM_WEST[p_]} m from W wall, {min(X_FROM_WEST[p_], DIKE_WIDTH - X_FROM_WEST[p_]):.1f} m from nearest wall\n(6 × 6 mm area, 15 µm step)', fontsize=10); ax.axis('off')
fig.legend(handles=[Patch(color=cols[1], label='feldspar'), Patch(color=cols[5], label='augite'), Patch(color=cols[3], label='ilmenite'),
                    Patch(color=cols[7], label='magnetite / Ti-magnetite'), Patch(color=cols[2], label='apatite / quartz (minor, partly misindexed)'),
                    Patch(color=cols[0], label='unindexed')], loc='lower center', ncol=3)
fig.savefig(OUT/"figures"/"04_ebsd_maps_margin_center_margin.png", dpi=200, bbox_inches='tight')

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(M.measurement_order, M.vis_slope, 'o-'); axes[0].set(xlabel='Measurement order (140 first → 1260 last)', ylabel='Visible slope (R800/R500)', title=stat('vis_slope', 'measurement_order'))
axes[1].plot(M.measurement_order, M.band1_depth, 'o-', color='tab:orange'); axes[1].set(xlabel='Measurement order', ylabel='1 µm band depth', title=stat('band1_depth', 'measurement_order'))
fig.tight_layout(); fig.savefig(OUT/"figures"/"05_measurement_order_check.png", dpi=300)

print("\nFigures and tables written to", OUT.resolve())
