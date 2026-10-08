"""
PR-6 tunnel dike: MCMC spectral mixing model for mineral proportions
====================================================================
Estimates how much each library mineral contributes to each sample's absorption
bands, with 95% uncertainty ranges, then checks the estimates against EBSD.

Usage:
    python run_mcmc_pr6.py "path/to/hyperspec x EBSD x XRF" path/to/corpus_hullq.npz path/to/master_table.csv
      - corpus_hullq.npz : spectral-workbench/_normalized/corpus_hullq.npz (Tushar's library corpus)
      - master_table.csv : pr6_results/tables/master_table.csv from run_pr6_analysis.py (EBSD + XRF values)

Model, in plain terms
  1. Continuum removal: divide every spectrum by a straight-line "roof" (upper convex hull) so only
     the absorption dips remain. The SAME window (750-2400 nm) and method are applied to the
     samples and to the library minerals, so they are directly comparable.
  2. Absorption = 1 - continuum-removed spectrum.
  3. Mixing: sample absorption ~= sum of (weight x mineral absorption). The weights are "apparent
     abundances": how strongly each mineral's bands show up. They are NOT volume %, because band
     strength also depends on grain size and on dark minerals that suppress other bands.
  4. MCMC: an ensemble sampler (Goodman & Weare 2010, the algorithm used by emcee) explores all
     weight combinations consistent with the data and gives a posterior range for each weight.
Checks built in: a known-answer test on synthetic mixtures, posterior correlations (which minerals
trade off against each other), a pool-sensitivity test, and comparison with EBSD.
"""
import sys, re, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import nnls
from scipy.stats import rankdata

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("hyperspec x EBSD x XRF")
CORPUS = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("corpus_hullq.npz")
MASTER = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("master_table.csv")
OUT = Path("mcmc_results"); (OUT / "figures").mkdir(parents=True, exist_ok=True); (OUT / "tables").mkdir(exist_ok=True)

LO, HI = 750, 2400      # fit window (nm). Below 750 nm, iron charge-transfer/weathering effects dominate
THIN = 5                # use every 5th nm (reduces correlated noise, speeds up)
# Library end-members: index in the corpus -> label. Chosen by the pool test described in the README.
END_MEMBERS = {"augite": "Augite_WS592_Pyroxene_BECKb_AREF.txt",
               "magnetite": "Magnetite_HS195.3B_ASDFRb_AREF.txt",
               "zeolite (stilbite)": "Stilbite_HS482.1B_Zeolite_ASDFRb_AREF.txt"}
# Also tested (see pool test): plagioclase, ilmenite, chlorite contributed nothing; saponite fit worse than zeolite
EXTRA_TESTED = {"plagioclase": "Labradorite_HS17.3B_ASDFRc_AREF.txt", "ilmenite": "Ilmenite_HS231.3B_ASDFRb_AREF.txt",
                "chlorite": "Chlorite_HS197.3B_BECKb_AREF.txt", "smectite (saponite)": "Saponite_SapCa-1_BECKb_AREF.txt"}
rng = np.random.default_rng(42)
X_FROM_WEST = {140: 1.6, 280: 3.3, 420: 4.7, 560: 6.3, 700: 7.3, 840: 8.7, 980: 10.3, 1120: 12.3, 1260: 13.7}  # m, Currier et al. 2026 Fig. 1c
DIKE_WIDTH = 16.7

# ---------------------------------------------------------------- continuum removal
def upper_hull(wl, y):
    """Upper convex hull: the 'roof' over a spectrum, made of straight lines between its high points."""
    pts = []
    for x, v in zip(wl, y):
        while len(pts) >= 2:
            (x1, y1), (x2, y2) = pts[-2], pts[-1]
            if (x2 - x1) * (v - y1) - (y2 - y1) * (x - x1) >= 0: pts.pop()   # middle point is under the roof
            else: break
        pts.append((x, v))
    hx, hy = zip(*pts); return np.interp(wl, hx, hy)

corpus = np.load(CORPUS, allow_pickle=True); WL = corpus["wl"]; LIB = corpus["M"]
import ast
ROWS = [r if isinstance(r, dict) else ast.literal_eval(str(r)) for r in corpus["rows"]]
FILE_INDEX = {r["file"]: i for i, r in enumerate(ROWS) if r["library"].startswith("USGS_Minerals")}  # file name -> row
win = (WL >= LO) & (WL <= HI); w = WL[win]; idx = np.arange(win.sum())[::THIN]; wf = w[idx]

def absorption(spectrum_full_range):
    """1 - (spectrum / its hull) inside the fit window, thinned."""
    s = spectrum_full_range[win]; return 1 - (s / upper_hull(w, s))[idx]

def library_absorption(file):
    i = FILE_INDEX[file]
    q = LIB[i] / LIB[i].max()       # corpus rows are unit-length continuum-removed spectra; max rescales to 1
    return absorption(q)

names = list(END_MEMBERS); E = np.stack([library_absorption(f) for f in END_MEMBERS.values()])

# ---------------------------------------------------------------- samples
spots = {}
for f in sorted(glob.glob(str(DATA / "PR-6_Hyperspec" / "*.sed"))):
    t = open(f, "rb").read().decode("latin-1")
    pos = int(re.search(r"PR6-(\d+)", re.search(r"Comment:(.*)", t).group(1)).group(1))
    d = np.array([[float(x) for x in l.split()] for l in t.split("Wvl")[1].splitlines()[1:] if l.strip()])
    r = savgol_filter(np.interp(WL, d[:, 0], d[:, 1] / 100), 11, 2)     # light smoothing
    spots.setdefault(pos, []).append(absorption(r))
spots = {p: np.vstack(v) for p, v in sorted(spots.items())}

# ---------------------------------------------------------------- MCMC
def log_post(theta, A, E):
    a, logs = theta[:-1], theta[-1]
    if np.any(a < 0) or a.sum() > 1 or not (-9 < logs < 0): return -np.inf   # priors: weights >= 0, sum <= 1
    r = A - a @ E; s = np.exp(logs)
    return -0.5 * np.sum((r / s) ** 2) - len(A) * logs                    # Gaussian likelihood, unknown noise

def ensemble_mcmc(A, E, n_walkers=40, n_steps=4000, burn=1500, seed=0):
    """Affine-invariant 'stretch move' ensemble sampler (Goodman & Weare 2010)."""
    g = np.random.default_rng(seed); K = E.shape[0]; D = K + 1
    a0, _ = nnls(E.T, A); a0 = np.clip(a0, 1e-3, None)                      # start near the best fit
    s0 = np.log(np.std(A - a0 @ E) + 1e-4)
    P = np.column_stack([np.abs(a0 + 1e-3 * g.standard_normal((n_walkers, K))), s0 + 0.01 * g.standard_normal(n_walkers)])
    lp = np.array([log_post(p, A, E) for p in P]); chain = np.empty((n_steps, n_walkers, D)); acc = 0; h = n_walkers // 2
    for t in range(n_steps):
        for S, C in [(slice(0, h), slice(h, None)), (slice(h, None), slice(0, h))]:
            Ps, Pc = P[S], P[C]; n = len(Ps)
            z = (g.random(n) + 1) ** 2 / 2                                     # stretch factor (a = 2)
            partner = Pc[g.integers(0, len(Pc), n)]; prop = partner + z[:, None] * (Ps - partner)
            lpp = np.array([log_post(p, A, E) for p in prop])
            ok = np.log(g.random(n)) < (D - 1) * np.log(z) + lpp - lp[S]
            Ps[ok] = prop[ok]; l = lp[S]; l[ok] = lpp[ok]; P[S] = Ps; lp[S] = l; acc += ok.sum()
        chain[t] = P
    return chain[burn:].reshape(-1, D), acc / (n_steps * n_walkers)

# ---------------------------------------------------------------- 1. known-answer test
cover = []
for t in range(10):
    a_true = rng.dirichlet(np.ones(len(names))) * rng.uniform(0.05, 0.25)
    A = a_true @ E + rng.normal(0, 0.005, len(idx))
    smp, _ = ensemble_mcmc(A, E, n_steps=3000, burn=1000, seed=100 + t)
    lo, hi = np.percentile(smp[:, :-1], [2.5, 97.5], 0); cover.append((a_true >= lo) & (a_true <= hi))
print(f"Known-answer test: true value inside the 95% range {100*np.mean(cover):.0f}% of the time (ideal ~95%)")

# ---------------------------------------------------------------- 2. pool-sensitivity test
pool_rows = []
full_pool = {**END_MEMBERS, **EXTRA_TESTED}
for label, pool in [("final: augite + magnetite + zeolite", list(END_MEMBERS)),
                    ("augite + magnetite + saponite (clay)", ["augite", "magnetite", "smectite (saponite)"]),
                    ("final + plagioclase, ilmenite, chlorite", list(END_MEMBERS) + ["plagioclase", "ilmenite", "chlorite"]),
                    ("augite + zeolite (no magnetite)", ["augite", "zeolite (stilbite)"])]:
    Ek = np.stack([library_absorption(full_pool[n]) for n in pool])
    rms = [np.sqrt(np.mean((s.mean(0) - nnls(Ek.T, s.mean(0))[0] @ Ek) ** 2)) for s in spots.values()]
    pool_rows.append(dict(pool=label, mean_rms_misfit=np.mean(rms)))
pools = pd.DataFrame(pool_rows); pools.to_csv(OUT / "tables" / "pool_sensitivity.csv", index=False); print(pools.round(4).to_string(index=False))

# ---------------------------------------------------------------- 3. fit every sample
post, rows = {}, []
for p, s in spots.items():
    A = s.mean(0); smp, acc = ensemble_mcmc(A, E, seed=p); post[p] = smp
    med = np.median(smp[:, :-1], 0); lo, hi = np.percentile(smp[:, :-1], [2.5, 97.5], 0)
    rows.append(dict(pos=p, acceptance=acc, rms_misfit=np.sqrt(np.mean((A - med @ E) ** 2)),
                     spot_noise=np.std(s, axis=0, ddof=1).mean(),
                     **{f"{n}": v for n, v in zip(names, med)}, **{f"{n}_lo95": v for n, v in zip(names, lo)},
                     **{f"{n}_hi95": v for n, v in zip(names, hi)}))
R = pd.DataFrame(rows).set_index("pos"); R.to_csv(OUT / "tables" / "mcmc_apparent_abundances.csv")
corr = pd.DataFrame(np.mean([np.corrcoef(post[p][:, :-1].T) for p in post], 0), index=names, columns=names)
corr.to_csv(OUT / "tables" / "posterior_correlations.csv")
print(R[["rms_misfit", "spot_noise"] + names].round(4).to_string()); print("\nPosterior correlations:\n", corr.round(2).to_string())

# ---------------------------------------------------------------- 4. compare with EBSD
Mt = pd.read_csv(MASTER, index_col="pos"); Mt["oxides_pct"] = Mt["Fe-Ti oxide_pct"] + Mt["ilmenite_pct"]
def shuffle_test(x, y, n=20000):
    rx, ry = rankdata(x), rankdata(y); obs = np.corrcoef(rx, ry)[0, 1]
    sh = np.array([rng.permutation(ry) for _ in range(n)]); sc = sh - sh.mean(1, keepdims=True); rc = rx - rx.mean()
    null = (sc @ rc) / (np.linalg.norm(sc, axis=1) * np.linalg.norm(rc)); return obs, (np.sum(abs(null) >= abs(obs)) + 1) / (n + 1)
checks = [("augite", "augite_pct", "EBSD augite %"), ("magnetite", "oxides_pct", "EBSD Fe-Ti oxides + ilmenite %"),
          ("zeolite (stilbite)", "h2o_1900", "1.9 µm water feature"),
          ("zeolite (stilbite)", "feldspar_char_len_um", "feldspar CSD characteristic length (µm)")]
val = []
for a, b, lab in checks:
    d = R[[a]].join(Mt[[b]]).dropna(); r, pv = shuffle_test(d[a].values, d[b].values)
    val.append(dict(model=a, compared_with=lab, spearman_rho=r, shuffle_p=pv, n=len(d)))
val = pd.DataFrame(val); val.to_csv(OUT / "tables" / "validation_vs_ebsd.csv", index=False); print("\n", val.round(3).to_string(index=False))

# ---------------------------------------------------------------- 5. figures
fig, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=True)
for ax, (p, s) in zip(axes.flat, spots.items()):
    A = s.mean(0); med = np.median(post[p][:, :-1], 0)
    for x in s: ax.plot(wf, x, color="0.8", lw=0.8)
    ax.plot(wf, A, "k", label="observed (mean of 3 spots)"); ax.plot(wf, med @ E, "r", label="model")
    ax.set_title(f"PR-6-{p}   misfit {R.loc[p,'rms_misfit']:.4f} vs spot noise {R.loc[p,'spot_noise']:.4f}", fontsize=9)
for ax in axes[-1]: ax.set_xlabel("Wavelength (nm)")
for ax in axes[:, 0]: ax.set_ylabel("Absorption (1 − CR)")
axes[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "figures" / "M1_model_fits.png", dpi=250)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (a, b, lab), v in zip(axes, checks[:2] + checks[3:], [val.iloc[0], val.iloc[1], val.iloc[3]]):
    d = R.join(Mt[[b]])
    ax.errorbar(d[b], d[a], yerr=[d[a] - d[f"{a}_lo95"], d[f"{a}_hi95"] - d[a]], fmt="o", color="k", ecolor="0.6", capsize=3)
    for p_, r_ in d.iterrows(): ax.annotate(str(p_), (r_[b], r_[a]), fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel(lab); ax.set_ylabel(f"Model apparent abundance: {a}"); ax.set_title(f"ρ = {v.spearman_rho:+.2f}, p = {v.shuffle_p:.3f} (n = {v.n})", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "figures" / "M2_model_vs_ebsd.png", dpi=250)

fig, ax = plt.subplots(figsize=(5, 4.5)); p = 700; smp = post[p]
ax.scatter(smp[::20, 0], smp[::20, 1], s=2, alpha=0.3)
ax.set(xlabel="augite apparent abundance", ylabel="magnetite apparent abundance",
       title=f"PR-6-{p}: augite and magnetite trade off\n(posterior correlation {corr.loc['augite','magnetite']:+.2f}, averaged over samples)")
fig.tight_layout(); fig.savefig(OUT / "figures" / "M3_augite_magnetite_tradeoff.png", dpi=250)

fig, ax = plt.subplots(figsize=(8, 4)); xm = np.array([X_FROM_WEST[p] for p in R.index])
for wall in (0, DIKE_WIDTH): ax.axvline(wall, color="firebrick", lw=2)
for n, c in zip(names, ["tab:green", "k", "tab:blue"]):
    ax.errorbar(xm, R[n], yerr=[R[n] - R[f"{n}_lo95"], R[f"{n}_hi95"] - R[n]], fmt="o-", color=c, capsize=3, label=n)
ax.set(xlabel="Distance from west wall (m)", ylabel="Apparent abundance (median, 95% range)", title="MCMC mixing model across the dike", xlim=(-0.3, DIKE_WIDTH + 0.3))
ax.legend(); fig.tight_layout(); fig.savefig(OUT / "figures" / "M4_model_profile.png", dpi=250)

fig, ax = plt.subplots(figsize=(8, 4))
for n, e in zip(names, E): ax.plot(wf, e, label=n)
ax.set(xlabel="Wavelength (nm)", ylabel="Absorption (1 − CR)", title="End-member absorption spectra (750–2400 nm)"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "figures" / "M5_end_members.png", dpi=250)
print("\nDone. Results in", OUT.resolve())
