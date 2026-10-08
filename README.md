# Tunnel Dike spectroscopy

Code for my GSA 2026 poster, *The Value of Field Spectrometers: Can Hyperspectral Data Distinguish between Magma Pulses in Tunnel Dike, Columbia River Flood Basalts?* (Keya Hopwood, Paulo Hidalgo, Tushar Mittal, and Ryan Currier).

The Tunnel Dike (PR-6) in the Chief Joseph Dike Swarm was built by at least two pulses of magma, an aphanitic first pulse along the margins and a crystal-rich porphyritic core (Currier et al., 2026). For this project we took hand-held reflectance spectra (NaturaSpec Plus, 350–2500 nm) on the billet faces left over from thin-section preparation and compared them with EBSD, EDS, XRF, and ICP-MS data from the same nine samples. The 1.9 µm water feature separates the two pulses in the same way plagioclase crystal shape does, while the usual grain-size spectral indicators don't show up in these dark, oxide-rich rocks.

## What's in here

```
analysis/        main analysis: spectra, EBSD grain reconstruction, crystal sizes, statistics
spectra/         script I used to rename and clean the raw .sed files from the spectrometer
mixing_model/    MCMC spectral mixing model (preliminary)
figures/         scripts for poster Figs. 3, 4, 6, 7 and the sample transect strip
results/         output tables and the figures as they appear on the poster
data/            published crystal size and shape data (Currier et al., 2026, Table ST2)
```

`analysis/run_pr6_analysis.py` does most of the work. It reads the 27 spectra (three spots per sample), trims the noisy ends and detector joins, and measures brightness, the 1 µm band depth, and the 1.9 µm water feature. It then reads the EBSD maps (.ctf), rebuilds grains in Python (neighboring pixels of the same mineral within 10°, feldspar twins merged, grains under 10 pixels or touching the map edge dropped), and fits 2D crystal size distributions. Everything is compared using Spearman correlations with permutation tests. `grains.py` and `ebsdio.py` are the helper files it imports.

The mixing model in `mixing_model/` fits each spectrum as a mix of augite, magnetite, and zeolite (stilbite) spectra from the USGS Spectral Library v7 (Kokaly et al., 2017), using an ensemble MCMC sampler to get uncertainty on each weight. The weights are apparent abundances, not volume percent, so treat them as a work in progress.

## Running it

Python 3.10 or newer.

```
pip install -r requirements.txt

cd analysis
python run_pr6_analysis.py "path/to/hyperspec x EBSD x XRF" path/to/xrf_tunnel_dike_clean.xlsx

cd ../figures
python make_fig3_profile.py ../data/currier2026_ST2_plagioclase.csv ../results/tables/master_table.csv
python make_fig6_geochemistry.py ../data/pr6_geochemistry_anhydrous.csv
python make_fig7_fusion_loss.py ../data/pr6_fusion_loss.csv
python make_fig4_eds_maps.py path/to/eds_maps
python make_transect_strip.py

cd ../mixing_model
python run_mcmc_pr6.py "path/to/hyperspec x EBSD x XRF" path/to/corpus_hullq.npz ../results/tables/master_table.csv
```

The raw spectra, EBSD maps, and geochemistry are not in this repository yet (see `data/README.md`). The tables in `results/tables` are the outputs the poster numbers come from, so Fig. 3 can be remade from what's here.

## A few things to keep in mind

- Nine samples, with only two from pulse 1 that have geochemistry, so this is a feasibility test.
- The spectra were measured in position order. Visible slope drifts with that order, but the water feature does not.
- My crystal sizes are 2D and not stereologically corrected. The conclusions use the published 3D values.
- Fusion mass loss from making the XRF glass disks is used as a relative volatile measure. It is not a true LOI.
- Sample positions across the dike (meters from the west wall) are from Currier et al. (2026), Fig. 1.

## References

Currier, R.M., Hidalgo, P.J., Mittal, T., et al., 2026, Multi-pulse and crystal-laden emplacement of the Tunnel Dike: Earth and Planetary Science Letters, v. 679, 119895.

Kokaly, R.F., et al., 2017, USGS Spectral Library Version 7: U.S. Geological Survey Data Series 1035.

## Contact

Keya Hopwood, Department of Geosciences, Georgia State University (lhopwood1@gsu.edu)
