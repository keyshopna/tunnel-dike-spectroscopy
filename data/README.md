# Data

`currier2026_ST2_plagioclase.csv` is from the supplementary material of Currier et al. (2026), Table ST2: the number of plagioclase crystals measured in each sample, the ShapeCalc short : intermediate : long shape (only I and L are listed, S = 1), and the mean 3D crystal size in µm. `x_from_west_m` is each sample's distance from the west wall of the dike, from Fig. 1 of the same paper.

The rest of the data is unpublished and isn't included here yet. If you'd like to use it, please get in touch. The scripts expect:

- `hyperspec x EBSD x XRF/PR-6_Hyperspec/*.sed`: NaturaSpec Plus spectra, three spots per sample
- `hyperspec x EBSD x XRF/PR6_EBSD/<map folder>/*.ctf`: EBSD maps exported from AZtec
- `xrf_tunnel_dike_clean.xlsx`: XRF major elements
- `pr6_geochemistry_anhydrous.csv`: one row per sample, majors normalized to 100% anhydrous (columns like `MgO*`), traces in ppm, plus the ratios La/Yb, Zr/Nb, Nb/Y, Th/Nb
- `pr6_fusion_loss.csv`: columns `pos, x_from_west_m, pulse, disk_recipe, fusion_loss_pct, h2o_1900`
- EDS maps saved as `PR-6-<sample>-SiAlmap.png`
- `corpus_hullq.npz`: continuum-removed library spectra used by the mixing model
