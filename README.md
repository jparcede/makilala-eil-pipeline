# Closed-form earthquake-induced landslide hazard assessment

Analysis code for the paper *Calibration transfers but discrimination does not:
spatially blocked validation of a closed-form co-seismic landslide model,
Makilala, Philippines*.

The framework computes, at every pixel and in closed form, the pseudo-static
Factor of Safety for shallow translational failure, the critical seismic
coefficient `k_c`, and a sharp bound on the sensitivity of the FoS to seismic
loading,

```
|S_kh|_max = 1 - 1 / FoS_static
```

which cancels every material strength parameter and therefore depends only on
the static safety margin. It is applied over 7.10 million pixels at 5 m
resolution and validated against 1,703 inventory polygons of the 2019
Cotabato-Davao del Sur seismic sequence.

No parameter of the physical model is fitted to the inventory. Exactly one
scalar is -- the operating threshold `theta*`, chosen to maximise the Youden
index -- and everything else comes from soil properties, topography and
modelled ground motion. To run the framework somewhere else, substitute that
municipality's rasters; note that an operating threshold must still be set
there, either borrowed from a calibrated area or estimated from a local
inventory. The paper's blocked validation is the evidence on what borrowing one
costs.

Everything is computed **pixel-wise** on the 5 m grid: the factor of safety, the
critical seismic coefficient and the sensitivity bound are element-wise functions
of the input rasters, with no slope units, terrain segmentation or zonal
aggregation anywhere in the pipeline. Spatial blocking for cross-validation is
applied afterwards, as 2 km square tiles built from pixel row/column indices.

## Layout

```
src/pipeline/     factor of safety, critical seismic coefficient, domain masking
src/validation/   pooled ROC/AUC, polygon and ESI-stratified detection,
                  Form B zone statistics
src/sensitivity/  delta-method uncertainty propagation into k_c;
                  Sobol variance-based global sensitivity (Saltelli, N=4096)
src/crs/          coordinate reference system diagnostics (see below)
src/figures/      figure regeneration
notebooks/        v9  -- the pipeline the first draft was written against
                  v10 -- the same model, rebuilt so that every published number
                         is recomputed and checked against its published value
                  v11 -- CURRENT. v10 plus the threshold-plateau test, the
                         block-size sweep and the blocked-validation figure.
                         The manuscript's numbers are the v11 numbers.
tests/            anti-drift corruption test and a synthetic fixture
docs/             CRS root-cause write-up
data/             input data documentation; rasters are not distributed here
```

## Running it

```bash
pip install -r requirements.txt
export MAKILALA_DIR=/path/to/rasters
python src/validation/fos_means.py
```

The restricted input rasters are not distributed here (see `data/README.md`), so
most scripts need `MAKILALA_DIR` to point at them. One check does not, and it is
the one to run first:

```bash
python tests/make_fixture.py     # synthetic rasters, no real data needed
python tests/guard_test.py       # expect 4/4
```

Both were last run clean on 25 Nov 2026 from a fresh clone.

With `MAKILALA_DIR` set, `src/validation/fos_means.py` and
`src/validation/formb_zones.py` were also run against the real rasters on the
same date and reproduce the manuscript: 7,104,441 scored pixels against the
published 7,104,440 (the one-pixel difference is float32 against the
notebook's float64 at the mask boundary), Form B defined on 83.1% of the
scored domain with a mean of 0.3865, and zone means of 0.088, 0.320 and 0.596
for the steep, transitional and lowland bands.

Every script resolves its input directory from `MAKILALA_DIR`, falling back to
`./data`. See `data/README.md` for the expected files, and for which inputs are
restricted and how to request them.

## Verification

`notebooks/Makilala_FoS_Pipeline_v11.ipynb` is the version to run, and the one
the manuscript's numbers come from. It computes the same model as v9 and v10 and
checks **81 quantities** against the values printed in the paper -- the domain cascade, all twelve cohesion x depth-cap
configurations, the parameter table over both its domains, the pixel-level
confusion matrix, Form B, polygon-level and ESI-stratified detection, the
spatially blocked validation including leave-one-block-out threshold transfer,
the saturation sweep, and the k_c variance decomposition with its
class-probability treatment of the threshold-assigned parameters. Run All, then
append `src/validation/v10_extended_checks.py` as a final cell. Results are
written to `output_maps_v10/master_summary.txt` and `extended_checks.txt`.

Eighty of the eighty-one targets are values published in the manuscript. One --
the fraction of pixels whose most probable cohesion class falls below even odds
-- is a regression target taken from v10 itself, because the originally
submitted 7.4% was traced to a masking error in
`src/sensitivity/kc_uncertainty_propagation.py` (`nan < 0.5` evaluates to
`False`, so no-data pixels entered the denominator). That is fixed here and the
manuscript now reports 11.1%; the code comment records the derivation.

v10 also guards the failure that motivated it. In v9 the evaluation mask was a
loose module global that a later cell could rebind, and in a full Run All it
did: `alpha[scored].min()` came back 0.10 where the mask enforces
`alpha >= 10`. Nothing raised. In v10 the canonical arrays live in a frozen
namespace, the mask is a module constant, and `guard()` re-asserts both at the
top of every cell, so drift becomes an exception rather than a wrong number.
`tests/guard_test.py` corrupts that state three ways and checks the guard fires
each time:

```bash
python tests/make_fixture.py     # synthetic rasters, no real data needed
python tests/guard_test.py       # expect 4/4
```

## Which script produces which result

| Manuscript item | Produced by |
|---|---|
| Table 2 domain cascade; Table 3 parameter statistics | `notebooks/...v11.ipynb` §16 |
| Figure 3 six-panel output map | `src/figures/regen_figs_v9.py` |
| Table 5 cohesion x depth-cap sweep; Figures 4-5 | `notebooks/...v11.ipynb` §16 |
| Pixel, polygon and ESI-stratified validation (Tables 6-8) | `src/validation/v7_cell17_final_validation_v2.py` |
| Form B zone statistics | `src/validation/formb_zones.py` |
| Spatially blocked validation, Table 10 and Figure 8 | `notebooks/...v11.ipynb` §28 |
| Threshold plateau and spatial K-fold transfer (Section 6.4) | `src/validation/v10_theta_plateau_test.py` |
| Variogram ranges and block-size sweep (Table 11) | `src/validation/v10_block_size.py` |
| Point-level validation (Section 6.5) | `notebooks/...v11.ipynb` §25 |
| k_c variance decomposition (Table 12) | `notebooks/...v11.ipynb` §12 |
| Sobol indices (Appendix C) | `src/sensitivity/regen_fig8_sobol.py` |
| CRS diagnosis (Section 4.4, Supplementary S1) | `src/crs/crs_offset_scan.py`, `crs_decisive_test.py` |

### Resolved: the soil-depth variance share

v11 cell 12 originally reported the soil-depth share of `Var(k_c)` as 69.9%
against the manuscript's 71.5%, with first-order shares summing to 98.1% rather
than 100%. The cause was an averaging-domain mismatch: each term's mean was
taken over the pixels where that term was defined, while the denominator was
the mean of the sum, which is undefined wherever any term is. Averaging
everything over the same support gives **71.4%** and a total of 100.0%,
confirming the published figure. `src/sensitivity/kc_variance_common_mask.py`
rebuilds the parameter stack and the scored mask from the rasters independently
of the notebook -- it returns SCORED = 7,104,440, matching the published count
-- and prints both decompositions side by side together with the sensitivity of
the depth share to the assumed relative error on depth (38.5% at 15%, 71.4% at
the assumed 30%, 87.4% at 50%). Cell 12 is corrected.

The decomposition's honest support is 4,754,653 px, 66.9% of the scored domain,
because that is where all four EBK standard-error surfaces exist.

### Resolved: the point-level subset

Section 6.5 of the submitted manuscript already uses this cell's rebuilt
figures -- 31 sites, J = 0.127, 4 of 31 classified differently. The apparent
disagreement with "27 sites, J = 0.097" came from the `(paper: ...)` comparison
literals inside cell 25, which were left over from an earlier draft; they are
now synchronised. `src/validation/point_screen_reconciliation.py` rebuilds the
scored mask independently (SCORED = 7,104,440) and enumerates the candidate
screens: 36 sites in domain, 31 complete on the five properties the
correlations consume, 30 complete on all nine laboratory fields. No screen
yields 27, and the five-property screen is the defensible one because LL, PL,
gravel and dry unit weight are not inputs to the model.

Cell 25 also resolved spreadsheet columns by substring, so `col("PL")` matched
`SAMPLE LABEL` -- "sam**PL**e label" -- before the real PL column, reporting PL
as present at 0 of 36 sites when it is present at 42 of 43. That is what
collapsed the "full laboratory record" screen to zero.

The co-location counts in Section 6.5 were corrected from 15 and 18 within
25 m and 50 m to **14 and 18**, verified by
`src/validation/colocation_check.py`, which measures both to the nearest
centre-rasterised inventory pixel (3 / 14 / 18, median 25.0 m) and to the
polygon geometry (3 / 16 / 18, median 21.9 m).

## The CRS diagnostics

`src/crs/` is independently useful. The eight interpolated rasters share one
grid, but four declared PRS 1992 (ESRI:102457) and four declared WGS 84
(EPSG:32651) while all eight in fact carried WGS 84 coordinates. Reprojecting
the correctly-labelled subset therefore split the stack by roughly 168 m -- a
failure that produces no error and no visible artefact.

`crs_offset_scan.py` diagnoses this without assuming the answer: it searches a
free 2-D offset over +/- 288 m against ground measurements and reports the
optimum. On this dataset it recovers +168.0 m E, -48.0 m N, within 1.2 m of the
independently computed PRS 1992 to WGS 84 datum shift at the study centroid.
The technique transfers to any raster stack suspected of a datum mislabelling.

## Citation

Cite both the paper and this archive. See `CITATION.cff`.

## Licence

MIT for the code. The input data are under the terms described in
`data/README.md` and are not covered by this licence.
