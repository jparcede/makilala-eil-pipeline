# Closed-form earthquake-induced landslide hazard assessment

Analysis code for the paper *Closed-Form Earthquake-Induced Landslide Hazard
Assessment for Makilala, North Cotabato: Regional Application and Validation of
the Analytical Factor of Safety Framework*.

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

There is no training step. One number is calibrated from the inventory -- the
operating threshold `theta*` -- and everything else comes from soil properties,
topography and modelled ground motion. To run the framework somewhere else,
substitute that municipality's rasters.

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
notebooks/        v9  -- the pipeline as the paper was written against
                  v10 -- the same model, rebuilt so that every published number
                  is recomputed and checked against its published value
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

Every script resolves its input directory from `MAKILALA_DIR`, falling back to
`./data`. See `data/README.md` for the expected files, and for which inputs are
restricted and how to request them.

## Verification

`notebooks/Makilala_FoS_Pipeline_v10.ipynb` is the version to run. It computes
the same model as v9 and additionally checks **81 quantities** against the values
printed in the paper -- the domain cascade, all twelve cohesion x depth-cap
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
