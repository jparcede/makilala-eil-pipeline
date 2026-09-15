# Root cause of the Makilala CRS problem

## Finding

The eight EBK rasters carry **contradictory CRS labels on a single identical grid**.
All eight share the same geotransform — origin 719879.943 / 774735.143, 5709 x 3697
at 5 m — but four declare PRS 1992 / UTM 51N and four declare WGS 84 / UTM 51N.

| Layer | Declared CRS | Correct? |
|---|---|---|
| Mak_BulkUnitWeight.tif | ESRI:102457 (PRS 1992) | **no** |
| Mak_LiquidLimit.tif | ESRI:102457 (PRS 1992) | **no** |
| Mak_PlasticLimit.tif | ESRI:102457 (PRS 1992) | **no** |
| Mak_PlasticityIndex.tif | ESRI:102457 (PRS 1992) | **no** |
| Mak_Fines.tif | EPSG:32651 (WGS 84) | yes |
| Mak_Gravel.tif | EPSG:32651 (WGS 84) | yes |
| Mak_Sand.tif | EPSG:32651 (WGS 84) | yes |
| Mak_WaterContent.tif | EPSG:32651 (WGS 84) | yes |

A single physical location cannot have identical easting and northing in two datums
separated by 168 m, so at least one set of labels had to be wrong.

## Which set is wrong

Test: the EBK surfaces were interpolated *from* the 37 field samples, so at each
sample's own GPS position the surface must reproduce that sample's laboratory value.
A free 2-D search (+/-240 m, 12 m steps) was run for the offset minimising that
residual — separately for each label group.

```
datum shift PRS92 -> WGS84 (computed):   dx +168.0 m   dy  -46.8 m

group labelled PRS92:   optimum  dx +168.0   dy -48.0   1.2 m from the datum shift
group labelled WGS84:   optimum  dx +168.0   dy -48.0   1.2 m from the datum shift
```

Both groups land on the same optimum, and it is the datum shift. The coordinates are
**WGS 84** throughout; the four PRS 1992 labels are wrong. This is consistent with the
identical geotransforms: the layers were always one grid, and always the same grid.

## What the reprojected set did

Reprojecting from the file metadata was the correct procedure given what the metadata
said. The tool moved the four layers whose labels were accurate and left untouched the
four that already claimed PRS 1992:

```
Mak_Fines / Gravel / Sand / WaterContent   ->  origin moved -168.355 / +47.341  (true reprojection)
Mak_BulkUnitWeight / LiquidLimits /
  PlasticLimit / PlasticityIndex           ->  origin unchanged                 (already "PRS 1992")
```

The consequence is that the eight EBK layers, previously pixel-aligned on one grid,
are now split into two groups 168 m apart. This is a property of the contradictory
input metadata, not of the reprojection work.

Two things in the reprojected set are improvements and should be kept:

- `Mak_SoilDepth_PRS92.tif` is clipped to the Makilala window (126 x 81 at 229 m)
  rather than the full province — the correct clip.
- `Mak_Slope_PRS92.tif` is unchanged from the original, correctly, since the slope
  raster is genuinely PRS 1992. Both analyses agree on this.

## Minimal fix

Relabel the four mislabelled files to EPSG:32651 — Define Projection, not Project,
because the coordinates are already correct and must not be moved:

```
Mak_BulkUnitWeight.tif
Mak_LiquidLimit.tif
Mak_PlasticLimit.tif
Mak_PlasticityIndex.tif
```

After that all eight agree, and the stack can either be used as-is in EPSG:32651 or
reprojected as a whole to PRS 1992 in one consistent operation.

## Effect on the manuscript

None. The v9 pipeline sets `GRID_CRS = EPSG:32651` and treats the whole EBK stack as
sitting on that grid, which is what the evidence above says it does. The registered
results in the paper stand.

## Where the inventory error came from

`LSI_North Cotabato.shp` arrived from PHIVOLCS already labelled PRS 1992 and was not
re-projected after receipt. Its coordinates match the WGS 84 grid, so the mislabelling
is upstream of this project and would affect any other work using that shapefile. Worth
raising with PHIVOLCS.

## Reproducing this

`artifacts/crs_offset_scan.py` and `artifacts/crs_decisive_test.py` run against the
original files and the field spreadsheet. Neither uses any raster as a reference for
another; both compare interpolated surfaces against the laboratory measurements they
were built from.
