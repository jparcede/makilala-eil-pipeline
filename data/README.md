# Input data

The code in this repository expects a directory of input rasters. Point the
environment variable `MAKILALA_DIR` at it, or place the files in this `data/`
directory:

```bash
export MAKILALA_DIR=/path/to/rasters
```

## What is not distributed here, and why

Two of the inputs are not ours to redistribute:

| Input | Owner | How to obtain |
|---|---|---|
| Co-seismic landslide inventory (2019 Cotabato-Davao del Sur sequence) | DOST-PHIVOLCS | On reasonable request to PHIVOLCS |
| REDAS-modelled peak ground acceleration rasters | DOST-PHIVOLCS | On reasonable request to PHIVOLCS |
| Geotechnical field measurements (37 sites) and the interpolated parameter rasters derived from them | ML-PREP Project, Caraga State University | On reasonable request to the corresponding author |

## What is public

| Input | Source |
|---|---|
| Soil depth | SoilGrids 2.0, doi:10.5194/soil-7-217-2021 |
| Terrain / slope | Derived from a 5 m IfSAR DEM (NAMRIA) |

## Expected files

```
Mak_BulkUnitWeight.tif     bulk unit weight, kN/m^3
Mak_Fines.tif              fines content, %
Mak_PlasticityIndex.tif    plasticity index
Mak_WaterContent.tif       gravimetric water content, %
Mak_Sand.tif               sand content, %
Mak_Gravel.tif             gravel content, %
Mak_LiquidLimit.tif        liquid limit
Mak_PlasticLimit.tif       plastic limit
Slope_PRSTM51N.tif         slope angle, degrees
Soil Depth.tif             soil depth, m
PGA 2019 EQ Events.tif     peak ground acceleration, g
LSI_North Cotabato.shp     landslide inventory polygons
```

## A note on coordinate reference systems

Four of the interpolated rasters were originally written with an incorrect CRS
label (ESRI:102457, PRS 1992 UTM 51N) while their coordinates are in fact
WGS 84 UTM 51N (EPSG:32651). The pipeline treats `EPSG:32651` as the grid CRS
throughout. `src/crs/` contains the diagnostics that established this; see
`docs/crs_root_cause.md`. If you supply your own rasters, verify their labels
before use -- the failure is silent and shifts affected layers by about 168 m.
