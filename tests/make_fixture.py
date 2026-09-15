"""Synthetic fixture mirroring the real data's STRUCTURE (not its values).

Purpose: exercise every code path in v10 end to end -- registration with the
pass-through rule, the inventory relabel, the cascade, the twelve-config sweep,
blocked validation, the variance decomposition, the map writes and the figures.

The PASS/FAIL checks are EXPECTED to fail here; the data is invented. What this
test asserts is that nothing raises.
"""
import os, shutil, numpy as np, rasterio, geopandas as gpd
from rasterio.transform import from_origin
from rasterio.crs import CRS
from shapely.geometry import Polygon

OUT = os.environ.get("FIXTURE_DIR",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixture"))
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(os.path.join(OUT, "standard_error"), exist_ok=True)

H, W, RES = 1200, 1600, 5.0          # 6 km x 8 km -> 3 x 4 = 12 tiles of 2 km
X0, Y0 = 720000.0, 775000.0
TR = from_origin(X0, Y0, RES, RES)
rng = np.random.default_rng(20260915)

yy, xx = np.mgrid[0:H, 0:W]
ridge = np.exp(-((xx - W*0.62)**2)/(2*(W*0.17)**2))     # steep NE flank


def write(path, arr, crs, transform=TR, nodata=-9999.0):
    a = np.where(np.isfinite(arr), arr, nodata).astype('float32')
    with rasterio.open(path, 'w', driver='GTiff', height=a.shape[0], width=a.shape[1],
                       count=1, dtype='float32', crs=CRS.from_string(crs),
                       transform=transform, nodata=nodata, compress='lzw') as d:
        d.write(a, 1)


# ---- the Mak_* family: ONE shared grid, mutually inconsistent CRS tags,
#      exactly as in the real data. gamma is the reference grid.
hole = (xx > W*0.88) | (yy > H*0.93)          # a nodata margin, as in the real rasters

gamma = 11.0 + 2.8*ridge + rng.normal(0, .18, (H, W))
gamma[hole] = np.nan
write(f"{OUT}/Mak_BulkUnitWeight.tif", gamma, "ESRI:102457")

water = 36.0 + 9.0*ridge + rng.normal(0, .7, (H, W))   # spans one saturation band
water[hole] = np.nan
write(f"{OUT}/Mak_WaterContent.tif", water, "EPSG:32651")

pi = 8.0 + 42.0*ridge + rng.normal(0, 3.0, (H, W)); pi[hole] = np.nan
write(f"{OUT}/Mak_PlasticityIndex.tif", pi, "ESRI:102457")

fines = 3.0 + 17.0*ridge + rng.normal(0, 2.0, (H, W)); fines[hole] = np.nan
write(f"{OUT}/Mak_Fines.tif", np.clip(fines, 0, 100), "EPSG:32651")

sand = 90.0 - 25.0*ridge + rng.normal(0, 3.0, (H, W)); sand[hole] = np.nan
write(f"{OUT}/Mak_Sand.tif", np.clip(sand, 0, 100), "EPSG:32651")

# ---- external layers on their OWN grid, so they must be reprojected --------
H2, W2, RES2 = 620, 820, 9.7
TR2 = from_origin(X0 - 400, Y0 + 400, RES2, RES2)
y2, x2 = np.mgrid[0:H2, 0:W2]
ridge2 = np.exp(-((x2 - W2*0.62)**2)/(2*(W2*0.17)**2))

slope = 3.0 + 46.0*ridge2 + rng.normal(0, 2.5, (H2, W2))
write(f"{OUT}/Slope_PRSTM51N.tif", np.clip(slope, 0, 80), "ESRI:102457", TR2)

depth = 1.0 + 18.0*(1 - ridge2) + rng.normal(0, 1.2, (H2, W2))
depth = np.clip(depth, 0.05, 60)
depth[rng.random((H2, W2)) < 0.004] = np.nan        # sparse gaps
write(f"{OUT}/Soil Depth.tif", depth, "ESRI:102457", TR2)

pga = 0.42 - 0.22*ridge2 + rng.normal(0, .012, (H2, W2))   # anti-correlated
write(f"{OUT}/PGA 2019 EQ Events.tif", np.clip(pga, .05, .9), "ESRI:102457", TR2)

# ---- standard errors: different CRS AND extent, as in the real ones -------
H3, W3 = 150, 200
TR3 = from_origin(124.90, 7.00, 0.0009, 0.0009)
for fn, lo, hi in [("buk_v1_standarderror.tif", .05, .35),
                   ("watercontent_v11standarderror.tif", .4, 2.2),
                   ("siltclay_v1_standarderror.tif", .5, 3.0),
                   ("PI_v1_standarderror.tif", 1.0, 6.0)]:
    write(f"{OUT}/standard_error/{fn}", rng.uniform(lo, hi, (H3, W3)), "EPSG:4326", TR3)

# ---- inventory: polygons on the steep flank, spread across tiles ----------
polys, esi = [], []
for i in range(320):
    t = rng.random()
    cx = X0 + RES*(W*0.62 + rng.normal(0, W*0.13))
    cy = Y0 - RES*(rng.uniform(0.04, 0.92)*H)
    r = 18 + 120*rng.random()**1.7
    k = rng.integers(5, 9)
    ang = np.sort(rng.uniform(0, 2*np.pi, k))
    pts = [(cx + r*(0.6+0.8*rng.random())*np.cos(a),
            cy + r*(0.6+0.8*rng.random())*np.sin(a)) for a in ang]
    polys.append(Polygon(pts))
    esi.append(int(rng.integers(6, 11)))

# plus a handful far outside, to exercise the "outside raster extent" branch
for i in range(40):
    cx = X0 + RES*W*2.5 + rng.normal(0, 400)
    cy = Y0 - RES*H*2.5 + rng.normal(0, 400)
    polys.append(Polygon([(cx, cy), (cx+60, cy), (cx+60, cy+60), (cx, cy+60)]))
    esi.append(int(rng.integers(6, 11)))

gdf = gpd.GeoDataFrame({'ESI_local': esi}, geometry=polys,
                       crs=CRS.from_string("ESRI:102457"))
gdf.to_file(f"{OUT}/LSI_North Cotabato.shp")

print(f"fixture written to {OUT}")
print(f"  reference grid {H} x {W} = {H*W:,} cells at {RES:g} m")
print(f"  external grid  {H2} x {W2} at {RES2:g} m, different CRS -> must reproject")
print(f"  polygons       {len(gdf)} ({len(gdf)-40} inside, 40 far outside)")
print(f"  water content  {np.nanmin(water):.2f} - {np.nanmax(water):.2f} %"
      f"  (single band -> m == 0.6 expected)")
