import os
"""
build_v9.py -- patch Makilala_FoS_Pipeline_v8.ipynb into v9.

v9 changes exactly one thing about the science: how rasters are brought onto the
reference grid. Everything downstream is untouched.

WHY
  v8 §3 aligned every input with scipy.ndimage.zoom, a pure array rescale that
  ignores CRS and bounds. Its markdown asserted this was safe "because every
  raster here already shares the reference CRS and extent". Three of the eight
  do not:

      Slope_PRSTM51N.tif      3716x5731 @ 5 m,   28.7 x 18.6 km, origin offset
      Soil Depth.tif           444x461  @ 229 m, 105.6 x 102.7 km
      PGA 2019 EQ Events.tif   702x701  @ 555 m, 389.0 x 389.6 km

  zoom() stretched a 105.6 km raster and a 389 km raster onto a 28.5 km window.
  Registered properly, depth correlates -0.245 with its zoomed self and PGA
  0.020; mean PGA over the municipality is 0.302 g, not 0.038 g.

  Separately, the reference grid and the landslide inventory both carry WGS 84 /
  UTM 51N coordinates under a .prj that declares PRS 1992 / UTM 51N. Four
  independent tests agree, each matching the PRS92 -> WGS84 datum shift at the
  study centroid (+167.9 m E, -46.8 m N) with no free parameters:

    1. free 2-D offset fit, slope vs inventory        -165 m E, +40 m N
    2. datum transform recovers that fit              contrast 20.06 vs 20.12
    3. field GPS to nearest polygon        median 67.5 m -> 21.9 m corrected
    4. free 2-D offset fit, EBK layers vs 38 GPS sites  +170 m E, -45 m N

  Because the grid and the inventory carry the SAME error they agree with each
  other, so the polygon-to-pixel geometry was never wrong. Only the three
  external layers are displaced.

HOW TO USE
  Set GRID_CRS / LSI_CRS in §0 below. They are the only lines that change if
  J.M.B. Segales reports different provenance. Nothing else in the notebook
  refers to a CRS.

    python build_v9.py            # writes Makilala_FoS_Pipeline_v9.ipynb
"""
import json, copy, sys, os

SRC = "Makilala_FoS_Pipeline_v8.ipynb"
DST = "Makilala_FoS_Pipeline_v9.ipynb"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


# ---------------------------------------------------------------- §0 config
CELL4 = '''INPUT_DIR = os.environ.get("MAKILALA_DIR", "data")
OUTPUT_DIR = os.path.join(INPUT_DIR, "output_maps_v9")

FILE_SLOPE = "Slope_PRSTM51N.tif"; FILE_DEPTH = "Soil Depth.tif"
FILE_PGA = "PGA 2019 EQ Events.tif"; FILE_GAMMA = "Mak_BulkUnitWeight.tif"
FILE_WATER = "Mak_WaterContent.tif"; FILE_PI = "Mak_PlasticityIndex.tif"
FILE_FINES = "Mak_Fines.tif"; FILE_SAND = "Mak_Sand.tif"
FILE_LSI = "LSI_North Cotabato.shp"

REFERENCE_GRID = 'gamma'
ALPHA_MIN = 10       # Minimum slope angle for failure consideration (degrees)
GAMMA_W = 9.81; G_ACC = 9.81; NODATA = -9999.0

# ============================================================================
#  COORDINATE REFERENCE SYSTEMS
#
#  These are the only lines in the notebook that name a CRS. If J.M.B. Segales
#  reports different provenance for the EBK export or the inventory, change
#  them here and rerun; nothing downstream needs editing.
#
#  GRID_CRS is what the Mak_*.tif rasters ACTUALLY are, which is not what all
#  of them say. Mak_Fines, Mak_Sand and Mak_WaterContent are tagged EPSG:32651;
#  Mak_BulkUnitWeight and Mak_PlasticityIndex are tagged ESRI:102457. All five
#  share a byte-identical grid, so at most one tag can be right. Sampling all
#  five at the 38 GPS sites with complete laboratory determinations puts the
#  best fit at +170 m E, -45 m N from the ESRI:102457 reading -- against a
#  PRS92 -> WGS84 datum shift of +167.9 m E, -46.8 m N at the study centroid.
#  The EPSG:32651 tag is the correct one.
#
#  LSI_CRS is what LSI_North Cotabato.shp ACTUALLY is. Its .prj declares
#  ESRI:102457. Under that reading the 25 field sites recorded AT landslide
#  sites sit a median 67.5 m from the nearest polygon; read as EPSG:32651 the
#  median is 21.9 m, and sites within 50 m go from 8/25 to 18/25. Handheld GPS
#  reports WGS 84 by construction, so the GPS is the ground truth here.
#
#  The external layers keep whatever their own files declare, unless overridden
#  below. Slope is PRS 1992 as labelled: it is the layer the other two disagree
#  with, by exactly one datum shift.
# ============================================================================
GRID_CRS = "EPSG:32651"      # WGS 84 / UTM 51N -- the Mak_*.tif reference grid
LSI_CRS = "EPSG:32651"      # WGS 84 / UTM 51N -- the inventory shapefile

# Per-layer overrides for the external rasters. None = trust the file's own CRS.
CRS_OVERRIDE = {
    'slope': None,           # ESRI:102457, verified correct against terrain
    'depth': None,           # SoilGrids -> reprojected by a third party; unverified
    'pga': None,           # PHIVOLCS REDAS; unverified
}

# Zero-depth handling. The SoilGrids raster carries 286 literal zeros which the
# resample spreads across neighbouring pixels. Because z_base is clipped to a
# 0.1 m floor before the validity test, those pixels enter the scored domain at
# 10 cm of soil and are predicted stable almost by construction. Under v8's
# zoom this affected 3,688 px; under v9's reprojection the count differs and is
# reported by §3. Left False for continuity with v8.
EXCLUDE_ZERO_DEPTH = False

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Input:  {INPUT_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"Grid CRS: {GRID_CRS}   Inventory CRS: {LSI_CRS}")
print(f"Min slope for failure: {ALPHA_MIN}°")
'''

# ---------------------------------------------------------------- §3 markdown
CELL7 = '''## §3 Rasters: load and register

Alignment is by `rasterio.warp.reproject`, which maps each source pixel through
its own coordinate system onto the reference grid. v8 used `scipy.ndimage.zoom`,
a pure array rescale that ignores CRS and bounds, on the stated assumption that
every raster already shared the reference extent. Three do not:

| layer | grid | cell | extent |
|---|---|---|---|
| `Slope_PRSTM51N` | 3716 × 5731 | 5 m | 28.7 × 18.6 km, origin offset |
| `Soil Depth` | 444 × 461 | 229 m | 105.6 × 102.7 km |
| `PGA 2019 EQ Events` | 702 × 701 | 555 m | 389.0 × 389.6 km |

The reference window is 28.5 × 18.5 km, so `zoom` squeezed a 105.6 km raster and
a 389 km raster onto it. Registered properly, depth correlates **−0.245** with its
zoomed self and PGA **0.020**; mean PGA over the municipality is **0.302 g**, not
0.038 g, against an instrumental 0.191 g recorded at KCPA 1 km from the municipal
centre during a *smaller* aftershock.

The five `Mak_*` rasters share the reference grid exactly and are passed through
untouched — reprojecting them would be a no-op at best and, because three of the
five are tagged `EPSG:32651` and two `ESRI:102457`, would silently apply a datum
shift to some and not others. `GRID_CRS` in §0 settles which tag is right; the
files themselves are never re-read for their CRS.

`gamma` defines the reference grid. This cell reports, for every layer, whether
it was passed through or reprojected and from what.
'''

# ---------------------------------------------------------------- §3 code
CELL8 = '''from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS


def load_raster(fp, label=""):
    with rasterio.open(fp) as src:
        d = src.read(1).astype(np.float64); p = src.profile.copy()
        if src.nodata is not None: d[d == src.nodata] = np.nan
        v = np.isfinite(d) & (d != 0)
        print(f"  ✓ {label}: {d.shape} @ {src.res[0]:g} {'m' if src.crs and src.crs.is_projected else 'deg'}"
              f", {src.crs}, valid={v.sum():,}, [{np.nanmin(d[v]):.3f}, {np.nanmax(d[v]):.3f}]")
        return d, p


rasters, profiles = {}, {}
for key, (fn, lb) in {'slope': (FILE_SLOPE,"α"), 'depth': (FILE_DEPTH,"z"),
    'pga': (FILE_PGA,"PGA"), 'gamma': (FILE_GAMMA,"γ"), 'water': (FILE_WATER,"w"),
    'pi': (FILE_PI,"PI"), 'fines': (FILE_FINES,"fines"), 'sand': (FILE_SAND,"sand")}.items():
    fp = os.path.join(INPUT_DIR, fn)
    if os.path.exists(fp):
        d, p = load_raster(fp, lb); rasters[key] = d; profiles[key] = p
    else:
        print(f"  ✗ {lb}: NOT FOUND")

ref_shape = rasters[REFERENCE_GRID].shape
ref_profile = profiles[REFERENCE_GRID]
ref_transform = ref_profile['transform']
ref_crs = CRS.from_string(GRID_CRS)          # §0, NOT the file's own tag
ref_profile = {**ref_profile, 'crs': ref_crs}

print(f"\\nReference grid: {ref_shape[0]}×{ref_shape[1]} @ {ref_transform.a:g} m")
print(f"  file says : {profiles[REFERENCE_GRID].get('crs')}")
print(f"  GRID_CRS  : {ref_crs}   <- used")
if str(profiles[REFERENCE_GRID].get('crs')) != str(ref_crs):
    print("  (the file's tag is overridden; see the CRS note in §0)")

# ---------------------------------------------------------------------------
#  Registration. Layers already on the reference grid pass through untouched --
#  never reproject them, because the Mak_* tags are mutually inconsistent and a
#  reprojection would shift some by a datum and leave others alone.
# ---------------------------------------------------------------------------
aligned = {}
for key, data in rasters.items():
    same_grid = (data.shape == ref_shape
                 and profiles[key]['transform'].almost_equals(ref_transform, precision=1e-6))
    if same_grid:
        aligned[key] = data
        print(f"  = {key:<6}: on the reference grid, passed through")
        continue

    src_crs = profiles[key]['crs']
    override = CRS_OVERRIDE.get(key)
    if override is not None:
        src_crs = CRS.from_string(override)
    if src_crs is None:
        raise RuntimeError(f"{key}: no CRS on the file and no CRS_OVERRIDE entry; "
                           f"cannot register it. Set CRS_OVERRIDE['{key}'] in §0.")

    dst = np.full(ref_shape, np.nan, dtype=np.float64)
    reproject(source=data, destination=dst,
              src_transform=profiles[key]['transform'], src_crs=src_crs,
              dst_transform=ref_transform, dst_crs=ref_crs,
              resampling=Resampling.bilinear,
              src_nodata=np.nan, dst_nodata=np.nan)
    aligned[key] = dst
    tag = f" (CRS_OVERRIDE -> {override})" if override else ""
    cov_before = np.isfinite(data).mean(); cov_after = np.isfinite(dst).mean()
    print(f"  ↳ {key:<6}: {str(data.shape):>13} {str(src_crs):<13}{tag} → reprojected, "
          f"coverage {100*cov_after:.1f}% of the grid")

print("\\n✓ All layers on the reference grid")

# ---------------------------------------------------------------------------
#  Registration sanity check. Landslides occur on steep ground, so a correctly
#  registered slope raster must show a large mean-slope contrast at inventory
#  pixels. This is a smoke test, not a result: it runs before `observed` exists,
#  so it rasterises the inventory itself, cheaply and centre-only.
# ---------------------------------------------------------------------------
if gpd is not None and os.path.exists(os.path.join(INPUT_DIR, FILE_LSI)):
    from rasterio.features import rasterize as _rz
    from shapely.geometry import mapping as _mp
    _g = gpd.read_file(os.path.join(INPUT_DIR, FILE_LSI))
    _g = _g.set_crs(LSI_CRS, allow_override=True).to_crs(ref_crs)
    _o = _rz([(_mp(x), 1) for x in _g.geometry if x is not None], out_shape=ref_shape,
             transform=ref_transform, fill=0, dtype=np.uint8, all_touched=False)
    _s = aligned['slope']; _m = np.isfinite(_s)
    _ls, _bg = _s[_m & (_o == 1)].mean(), _s[_m & (_o == 0)].mean()
    print(f"\\nRegistration check: mean slope at landslide pixels {_ls:.2f}° "
          f"vs background {_bg:.2f}°, contrast {_ls-_bg:.2f}°")
    if _ls - _bg < 18.0:
        print("  ⚠ contrast below 18° — expected ~20° when slope, grid and inventory")
        print("    are mutually registered. Check GRID_CRS / LSI_CRS in §0.")
    else:
        print("  ✓ consistent with correct registration")
    del _g, _o, _s, _m

# ---------------------------------------------------------------------------
#  Second check, for GRID_CRS specifically. The one above tests slope against
#  the inventory, which is insensitive to GRID_CRS: if both are read in the same
#  frame they agree whichever frame that is. This one pins the grid itself.
#  EBK produces a conditional-mean surface, so the interpolated value at a
#  sample site should reproduce that site's measured value. Sampling all five
#  layers at the GPS sites, the correct GRID_CRS gives the lower mismatch.
# ---------------------------------------------------------------------------
_xl = os.path.join(INPUT_DIR, "Makilala Geotechnical Data.xlsx")
if os.path.exists(_xl):
    import pandas as _pd
    from shapely.geometry import Point as _Pt
    _raw = _pd.read_excel(_xl, sheet_name='GEOTECH MAPUA', header=None)
    _d = _raw.iloc[2:].copy()
    _d.columns = [str(x).strip() for x in _raw.iloc[1].tolist()]
    _pairs = [('sand', 'Sand %'), ('fines', 'Silt/Clay %'), ('pi', 'PI'),
              ('water', 'WATER CONTENT (%)'), ('gamma', 'BULK UNIT WEIGHT (kN/m3)')]
    for _c in ['LATITUDE', 'LONGITUDE'] + [b for _, b in _pairs]:
        _d[_c] = _pd.to_numeric(_d[_c], errors='coerce')
    _d = _d.dropna(subset=['LATITUDE', 'LONGITUDE'] + [b for _, b in _pairs])
    _p = gpd.GeoDataFrame(_d, geometry=[_Pt(x, y) for x, y in
                          zip(_d['LONGITUDE'], _d['LATITUDE'])],
                          crs='EPSG:4326').to_crs(ref_crs)
    _tot, _n = 0.0, 0
    for _k, _col in _pairs:
        _e = []
        for _pt, _meas in zip(_p.geometry, _d[_col]):
            _c2, _r2 = ~ref_transform * (_pt.x, _pt.y)
            _r2, _c2 = int(_r2), int(_c2)
            if 0 <= _r2 < ref_shape[0] and 0 <= _c2 < ref_shape[1]:
                _val = aligned[_k][_r2, _c2]
                if np.isfinite(_val):
                    _e.append(abs(_val - _meas))
        if _e:
            _tot += np.mean(_e) / max(abs(_d[_col].mean()), 1e-9); _n += 1
    print(f"\\nGrid check: interpolated vs measured at {len(_d)} GPS sites, "
          f"normalised error {_tot:.3f} over {_n} layers")
    if _tot > 0.65:
        print("  ⚠ above 0.65 — the EBK layers do not reproduce their own sample")
        print("    values. GRID_CRS is probably wrong; ~0.56 is expected when right.")
    else:
        print("  ✓ EBK layers reproduce their sample values; GRID_CRS consistent")
    del _raw, _d, _p
else:
    print("\\nGrid check skipped: Makilala Geotechnical Data.xlsx not found")
'''

INV_PATCH_OLD = "lsi = gpd.read_file(lsi_path) if gpd and os.path.exists(lsi_path) else None"
INV_PATCH_NEW = '''lsi = gpd.read_file(lsi_path) if gpd and os.path.exists(lsi_path) else None

# The shapefile's .prj declares ESRI:102457 but its coordinates are WGS 84 /
# UTM 51N; see the CRS note in §0. set_crs() relabels without moving anything,
# to_crs() then puts it on whatever GRID_CRS says the reference grid is. When
# LSI_CRS == GRID_CRS this is a no-op and the geometry is untouched.
if lsi is not None:
    lsi = lsi.set_crs(LSI_CRS, allow_override=True).to_crs(ref_crs)
    print(f"Inventory read as {LSI_CRS}, placed on the reference grid ({ref_crs})")'''


def main():
    if not os.path.exists(SRC):
        sys.exit(f"{SRC} not found; run from the directory that holds it.")
    nb = json.load(open(SRC))
    cells = copy.deepcopy(nb["cells"])

    # cell 0 -- retitle and stop asserting superseded counts
    head = "".join(cells[0]["source"])
    head = head.replace("# Makilala FoS Pipeline — v8", "# Makilala FoS Pipeline — v9")
    head = head.replace(
        "| scored domain | 3,870,885 px = 96.8 of 340.2 km² |",
        "| scored domain | **re-measured by this run** — v8 reported 3,870,885 px, "
        "but that figure came from a misregistered soil-depth raster |")
    head = head.replace(
        "Everything reported in the manuscript comes from **one** configuration, pinned\nin Part I §7:",
        "Everything reported in the manuscript comes from **one** configuration, pinned\n"
        "in Part I §7. **All numeric results below are superseded by this run:** v9\n"
        "corrects the raster registration in §3, which changes every quantity that\n"
        "depends on slope, soil depth or ground motion.\n")
    cells[0]["source"] = head.splitlines(keepends=True)

    cells[4] = code(CELL4)
    cells[7] = md(CELL7)
    cells[8] = code(CELL8)

    src12 = "".join(cells[12]["source"])
    assert INV_PATCH_OLD in src12, "cell 12 does not match the expected v8 text"
    cells[12]["source"] = src12.replace(INV_PATCH_OLD, INV_PATCH_NEW, 1).splitlines(keepends=True)

    nb["cells"] = cells
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []; c["execution_count"] = None
    json.dump(nb, open(DST, "w"), indent=1)
    print(f"wrote {DST}: {len(cells)} cells")
    print("  §0  cell 4  -> GRID_CRS / LSI_CRS / CRS_OVERRIDE")
    print("  §3  cell 7  -> markdown rewritten")
    print("  §3  cell 8  -> reproject, pass-through guard, registration smoke test")
    print("      cell 12 -> inventory relabelled via LSI_CRS")


if __name__ == "__main__":
    main()
