"""
CRS OFFSET SCAN -- no hypotheses, no fitting to another raster.

Rather than compare two candidate CRSs, this asks an open question: at what
spatial offset does the EBK surface best reproduce the laboratory measurements it
was interpolated from?

Procedure. Take each field sample's GPS coordinate (WGS 84, ground truth).
Transform it into the raster's DECLARED CRS, PRS 1992 / UTM 51N. Then displace by
(dx, dy) over a grid, sample the surface, and compute RMSE against the laboratory
value. Whichever (dx, dy) minimises RMSE is where the data actually sit relative
to the declared georeference.

  - If the declaration is correct, the minimum lies at (0, 0).
  - If the files carry a PRS 1992 label over WGS 84 coordinates, the minimum lies
    at the PRS92 -> WGS84 datum shift for this location, which is computed below
    from the projection definitions rather than assumed.

RMSE is pooled across five parameters after standardising each by its own spread,
so no single noisy field dominates.
"""
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform

# --- data directory resolution -----------------------------------------------
# This script lives in <data>/artifacts/, so the data directory is its parent's
# parent. Falls back to MAKILALA_DIR, then to common locations, so it runs
# unchanged on a laptop, in a sandbox, or from any working directory.
def _find_data_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.path.dirname(here),
             os.environ.get('MAKILALA_DIR', ''),
             os.path.expanduser('~/Downloads/makilala_data'),
             os.path.expanduser('~/mnt/makilala_data'),
             os.getcwd()]
    for c in cands:
        if c and os.path.exists(os.path.join(c, 'Mak_BulkUnitWeight.tif')):
            return c
    msg = ['Could not locate makilala_data (no Mak_BulkUnitWeight.tif found).',
           'Tried:']
    msg += ['  ' + x for x in cands if x]
    msg += ['Set it explicitly:  MAKILALA_DIR=/path/to/makilala_data python '
            + os.path.basename(__file__)]
    raise SystemExit(chr(10).join(msg))


IN = _find_data_dir()
print('data directory: ' + IN)
OUT = os.path.join(IN, "artifacts", "crs_offset_scan.txt")

PAIRS = [
    ("Mak_BulkUnitWeight.tif",  "BULK UNIT WEIGHT (kN/m3)"),
    ("Mak_WaterContent.tif",    "WATER CONTENT (%)"),
    ("Mak_PlasticityIndex.tif", "PI"),
    ("Mak_Sand.tif",            "Sand %"),
    ("Mak_Fines.tif",           "Silt/Clay %"),
]

raw = pd.read_excel(os.path.join(IN, "Makilala Geotechnical Data.xlsx"),
                    sheet_name="GEOTECH MAPUA", header=None)
d = raw.iloc[2:].copy()
d.columns = [str(x).strip() for x in raw.iloc[1].tolist()]
for c in ("LATITUDE", "LONGITUDE"):
    d[c] = pd.to_numeric(d[c], errors="coerce")
d = d.dropna(subset=["LATITUDE", "LONGITUDE"]).reset_index(drop=True)
lon = d["LONGITUDE"].to_numpy()
lat = d["LATITUDE"].to_numpy()

# the datum shift, computed rather than assumed
x_prs, y_prs = warp_transform("EPSG:4326", "ESRI:102457", list(lon), list(lat))
x_wgs, y_wgs = warp_transform("EPSG:4326", "EPSG:32651", list(lon), list(lat))
dx_true = float(np.mean(np.array(x_wgs) - np.array(x_prs)))
dy_true = float(np.mean(np.array(y_wgs) - np.array(y_prs)))

# preload rasters once
layers = []
for tif, col in PAIRS:
    meas = pd.to_numeric(d[col], errors="coerce").to_numpy()
    with rasterio.open(os.path.join(IN, tif)) as s:
        arr = s.read(1).astype(float)
        if s.nodata is not None:
            arr[arr == s.nodata] = np.nan
        layers.append((arr, s.transform, s.height, s.width, meas, np.nanstd(meas)))

STEP, HALF = 12.0, 288.0
grid = np.arange(-HALF, HALF + STEP, STEP)


def pooled_rmse(dx, dy):
    zs = []
    for arr, T, H, W, meas, sd in layers:
        vals = np.full(len(lon), np.nan)
        for i, (x, y) in enumerate(zip(x_prs, y_prs)):
            c, r = ~T * (x + dx, y + dy)
            r, c = int(r), int(c)
            if 0 <= r < H and 0 <= c < W:
                vals[i] = arr[r, c]
        m = np.isfinite(vals) & np.isfinite(meas)
        if m.sum() < 5:
            continue
        zs.append(((vals[m] - meas[m]) / sd) ** 2)
    return float(np.sqrt(np.mean(np.concatenate(zs)))) if zs else np.inf


best, bxy = np.inf, (0, 0)
surf = np.full((len(grid), len(grid)), np.nan)
for i, dy in enumerate(grid):
    for j, dx in enumerate(grid):
        v = pooled_rmse(dx, dy)
        surf[i, j] = v
        if v < best:
            best, bxy = v, (dx, dy)

L = []
L.append("=" * 78)
L.append("CRS OFFSET SCAN -- where does the EBK surface best fit its own source data?")
L.append("=" * 78)
L.append("")
L.append(f"field samples            {len(lon)}")
L.append(f"parameters pooled        {len(layers)}")
L.append(f"scan grid                +/-{HALF:.0f} m in {STEP:.0f} m steps")
L.append("")
L.append("Datum shift computed from the projection definitions (not assumed):")
L.append(f"  PRS 1992 -> WGS 84 at this location:  dx {dx_true:+8.1f} m   dy {dy_true:+8.1f} m")
L.append("")
L.append("Best-fitting offset found by the scan:")
L.append(f"  argmin RMSE:                          dx {bxy[0]:+8.1f} m   dy {bxy[1]:+8.1f} m")
L.append(f"  pooled standardised RMSE at optimum:  {best:.4f}")
L.append(f"  pooled standardised RMSE at (0,0):    {pooled_rmse(0,0):.4f}")
L.append(f"  pooled standardised RMSE at datum shift: {pooled_rmse(dx_true, dy_true):.4f}")
L.append("")
resid = float(np.hypot(bxy[0] - dx_true, bxy[1] - dy_true))
L.append(f"  distance from scan optimum to predicted datum shift: {resid:.1f} m")
L.append(f"  distance from scan optimum to zero offset:           {np.hypot(*bxy):.1f} m")
L.append("")
L.append("=" * 78)
if resid < np.hypot(*bxy):
    L.append("VERDICT: the optimum sits at the DATUM SHIFT, not at zero.")
    L.append("The rasters carry a PRS 1992 label over WGS 84 coordinates.")
else:
    L.append("VERDICT: the optimum sits at ZERO offset.")
    L.append("The PRS 1992 declaration is correct and no correction is needed.")
L.append("=" * 78)
L.append("")
L.append("Note: the scan grid is a free two-dimensional search. It was given no")
L.append("knowledge of the datum shift; that value is printed only for comparison.")

txt = "\n".join(L)
print(txt)
open(OUT, "w").write(txt + "\n")
print(f"\nwrote {OUT}")
