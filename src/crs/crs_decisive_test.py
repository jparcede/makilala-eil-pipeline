"""
DECISIVE CRS TEST -- uses the laboratory measurements as ground truth.

The question: are the Mak_*.tif EBK surfaces genuinely in PRS 1992 / UTM 51N
(ESRI:102457), as their headers say and as the pipeline intended, or are they in
WGS 84 / UTM 51N (EPSG:32651) carrying a PRS 1992 label?

Every previous test rested on comparing one raster against another, so a sceptic
could argue the reference was the thing at fault. This one does not. The EBK
surfaces were INTERPOLATED FROM the 37 field samples. At a sample's own location
the kriged surface must reproduce that sample's measured value closely -- that is
what interpolation means. So:

  - take each sample's GPS coordinate (WGS 84 lat/lon from the field spreadsheet,
    ground truth, independent of every raster header)
  - sample the EBK surface at that point under BOTH CRS interpretations
  - compare against the LABORATORY MEASURED value for that sample

The interpretation that reproduces the laboratory measurements is the correct one.
The wrong one samples a point displaced by the PRS92/WGS84 datum shift (~168 m at
this latitude) and will show visibly larger residuals.

This test is symmetric, uses no offset fitting, and can be re-run by anyone.
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
OUT = os.path.join(IN, "artifacts", "crs_decisive_test.txt")

PAIRS = [
    ("Mak_BulkUnitWeight.tif", "BULK UNIT WEIGHT (kN/m3)", "unit weight (kN/m3)"),
    ("Mak_WaterContent.tif",   "WATER CONTENT (%)",        "water content (%)"),
    ("Mak_PlasticityIndex.tif","PI",                       "plasticity index"),
    ("Mak_Sand.tif",           "Sand %",                   "sand (%)"),
    ("Mak_Fines.tif",          "Silt/Clay %",              "fines (%)"),
]

CANDIDATES = [
    ("PRS 1992 / UTM 51N  (as labelled)", "ESRI:102457"),
    ("WGS 84 / UTM 51N    (label wrong)", "EPSG:32651"),
]

raw = pd.read_excel(os.path.join(IN, "Makilala Geotechnical Data.xlsx"),
                    sheet_name="GEOTECH MAPUA", header=None)
d = raw.iloc[2:].copy()
d.columns = [str(x).strip() for x in raw.iloc[1].tolist()]
for c in ("LATITUDE", "LONGITUDE"):
    d[c] = pd.to_numeric(d[c], errors="coerce")
d = d.dropna(subset=["LATITUDE", "LONGITUDE"]).reset_index(drop=True)

L = []
L.append("=" * 78)
L.append("DECISIVE CRS TEST -- EBK surfaces against their own source measurements")
L.append("=" * 78)
L.append("")
L.append("At a field sample's own location an interpolated surface must reproduce")
L.append("that sample's measured value. Residuals are computed under both CRS")
L.append("interpretations; the smaller set identifies the true CRS.")
L.append("")

summary = {}
for tif, col, label in PAIRS:
    meas = pd.to_numeric(d[col], errors="coerce")
    ok = meas.notna()
    lon = d.loc[ok, "LONGITUDE"].to_numpy()
    lat = d.loc[ok, "LATITUDE"].to_numpy()
    y_meas = meas[ok].to_numpy()

    L.append(f"--- {label}   (n = {ok.sum()} samples with a laboratory value)")
    for name, crs in CANDIDATES:
        with rasterio.open(os.path.join(IN, tif)) as s:
            xs, ys = warp_transform("EPSG:4326", crs, list(lon), list(lat))
            vals = []
            for x, y in zip(xs, ys):
                col_, row_ = ~s.transform * (x, y)
                r, c = int(row_), int(col_)
                if 0 <= r < s.height and 0 <= c < s.width:
                    v = float(s.read(1, window=rasterio.windows.Window(c, r, 1, 1))[0, 0])
                    if s.nodata is not None and v == s.nodata:
                        v = np.nan
                else:
                    v = np.nan
                vals.append(v)
        vals = np.array(vals, float)
        m = np.isfinite(vals)
        if m.sum() == 0:
            L.append(f"    {name}: no valid samples"); continue
        res = vals[m] - y_meas[m]
        rmse = float(np.sqrt(np.mean(res ** 2)))
        mae = float(np.mean(np.abs(res)))
        rng = float(np.nanmax(y_meas) - np.nanmin(y_meas))
        L.append(f"    {name}:  n={m.sum():>2}  RMSE {rmse:8.3f}  MAE {mae:8.3f}"
                 f"  RMSE/range {100*rmse/rng:5.1f}%")
        summary.setdefault(label, {})[name] = rmse
    L.append("")

L.append("=" * 78)
L.append("VERDICT")
L.append("=" * 78)
wins = {n: 0 for n, _ in CANDIDATES}
for label, res in summary.items():
    if len(res) < 2:
        continue
    best = min(res, key=res.get)
    wins[best] += 1
    other = [k for k in res if k != best][0]
    ratio = res[other] / res[best] if res[best] > 0 else float("inf")
    L.append(f"  {label:<22} -> {best.strip()}  "
             f"({ratio:.2f}x lower RMSE than the alternative)")
L.append("")
for n, w in wins.items():
    L.append(f"  {n}: wins on {w} of {len(summary)} parameters")
L.append("")
L.append("If 'PRS 1992 (as labelled)' wins, the pipeline is correct as built and the")
L.append("registration correction should be withdrawn. If 'WGS 84 (label wrong)' wins,")
L.append("the files carry a PRS 1992 label over WGS 84 coordinates -- the signature of")
L.append("a Define-Projection step, or a Project step with no datum transformation,")
L.append("rather than a true reprojection.")

txt = "\n".join(L)
print(txt)
open(OUT, "w").write(txt + "\n")
print(f"\nwrote {OUT}")
