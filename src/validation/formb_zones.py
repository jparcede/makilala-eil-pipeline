"""Form B zone statistics quoted in Section 5.4 and the discussion."""
import os, gc
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

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
OUT = os.path.join(IN, "artifacts", "formb_zones.txt")
F = np.float32
GW = 9.81
ALPHA_MIN, Z_CAP, C_OFF = 10.0, 5.0, 10.0

with rasterio.open(os.path.join(IN, "Mak_BulkUnitWeight.tif")) as r:
    RT, REF = r.transform, (r.height, r.width)
CRS = "EPSG:32651"


def align(fn, assumed=None):
    with rasterio.open(os.path.join(IN, fn)) as s:
        a = s.read(1).astype(F)
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        if (s.height, s.width) == REF and s.transform == RT:
            return a
        out = np.full(REF, np.nan, F)
        reproject(a, out, src_transform=s.transform, src_crs=assumed or s.crs,
                  dst_transform=RT, dst_crs=CRS, resampling=Resampling.bilinear,
                  src_nodata=np.nan, dst_nodata=np.nan)
        return out


# Same defect as src/validation/fos_means.py, fixed 25 Nov 2026: passing CRS as
# `assumed` overrode the slope raster's true PRS 1992 declaration and
# reprojected it as though it were already EPSG:32651, reintroducing the ~168 m
# datum error diagnosed in Section 4.4. Let align() read the file's own CRS.
alpha = np.clip(align("Slope_PRSTM51N.tif"), 0.1, 80.).astype(F)
z_raw = align("Soil Depth.tif").astype(F)
gam = align("Mak_BulkUnitWeight.tif").astype(F)
fines = align("Mak_Fines.tif").astype(F)
sand = align("Mak_Sand.tif").astype(F)
pi = align("Mak_PlasticityIndex.tif").astype(F)
water = align("Mak_WaterContent.tif").astype(F)
gc.collect()

phi = np.full_like(pi, 30.)
v_ = (pi > 5) & np.isfinite(pi)
phi[v_] = 35. - 0.1 * pi[v_]
no = ~v_ & np.isfinite(sand)
phi[no & (sand > 85) & (fines < 10)] = 33.
phi[no & (sand > 70) & (fines >= 10) & (fines < 20)] = 31.
phi[no & (fines >= 20)] = 28.
tanphi = np.tan(np.radians(np.clip(phi, 20., 38.))).astype(F)

cb = np.full_like(fines, 5.)
vv = np.isfinite(fines) & np.isfinite(sand)
cb[vv & (fines < 5)] = 1.
cb[vv & (fines >= 5) & (fines < 12)] = 3.
cb[vv & (fines >= 12) & (fines < 20)] = 6.
cb[vv & (fines >= 20)] = 10.
c = (cb + C_OFF).astype(F)

m = np.full_like(water, 0.5)
w_ = np.isfinite(water) & (water > 0)
m[w_ & (water < 20)] = 0.2
m[w_ & (water >= 20) & (water < 35)] = 0.4
m[w_ & (water >= 35) & (water < 50)] = 0.6
m = m.astype(F)
del pi, fines, sand, water, cb, phi
gc.collect()

a_ = np.radians(alpha).astype(F)
ca, sa = np.cos(a_).astype(F), np.sin(a_).astype(F)
del a_
z = np.clip(z_raw, 0.1, Z_CAP).astype(F)
R = (c + (gam - m * GW) * z * ca * ca * tanphi).astype(F)
D_st = (gam * z * sa * ca).astype(F)

# Depth validity on the RAW raster, before the 0.1 m floor -- see the note in
# src/validation/fos_means.py. Clipping first admits SoilGrids' 286 zero-depth
# cells as 0.1 m columns; one survives into the scored domain.
valid = (np.isfinite(alpha) & (alpha >= ALPHA_MIN) & np.isfinite(z_raw) & (z_raw > 0)
         & np.isfinite(gam) & (gam > GW) & np.isfinite(tanphi)
         & np.isfinite(c) & np.isfinite(m))
fst = np.full(REF, np.nan, F)
fst[valid] = np.where(D_st[valid] > 0, R[valid] / D_st[valid], np.nan)

stb = valid & np.isfinite(fst) & (fst > 1)
fb = np.full(REF, np.nan, F)
fb[stb] = 1.0 - 1.0 / fst[stb]

L = []
L.append(f"scored (alpha>=10, valid)      {int(valid.sum()):,}")
L.append(f"Form B defined (FoS_static>1)  {int(stb.sum()):,} "
         f"({100*stb.sum()/int(valid.sum()):.1f}% of scored)")
L.append(f"Form B  mean {np.nanmean(fb[stb]):.4f}  max {np.nanmax(fb[stb]):.4f}")
L.append("")
L.append("ZONES (over pixels where Form B is defined)")
for name, msk in (("alpha > 30", stb & (alpha > 30)),
                  ("15 < alpha <= 30", stb & (alpha > 15) & (alpha <= 30)),
                  ("alpha <= 15", stb & (alpha <= 15))):
    if msk.sum() == 0:
        L.append(f"  {name:<18} no pixels"); continue
    v = fb[msk]
    q1, q3 = np.percentile(v, [25, 75])
    L.append(f"  {name:<18} n={int(msk.sum()):>9,}  mean {v.mean():.3f}  "
             f"IQR {q1:.2f}-{q3:.2f}  below 0.3: {100*(v<0.3).mean():.1f}%")
L.append("")
L.append("ZONES (over ALL scored pixels, incl. statically unstable)")
for name, msk in (("alpha > 30", valid & (alpha > 30)),
                  ("15 < alpha <= 30", valid & (alpha > 15) & (alpha <= 30)),
                  ("alpha <= 15", valid & (alpha <= 15))):
    n = int(msk.sum())
    d = int((msk & stb).sum())
    L.append(f"  {name:<18} n={n:>9,}  Form B defined on {100*d/max(n,1):5.1f}%")
txt = "\n".join(L)
print(txt)
open(OUT, "w").write(txt + "\n")
print(f"\nwrote {OUT}")
