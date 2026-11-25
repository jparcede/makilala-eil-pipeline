"""
Static and pseudo-static FoS moments over the scored domain, primary configuration.
Three sentences in the manuscript depend on these and currently carry v7 values.
Writes a small text file so the run survives a shell timeout.
"""
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
OUT = os.path.join(IN, "artifacts", "fos_means.txt")
F = np.float32
GW = 9.81
ALPHA_MIN, Z_CAP, C_OFF = 10.0, 5.0, 10.0

with rasterio.open(os.path.join(IN, "Mak_BulkUnitWeight.tif")) as r:
    RT, REF = r.transform, (r.height, r.width)
CRS = "EPSG:32651"


def align(fn, assumed=None):
    out = np.full(REF, np.nan, F)
    with rasterio.open(os.path.join(IN, fn)) as s:
        if (s.height, s.width) == REF and s.transform == RT:
            a = s.read(1).astype(F)
            if s.nodata is not None:
                a[a == s.nodata] = np.nan
            return a
        a = s.read(1).astype(F)
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        reproject(a, out, src_transform=s.transform, src_crs=assumed or s.crs,
                  dst_transform=RT, dst_crs=CRS, resampling=Resampling.bilinear,
                  src_nodata=np.nan, dst_nodata=np.nan)
    return out


# The slope raster is GENUINELY PRS 1992 as declared, which Section 4.4 of the
# manuscript establishes by free-offset scan against the field samples. Passing
# CRS here as `assumed` overrode that and reprojected it as though it were
# already EPSG:32651 -- reintroducing the exact ~168 m datum error the paper
# exists to diagnose. It silently cost 34,406 scored pixels (7,070,034 against
# the published 7,104,440) and moved the Form B defined fraction from 83.1% to
# 82.9%. Fixed 25 Nov 2026; let align() read the file's own CRS.
alpha = np.clip(align("Slope_PRSTM51N.tif"), 0.1, 80.).astype(F)
z_raw = align("Soil Depth.tif").astype(F)
pga = align("PGA 2019 EQ Events.tif").astype(F)
gam = align("Mak_BulkUnitWeight.tif").astype(F)
fines = align("Mak_Fines.tif").astype(F)
sand = align("Mak_Sand.tif").astype(F)
pi = align("Mak_PlasticityIndex.tif").astype(F)
water = align("Mak_WaterContent.tif").astype(F)
gc.collect()

mx = np.nanmax(pga)
pg = pga / 981. if mx > 100 else (pga / 9.81 if mx > 2 else pga)
k_h = np.clip(0.5 * pg, 0, 1).astype(F)
del pga, pg

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
D_ps = (gam * z * sa * ca + k_h * gam * z * ca * ca).astype(F)
D_st = (gam * z * sa * ca).astype(F)

valid = (np.isfinite(alpha) & (alpha >= ALPHA_MIN) & np.isfinite(z) & (z > 0)
         & np.isfinite(gam) & (gam > GW) & np.isfinite(tanphi)
         & np.isfinite(c) & np.isfinite(m) & np.isfinite(k_h))

fps = np.full(REF, np.nan, F)
fps[valid] = np.where(D_ps[valid] > 0, R[valid] / D_ps[valid], np.nan)
fst = np.full(REF, np.nan, F)
fst[valid] = np.where(D_st[valid] > 0, R[valid] / D_st[valid], np.nan)

sc = valid & np.isfinite(fps)
L = []
L.append(f"scored pixels                {int(sc.sum()):,}")
L.append(f"pseudo-static FoS  mean      {np.nanmean(fps[sc]):.4f}")
L.append(f"pseudo-static FoS  median    {np.nanmedian(fps[sc]):.4f}")
st = sc & np.isfinite(fst)
L.append(f"static FoS         mean      {np.nanmean(fst[st]):.4f}")
L.append(f"static FoS         median    {np.nanmedian(fst[st]):.4f}")
L.append(f"mean difference (static - pseudo) {np.nanmean(fst[st]) - np.nanmean(fps[st]):.4f}")
stb = st & (fst > 1)
fb = 1.0 - 1.0 / fst[stb]
L.append(f"Form B defined on            {int(stb.sum()):,} px "
         f"({100*stb.sum()/max(int(sc.sum()),1):.1f}% of scored)")
L.append(f"Form B  mean {fb.mean():.4f}  median {np.median(fb):.4f}  max {fb.max():.4f}")
txt = "\n".join(L)
print(txt)
open(OUT, "w").write(txt + "\n")
print(f"\nwrote {OUT}")
