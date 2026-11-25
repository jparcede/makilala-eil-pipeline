#!/usr/bin/env python3
"""Diff the REAL fos_means.py scored mask against the kc_variance formulation.

Three hypotheses for the one-pixel disagreement (7,104,441 against the
published 7,104,440) have now been tested and refuted by measurement:
zero-depth cells, differing mask clauses, and float32-vs-float64 precision.
Each was refuted because the comparison was made against a REIMPLEMENTATION of
fos_means.py rather than against fos_means.py itself.

This script executes the actual module, takes the actual mask it builds, and
diffs it against a kc_variance-style mask computed from the same files in the
same process. Whatever the cause is, it is inside fos_means.py, so this cannot
miss it the way a reimplementation can.

    python tools/scored_mask_trace.py
"""
import os, runpy, sys
import numpy as np
import rasterio
from rasterio.crs import CRS as RCRS
from rasterio.warp import reproject, Resampling

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FOS = os.path.join(REPO, "src", "validation", "fos_means.py")

print("executing", FOS, "...\n")
ns = runpy.run_path(FOS, run_name="__not_main__")

sc = ns.get("sc")
if sc is None:
    raise SystemExit("fos_means.py did not expose `sc`; inspect it by hand.")
print(f"\nfos_means.py own scored mask : {int(sc.sum()):,}")

IN = ns["IN"]
RT, REF = ns["RT"], ns["REF"]
GAMMA_W, ALPHA_MIN = 9.81, 10.0

# --- kc_variance-style alignment: almost_equals, float64 --------------------
def align64(fn):
    with rasterio.open(os.path.join(IN, fn)) as s:
        a = s.read(1).astype(np.float64)
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        if (s.height, s.width) == REF and s.transform.almost_equals(RT, precision=1e-6):
            return a, True
        out = np.full(REF, np.nan)
        reproject(a, out, src_transform=s.transform, src_crs=s.crs,
                  dst_transform=RT, dst_crs=RCRS.from_string("EPSG:32651"),
                  resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
        return out, False

print("\nalignment path taken per raster (fos_means uses EXACT transform equality,")
print("kc_variance uses almost_equals(precision=1e-6)):")
names = {'slope': "Slope_PRSTM51N.tif", 'depth': "Soil Depth.tif",
         'pga': "PGA 2019 EQ Events.tif", 'gamma': "Mak_BulkUnitWeight.tif",
         'water': "Mak_WaterContent.tif", 'pi': "Mak_PlasticityIndex.tif",
         'fines': "Mak_Fines.tif", 'sand': "Mak_Sand.tif"}
A = {}
for k, fn in names.items():
    with rasterio.open(os.path.join(IN, fn)) as s:
        exact = (s.height, s.width) == REF and s.transform == RT
        near = (s.height, s.width) == REF and s.transform.almost_equals(RT, precision=1e-6)
    A[k], passthru = align64(fn)
    flag = "  <-- DIFFERENT PATH" if (exact != near) else ""
    print(f"  {k:<7} exact={str(exact):<5} almost={str(near):<5}{flag}")

alpha = np.clip(A['slope'], 0.1, 80.0)
z_raw, water, gam = A['depth'], A['water'], A['gamma']
pi, sand, fines = A['pi'], A['sand'], A['fines']

phi = np.full_like(pi, 30.0)
v_ = (pi > 5) & np.isfinite(pi)
phi[v_] = 35.0 - 0.1*pi[v_]
no = ~v_ & np.isfinite(sand)
phi[no & (sand > 85) & (fines < 10)] = 33.0
phi[no & (sand > 70) & (fines >= 10) & (fines < 20)] = 31.0
phi[no & (fines >= 20)] = 28.0
tanphi = np.tan(np.radians(np.clip(phi, 20.0, 38.0)))

cb = np.full_like(fines, 5.0)
vv = np.isfinite(fines) & np.isfinite(sand)
cb[vv & (fines < 5)] = 1.0
cb[vv & (fines >= 5) & (fines < 12)] = 3.0
cb[vv & (fines >= 12) & (fines < 20)] = 6.0
cb[vv & (fines >= 20)] = 10.0

m = np.full_like(water, 0.5)
w_ = np.isfinite(water) & (water > 0)
m[w_ & (water < 20)] = 0.2
m[w_ & (water >= 20) & (water < 35)] = 0.4
m[w_ & (water >= 35) & (water < 50)] = 0.6
m[w_ & (water >= 50)] = 0.8

pga = A['pga']
mx = np.nanmax(pga[np.isfinite(pga) & (pga > 0)])
pg = pga/981.0 if mx > 100 else (pga/9.81 if mx > 2 else pga)
k_h = np.clip(0.5*pg, 0.0, 1.0)

z5 = np.clip(z_raw, 0.1, 5.0)
ca, sa = np.cos(np.radians(alpha)), np.sin(np.radians(alpha))
R = cb + 10.0 + (gam - m*GAMMA_W)*z5*ca*ca*tanphi
D = gam*z5*sa*ca + k_h*gam*z5*ca*ca
fos = np.full(REF, np.nan)
g = np.isfinite(D) & (D > 0)
fos[g] = R[g]/D[g]

clauses = {
    "isfinite(alpha)": np.isfinite(alpha), "alpha >= 10": alpha >= ALPHA_MIN,
    "isfinite(z_raw)": np.isfinite(z_raw), "z_raw > 0": z_raw > 0,
    "isfinite(gam)": np.isfinite(gam), "gam > GW": gam > GAMMA_W,
    "isfinite(tanphi)": np.isfinite(tanphi), "isfinite(c)": np.isfinite(cb),
    "isfinite(m)": np.isfinite(m), "isfinite(k_h)": np.isfinite(k_h),
    "isfinite(FoS)": np.isfinite(fos),
}
ref_mask = np.ones(REF, bool)
for v in clauses.values():
    ref_mask &= v

print(f"\nreference (float64, almost_equals) : {int(ref_mask.sum()):,}")
print(f"fos_means.py                       : {int(sc.sum()):,}")
diff = np.asarray(sc, bool) ^ ref_mask
print(f"differing pixels                   : {int(diff.sum()):,}")

for (r, c_) in np.argwhere(diff)[:10]:
    side = "fos_means only" if sc[r, c_] else "reference only"
    failed = [k for k, v in clauses.items() if not v[r, c_]]
    print(f"\n  row {r} col {c_}  ({side})")
    print(f"    reference clauses failing: {failed or 'none'}")
    for k in ('slope', 'depth', 'gamma', 'water', 'pga', 'pi', 'fines', 'sand'):
        print(f"      {k:<7} = {A[k][r, c_]!r}")
    print(f"      alpha={alpha[r,c_]!r} m={m[r,c_]!r} k_h={k_h[r,c_]!r} FoS={fos[r,c_]!r}")
