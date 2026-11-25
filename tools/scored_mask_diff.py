#!/usr/bin/env python3
"""Locate the pixels where fos_means.py and kc_variance_common_mask.py disagree.

The two scripts build the scored domain from the same rasters and report
different totals: 7,104,441 against 7,104,440, the latter matching the
manuscript and the v11 notebook. A one-pixel difference is immaterial to every
published statistic, but two scripts in the same archive disagreeing on the
size of the domain is the kind of thing a referee checks first, and it was
twice attributed to the wrong cause before anyone measured it.

This script rebuilds both masks in one process, from one alignment, and prints
the pixels in the symmetric difference together with the clause responsible for
each. It also counts valid pixels in the water >= 50 saturation band, which was
missing from fos_means.py and formb_zones.py until 25 Nov 2026, so that the
effect of restoring it is known rather than assumed.

    python tools/scored_mask_diff.py
"""
import os, sys
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling

GAMMA_W, ALPHA_MIN, GRID_CRS, REF = 9.81, 10.0, "EPSG:32651", "gamma"
FILES = {'slope': "Slope_PRSTM51N.tif", 'depth': "Soil Depth.tif",
         'pga': "PGA 2019 EQ Events.tif", 'gamma': "Mak_BulkUnitWeight.tif",
         'water': "Mak_WaterContent.tif", 'pi': "Mak_PlasticityIndex.tif",
         'fines': "Mak_Fines.tif", 'sand': "Mak_Sand.tif"}


def _find_data_dir():
    cands = [os.environ.get("MAKILALA_DIR", ""),
             os.path.expanduser("~/Downloads/makilala_data"),
             os.path.expanduser("~/mnt/makilala_data"), os.getcwd()]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "Mak_BulkUnitWeight.tif")):
            return c
    raise SystemExit("Could not locate makilala_data; set MAKILALA_DIR.")


IN = _find_data_dir()
print("data directory:", IN)

raw, prof = {}, {}
for k, fn in FILES.items():
    with rasterio.open(os.path.join(IN, fn)) as src:
        d = src.read(1).astype(np.float64)
        if src.nodata is not None:
            d[d == src.nodata] = np.nan
        raw[k], prof[k] = d, src.profile.copy()

shape, tr = raw[REF].shape, prof[REF]['transform']
crs = CRS.from_string(GRID_CRS)
A = {}
for k, d in raw.items():
    if d.shape == shape and prof[k]['transform'].almost_equals(tr, precision=1e-6):
        A[k] = d
        continue
    dst = np.full(shape, np.nan)
    reproject(source=d, destination=dst, src_transform=prof[k]['transform'],
              src_crs=CRS.from_user_input(prof[k]['crs']), dst_transform=tr,
              dst_crs=crs, resampling=Resampling.bilinear,
              src_nodata=np.nan, dst_nodata=np.nan)
    A[k] = dst
del raw

alpha = np.clip(A['slope'], 0.1, 80.0)
z_raw, water = A['depth'], A['water']
pi, sand, fines = A['pi'], A['sand'], A['fines']

phi = np.full_like(pi, 30.0)
v_ = (pi > 5) & np.isfinite(pi)
phi[v_] = 35.0 - 0.1 * pi[v_]
no = ~v_ & np.isfinite(sand)
phi[no & (sand > 85) & (fines < 10)] = 33.0
phi[no & (sand > 70) & (fines >= 10) & (fines < 20)] = 31.0
phi[no & (fines >= 20)] = 28.0
phi = np.clip(phi, 20.0, 38.0)

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
ca = np.cos(np.radians(alpha)); sa = np.sin(np.radians(alpha))
tanphi = np.tan(np.radians(phi))

# --- the two formulations, each exactly as its own script writes it ----------
gam_F = A['gamma']                                    # fos_means: unclipped
gam_K = np.clip(A['gamma'], GAMMA_W+0.1, 30.0)        # kc_variance: clipped

def fos_of(gam):
    c = cb + 10.0
    R = c + (gam - m*GAMMA_W) * z5 * ca * ca * tanphi
    D = gam * z5 * sa * ca + k_h * gam * z5 * ca * ca
    out = np.full(shape, np.nan)
    good = np.isfinite(D) & (D > 0)
    out[good] = R[good] / D[good]
    return out

clauses_F = {
    "isfinite(alpha)":    np.isfinite(alpha),
    "alpha >= 10":        alpha >= ALPHA_MIN,
    "isfinite(z_raw)":    np.isfinite(z_raw),
    "z_raw > 0":          z_raw > 0,
    "isfinite(gam)":      np.isfinite(gam_F),
    "gam > GW  [UNCLIPPED]": gam_F > GAMMA_W,
    "isfinite(tanphi)":   np.isfinite(tanphi),
    "isfinite(c)":        np.isfinite(cb),
    "isfinite(m)":        np.isfinite(m),
    "isfinite(k_h)":      np.isfinite(k_h),
}
clauses_K = {
    "isfinite(alpha)":    np.isfinite(alpha),
    "alpha >= 10":        alpha >= ALPHA_MIN,
    "isfinite(z_raw)":    np.isfinite(z_raw),
    "z_raw > 0":          z_raw > 0,
    "isfinite(gamma)":    np.isfinite(gam_K),
    "gamma > GW  [CLIPPED]": gam_K > GAMMA_W,
    "isfinite(phi)":      np.isfinite(phi),
    "phi > 0":            phi > 0,
    "isfinite(c)":        np.isfinite(cb),
    "c >= 0":             cb >= 0,
    "isfinite(m)":        np.isfinite(m),
    "m >= 0":             m >= 0,
    "m <= 1":             m <= 1,
}

def build(clauses):
    out = np.ones(shape, bool)
    for v in clauses.values():
        out &= v
    return out

vF, vK = build(clauses_F), build(clauses_K)
SF = vF & np.isfinite(fos_of(gam_F))
SK = vK & np.isfinite(fos_of(gam_K))

print(f"\nfos_means-style  SCORED : {int(SF.sum()):,}")
print(f"kc_variance-style SCORED: {int(SK.sum()):,}")
print(f"symmetric difference    : {int((SF ^ SK).sum()):,}")

for name, extra, base in (("in fos_means only", SF & ~SK, clauses_K),
                          ("in kc_variance only", SK & ~SF, clauses_F)):
    idx = np.argwhere(extra)
    print(f"\n{name}: {len(idx)} pixel(s)")
    for (r, c_) in idx[:10]:
        failed = [k for k, v in base.items() if not v[r, c_]]
        print(f"  row {r} col {c_}: fails in the other script -> "
              f"{failed if failed else 'no clause; difference is in FoS finiteness'}")
        print(f"      alpha={alpha[r,c_]:.4f}  z_raw={z_raw[r,c_]:.4f}  "
              f"gamma={A['gamma'][r,c_]:.6f}  water={water[r,c_]:.4f}  m={m[r,c_]:.2f}")

band = w_ & (water >= 50)
print(f"\nwater >= 50 band (the branch missing from fos_means/formb_zones):")
print(f"  cells in band            : {int(band.sum()):,}")
print(f"  of those, scored         : {int((band & SK).sum()):,}")
print("  -> restoring m=0.8 there changes the published moments only if this is > 0")

# --- working precision -------------------------------------------------------
# fos_means.py and formb_zones.py set F = np.float32 and carry every array at
# that precision; kc_variance_common_mask.py and the v11 notebook work in
# float64. The clause sets above are identical and in float64 agree exactly, so
# if the shipped scripts still disagree the cause is precision at a mask
# boundary. This pass rebuilds the fos_means clause set in float32 and names the
# pixel and the clause that flips.
f32 = np.float32
alpha32, zr32, gam32 = alpha.astype(f32), z_raw.astype(f32), A['gamma'].astype(f32)
tanphi32, cb32, m32, kh32 = (tanphi.astype(f32), cb.astype(f32),
                             m.astype(f32), k_h.astype(f32))
clauses_F32 = {
    "isfinite(alpha)": np.isfinite(alpha32), "alpha >= 10": alpha32 >= f32(ALPHA_MIN),
    "isfinite(z_raw)": np.isfinite(zr32),    "z_raw > 0":    zr32 > 0,
    "isfinite(gam)":   np.isfinite(gam32),   "gam > GW":     gam32 > f32(GAMMA_W),
    "isfinite(tanphi)": np.isfinite(tanphi32), "isfinite(c)": np.isfinite(cb32),
    "isfinite(m)":     np.isfinite(m32),     "isfinite(k_h)": np.isfinite(kh32),
}
vF32 = build(clauses_F32)
z532 = z5.astype(f32); ca32 = ca.astype(f32); sa32 = sa.astype(f32)
R32 = cb32 + f32(10.0) + (gam32 - m32*f32(GAMMA_W)) * z532 * ca32 * ca32 * tanphi32
D32 = gam32*z532*sa32*ca32 + kh32*gam32*z532*ca32*ca32
fos32 = np.full(shape, np.nan, f32)
g32 = np.isfinite(D32) & (D32 > 0)
fos32[g32] = R32[g32] / D32[g32]
SF32 = vF32 & np.isfinite(fos32)

print(f"\nfos_means clause set, float32 : {int(SF32.sum()):,}")
print(f"fos_means clause set, float64 : {int(SF.sum()):,}")
flip = SF32 ^ SF
print(f"pixels that flip on precision : {int(flip.sum()):,}")
for (r, c_) in np.argwhere(flip)[:10]:
    side = "float32 only" if SF32[r, c_] else "float64 only"
    f64f = [k for k, v in clauses_F.items() if not v[r, c_]]
    f32f = [k for k, v in clauses_F32.items() if not v[r, c_]]
    print(f"  row {r} col {c_}  ({side})")
    print(f"      fails in float64: {f64f or 'none'}")
    print(f"      fails in float32: {f32f or 'none'}")
    print(f"      alpha={alpha[r,c_]!r}  gamma={A['gamma'][r,c_]!r}")
