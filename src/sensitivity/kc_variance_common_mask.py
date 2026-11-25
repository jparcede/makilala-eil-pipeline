#!/usr/bin/env python3
"""k_c variance decomposition on a COMMON mask.

Reproduces cells 0-5 of Makilala_FoS_Pipeline_v11 (configuration, helpers,
registration, derived parameters, domain cascade) and then recomputes the
first-order variance decomposition of cell 12 two ways:

  ORIGINAL  each term's mean is taken over the pixels where that term is
            defined, while the denominator is the mean of the SUM, which is
            NaN wherever ANY term is undefined. Numerator and denominator
            therefore average over different pixel populations and the shares
            do not sum to 100%. v11 prints 98.1%.

  COMMON    every term and the denominator are averaged over the same mask:
            SCORED and all six terms finite. Shares sum to 100% by
            construction.

Also sweeps the assumed relative standard error on soil depth, which is the
decomposition's dominant free assumption and enters quadratically.
"""
import os, sys, time
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling

t0 = time.time()
# --- data directory resolution -----------------------------------------------
# Was hardcoded to a sandbox path, so this script could not find the rasters on
# any other machine. Mirrors the resolution used in src/validation/fos_means.py:
# MAKILALA_DIR first, then the usual locations, then the working directory.
def _find_data_dir():
    cands = [os.environ.get("MAKILALA_DIR", ""),
             os.path.expanduser("~/Downloads/makilala_data"),
             os.path.expanduser("~/mnt/makilala_data"),
             os.getcwd()]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "Mak_BulkUnitWeight.tif")):
            return c
    msg = ["Could not locate makilala_data (no Mak_BulkUnitWeight.tif found).",
           "Tried:"]
    msg += ["  " + x for x in cands if x]
    msg += ["Set it explicitly:  MAKILALA_DIR=/path/to/makilala_data python "
            + os.path.basename(__file__)]
    raise SystemExit(chr(10).join(msg))


INPUT_DIR = _find_data_dir()
GAMMA_W, ALPHA_MIN, GRID_CRS, REF = 9.81, 10, "EPSG:32651", 'gamma'
REL_SE_Z = 0.30
DC_DFINES, DM_DWATER = 0.3989, 0.016766

FILES = {'slope':"Slope_PRSTM51N.tif", 'depth':"Soil Depth.tif",
         'pga':"PGA 2019 EQ Events.tif", 'gamma':"Mak_BulkUnitWeight.tif",
         'water':"Mak_WaterContent.tif", 'pi':"Mak_PlasticityIndex.tif",
         'fines':"Mak_Fines.tif", 'sand':"Mak_Sand.tif"}
SE_FILES = {'gamma':'buk_v1_standarderror.tif', 'water':'watercontent_v11standarderror.tif',
            'fines':'siltclay_v1_standarderror.tif', 'pi':'PI_v1_standarderror.tif'}

def estimate_phi(pi, sand, fines):
    phi = np.full_like(pi, 30.0, dtype=np.float64)
    v = (pi > 5) & np.isfinite(pi); phi[v] = 35.0 - 0.1*pi[v]
    no = ~v & np.isfinite(sand)
    phi[no & (sand > 85) & (fines < 10)] = 33.0
    phi[no & (sand > 70) & (fines >= 10) & (fines < 20)] = 31.0
    phi[no & (fines >= 20)] = 28.0
    return np.clip(phi, 20.0, 38.0)

def estimate_c(fines, sand):
    c = np.full_like(fines, 5.0, dtype=np.float64)
    v = np.isfinite(fines) & np.isfinite(sand)
    c[v & (fines < 5)] = 1.0
    c[v & (fines >= 5) & (fines < 12)] = 3.0
    c[v & (fines >= 12) & (fines < 20)] = 6.0
    c[v & (fines >= 20)] = 10.0
    return c

def estimate_m(water):
    m = np.full_like(water, 0.5, dtype=np.float64)
    v = np.isfinite(water) & (water > 0)
    m[v & (water < 20)] = 0.2
    m[v & (water >= 20) & (water < 35)] = 0.4
    m[v & (water >= 35) & (water < 50)] = 0.6
    m[v & (water >= 50)] = 0.8
    return m

def compute_FoS(c, phi_deg, alpha_deg, m, gamma, z, k_h):
    phi, alpha = np.radians(phi_deg), np.radians(alpha_deg)
    R = c + (gamma - m*GAMMA_W)*z*np.cos(alpha)**2*np.tan(phi)
    D = gamma*z*np.sin(alpha)*np.cos(alpha) + k_h*gamma*z*np.cos(alpha)**2
    return np.where(D > 0, R/D, np.nan)

def load(path):
    with rasterio.open(path) as src:
        d = src.read(1).astype(np.float64)
        if src.nodata is not None: d[d == src.nodata] = np.nan
        return d, src.profile.copy()

print(f"input: {INPUT_DIR}", flush=True)
rasters, profiles = {}, {}
for k, fn in FILES.items():
    rasters[k], profiles[k] = load(os.path.join(INPUT_DIR, fn))
    print(f"  loaded {k:<6} {rasters[k].shape}  {time.time()-t0:.0f}s", flush=True)

ref_shape = rasters[REF].shape
ref_transform = profiles[REF]['transform']
ref_crs = CRS.from_string(GRID_CRS)

aligned = {}
for k, d in rasters.items():
    if d.shape == ref_shape and profiles[k]['transform'].almost_equals(ref_transform, precision=1e-6):
        aligned[k] = d; continue
    dst = np.full(ref_shape, np.nan)
    reproject(source=d, destination=dst, src_transform=profiles[k]['transform'],
              src_crs=CRS.from_user_input(profiles[k]['crs']), dst_transform=ref_transform,
              dst_crs=ref_crs, resampling=Resampling.bilinear,
              src_nodata=np.nan, dst_nodata=np.nan)
    aligned[k] = dst
    print(f"  reprojected {k}  {time.time()-t0:.0f}s", flush=True)
del rasters

alpha = np.clip(aligned['slope'], 0.1, 80.0)
gamma = np.clip(aligned['gamma'], GAMMA_W+0.1, 30.0)
phi   = estimate_phi(aligned['pi'], aligned['sand'], aligned['fines'])
c     = estimate_c(aligned['fines'], aligned['sand'])
m     = estimate_m(aligned['water'])
z_raw = aligned['depth']
pga   = aligned['pga']
pmax  = np.nanmax(pga[np.isfinite(pga) & (pga > 0)])
pga_g = pga/981.0 if pmax > 100 else (pga/9.81 if pmax > 2 else pga)
k_h   = np.clip(0.5*pga_g, 0.0, 1.0)
del aligned, pga, pga_g

z5 = np.clip(z_raw, 0.1, 5.0)
valid = (np.isfinite(alpha) & (alpha >= ALPHA_MIN) & np.isfinite(z_raw) & (z_raw > 0)
         & np.isfinite(gamma) & (gamma > GAMMA_W) & np.isfinite(phi) & (phi > 0)
         & np.isfinite(c) & (c >= 0) & np.isfinite(m) & (m >= 0) & (m <= 1))
fos = np.full(ref_shape, np.nan)
fos[valid] = compute_FoS((c+10.0)[valid], phi[valid], alpha[valid], m[valid],
                         gamma[valid], z5[valid], k_h[valid])
SCORED = valid & np.isfinite(fos)
n_scored = int(SCORED.sum())
print(f"\nSCORED pixels: {n_scored:,}   (v11 published 7,104,440)", flush=True)
del fos, valid, k_h

se = {}
for k, fn in SE_FILES.items():
    p = os.path.join(INPUT_DIR, "standard_error", fn)
    with rasterio.open(p) as src:
        a = src.read(1).astype(np.float64)
        if src.nodata is not None: a[a == src.nodata] = np.nan
        dst = np.full(ref_shape, np.nan)
        reproject(a, dst, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref_transform, dst_crs=ref_crs,
                  resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    se[k] = dst[SCORED]
    print(f"  SE {k:<6} coverage {100*np.isfinite(se[k]).mean():.1f}% of scored", flush=True)

# ---- collapse to 1-D over SCORED ----
a_  = np.radians(alpha[SCORED]); ph_ = np.radians(phi[SCORED])
g_  = gamma[SCORED]; m_ = m[SCORED]; z_ = z5[SCORED]; c10 = c[SCORED] + 10.0
del alpha, gamma, phi, c, m, z_raw, z5, SCORED

d   = g_*z_*np.cos(a_)**2
d_c     = 1.0/d
d_z     = -c10/(g_*z_**2*np.cos(a_)**2)
d_m     = -(GAMMA_W/g_)*np.tan(ph_)
d_phi   = (1 - m_*GAMMA_W/g_)/np.cos(ph_)**2
d_gamma = -c10/(g_**2*z_*np.cos(a_)**2) + (m_*GAMMA_W/g_**2)*np.tan(ph_)
d_alpha = (2*c10*np.sin(a_)/(g_*z_*np.cos(a_)**3)) - 1.0/np.cos(a_)**2

def shares(rel_se_z):
    var = {'z': (rel_se_z*z_)**2, 'gamma': se['gamma']**2,
           'phi': (np.radians(0.1)*se['pi'])**2, 'alpha': np.full(z_.shape, np.radians(2.0)**2),
           'c': (DC_DFINES*se['fines'])**2, 'm': (DM_DWATER*se['water'])**2}
    terms = {'z': d_z**2*var['z'], 'c(fines)': d_c**2*var['c'], 'm(water)': d_m**2*var['m'],
             'phi(PI)': d_phi**2*var['phi'], 'gamma': d_gamma**2*var['gamma'],
             'alpha': d_alpha**2*var['alpha']}
    tot = sum(terms.values())
    common = np.isfinite(tot)
    orig = {k: 100*np.nanmean(v)/np.nanmean(tot) for k, v in terms.items()}
    comm = {k: 100*np.mean(v[common])/np.mean(tot[common]) for k, v in terms.items()}
    return orig, comm, int(common.sum())

orig, comm, n_common = shares(REL_SE_Z)
print(f"\ncommon mask (all six terms finite): {n_common:,} px"
      f"  = {100*n_common/n_scored:.1f}% of scored", flush=True)
print(f"\n{'term':<12}{'ORIGINAL':>12}{'COMMON MASK':>14}")
print("-"*38)
for k in sorted(comm, key=lambda x: -comm[x]):
    print(f"{k:<12}{orig[k]:>11.1f}%{comm[k]:>13.1f}%")
print("-"*38)
print(f"{'TOTAL':<12}{sum(orig.values()):>11.1f}%{sum(comm.values()):>13.1f}%")
print(f"\npublished in the manuscript: 71.4%  (corrected from 71.5% on 25 Nov 2026)")
print(f"v11 cell 12 prints          : 69.9%  (= the ORIGINAL column)")
print(f"common-mask value           : {comm['z']:.1f}%")

print(f"\nsensitivity of the soil-depth share to the assumed relative SE on z")
print(f"  (the share is quadratic in this; the manuscript assumes 30%)")
print(f"\n  {'rel SE on z':<14}{'depth share':>13}")
for r in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
    _, cm, _ = shares(r)
    mark = "   <- assumed" if abs(r-0.30) < 1e-9 else ""
    print(f"  {r:>8.0%}      {cm['z']:>11.1f}%{mark}")
print(f"\ndone in {time.time()-t0:.0f}s")
