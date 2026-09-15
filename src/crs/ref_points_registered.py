"""
Re-derive the three reference points on the registered rasters.

The old Table 5 set (c'=8.0, phi'=30.9, alpha=20.1, m=0.40, gamma=11.6, z=5.0,
k_h=0.190) is only partly a raster read:
  - gamma 11.6  matches the EBK grid in Villaflores
  - k_h 0.190   matches the REGISTERED PGA (0.38 g), not the broken one
  - z 5.0       is the depth cap, not a depth read
  - c' 8.0      is c_base=3 (fines 5-12%) under the c'+5 scenario, not the primary c'+10
  - phi' 30.9   implies PI = 41.0 via phi = 35 - 0.1*PI; quoted as "near the regional mean"
  - alpha 20.1  has no matching field sample, but 35k pixels in the barangay carry it

So the old set was assembled representatively rather than sampled at one pixel. Rather
than reverse-engineer it, this picks a reference pixel per barangay by a STATED RULE and
reads all seven parameters there under the primary configuration (c'+10, z-cap 5 m).

Rule: among pixels within 2 km of that barangay's field samples that pass the
alpha >= 10 deg filter and have finite values in every layer, take the pixel whose
slope angle is closest to the barangay's median slope over that same set. That is
reproducible, defensible, and needs nothing from a co-author.
"""
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform, reproject, Resampling

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
REF_TIF = os.path.join(IN, "Mak_BulkUnitWeight.tif")
REF_CRS = "EPSG:32651"
GAMMA_W = 9.81
RADIUS_M = 2000.0
C_OFFSET = 10.0        # primary configuration
Z_CAP = 5.0

with rasterio.open(REF_TIF) as ref:
    RT, RH, RW = ref.transform, ref.height, ref.width

raw = pd.read_excel(os.path.join(IN, "Makilala Geotechnical Data.xlsx"),
                    sheet_name="GEOTECH MAPUA", header=None)
d = raw.iloc[2:].copy()
d.columns = [str(x).strip() for x in raw.iloc[1].tolist()]
for c in ("LATITUDE", "LONGITUDE"):
    d[c] = pd.to_numeric(d[c], errors="coerce")
d = d.dropna(subset=["LATITUDE", "LONGITUDE"])
d["BGY"] = d["BARANGAY"].astype(str).str.strip().str.lower()


def window_for(pts):
    xs, ys = warp_transform("EPSG:4326", REF_CRS,
                            [p[0] for p in pts], [p[1] for p in pts])
    x0, x1 = min(xs) - RADIUS_M, max(xs) + RADIUS_M
    y0, y1 = min(ys) - RADIUS_M, max(ys) + RADIUS_M
    c0, r1 = ~RT * (x0, y0)
    c1, r0 = ~RT * (x1, y1)
    c0, c1 = int(max(0, min(c0, c1))), int(min(RW, max(c0, c1)))
    r0, r1 = int(max(0, min(r0, r1))), int(min(RH, max(r0, r1)))
    return rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)


def layers(win):
    WT = rasterio.windows.transform(win, RT)
    SH = (int(win.height), int(win.width))

    def rp(path, assumed=None):
        out = np.full(SH, np.nan, np.float32)
        with rasterio.open(path) as src:
            a = src.read(1).astype(np.float32)
            if src.nodata is not None:
                a[a == src.nodata] = np.nan
            reproject(a, out, src_transform=src.transform,
                      src_crs=assumed or src.crs, dst_transform=WT,
                      dst_crs=REF_CRS, resampling=Resampling.bilinear,
                      src_nodata=np.nan, dst_nodata=np.nan)
        return out

    def rw(path):
        with rasterio.open(path) as src:
            a = src.read(1, window=win).astype(np.float32)
            if src.nodata is not None:
                a[a == src.nodata] = np.nan
            return a

    L = dict(
        alpha=rp(os.path.join(IN, "Slope_PRSTM51N.tif"), "EPSG:32651"),
        depth=rp(os.path.join(IN, "Soil Depth.tif")),
        pga=rp(os.path.join(IN, "PGA 2019 EQ Events.tif")),
        gam=rw(os.path.join(IN, "Mak_BulkUnitWeight.tif")),
        fines=rw(os.path.join(IN, "Mak_Fines.tif")),
        sand=rw(os.path.join(IN, "Mak_Sand.tif")),
        pi=rw(os.path.join(IN, "Mak_PlasticityIndex.tif")),
        water=rw(os.path.join(IN, "Mak_WaterContent.tif")),
    )
    L["WT"] = WT
    return L


def derive(L):
    pi, fines, sand, water = L["pi"], L["fines"], L["sand"], L["water"]
    phi = np.full_like(pi, 30.0)
    v = (pi > 5) & np.isfinite(pi)
    phi[v] = 35.0 - 0.1 * pi[v]
    no = ~v & np.isfinite(sand)
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
    w = np.isfinite(water) & (water > 0)
    m[w & (water < 20)] = 0.2
    m[w & (water >= 20) & (water < 35)] = 0.4
    m[w & (water >= 35) & (water < 50)] = 0.6

    pga = L["pga"]
    mx = np.nanmax(pga)
    pg = pga / 981.0 if mx > 100 else (pga / 9.81 if mx > 2 else pga)
    kh = np.clip(0.5 * pg, 0, 1)
    return phi, cb, m, kh


def fos(c, phi, al, m, g, z, kh):
    a = np.radians(al); ca = np.cos(a); sa = np.sin(a)
    tp = np.tan(np.radians(phi))
    R = c + (g - m * GAMMA_W) * z * ca * ca * tp
    Dd = g * z * sa * ca + kh * g * z * ca * ca
    return R / Dd, R / (g * z * sa * ca)


def kc_of(c, phi, al, m, g, z):
    a = np.radians(al); ca = np.cos(a)
    return c / (g * z * ca * ca) + (1 - m * GAMMA_W / g) * np.tan(np.radians(phi)) - np.tan(a)


print("=" * 100)
print("REFERENCE POINTS RE-DERIVED ON THE REGISTERED RASTERS")
print(f"primary configuration: c' + {C_OFFSET:.0f} kPa, z-cap {Z_CAP:.0f} m, alpha >= 10 deg")
print("=" * 100)

for bgy in ("villaflores", "rodero 2", "sto. nino"):
    sub = d[d["BGY"] == bgy]
    pts = list(zip(sub["LONGITUDE"], sub["LATITUDE"]))
    win = window_for(pts)
    L = layers(win)
    phi, cb, m, kh = derive(L)
    alpha, depth, gam = L["alpha"], L["depth"], L["gam"]
    z = np.clip(depth, 0.1, Z_CAP)
    c = cb + C_OFFSET

    ok = (np.isfinite(alpha) & (alpha >= 10) & np.isfinite(z) & (z > 0)
          & np.isfinite(gam) & (gam > GAMMA_W) & np.isfinite(phi)
          & np.isfinite(c) & np.isfinite(m) & np.isfinite(kh))
    if not ok.sum():
        print(f"\n{bgy}: no valid pixels"); continue

    med = np.median(alpha[ok])
    dist = np.where(ok, np.abs(alpha - med), np.inf)
    r, cc = np.unravel_index(np.argmin(dist), alpha.shape)
    x, y = L["WT"] * (cc + 0.5, r + 0.5)
    lo, la = warp_transform(REF_CRS, "EPSG:4326", [x], [y])

    P = dict(c=float(c[r, cc]), phi=float(phi[r, cc]), alpha=float(alpha[r, cc]),
             m=float(m[r, cc]), gam=float(gam[r, cc]), z=float(z[r, cc]),
             kh=float(kh[r, cc]))
    F, Fs = fos(P["c"], P["phi"], P["alpha"], P["m"], P["gam"], P["z"], P["kh"])
    KC = kc_of(P["c"], P["phi"], P["alpha"], P["m"], P["gam"], P["z"])
    SB = 1 - 1 / Fs if Fs > 1 else float("nan")

    print(f"\n{bgy.title():<14} window {ok.sum():,} valid px   median alpha {med:.2f} deg")
    print(f"  reference pixel   lon {lo[0]:.5f}   lat {la[0]:.5f}")
    print(f"    c'    {P['c']:8.2f} kPa      (c_base {cb[r,cc]:.0f} + {C_OFFSET:.0f})")
    print(f"    phi'  {P['phi']:8.2f} deg      (PI {L['pi'][r,cc]:.1f})")
    print(f"    alpha {P['alpha']:8.2f} deg")
    print(f"    m     {P['m']:8.2f}          (water {L['water'][r,cc]:.1f}%)")
    print(f"    gamma {P['gam']:8.2f} kN/m3")
    print(f"    z     {P['z']:8.2f} m        (raw {depth[r,cc]:.2f})")
    print(f"    k_h   {P['kh']:8.4f}          (PGA {L['pga'][r,cc]:.4f} g)")
    print(f"  ->  FoS {F:.4f}   FoS_static {Fs:.4f}   k_c {KC:.4f}   "
          f"|S_kh|max {SB:.4f}")
    print(f"      alpha percentiles in window: "
          f"p10 {np.percentile(alpha[ok],10):.1f}  p50 {med:.1f}  "
          f"p90 {np.percentile(alpha[ok],90):.1f}  max {alpha[ok].max():.1f}")

print("\ndone.")
