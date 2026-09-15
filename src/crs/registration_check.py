"""
registration_check.py — reproduces the two registration defects and their effect.

Run from anywhere; set U to the makilala_data folder. Needs rasterio, geopandas,
pyproj, scipy, pandas, openpyxl. Roughly 3 GB peak, a few minutes.

Findings it reproduces, in order:

  1. Slope, Soil Depth and PGA do not share the reference grid's extent, so §3's
     scipy.ndimage.zoom stretches them instead of registering them. Recomputing
     FoS with zoom reproduces output_maps_v7/FoS_map.tif exactly; recomputing it
     with rasterio.warp.reproject does not.

  2. LSI_North Cotabato.shp carries WGS 84 / UTM 51N coordinates while its .prj
     declares PRS 1992 / UTM 51N, displacing every polygon ~168 m E / 47 m N.
     Three independent tests: a free offset fit, the datum transform itself, and
     the field GPS.
"""
import os, gc, warnings
import numpy as np
warnings.filterwarnings("ignore")
from scipy.ndimage import zoom
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from shapely.geometry import mapping, Point
import geopandas as gpd
import pandas as pd
import pyproj

U = os.environ.get("MAKILALA_DIR", "data")
V7 = os.path.join(U, "output_maps_v7")
F = np.float32
TH = np.arange(0.40, 2.005, 0.05)
REF_CRS = "ESRI:102457"        # PRS 1992 / UTM zone 51N
ALT_CRS = "EPSG:32651"         # WGS 84  / UTM zone 51N


def rd(p):
    with rasterio.open(p) as s:
        a = s.read(1).astype(F)
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        return a, s.profile.copy()


gam0, rp = rd(f"{U}/Mak_BulkUnitWeight.tif")
ref = gam0.shape


# ---------------------------------------------------------------- 1. the audit
def audit():
    print("\n=== raster registration audit (reference = Mak_BulkUnitWeight) ===")
    for fn in sorted(f for f in os.listdir(U) if f.endswith(".tif")):
        with rasterio.open(f"{U}/{fn}") as s:
            b = s.bounds
            same = (s.shape == ref
                    and abs(s.transform.c - rp['transform'].c) < 1e-6
                    and abs(s.transform.f - rp['transform'].f) < 1e-6)
            print(f"  {fn:<28} {str(s.shape):<14} {s.res[0]:>7.2f} m  {str(s.crs):<13} "
                  f"{(b.right-b.left)/1000:6.1f} x {(b.top-b.bottom)/1000:5.1f} km  "
                  f"{'identical' if same else 'RESAMPLED BY zoom'}")


def AL(fn, mode):
    """mode='zoom' reproduces the published pipeline; 'reproject' registers."""
    a, p = rd(f"{U}/{fn}")
    if a.shape == ref:
        return a
    if mode == 'zoom':
        return zoom(a, (ref[0]/a.shape[0], ref[1]/a.shape[1]),
                    order=1)[:ref[0], :ref[1]].astype(F)
    d = np.full(ref, np.nan, F)
    reproject(a, d, src_transform=p['transform'], src_crs=p['crs'],
              dst_transform=rp['transform'], dst_crs=rp['crs'],
              resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    return d


def field_comparison():
    print("\n=== zoom vs registered, over the reference grid ===")
    for fn, lab in [("Slope_PRSTM51N.tif", "slope"), ("Soil Depth.tif", "depth"),
                    ("PGA 2019 EQ Events.tif", "PGA")]:
        z, r = AL(fn, 'zoom'), AL(fn, 'reproject')
        b = np.isfinite(z) & np.isfinite(r)
        print(f"  {lab:<6} zoom mean {np.nanmean(z[np.isfinite(z)]):8.4f}   "
              f"registered mean {np.nanmean(r[np.isfinite(r)]):8.4f}   "
              f"corr {np.corrcoef(z[b], r[b])[0,1]:+.4f}")
        del z, r, b; gc.collect()


# ------------------------------------------------- shared material fields
pi = AL("Mak_PlasticityIndex.tif", 'x'); sand = AL("Mak_Sand.tif", 'x')
fines = AL("Mak_Fines.tif", 'x'); water = AL("Mak_WaterContent.tif", 'x')
gam = np.clip(gam0, 9.91, 30.).astype(F)
phi = np.full_like(pi, 30.); v_ = (pi > 5) & np.isfinite(pi); phi[v_] = 35. - 0.1*pi[v_]
no = ~v_ & np.isfinite(sand)
phi[no & (sand > 85) & (fines < 10)] = 33.
phi[no & (sand > 70) & (fines >= 10) & (fines < 20)] = 31.
phi[no & (fines >= 20)] = 28.
phi = np.clip(phi, 20., 38.).astype(F)
c0 = np.full_like(fines, 5.); vv = np.isfinite(fines) & np.isfinite(sand)
c0[vv & (fines < 5)] = 1.; c0[vv & (fines >= 5) & (fines < 12)] = 3.
c0[vv & (fines >= 12) & (fines < 20)] = 6.; c0[vv & (fines >= 20)] = 10.
m = np.full_like(water, 0.5); w_ = np.isfinite(water) & (water > 0)
m[w_ & (water < 20)] = 0.2; m[w_ & (water >= 20) & (water < 35)] = 0.4
m[w_ & (water >= 35) & (water < 50)] = 0.6; m[w_ & (water >= 50)] = 0.8
m = m.astype(F); tanphi = np.tan(np.radians(phi)).astype(F)
del pi, sand, fines, water, phi; gc.collect()

base = gpd.read_file(f"{U}/LSI_North Cotabato.shp")
corrected = base.set_crs(ALT_CRS, allow_override=True).to_crs(REF_CRS)


def inventory(fix):
    g = corrected if fix else base
    cov = np.isfinite(gam0)
    pid = rasterize([(mapping(x), i+1) for i, x in enumerate(g.geometry) if x is not None],
                    out_shape=ref, transform=rp["transform"], fill=0,
                    dtype=np.int32, all_touched=True)
    inm = np.bincount(pid.ravel(), weights=cov.ravel(), minlength=len(g)+1)[1:] > 0
    obs = rasterize([(mapping(x), 1) for x in g.loc[inm].geometry if x is not None],
                    out_shape=ref, transform=rp["transform"], fill=0,
                    dtype=np.uint8, all_touched=False)
    return obs, pid, int(inm.sum())


# ------------------------------------------------------- 2. the datum evidence
def datum_evidence():
    obs, _, _ = inventory(False)
    a, p = rd(f"{U}/Slope_PRSTM51N.tif")
    rr, cc = np.nonzero(obs == 1)
    rng = np.random.default_rng(0)
    br = rng.integers(0, ref[0], 400_000); bc = rng.integers(0, ref[1], 400_000)
    k = obs[br, bc] == 0; br, bc = br[k], bc[k]
    r0 = (rp['transform'].f - p['transform'].f) / p['transform'].e
    c0_ = (rp['transform'].c - p['transform'].c) / rp['transform'].a

    def contrast(dr, dc):
        R = np.rint(rr + r0 + dr).astype(int); C = np.rint(cc + c0_ + dc).astype(int)
        ok = (R >= 0) & (R < a.shape[0]) & (C >= 0) & (C < a.shape[1])
        v = a[R[ok], C[ok]]; v = v[np.isfinite(v)]
        R2 = np.rint(br + r0 + dr).astype(int); C2 = np.rint(bc + c0_ + dc).astype(int)
        o2 = (R2 >= 0) & (R2 < a.shape[0]) & (C2 >= 0) & (C2 < a.shape[1])
        b = a[R2[o2], C2[o2]]; b = b[np.isfinite(b)]
        return v.mean() - b.mean()

    best = max(((contrast(dr, dc), dr, dc)
                for dr in range(-60, 61, 4) for dc in range(-60, 61, 4)))
    best = max(((contrast(dr, dc), dr, dc)
                for dr in range(best[1]-4, best[1]+5) for dc in range(best[2]-4, best[2]+5)))
    E, N = 734150., 765500.
    dE, dN = pyproj.Transformer.from_crs(REF_CRS, ALT_CRS, always_xy=True).transform(E, N)
    print("\n=== evidence 1: free 2-D offset fit vs the datum shift ===")
    print(f"  free fit optimum      : {best[2]*5:+7.1f} m E, {-best[1]*5:+6.1f} m N  (contrast {best[0]:.3f})")
    print(f"  PRS92 -> WGS84 at site: {dE-E:+7.1f} m E, {dN-N:+6.1f} m N  (zero free parameters)")

    print("\n=== evidence 2: the datum transform recovers the free fit ===")
    sl = np.full(ref, np.nan, F)
    reproject(a, sl, src_transform=p['transform'], src_crs=p['crs'],
              dst_transform=rp['transform'], dst_crs=rp['crs'],
              resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    for lab, g in (("inventory as labelled  ", base), ("inventory -> PRS92     ", corrected)):
        o = rasterize([(mapping(x), 1) for x in g.geometry if x is not None],
                      out_shape=ref, transform=rp["transform"], fill=0,
                      dtype=np.uint8, all_touched=False)
        mk = np.isfinite(sl); ls = sl[mk & (o == 1)]; bg = sl[mk & (o == 0)]
        pos = np.sort(ls); neg = np.sort(bg)
        le = np.searchsorted(neg, pos, "right"); lt = np.searchsorted(neg, pos, "left")
        auc = 1 - ((len(neg)-le).sum() + 0.5*(le-lt).sum())/(len(pos)*len(neg))
        print(f"  {lab} slope contrast {ls.mean()-bg.mean():6.3f}   AUC(slope alone) {auc:.4f}")
    del sl; gc.collect()

    print("\n=== evidence 3: field GPS (WGS 84 by construction, not used to fit) ===")
    raw = pd.read_excel(f"{U}/Makilala Geotechnical Data.xlsx",
                        sheet_name='GEOTECH MAPUA', header=None)
    d = raw.iloc[2:].copy(); d.columns = [str(h).strip() for h in raw.iloc[1].tolist()]
    d = d[pd.to_numeric(d['LATITUDE'], errors='coerce').notna()].copy()
    ls_sites = d[d['SITE TYPE'].astype(str).str.strip().str.lower() == 'landslide']
    pts = gpd.GeoDataFrame(ls_sites, geometry=[Point(float(x), float(y)) for x, y in
                           zip(ls_sites['LONGITUDE'], ls_sites['LATITUDE'])],
                           crs='EPSG:4326').to_crs(REF_CRS)
    for lab, g in (("as labelled", base), ("-> PRS92   ", corrected)):
        uni = g.geometry.union_all()
        dist = np.array([q.distance(uni) for q in pts.geometry])
        print(f"  inventory {lab}: median {np.median(dist):6.1f} m   <50 m {(dist<50).sum():2d}/{len(dist)}"
              f"   inside {sum(uni.contains(q) for q in pts.geometry)}")


# ---------------------------------------------------------------- 3. the impact
INV = {}


def run(mode, fix, cap, coff, label):
    if fix not in INV:
        INV[fix] = inventory(fix)
    obs, pid, npoly = INV[fix]
    alpha = np.clip(AL("Slope_PRSTM51N.tif", mode), 0.1, 80.).astype(F)
    z = np.clip(AL("Soil Depth.tif", mode), 0.1, cap).astype(F)
    pga = AL("PGA 2019 EQ Events.tif", mode)
    mx = np.nanmax(pga[np.isfinite(pga) & (pga > 0)])
    pg = pga/981. if mx > 100 else (pga/9.81 if mx > 2 else pga)
    kh = np.clip(0.5*pg, 0, 1).astype(F); del pga, pg
    c = (c0 + coff).astype(F); a_ = np.radians(alpha).astype(F)
    ca = np.cos(a_).astype(F); sa = np.sin(a_).astype(F)
    R = (c + (gam - m*9.81)*z*ca*ca*tanphi).astype(F)
    D = (gam*z*sa*ca + kh*gam*z*ca*ca).astype(F)
    v = (np.isfinite(alpha) & (alpha >= 10) & np.isfinite(z) & (z > 0)
         & np.isfinite(gam) & (gam > 9.81) & np.isfinite(tanphi)
         & np.isfinite(c) & np.isfinite(m))
    f = np.full(ref, np.nan, F); f[v] = np.where(D[v] > 0, R[v]/D[v], np.nan)
    sc = v & np.isfinite(f)
    pos = np.sort(f[sc & (obs == 1)]); neg = np.sort(f[sc & (obs == 0)])
    npos, nneg = len(pos), len(neg)
    hr = np.array([np.searchsorted(pos, t, "left")/npos for t in TH])
    far = np.array([np.searchsorted(neg, t, "left")/nneg for t in TH])
    J = hr - far; ib = int(np.nanargmax(J))
    le = np.searchsorted(neg, pos, "right"); lt = np.searchsorted(neg, pos, "left")
    auc = ((nneg-le).sum() + 0.5*(le-lt).sum())/(npos*nneg)
    acc = (hr[ib]*npos + (1-far[ib])*nneg)/(npos+nneg)
    fail = sc & (f < TH[ib])
    ns = np.bincount(pid.ravel(), weights=sc.ravel(), minlength=npoly+2)
    nf = np.bincount(pid.ravel(), weights=(fail & (pid > 0)).ravel(), minlength=npoly+2)
    sp = ns[1:] > 0
    det10 = 100*np.mean(np.divide(nf[1:], np.maximum(ns[1:], 1))[sp] >= 0.10)
    print(f"  {label:<30} n={int(sc.sum()):>9,} poly {int(sp.sum()):>5}  th* {TH[ib]:4.2f}  "
          f"J {J[ib]:.3f}  hit {100*hr[ib]:5.1f}%  FAR {100*far[ib]:5.1f}%  "
          f"acc {100*acc:5.1f}%  AUC {auc:.4f}  det10 {det10:5.1f}%")
    del alpha, z, kh, c, a_, ca, sa, R, D, v, f, sc, pos, neg, fail; gc.collect()


def reproduces_v7():
    """The published maps came from zoom. Confirm before trusting anything else."""
    p = os.path.join(V7, "FoS_map.tif")
    if not os.path.exists(p):
        print("\n(skipping v7 reproduction check: output_maps_v7/FoS_map.tif not found)")
        return
    stored, _ = rd(p)
    for mode in ("zoom", "reproject"):
        alpha = np.clip(AL("Slope_PRSTM51N.tif", mode), 0.1, 80.).astype(F)
        z = np.clip(AL("Soil Depth.tif", mode), 0.1, 5.).astype(F)
        pga = AL("PGA 2019 EQ Events.tif", mode)
        mx = np.nanmax(pga[np.isfinite(pga) & (pga > 0)])
        pg = pga/981. if mx > 100 else (pga/9.81 if mx > 2 else pga)
        kh = np.clip(0.5*pg, 0, 1).astype(F)
        c = (c0 + 10.).astype(F); a_ = np.radians(alpha).astype(F)
        R = (c + (gam - m*9.81)*z*np.cos(a_)**2*tanphi).astype(F)
        D = (gam*z*np.sin(a_)*np.cos(a_) + kh*gam*z*np.cos(a_)**2).astype(F)
        v = (np.isfinite(alpha) & (alpha >= 10) & np.isfinite(z) & (z > 0)
             & np.isfinite(gam) & (gam > 9.81) & np.isfinite(tanphi)
             & np.isfinite(c) & np.isfinite(m))
        f = np.full(ref, np.nan, F); f[v] = np.where(D[v] > 0, R[v]/D[v], np.nan)
        b = v & np.isfinite(f) & np.isfinite(stored)
        d = f[b] - stored[b]
        print(f"  {mode:<10} vs stored FoS_map: RMS {np.sqrt((d**2).mean()):.6f}  "
              f"corr {np.corrcoef(f[b], stored[b])[0,1]:.6f}")
        del alpha, z, pga, pg, kh, c, a_, R, D, v, f, b, d; gc.collect()


if __name__ == "__main__":
    audit()
    field_comparison()
    print("\n=== which alignment produced the published maps? ===")
    reproduces_v7()
    datum_evidence()
    print("\n=== impact at the primary configuration (c'+10, z-cap 5 m) ===")
    run('zoom',      False, 5., 10., "A  as published")
    run('zoom',      True,  5., 10., "B  datum fixed only")
    run('reproject', True,  5., 10., "C  both fixed")
    print("\n=== C: cohesion migration, 5 m cap ===")
    for coff, lab in [(0., 'baseline 3.4'), (3.375, 'x2 6.8'), (5., '+5 8.4'),
                      (10., '+10 13.4'), (15., '+15 18.4'), (20., '+20 23.4')]:
        run('reproject', True, 5., coff, f"C  {lab}")
    print("\n=== C: depth cap at c'+10 ===")
    for cap, lab in [(1e9, '20 m'), (5., '5 m'), (3., '3 m')]:
        run('reproject', True, cap, 10., f"C  z-cap {lab}")
