"""
regen_figs_v9.py -- regenerate Figs 6 and 7 under the v9 registration.

Run from anywhere:   python regen_figs_v9.py

WHAT IT PRODUCES
  companion_figures/cprime_sensitivity.png            (manuscript Fig. 6)
  companion_figures/decoupling_threshold_migration.png (manuscript Fig. 7)

WHY THESE TWO FIRST
  Both are still v7 artifacts and both are quoted directly in the text.
  Fig. 6 panel (c) carries the ESI-by-cohesion-scenario bars, which are the
  only numbers in Section 5.7 that could not be verified from the v9 notebook
  output -- Section 7 prints ESI at the primary configuration only. Running
  this both regenerates the figure and confirms (or corrects) that paragraph.
  Fig. 7 plots dtheta*/dc-bar, which changed from 0.010/0.040/0.065 kPa^-1 to
  0.0204/0.0355/0.0516 under the corrected registration.

IT ALSO PRINTS
  - the twelve-configuration sweep, to cross-check Table 2
  - ESI hit rates per cohesion scenario, for Section 5.7 panel (c)
  - dtheta*/dc-bar fits per depth cap, for Section 5.9
  - parameter values at the three Sobol reference sites, needed before
    Figs 5 and 8 can be regenerated

REGISTRATION
  Identical to Makilala_FoS_Pipeline_v9.ipynb section 3: external layers are
  reprojected, the Mak_* rasters pass through, and GRID_CRS / LSI_CRS below are
  the only lines naming a CRS. Change them here if Segales reports otherwise.

MEMORY
  One configuration is resident at a time; full-grid arrays are float32 and
  freed after each. Peak is well under the notebook's section 6.
"""
import os, gc, warnings
import numpy as np
warnings.filterwarnings("ignore")
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from rasterio.features import rasterize
from shapely.geometry import mapping
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- config
INPUT_DIR = os.environ.get("MAKILALA_DIR", "data")
FIG_DIR = os.path.join(INPUT_DIR, "overleaf_edited", "companion_figures")

GRID_CRS = "EPSG:32651"      # what the Mak_*.tif reference grid actually is
LSI_CRS  = "EPSG:32651"      # what LSI_North Cotabato.shp actually is
CRS_OVERRIDE = {'slope': None, 'depth': None, 'pga': None}

ALPHA_MIN, GAMMA_W = 10, 9.81
THRESHOLDS = np.arange(0.40, 2.005, 0.05)
F = np.float32

os.makedirs(FIG_DIR, exist_ok=True)


def rd(fn):
    with rasterio.open(os.path.join(INPUT_DIR, fn)) as s:
        a = s.read(1).astype(F)
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        return a, s.profile.copy()


gam0, prof = rd("Mak_BulkUnitWeight.tif")
REF = gam0.shape
RT = prof["transform"]
RC = CRS.from_string(GRID_CRS)
print(f"reference grid {REF[0]}x{REF[1]} @ {RT.a:g} m, read as {RC}")


def align(fn, key=None):
    """v9 convention: pass through if already on the grid, else reproject."""
    a, p = rd(fn)
    if a.shape == REF and p["transform"].almost_equals(RT, precision=1e-6):
        return a
    src = p["crs"]
    ov = CRS_OVERRIDE.get(key)
    if ov is not None:
        src = CRS.from_string(ov)
    d = np.full(REF, np.nan, F)
    reproject(a, d, src_transform=p["transform"], src_crs=src,
              dst_transform=RT, dst_crs=RC,
              resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    print(f"  reprojected {fn} from {src}")
    return d


alpha = np.clip(align("Slope_PRSTM51N.tif", 'slope'), 0.1, 80.).astype(F)
z_raw = align("Soil Depth.tif", 'depth')
pga = align("PGA 2019 EQ Events.tif", 'pga')
gam = np.clip(gam0, GAMMA_W + 0.1, 30.).astype(F)
pi = align("Mak_PlasticityIndex.tif")
sand = align("Mak_Sand.tif")
fines = align("Mak_Fines.tif")
water = align("Mak_WaterContent.tif")

mx = np.nanmax(pga[np.isfinite(pga) & (pga > 0)])
pg = pga / 981. if mx > 100 else (pga / 9.81 if mx > 2 else pga)
k_h = np.clip(0.5 * pg, 0, 1).astype(F)
del pga, pg, gam0
gc.collect()

# ---------------------------------------------------------------- derived fields
phi = np.full_like(pi, 30.)
v_ = (pi > 5) & np.isfinite(pi)
phi[v_] = 35. - 0.1 * pi[v_]
no = ~v_ & np.isfinite(sand)
phi[no & (sand > 85) & (fines < 10)] = 33.
phi[no & (sand > 70) & (fines >= 10) & (fines < 20)] = 31.
phi[no & (fines >= 20)] = 28.
phi = np.clip(phi, 20., 38.).astype(F)
tanphi = np.tan(np.radians(phi)).astype(F)

c_base = np.full_like(fines, 5.)
vv = np.isfinite(fines) & np.isfinite(sand)
c_base[vv & (fines < 5)] = 1.
c_base[vv & (fines >= 5) & (fines < 12)] = 3.
c_base[vv & (fines >= 12) & (fines < 20)] = 6.
c_base[vv & (fines >= 20)] = 10.
c_base = c_base.astype(F)

m = np.full_like(water, 0.5)
w_ = np.isfinite(water) & (water > 0)
m[w_ & (water < 20)] = 0.2
m[w_ & (water >= 20) & (water < 35)] = 0.4
m[w_ & (water >= 35) & (water < 50)] = 0.6
m[w_ & (water >= 50)] = 0.8
m = m.astype(F)
del pi, sand, water
gc.collect()

# ---------------------------------------------------------------- inventory
lsi = gpd.read_file(os.path.join(INPUT_DIR, "LSI_North Cotabato.shp"))
lsi = lsi.set_crs(LSI_CRS, allow_override=True).to_crs(RC)
covered = np.isfinite(gam) & np.isfinite(alpha)
pid = rasterize([(mapping(g), i + 1) for i, g in enumerate(lsi.geometry) if g is not None],
                out_shape=REF, transform=RT, fill=0, dtype=np.int32, all_touched=True)
inmak = np.bincount(pid.ravel(), weights=covered.ravel(), minlength=len(lsi) + 1)[1:] > 0
lsi_mak = lsi.loc[inmak].copy()
observed = rasterize([(mapping(g), 1) for g in lsi_mak.geometry if g is not None],
                     out_shape=REF, transform=RT, fill=0, dtype=np.uint8, all_touched=False)
esi_r = {}
for e in sorted(lsi_mak["ESI_local"].unique()):
    sub = lsi_mak[lsi_mak["ESI_local"] == e]
    esi_r[int(e)] = rasterize([(mapping(g), 1) for g in sub.geometry if g is not None],
                              out_shape=REF, transform=RT, fill=0, dtype=np.uint8,
                              all_touched=False)
del pid, covered
gc.collect()
print(f"inventory: {len(lsi_mak)} polygons in footprint, "
      f"{int((observed == 1).sum()):,} landslide px, ESI {sorted(esi_r)}")

# ---------------------------------------------------------------- sweep
COH = [("baseline",  0.0,   "$c'_{\\mathrm{baseline}}$", "#C0392B"),
       ("c_times2",  None,  "$c' \\times 2$",            "#2E86C1"),
       ("c_plus5",   5.0,   "$c' + 5$ kPa",              "#E67E22"),
       ("c_plus10", 10.0,   "$c' + 10$ kPa",             "#1E8449")]
# z_original is capped at 20 m, matching the notebook: leaving it uncapped gives
# z_mean 11.80 against the notebook's 11.48 and shifts the 1/z prediction from
# 3.85 to 3.97. theta* is unaffected either way.
CAPS = [("z_original", 20.0, "20 m"), ("z_cap_5m", 5.0, "5 m"), ("z_cap_3m", 3.0, "3 m")]

a_ = np.radians(alpha).astype(F)
ca = np.cos(a_).astype(F)
sa = np.sin(a_).astype(F)
del a_
gc.collect()


def evaluate(coff, cap):
    c = (c_base * 2.0 if coff is None else c_base + coff).astype(F)
    z = np.clip(z_raw, 0.1, cap).astype(F)
    R = (c + (gam - m * GAMMA_W) * z * ca * ca * tanphi).astype(F)
    D = (gam * z * sa * ca + k_h * gam * z * ca * ca).astype(F)
    v = (np.isfinite(alpha) & (alpha >= ALPHA_MIN) & np.isfinite(z) & (z > 0)
         & np.isfinite(gam) & (gam > GAMMA_W) & np.isfinite(tanphi)
         & np.isfinite(c) & np.isfinite(m))
    f = np.full(REF, np.nan, F)
    f[v] = np.where(D[v] > 0, R[v] / D[v], np.nan)
    sc = v & np.isfinite(f)
    pos = np.sort(f[sc & (observed == 1)])
    neg = np.sort(f[sc & (observed == 0)])
    npos, nneg = len(pos), len(neg)
    hr = np.array([np.searchsorted(pos, t, "left") / npos for t in THRESHOLDS])
    far = np.array([np.searchsorted(neg, t, "left") / nneg for t in THRESHOLDS])
    J = hr - far
    ib = int(np.nanargmax(J))
    le = np.searchsorted(neg, pos, "right")
    lt = np.searchsorted(neg, pos, "left")
    auc = ((nneg - le).sum() + 0.5 * (le - lt).sum()) / (npos * nneg)
    fail = sc & (f < THRESHOLDS[ib])
    esi_hit = {}
    for e, r_ in esi_r.items():
        oe = (r_ == 1) & sc
        esi_hit[e] = 100 * np.sum(oe & fail) / max(int(oe.sum()), 1)
    out = dict(hr=hr, far=far, J=J, ib=ib, th=float(THRESHOLDS[ib]), auc=float(auc),
               cmean=float(np.nanmean(c[sc])), zmean=float(np.nanmean(z[sc])),
               fosmean=float(np.nanmean(f[sc])), n=int(sc.sum()), esi=esi_hit,
               censored=bool(ib in (0, len(THRESHOLDS) - 1)))
    del c, z, R, D, v, f, sc, pos, neg, fail
    gc.collect()
    return out


print("\n" + "=" * 96)
print("TWELVE-CONFIGURATION SWEEP  (cross-check against Table 2)")
print("=" * 96)
print(f"{'cohesion':<12}{'z-cap':<8}{'c_mn':>7}{'z_mn':>7}{'FoS_mn':>8}"
      f"{'theta*':>8}{'J':>7}{'hit':>8}{'FAR':>8}{'AUC':>8}  cens")
GRID = {}
for cn, coff, _, _ in COH:
    for zn, cap, zl in CAPS:
        r = evaluate(coff, cap)
        GRID[(cn, zn)] = r
        print(f"{cn:<12}{zl:<8}{r['cmean']:>7.2f}{r['zmean']:>7.2f}{r['fosmean']:>8.3f}"
              f"{r['th']:>8.2f}{r['J'][r['ib']]:>7.3f}{100*r['hr'][r['ib']]:>7.1f}%"
              f"{100*r['far'][r['ib']]:>7.1f}%{r['auc']:>8.4f}  {'YES' if r['censored'] else 'no'}")

# ---------------------------------------------------------------- Fig 6
res5 = [(lbl, col, GRID[(cn, "z_cap_5m")]) for cn, _, lbl, col in COH]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))

ax = axes[0]
for lbl, col, r in res5:
    ax.plot(r["far"], r["hr"], "-", color=col, lw=1.8,
            label=f"{lbl}  ($\\theta^*$={r['th']:.2f}, AUC={r['auc']:.3f})")
    ax.plot(r["far"][r["ib"]], r["hr"][r["ib"]], "*", color=col, ms=16, zorder=5)
ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.6, label="Random")
ax.set_xlim(0, 0.8); ax.set_ylim(0.3, 1.0)
ax.set_xlabel("False Alarm Rate"); ax.set_ylabel("Hit Rate")
ax.set_title("(a) ROC curves by $c'$ scenario ($z$-cap 5 m)")
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)

ax = axes[1]
for lbl, col, r in res5:
    ax.plot(THRESHOLDS, r["J"], "-", color=col, lw=1.8, label=lbl)
    ax.axvline(r["th"], color=col, ls=":", alpha=0.55)
ax.axvline(1.00, color="k", ls="--", lw=1.1, label="FoS = 1.0")
ax.axvline(1.25, color="k", ls="-.", lw=1.1, alpha=0.6, label="FoS = 1.25 (engineering)")
ax.set_xlim(0.40, 2.00)
ax.set_xlabel("FoS Threshold"); ax.set_ylabel("Youden Index (HR $-$ FAR)")
ax.set_title("(b) Optimal threshold shifts with $c'$")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[2]
lv = sorted(esi_r); x = np.arange(len(lv)); w = 0.2
for i, (lbl, col, r) in enumerate(res5):
    ax.bar(x + (i - 1.5) * w, [r["esi"][e] for e in lv], w, color=col, label=lbl)
ax.set_xticks(x); ax.set_xticklabels([f"ESI {e}" for e in lv])
ax.set_ylabel("Hit Rate (%)"); ax.set_ylim(0, 100)
ax.set_title("(c) ESI-stratified hit rates by $c'$ scenario")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
p6 = os.path.join(FIG_DIR, "cprime_sensitivity.png")
plt.savefig(p6, dpi=200, bbox_inches="tight"); plt.close()
print(f"\nwrote {p6}")

print("\n" + "=" * 96)
print("PANEL (c): ESI hit rate by cohesion scenario, z-cap 5 m  -> Section 5.7")
print("=" * 96)
print("  " + "scenario".ljust(22) + "".join(f"ESI {e}".rjust(9) for e in lv))
for lbl, _, r in res5:
    nm = lbl.replace("$", "").replace("\\mathrm", "").replace("{", "").replace("}", "").replace("\\times", "x")
    print("  " + nm.ljust(22) + "".join(f"{r['esi'][e]:8.1f}%" for e in lv))

# ---------------------------------------------------------------- Fig 7
fig, ax = plt.subplots(figsize=(7.2, 5.2))
COLC = {"z_original": "#7D3C98", "z_cap_5m": "#1E8449", "z_cap_3m": "#B7950B"}
print("\n" + "=" * 96)
print("dtheta*/dc-bar BY DEPTH CAP  -> Section 5.9")
print("=" * 96)
fits = {}
for zn, cap, zl in CAPS:
    cs = np.array([GRID[(cn, zn)]["cmean"] for cn, _, _, _ in COH])
    th = np.array([GRID[(cn, zn)]["th"] for cn, _, _, _ in COH])
    o = np.argsort(cs); cs, th = cs[o], th[o]
    sl, ic = np.polyfit(cs, th, 1)
    r2 = np.corrcoef(cs, th)[0, 1] ** 2
    fits[zn] = (sl, ic, r2, GRID[(COH[0][0], zn)]["zmean"])
    ax.plot(cs, th, "o-", color=COLC[zn], lw=1.8, ms=7,
            label=f"$z$-cap {zl}:  $d\\theta^*/d\\bar{{c}}'$ = {sl:.4f} kPa$^{{-1}}$")
    xx = np.linspace(cs.min(), 20, 50)
    ax.plot(xx, sl * xx + ic, ":", color=COLC[zn], lw=1.1, alpha=0.7)
    print(f"  {zl:<6} dtheta*/dc = {sl:.4f} /kPa   R2 {r2:.3f}   z_mean {fits[zn][3]:.2f} m"
          f"   -> theta* at 15/20 kPa: {sl*15+ic:.2f} / {sl*20+ic:.2f}")
ax.axhline(1.00, color="k", ls="--", lw=1.1, alpha=0.8)
ax.axhspan(1.00, 1.25, color="k", alpha=0.06)
ax.text(19.6, 1.005, "FoS = 1.0", ha="right", va="bottom", fontsize=8)
ax.text(19.6, 1.245, "engineering band", ha="right", va="top", fontsize=8, alpha=0.7)
ax.axvspan(15, 20, color="grey", alpha=0.10)
ax.text(17.5, 0.45, "laboratory-realistic\n$c'$ range", ha="center", fontsize=8, alpha=0.75)
ax.set_xlabel("Mean effective cohesion $\\bar{c}'$ (kPa)")
ax.set_ylabel("Youden-optimal threshold $\\theta^*$")
ax.set_title("Migration of $\\theta^*$ with cohesion, by depth cap")
ax.legend(fontsize=8.5, loc="upper left"); ax.grid(alpha=0.3)
ax.set_xlim(2, 20.5)
plt.tight_layout()
p7 = os.path.join(FIG_DIR, "decoupling_threshold_migration.png")
plt.savefig(p7, dpi=200, bbox_inches="tight"); plt.close()
print(f"\nwrote {p7}")

s20, s5, s3 = fits["z_original"][0], fits["z_cap_5m"][0], fits["z_cap_3m"][0]
z20, z5, z3 = fits["z_original"][3], fits["z_cap_5m"][3], fits["z_cap_3m"][3]
print("\n  gradient ratios vs the 1/z prediction:")
print(f"    3m/20m  measured {s3/s20:.2f}   predicted {z20/z3:.2f}")
print(f"    3m/5m   measured {s3/s5:.2f}   predicted {z5/z3:.2f}")
print(f"    5m/20m  measured {s5/s20:.2f}   predicted {z20/z5:.2f}")

# ------------------------------------------- reference points for Figs 5 and 8
print("\n" + "=" * 96)
print("REFERENCE-POINT PARAMETERS  -> needed before Figs 5 and 8 can be regenerated")
print("=" * 96)
# Coordinates are read from the field spreadsheet, not hardcoded: the three
# reference sites are barangays in the campaign, and their centroid is the only
# defensible stand-in until the co-authors confirm the published pixels.
try:
    import pyproj, pandas as pd
    _raw = pd.read_excel(os.path.join(INPUT_DIR, "Makilala Geotechnical Data.xlsx"),
                         sheet_name="GEOTECH MAPUA", header=None)
    _d = _raw.iloc[2:].copy()
    _d.columns = [str(x).strip() for x in _raw.iloc[1].tolist()]
    _d["LATITUDE"] = pd.to_numeric(_d["LATITUDE"], errors="coerce")
    _d["LONGITUDE"] = pd.to_numeric(_d["LONGITUDE"], errors="coerce")
    _d = _d.dropna(subset=["LATITUDE", "LONGITUDE"])
    _d["BGY"] = _d["BARANGAY"].astype(str).str.strip().str.lower()
    SITES = {}
    for want, key in (("Villaflores", "villaflores"), ("Rodero 2", "rodero 2"),
                      ("Sto. Nino", "sto. nino")):
        g = _d[_d["BGY"] == key]
        if len(g):
            SITES[f"{want} (n={len(g)})"] = (g["LATITUDE"].mean(), g["LONGITUDE"].mean())
        else:
            print(f"  ! barangay '{key}' not found in the field data")
    tf = pyproj.Transformer.from_crs("EPSG:4326", GRID_CRS, always_xy=True)
    zc = np.clip(z_raw, 0.1, 5.0)
    cc = c_base + 10.0
    print(f"  {'site':<20}{'alpha':>8}{'z':>7}{'gamma':>8}{'phi':>8}{'c':>7}{'m':>6}{'k_h':>8}")
    for nm, (la, lo) in SITES.items():
        x, y = tf.transform(lo, la)
        col, row = ~RT * (x, y)
        r, c = int(row), int(col)
        if not (0 <= r < REF[0] and 0 <= c < REF[1]):
            print(f"  {nm:<20} outside the grid"); continue
        print(f"  {nm:<20}{alpha[r,c]:>8.1f}{zc[r,c]:>7.2f}{gam[r,c]:>8.2f}"
              f"{phi[r,c]:>8.2f}{cc[r,c]:>7.1f}{m[r,c]:>6.2f}{k_h[r,c]:>8.4f}")
    print("\n  NOTE: sampled at the centroid of that barangay's field sites, which is")
    print("  almost certainly NOT the pixel the published reference points used --")
    print("  Table 5 gives Villaflores 20.1 deg, Rodero 2 38.3 deg, Sto. Nino 28.8 deg,")
    print("  and a barangay spans a wide range of slope. Figs 5 and 8 cannot be")
    print("  regenerated until a co-author supplies the exact coordinates.")
except ImportError:
    print("  pyproj not available; skipped")

print("\ndone.")
