# ============================================================================
#  HARMONISED POINT-LEVEL VALIDATION
#
#  Replaces Makilala_Point_Validation.ipynb, which used a different parameter
#  scheme from the raster pipeline (a USCS lookup table plus a 1.3x multiplier
#  on cohesion where fines > 15%, neither documented in the manuscript). All
#  parameter estimation here calls the pipeline's own estimate_c(fines, sand),
#  estimate_phi(pi, sand, fines) and estimate_m(w), so the point-level and
#  pixel-level validations describe the same model.
#
#  Two scorings, answering different questions:
#    A. RASTER-AT-POINT  -- sample the FoS raster at each field site. Tests the
#       map as deployed.
#    B. SAMPLE-DIRECT    -- compute FoS from the site's own measured index
#       properties through the same correlations, with slope, depth and PGA
#       still from rasters (no point equivalent exists). Tests the correlations
#       without the interpolation step.
#
#  A vs B isolates what interpolation costs, at the only locations where the
#  ground truth is known.
#
#  Run as a cell after cell 17.
# ============================================================================

import pandas as pd
import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform

FIELD_XLSX = os.path.join(INPUT_DIR, "Makilala Geotechnical Data.xlsx")

df = pd.read_excel(FIELD_XLSX, header=1)
df.columns = [str(c).strip() for c in df.columns]
df = df[df['LATITUDE'].notna()].reset_index(drop=True)

LAT, LON, LABEL = 'LATITUDE', 'LONGITUDE', 'SITE TYPE'
FINES, SANDC, WATER, PI_C = 'Silt/Clay %', 'Sand %', 'WATER CONTENT (%)', 'PI'
BULK = [c for c in df.columns if 'BULK' in c.upper()][0]

print(f"field records: {len(df)}   (bulk column: '{BULK}')")
print(df[LABEL].value_counts().to_string())

# --- locate sites on the analysis grid -------------------------------------
xs, ys = warp_transform('EPSG:4326', ref_profile['crs'],
                        pd.to_numeric(df[LON], errors='coerce').tolist(),
                        pd.to_numeric(df[LAT], errors='coerce').tolist())
rr, cc = rasterio.transform.rowcol(ref_profile['transform'], xs, ys)
rr, cc = np.asarray(rr), np.asarray(cc)

in_grid = ((rr >= 0) & (rr < ref_shape[0]) & (cc >= 0) & (cc < ref_shape[1]))
rr_c, cc_c = np.clip(rr, 0, ref_shape[0] - 1), np.clip(cc, 0, ref_shape[1] - 1)
print(f"  inside grid extent: {int(in_grid.sum())} of {len(df)}")

def at_points(arr):
    out = np.full(len(df), np.nan)
    out[in_grid] = arr[rr_c[in_grid], cc_c[in_grid]]
    return out

# --- A. raster-at-point ----------------------------------------------------
fos_A = at_points(fos_best)
keep_A = in_grid.copy()
keep_A[in_grid] = mask[rr_c[in_grid], cc_c[in_grid]]

# --- B. sample-direct ------------------------------------------------------
alpha_pt = at_points(alpha)
z_pt     = at_points(z_scenarios[BEST_Z])
kh_pt    = at_points(k_h)

num = lambda col: pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)
fines_pt, sand_pt = num(FINES), num(SANDC)
pi_pt, w_pt, g_pt = num(PI_C), num(WATER), num(BULK)

def safe(fn, *args):
    out = np.full(len(df), np.nan)
    for i in range(len(df)):
        vals = [a[i] for a in args]
        if all(np.isfinite(v) for v in vals):
            try:
                out[i] = float(fn(*vals))
            except Exception:
                pass
    return out

c_pt   = safe(estimate_c,   fines_pt, sand_pt)          # (fines, sand)
phi_pt = safe(estimate_phi, pi_pt, sand_pt, fines_pt)   # (pi, sand, fines)
m_pt   = safe(estimate_m,   w_pt)

# primary configuration applies a +10 kPa cohesion offset
c_pt_off = c_pt + 10.0

keep_B = (in_grid & np.isfinite(c_pt_off) & np.isfinite(phi_pt)
          & np.isfinite(m_pt) & np.isfinite(g_pt) & (g_pt > GAMMA_W)
          & np.isfinite(alpha_pt) & (alpha_pt >= ALPHA_MIN)
          & np.isfinite(z_pt) & (z_pt > 0) & np.isfinite(kh_pt))

fos_B = np.full(len(df), np.nan)
if keep_B.any():
    fos_B[keep_B] = compute_FoS(c_pt_off[keep_B], phi_pt[keep_B], alpha_pt[keep_B],
                                m_pt[keep_B], g_pt[keep_B], z_pt[keep_B], kh_pt[keep_B])

# --- scoring ---------------------------------------------------------------
obs = df[LABEL].astype(str).str.strip().str.lower().str.startswith('landslide').to_numpy()

def score(fos, keep, label):
    k = keep & np.isfinite(fos)
    if k.sum() == 0:
        print(f"\n  {label}: no scorable sites")
        return
    pred, y = fos[k] < best_thresh, obs[k]
    tp, fn = int((pred & y).sum()), int((~pred & y).sum())
    fp, tn = int((pred & ~y).sum()), int((~pred & ~y).sum())
    hr  = tp / (tp + fn) if (tp + fn) else np.nan
    far = fp / (fp + tn) if (fp + tn) else np.nan
    print(f"\n  {label}  (n={int(k.sum())}: {int(y.sum())} LS, {int((~y).sum())} non-LS)")
    print(f"    TP {tp}  FN {fn}  FP {fp}  TN {tn}")
    print(f"    hit {100*hr:.1f}%  FAR {100*far:.1f}%  "
          f"precision {100*tp/max(tp+fp,1):.1f}%  "
          f"accuracy {100*(tp+tn)/k.sum():.1f}%  J {hr-far:.3f}")
    print(f"    mean FoS: LS {np.nanmean(fos[k][y]):.3f}, "
          f"non-LS {np.nanmean(fos[k][~y]):.3f}")

print(f"\n  threshold theta* = {best_thresh:.2f}  ({BEST_KEY[0]} / {BEST_KEY[1]})")
score(fos_A, keep_A, "A. raster-at-point")
score(fos_B, keep_B, "B. sample-direct")

# --- what interpolation costs ---------------------------------------------
both = keep_A & keep_B & np.isfinite(fos_A) & np.isfinite(fos_B)
if both.sum() > 1:
    d = fos_B[both] - fos_A[both]
    flip = int(((fos_A[both] < best_thresh) != (fos_B[both] < best_thresh)).sum())
    print(f"\n  sites scorable both ways: {int(both.sum())}")
    print(f"    FoS difference (direct - raster): mean {np.mean(d):+.3f}  "
          f"sd {np.std(d):.3f}  max|d| {np.max(np.abs(d)):.3f}")
    print(f"    Spearman rank correlation: "
          f"{pd.Series(fos_A[both]).corr(pd.Series(fos_B[both]), method='spearman'):.3f}")
    print(f"    classified differently by the two routes: {flip} "
          f"({100*flip/both.sum():.0f}%)")

    print(f"\n  parameter values at sites, measured vs interpolated:")
    for nm, pt, ras in [('fines', fines_pt, fines), ('sand', sand_pt, sand),
                        ('water', w_pt, water), ('gamma', g_pt, gamma)]:
        try:
            rv = at_points(ras)
            k = both & np.isfinite(pt) & np.isfinite(rv)
            if k.sum():
                print(f"    {nm:<6} measured {np.nanmean(pt[k]):7.2f}  "
                      f"interpolated {np.nanmean(rv[k]):7.2f}  "
                      f"mean|diff| {np.nanmean(np.abs(pt[k]-rv[k])):6.2f}")
        except NameError:
            pass
