# ============================================================================
#  UNCERTAINTY PROPAGATION FOR k_c
#  Runnable version. Paste as a new cell AFTER cell 19 (the pinned primary
#  configuration), because it needs `scored` and `best_r` from there.
#
#  Answers Reviewer 2 point 4 using the paper's own closed forms rather than a
#  bolted-on Monte Carlo.
#
#  (A) CONTINUOUS parameters (gamma, z, alpha, phi' via PI): first-order delta
#      method, Var(k_c) = sum_i (dk_c/dx_i)^2 sigma_i^2, sigma from the EBK
#      prediction standard-error rasters.
#  (B) THRESHOLD-ASSIGNED parameters (c' from fines, m from water content):
#      step functions, not differentiable. The underlying continuous variable
#      is treated as Normal(mu, sigma^2) and the class probabilities computed
#      exactly, giving E[.] and Var[.] per pixel.
#
#  ---------------------------------------------------------------------------
#  CHANGES FROM kc_uncertainty_propagation.py, and why it could not run as-is:
#
#  1. `align_to_reference()` did not exist. The comment said "same helper as
#     cell 7", but cell 7 has no such function -- it aligns inline with
#     scipy.ndimage.zoom inside a loop.
#
#  2. zoom() WOULD HAVE BEEN THE WRONG TOOL ANYWAY, and this is the important
#     one. The prediction rasters cell 7 aligns (Mak_*.tif) are already on the
#     reference CRS and extent, so a pure array rescale happens to be correct
#     for them. The standard-error rasters are NOT: they are EPSG:4326 in
#     degrees, ~74 m, spanning about 16.5 x 15.2 km, while the reference grid
#     is ESRI:102457 in metres, 5 m, spanning 28.5 x 18.5 km. zoom() would
#     stretch the smaller footprint across the larger one and misalign the
#     standard errors by kilometres, with no error raised. The helper below
#     reprojects properly.
#
#  3. `fines` and `water` were never bound as names. Cell 9 keeps them in the
#     `aligned` dict and only derives c, m, phi from them.
#
#  4. Missing imports: os, rasterio, the warp helpers.
#
#  ---------------------------------------------------------------------------
#  OPEN PREREQUISITE, unchanged from the original header. These SE rasters are
#  the v1 realisation. The prediction rasters the pipeline actually reads
#  (Mak_*.tif) match none of v1/v2/v3 exactly and sit on a different grid, so
#  the pairing of v1 standard errors with those predictions is not exact.
#  Run-to-run variability is under 16% of SE (ebk_version_check_v4.py), so this
#  is second-order -- but it should be stated, not assumed away.
# ============================================================================

import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.stats import norm

SE_DIR = os.path.join(INPUT_DIR, "standard_error")

SE_FILES = {
    'gamma': 'buk_v1_standarderror.tif',
    'water': 'watercontent_v11standarderror.tif',
    'fines': 'siltclay_v1_standarderror.tif',
    'pi'   : 'PI_v1_standarderror.tif',
}


def align_to_reference(path):
    """Reproject a raster onto the reference grid.

    Not zoom(): these rasters are on a different CRS and a different extent
    from the reference grid. See note 2 in the header.
    """
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float32)
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
        dst = np.full(ref_shape, np.nan, dtype=np.float32)
        reproject(source=a, destination=dst,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref_profile['transform'], dst_crs=ref_profile['crs'],
                  resampling=Resampling.bilinear,
                  src_nodata=np.nan, dst_nodata=np.nan)
    return dst


se = {}
for k, fn in SE_FILES.items():
    se[k] = align_to_reference(os.path.join(SE_DIR, fn))
    # EBK writes 0 where it has no prediction; that is nodata, not certainty.
    se[k] = np.where(se[k] > 0, se[k], np.nan)
    print(f"  SE {k:8s}: mean {np.nanmean(se[k]):.3f}  "
          f"[{np.nanmin(se[k]):.3f}, {np.nanmax(se[k]):.3f}]  "
          f"coverage {100*np.mean(np.isfinite(se[k])):.1f}% of grid")

# The continuous fields behind the two threshold rules (see note 3).
fines = aligned['fines']
water = aligned['water']


# ---------------------------------------------------------------------------
# (B) Threshold-assigned parameters -> class probabilities
# ---------------------------------------------------------------------------
def class_moments(mu, sigma, edges, values):
    """E[X] and Var[X] for a step function of a Normal variable."""
    mu = np.asarray(mu, float); sigma = np.asarray(sigma, float)
    cuts = [norm.cdf((e - mu) / sigma) for e in edges]
    probs = ([cuts[0]] +
             [cuts[i] - cuts[i - 1] for i in range(1, len(cuts))] +
             [1.0 - cuts[-1]])
    ex = sum(p * v for p, v in zip(probs, values))
    ex2 = sum(p * v * v for p, v in zip(probs, values))
    return ex, np.maximum(ex2 - ex ** 2, 0.0), probs


# cohesion from fines content (Das & Sobhan) -- matches estimate_c in cell 5
c_mean, c_var, c_probs = class_moments(
    fines, se['fines'], edges=[5.0, 12.0, 20.0], values=[1.0, 3.0, 6.0, 10.0])

# saturation from water content -- matches estimate_m in cell 5
m_mean, m_var, m_probs = class_moments(
    water, se['water'], edges=[20.0, 35.0, 50.0], values=[0.2, 0.4, 0.6, 0.8])

# --- match the cohesion scenario of the primary configuration ---------------
# class_moments returns the expectation of the BASELINE fines -> c' rule. The
# primary configuration applies a scenario offset on top of it (BEST_KEY[0]).
# Without this, k_c is computed at a different cohesion from everything else in
# the notebook: E[c'] came out at 3.53 kPa against the 13.38 kPa that §7 pins.
# An additive offset leaves the variance untouched; the multiplicative scenario
# scales it by the square.
_coh = BEST_KEY[0]
if   _coh == 'c_baseline': pass
elif _coh == 'c_plus5':    c_mean = c_mean + 5.0
elif _coh == 'c_plus10':   c_mean = c_mean + 10.0
elif _coh == 'c_times2':   c_mean, c_var = 2.0 * c_mean, 4.0 * c_var
else: raise ValueError(f"unhandled cohesion scenario {_coh!r}")

# the deterministic field at the same scenario, for a like-for-like comparison
c_det = {'c_baseline': c, 'c_plus5': c + 5.0,
         'c_times2': c * 2.0, 'c_plus10': c + 10.0}[_coh]

print(f"\n  cohesion scenario: {_coh}")
print(f"\n  E[c'] : mean {np.nanmean(c_mean[scored]):.2f} kPa   "
      f"sd {np.nanmean(np.sqrt(c_var[scored])):.2f}   "
      f"(deterministic assignment: {np.nanmean(c_det[scored]):.2f})")
print(f"  E[m]  : mean {np.nanmean(m_mean[scored]):.3f}   "
      f"sd {np.nanmean(np.sqrt(m_var[scored])):.3f}   "
      f"(deterministic: {np.nanmean(m[scored]):.3f})")

# NB: restrict to pixels where p_assigned is FINITE before taking the fraction.
# The SE rasters are NaN wherever EBK wrote 0, so p_assigned is NaN there too --
# and `nan < 0.5` evaluates to False, not NaN. Without the mask those no-data
# pixels enter the DENOMINATOR as "not below even odds" while the numerator can
# only draw from the pixels where the SE exists, diluting the fraction by the
# coverage ratio (4,754,653 / 7,104,440 = 0.669). That is what produced the 7.4%
# reported in the submitted manuscript; the correct figure on the domain where
# this analysis is defined is 11.1%.
p_assigned = np.nanmax(np.stack(c_probs), axis=0)
_conf = scored & np.isfinite(p_assigned)
print(f"  P(most likely c' class): mean {np.nanmean(p_assigned[_conf]):.2f}, "
      f"below 0.5 on {100*np.mean(p_assigned[_conf] < 0.5):.1f}% of pixels"
      f"  (over {int(_conf.sum()):,} px with valid EBK standard errors)")

# ---------------------------------------------------------------------------
# (A) Delta method over the continuous parameters
# ---------------------------------------------------------------------------
# float32 throughout: at 21.1 M pixels a float64 copy is 169 MB and this block
# holds ~30 of them. See the memory note in the header.
f32 = np.float32
a   = np.radians(alpha).astype(f32)
ph  = np.radians(phi).astype(f32)
ca2 = (np.cos(a) ** 2).astype(f32)
Z   = z_scenarios[BEST_Z].astype(f32)
gamma_  = gamma.astype(f32)
c_mean  = c_mean.astype(f32);  c_var = c_var.astype(f32)
m_mean  = m_mean.astype(f32);  m_var = m_var.astype(f32)

d_c     = (1.0 / (gamma_ * Z * ca2)).astype(f32)
d_gamma = -c_mean / (gamma_ ** 2 * Z * ca2) + (m_mean * GAMMA_W / gamma_ ** 2) * np.tan(ph)
d_z     = (-c_mean / (gamma_ * Z ** 2 * ca2)).astype(f32)
d_m     = (-(GAMMA_W / gamma_) * np.tan(ph)).astype(f32)
d_phi   = ((1.0 - m_mean * GAMMA_W / gamma_) / np.cos(ph) ** 2).astype(f32)      # per radian
d_alpha = (2.0 * c_mean * np.tan(a) / (gamma_ * Z * ca2) - 1.0 / ca2).astype(f32)

# phi' = 35 - 0.1 PI (degrees) on the linear branch of estimate_phi. Where the
# USCS fallback applies, phi does not depend on PI and this overstates sigma.
sigma_phi = np.radians(0.1 * se['pi'])

SIGMA_Z_FRAC = 0.30              # SoilGrids depth, 30% relative -- assumption
SIGMA_ALPHA  = np.radians(2.0)   # DEM-derived slope, 2 degrees -- assumption

var_kc = (d_c ** 2     * c_var
        + d_gamma ** 2 * se['gamma'] ** 2
        + d_m ** 2     * m_var
        + d_phi ** 2   * sigma_phi ** 2
        + d_z ** 2     * (SIGMA_Z_FRAC * Z) ** 2
        + d_alpha ** 2 * SIGMA_ALPHA ** 2)

sd_kc = np.sqrt(var_kc)
kc_mu = (c_mean / (gamma_ * Z * ca2)
         + (1.0 - m_mean * GAMMA_W / gamma_) * np.tan(ph)
         - np.tan(a))

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
s = scored & np.isfinite(sd_kc) & np.isfinite(kc_mu)
print(f"\n  pixels retained: {int(s.sum()):,} of {int(scored.sum()):,} scored "
      f"({100*s.sum()/scored.sum():.1f}%)")
print(f"  k_c    : mean {np.nanmean(kc_mu[s]):.4f}  "
      f"[{np.nanpercentile(kc_mu[s], 5):.4f}, {np.nanpercentile(kc_mu[s], 95):.4f}]")
print(f"  sd(k_c): mean {np.nanmean(sd_kc[s]):.4f}  "
      f"median ratio sd/|k_c| = {np.nanmedian(sd_kc[s]/np.abs(kc_mu[s])):.2f}")
print(f"  pixels where sd(k_c) exceeds |k_c| : "
      f"{100*np.nanmean(sd_kc[s] > np.abs(kc_mu[s])):.1f}%")

# Variance shares are computed one term at a time and reduced immediately, so
# six more full-grid arrays are never resident together.
def _term(name):
    if name == 'c(fines)': return d_c**2 * c_var
    if name == 'gamma'   : return d_gamma**2 * se['gamma']**2
    if name == 'm(water)': return d_m**2 * m_var
    if name == 'phi(PI)' : return d_phi**2 * sigma_phi**2
    if name == 'z'       : return d_z**2 * (SIGMA_Z_FRAC*Z)**2
    if name == 'alpha'   : return d_alpha**2 * SIGMA_ALPHA**2
denom = np.nanmean(var_kc[s])
shares = {}
for _n in ('c(fines)','gamma','m(water)','phi(PI)','z','alpha'):
    _t = _term(_n); shares[_n] = 100*np.nanmean(_t[s])/denom; del _t
print("\n  variance share of k_c:")
for k, v in sorted(shares.items(), key=lambda kv: -kv[1]):
    print(f"    {k:10s} {v:>5.1f}%")

p_fail = norm.cdf((k_h - kc_mu) / sd_kc)
p_fail = np.where(s, p_fail, np.nan)
print(f"\n  P(k_c < k_h): mean {np.nanmean(p_fail[s]):.3f}")
print(f"    on landslide pixels : {np.nanmean(p_fail[s & (observed==1)]):.3f}")
print(f"    on stable pixels    : {np.nanmean(p_fail[s & (observed==0)]):.3f}")

pv, yv = p_fail[s], (observed == 1)[s]
pos, neg = np.sort(pv[yv]), np.sort(pv[~yv])
lo = np.searchsorted(neg, pos, 'left'); hi = np.searchsorted(neg, pos, 'right')
auc_p = ((hi - lo).sum() * 0.5 + (lo).sum()) / (len(pos) * len(neg))
print(f"  AUC of P(k_c < k_h)  : {auc_p:.4f}   "
      f"(deterministic FoS ranking: {best_r['auc']:.4f})")
