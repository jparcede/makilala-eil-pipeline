# ============================================================================
#  UNCERTAINTY PROPAGATION FOR k_c  --  new cell, insert after cell 13
#
#  Answers Reviewer 2 point 4 ("parameter uncertainty is ignored ... uncertainty
#  ranges of k_c are needed") using the paper's own closed forms rather than a
#  bolted-on Monte Carlo.
#
#  Two distinct treatments, because the parameters enter differently:
#
#  (A) CONTINUOUS parameters (gamma, z, alpha, and phi' via PI). First-order
#      delta method:  Var(k_c) = sum_i (dk_c/dx_i)^2 * sigma_i^2
#      with sigma_i taken from the EBK prediction standard error rasters.
#
#  (B) THRESHOLD-ASSIGNED parameters (c' from fines, m from water content).
#      These are STEP functions -- not differentiable, so the delta method does
#      not apply. Instead we treat the underlying continuous variable as
#      Normal(mu, sigma^2) from the EBK output and compute the class
#      probabilities exactly, giving E[c'] and Var(c') per pixel.
#
#      This matters: the fines SE is ~4.6 percentage points against class
#      widths of 7-8 points, so the class assignment is genuinely uncertain
#      almost everywhere. The same is true of saturation.
#
#  Outputs per pixel: k_c, sd(k_c), a 95% band, and P(k_c < k_h) -- i.e. a
#  probabilistic hazard map derived analytically.
#
#  PREREQUISITE: confirm with J. Segales that the PREDICTION rasters behind the
#  v7 results are unchanged. He re-interpolated when generating the SE layers;
#  if the predictions also moved, every result in the paper must be recomputed
#  before this cell means anything.
# ============================================================================

import numpy as np
from scipy.stats import norm

SE_DIR = os.path.join(INPUT_DIR, "standard_error")   # adjust to your path

# EBK standard error rasters, aligned to the reference grid exactly as the
# prediction rasters were in cell 7.
SE_FILES = {
    'gamma': 'buk_v1_standarderror.tif',
    'water': 'watercontent_v11standarderror.tif',
    'fines': 'siltclay_v1_standarderror.tif',
    'pi'   : 'PI_v1_standarderror.tif',
}

se = {}
for k, fn in SE_FILES.items():
    se[k] = align_to_reference(os.path.join(SE_DIR, fn))   # same helper as cell 7
    # EBK writes 0 where it has no prediction; that is nodata, not certainty.
    se[k] = np.where(se[k] > 0, se[k], np.nan)
    print(f"  SE {k:8s}: mean {np.nanmean(se[k]):.3f}  "
          f"[{np.nanmin(se[k]):.3f}, {np.nanmax(se[k]):.3f}]")

# ---------------------------------------------------------------------------
# (B) Threshold-assigned parameters -> class probabilities
# ---------------------------------------------------------------------------
def class_moments(mu, sigma, edges, values):
    """E[X] and Var[X] for a step function of a Normal variable.

    edges  : ascending cut points, e.g. [5, 12, 20] for the fines->c' rule
    values : one more value than edges, e.g. [1, 3, 6, 10] kPa

    P(class k) = Phi((edge_k - mu)/sigma) - Phi((edge_{k-1} - mu)/sigma)
    """
    mu = np.asarray(mu, float); sigma = np.asarray(sigma, float)
    cuts = [norm.cdf((e - mu) / sigma) for e in edges]
    probs = ([cuts[0]] +
             [cuts[i] - cuts[i - 1] for i in range(1, len(cuts))] +
             [1.0 - cuts[-1]])
    ex  = sum(p * v for p, v in zip(probs, values))
    ex2 = sum(p * v * v for p, v in zip(probs, values))
    return ex, np.maximum(ex2 - ex ** 2, 0.0), probs

# cohesion from fines content (Das & Sobhan)
c_mean, c_var, c_probs = class_moments(
    fines, se['fines'], edges=[5.0, 12.0, 20.0], values=[1.0, 3.0, 6.0, 10.0])

# saturation from water content
m_mean, m_var, m_probs = class_moments(
    water, se['water'], edges=[20.0, 35.0, 50.0], values=[0.2, 0.4, 0.6, 0.8])

print(f"\n  E[c'] : mean {np.nanmean(c_mean[scored]):.2f} kPa   "
      f"sd {np.nanmean(np.sqrt(c_var[scored])):.2f}   "
      f"(deterministic assignment: {np.nanmean(c[scored]):.2f})")
print(f"  E[m]  : mean {np.nanmean(m_mean[scored]):.3f}   "
      f"sd {np.nanmean(np.sqrt(m_var[scored])):.3f}   "
      f"(deterministic: {np.nanmean(m[scored]):.3f})")

# How often is the deterministic class assignment actually confident?
#
# NB: restrict to pixels where p_assigned is FINITE before taking the fraction.
# The SE rasters are NaN wherever EBK wrote 0 (see above), so p_assigned is NaN
# there too -- and `nan < 0.5` evaluates to False, not NaN. Without the mask
# those no-data pixels enter the DENOMINATOR as "not below even odds" while the
# numerator can only draw from the pixels where the SE exists, diluting the
# fraction by the coverage ratio (4,754,653 / 7,104,440 = 0.669). That is what
# produced the 7.4% reported in the submitted manuscript; the correct figure on
# the domain where this analysis is defined is 11.1%.
p_assigned = np.nanmax(np.stack(c_probs), axis=0)
_conf = scored & np.isfinite(p_assigned)
print(f"  P(most likely c' class): mean {np.nanmean(p_assigned[_conf]):.2f}, "
      f"below 0.5 on {100*np.mean(p_assigned[_conf] < 0.5):.1f}% of pixels"
      f"  (over {int(_conf.sum()):,} px with valid EBK standard errors)")

# ---------------------------------------------------------------------------
# (A) Delta method over the continuous parameters
# ---------------------------------------------------------------------------
# k_c = c'/(gamma z cos^2 a) + (1 - m gamma_w/gamma) tan(phi') - tan(a)
a   = np.radians(alpha)
ph  = np.radians(phi)
ca2 = np.cos(a) ** 2
Z   = z_scenarios[BEST_Z]

d_c     = 1.0 / (gamma * Z * ca2)
d_gamma = -c_mean / (gamma ** 2 * Z * ca2) + (m_mean * GAMMA_W / gamma ** 2) * np.tan(ph)
d_z     = -c_mean / (gamma * Z ** 2 * ca2)
d_m     = -(GAMMA_W / gamma) * np.tan(ph)
d_phi   = (1.0 - m_mean * GAMMA_W / gamma) / np.cos(ph) ** 2      # per radian
d_alpha = 2.0 * c_mean * np.tan(a) / (gamma * Z * ca2) - 1.0 / ca2

# phi' = 35 - 0.1 PI  (degrees) -> sigma_phi in radians
sigma_phi = np.radians(0.1 * se['pi'])

# Soil depth and slope carry no EBK error (different sources). Stated
# assumptions, reported as such -- vary them to test robustness.
SIGMA_Z_FRAC = 0.30      # SoilGrids depth, 30% relative
SIGMA_ALPHA  = np.radians(2.0)   # DEM-derived slope, 2 degrees

var_kc = (d_c ** 2     * c_var
        + d_gamma ** 2 * se['gamma'] ** 2
        + d_m ** 2     * m_var
        + d_phi ** 2   * sigma_phi ** 2
        + d_z ** 2     * (SIGMA_Z_FRAC * Z) ** 2
        + d_alpha ** 2 * SIGMA_ALPHA ** 2)

sd_kc = np.sqrt(var_kc)
kc_mu = (c_mean / (gamma * Z * ca2)
         + (1.0 - m_mean * GAMMA_W / gamma) * np.tan(ph)
         - np.tan(a))

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
s = scored & np.isfinite(sd_kc) & np.isfinite(kc_mu)
print(f"\n  k_c    : mean {np.nanmean(kc_mu[s]):.4f}  "
      f"[{np.nanpercentile(kc_mu[s], 5):.4f}, {np.nanpercentile(kc_mu[s], 95):.4f}]")
print(f"  sd(k_c): mean {np.nanmean(sd_kc[s]):.4f}  "
      f"median ratio sd/|k_c| = {np.nanmedian(sd_kc[s]/np.abs(kc_mu[s])):.2f}")
print(f"  pixels where sd(k_c) exceeds |k_c| : "
      f"{100*np.nanmean(sd_kc[s] > np.abs(kc_mu[s])):.1f}%")

# Variance share per parameter -- the analytical counterpart of the Sobol
# analysis, and unlike it, driven by measured interpolation error.
terms = {'c(fines)': d_c**2 * c_var,
         'gamma'   : d_gamma**2 * se['gamma']**2,
         'm(water)': d_m**2 * m_var,
         'phi(PI)' : d_phi**2 * sigma_phi**2,
         'z'       : d_z**2 * (SIGMA_Z_FRAC*Z)**2,
         'alpha'   : d_alpha**2 * SIGMA_ALPHA**2}
print("\n  variance share of k_c:")
for k, v in sorted(terms.items(), key=lambda kv: -np.nanmean(kv[1][s])):
    print(f"    {k:10s} {100*np.nanmean(v[s])/np.nanmean(var_kc[s]):>5.1f}%")

# Probabilistic failure map: P(k_c < k_h), i.e. shaking exceeds capacity.
p_fail = norm.cdf((k_h - kc_mu) / sd_kc)
p_fail = np.where(s, p_fail, np.nan)
print(f"\n  P(k_c < k_h): mean {np.nanmean(p_fail[s]):.3f}")
print(f"    on landslide pixels : {np.nanmean(p_fail[s & (observed==1)]):.3f}")
print(f"    on stable pixels    : {np.nanmean(p_fail[s & (observed==0)]):.3f}")

# The probabilistic map is itself a classifier -- score it the same way.
pv, yv = p_fail[s], (observed == 1)[s]
pos, neg = np.sort(pv[yv]), np.sort(pv[~yv])
lo = np.searchsorted(neg, pos, 'left'); hi = np.searchsorted(neg, pos, 'right')
auc_p = ((hi - lo).sum() * 0.5 + (lo).sum()) / (len(pos) * len(neg))
print(f"  AUC of P(k_c < k_h)  : {auc_p:.4f}   "
      f"(deterministic FoS ranking: {best_r['auc']:.4f})")
