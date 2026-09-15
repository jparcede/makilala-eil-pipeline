# ============================================================================
# REPLACES CELL 17 -- final validation at the pinned primary configuration
#
# Produces the four things still missing before the Discussion can be written:
#   1. Confusion matrix at the primary configuration  (replaces draft Table 7)
#   2. ESI-stratified hit rate with POLYGON-based CIs (replaces draft Table 8)
#   3. Polygon-level detection rate
#   4. Spatially blocked AUC + out-of-sample theta*   (new; answers the
#      autocorrelation and threshold-fitting objections)
#
# Blocking uses square tiles rather than barangays: no administrative raster is
# needed, block size is tunable, and tiles are the standard construction for
# spatial cross-validation in susceptibility mapping.
#
# ---------------------------------------------------------------------------
# REVISED 26 Aug 2026. Three changes, all in parts 2 and 3:
#
#   (a) The ESI interval denominator was n = ALL polygons at that ESI within
#       Makilala. But the hit rate h is computed over SCORED pixels only, and
#       just 1,002 of 1,699 polygons contribute any. Counting polygons that
#       contribute nothing to the numerator shrinks se and makes the intervals
#       too narrow. n is now the number of polygons at that ESI carrying at
#       least one scored pixel. Both counts are printed so the change is
#       auditable.
#
#   (b) Adjacent-ESI interval separation is now computed and printed rather
#       than eyeballed off the table. Under (a) NO adjacent pair separates,
#       so the "disjoint confidence intervals" claim cannot be made. The
#       defensible claim is monotonic increase across ESI 6-8.
#
#   (c) Polygon attribution uses poly_id, built with all_touched=True, while
#       `observed` is deliberately rasterized centre-only (cell 11). That
#       asymmetry is the whole difference between 1,002 and 993 scored
#       polygons, and between 75.9% and 75.2% detection. Both are now
#       reported side by side so the methods can state which is used and why.
#       This supersedes the standalone polygon-detection cell, which looped
#       over the province-wide `lsi` (3,140) and rasterized the full grid
#       12,560 times to compute the centre-only column below.
#
# NOTE on the interval construction generally: se = sqrt(h(1-h)/n_poly) uses a
# PIXEL-level rate h with a POLYGON count n, as a deliberately conservative
# effective sample size. Polygons differ greatly in size (113 px mean among
# scored), so a properly clustered interval would be wider again. Whichever is
# adopted, Section 4 must say so explicitly.
# ============================================================================

BEST_KEY   = ('c_plus10', 'z_cap_5m')
BLOCK_PX   = 400          # 400 px x 5 m = 2 km tiles
MIN_BLOCK  = 200          # min positives AND negatives for a block to be scored

best_r      = cohesion_results[BEST_KEY]
BEST_Z      = BEST_KEY[1]
best_thresh = best_r['best_thresh']
fos_best    = best_r['fos']
scored      = best_r['scored']

print("=" * 92)
print(f"PRIMARY CONFIGURATION: {BEST_KEY[0]} / {BEST_KEY[1]}")
print("=" * 92)
print(f"  theta* {best_thresh:.2f}   J {best_r['best_youden']:.3f}   "
      f"pooled AUC {best_r['auc']:.4f}")
print(f"  c_mean {best_r['c_mean']:.2f} kPa   z_mean {best_r['z_mean']:.2f} m   "
      f"scored {best_r['n_scored']:,} px "
      f"({best_r['n_scored']*PIXEL_M2/1e6:.1f} km^2)")

fail_best = np.where(scored, np.where(fos_best < best_thresh, 1.0, 0.0), np.nan)
mask = scored
y    = (observed == 1)

# ---- 1. Confusion matrix ---------------------------------------------------
tp = int(np.sum(mask &  y & (fail_best == 1)))
fn = int(np.sum(mask &  y & (fail_best == 0)))
fp = int(np.sum(mask & ~y & (fail_best == 1)))
tn = int(np.sum(mask & ~y & (fail_best == 0)))
hr, far = tp / (tp + fn), fp / (fp + tn)

print(f"\n  {'':<20}{'Pred. failure':>16}{'Pred. stable':>16}")
print(f"  {'Observed landslide':<20}{tp:>16,}{fn:>16,}")
print(f"  {'Observed stable':<20}{fp:>16,}{tn:>16,}")
print(f"\n  hit {100*hr:.1f}%   FAR {100*far:.1f}%   "
      f"prec {100*tp/max(tp+fp,1):.1f}%   acc {100*(tp+tn)/(tp+fn+fp+tn):.1f}%   "
      f"J {hr-far:.3f}")

# ---- polygon-level bookkeeping (needed by BOTH parts 2 and 3) --------------
# poly_id is all_touched=True (cell 12): a polygon claims every cell it
# touches. Restricting to `mask` leaves only its scored cells.
pid_scored = np.where(mask, poly_id, 0)
n_px  = np.bincount(pid_scored.ravel(), minlength=len(lsi) + 1)[1:]
n_fal = np.bincount(pid_scored.ravel(),
                    weights=np.where(mask & (fail_best == 1), 1.0, 0.0).ravel(),
                    minlength=len(lsi) + 1)[1:]
has = n_px > 0                       # polygon carries >= 1 scored pixel

# ESI label per polygon, indexed over the full province-wide lsi so that it
# aligns with poly_id. -1 marks polygons outside the Makilala footprint.
esi_of_poly = np.full(len(lsi), -1, dtype=int)
esi_of_poly[np.where(in_mak)[0]] = lsi_mak['ESI_local'].to_numpy().astype(int)

# ---- 2. ESI-stratified, CI on the SCORED polygon count ---------------------
print(f"\n{'ESI':>4} {'poly':>6} {'scored':>7} {'LS_all':>10} {'LS_scored':>10} "
      f"{'TP':>8} {'hit':>7} {'95% CI':>16}")
print("-" * 84)
ci = {}
for esi in sorted(esi_rasters.keys()):
    oe = esi_rasters[esi] == 1
    at_esi   = (esi_of_poly == esi)
    npoly    = int(at_esi.sum())              # all polygons at this ESI
    npoly_sc = int((at_esi & has).sum())      # those carrying scored pixels
    n_all = int(oe.sum())
    t = int(np.sum(mask & oe & (fail_best == 1)))
    f = int(np.sum(mask & oe & (fail_best == 0)))
    h = t / (t + f) if (t + f) else 0.0
    se = (h * (1 - h) / npoly_sc) ** 0.5 if npoly_sc else 0.0
    lo, hi = max(0.0, h - 1.96 * se), min(1.0, h + 1.96 * se)
    ci[esi] = (lo, hi, h, npoly_sc)
    print(f"{esi:>4} {npoly:>6} {npoly_sc:>7} {n_all:>10,} {t+f:>10,} {t:>8,} "
          f"{100*h:>6.1f}% [{100*lo:>5.1f},{100*hi:>5.1f}]")

# Does the trend actually separate? Compute it rather than eyeball it.
lv = sorted(ci)
print("\n  adjacent-ESI interval separation:")
for a, b in zip(lv, lv[1:]):
    gap = ci[b][0] - ci[a][1]
    verdict = (f"disjoint by {100*gap:.2f} pp" if gap > 0
               else f"OVERLAP by {-100*gap:.2f} pp")
    print(f"    ESI {a} -> {b}: {100*ci[a][2]:>5.1f}% -> {100*ci[b][2]:>5.1f}%   {verdict}")
mono = all(ci[a][2] <= ci[b][2] for a, b in zip(lv, lv[1:]))
mono68 = all(ci[a][2] <= ci[b][2] for a, b in zip([6, 7], [7, 8]) if a in ci and b in ci)
print(f"    monotonic across all ESI: {mono}   across 6-8: {mono68}")
print("    -> claim monotonicity only; do NOT claim disjoint intervals unless"
      " every pair above reads 'disjoint'.")

# ---- 3. Polygon-level detection -------------------------------------------
# Centre-only attribution reported alongside, because `observed` is rasterized
# centre-only and the two give different denominators.
pid_centre = rasterize(
    [(mapping(g), i + 1) for i, g in enumerate(lsi.geometry) if g is not None],
    out_shape=ref_shape, transform=ref_profile['transform'],
    fill=0, dtype=np.int32, all_touched=False)
pidc_scored = np.where(mask, pid_centre, 0)
n_px_c  = np.bincount(pidc_scored.ravel(), minlength=len(lsi) + 1)[1:]
n_fal_c = np.bincount(pidc_scored.ravel(),
                      weights=np.where(mask & (fail_best == 1), 1.0, 0.0).ravel(),
                      minlength=len(lsi) + 1)[1:]
has_c = n_px_c > 0

print(f"\n  polygons with >=1 scored pixel")
print(f"    all_touched=True  (used) : {int(has.sum()):,} of {len(lsi_mak):,} in Makilala")
print(f"    centre-only              : {int(has_c.sum()):,}"
      f"   (difference {int(has.sum() - has_c.sum())} polygons)")
print(f"\n  {'DETECT_FRAC':>12} {'all_touched':>14} {'centre-only':>14}")
for fr in (0.01, 0.10, 0.25, 0.50):
    d  = int(np.sum((n_fal[has]   / n_px[has])     >= fr))
    dc = int(np.sum((n_fal_c[has_c] / n_px_c[has_c]) >= fr))
    print(f"  {fr:>11.0%} {d:>7,} ({100*d/has.sum():>4.1f}%) "
          f"{dc:>7,} ({100*dc/has_c.sum():>4.1f}%)")

# ---- 4. Spatial block cross-validation ------------------------------------
rows, cols = np.indices(ref_shape)
block = (rows // BLOCK_PX) * (ref_shape[1] // BLOCK_PX + 1) + (cols // BLOCK_PX)

def auc_of(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    pos, neg = np.sort(pos), np.sort(neg)
    lo = np.searchsorted(neg, pos, side='left')
    hi = np.searchsorted(neg, pos, side='right')
    return ((len(neg) - hi).sum() + 0.5 * (hi - lo).sum()) / (len(pos) * len(neg))

bid  = block[mask]
fv   = fos_best[mask]
yv   = y[mask]
uniq = np.unique(bid)

per_block, kept = [], []
for b in uniq:
    s = bid == b
    p, n = fv[s & yv], fv[s & ~yv]
    if len(p) >= MIN_BLOCK and len(n) >= MIN_BLOCK:
        per_block.append(auc_of(p, n))
        kept.append(b)
per_block = np.array(per_block)

print(f"\n  Spatial blocks: {BLOCK_PX*5/1000:.1f} km tiles, "
      f"{len(kept)} of {len(uniq)} scored (>= {MIN_BLOCK} px per class)")
print(f"    pooled AUC          : {best_r['auc']:.4f}")
print(f"    within-block AUC    : {np.nanmean(per_block):.4f} "
      f"+/- {np.nanstd(per_block):.4f}   "
      f"[{np.nanmin(per_block):.3f}, {np.nanmax(per_block):.3f}]")

# Leave-one-block-out: theta* chosen on the other blocks, scored on the held-out
oos = []
for b in kept:
    tr, te = ~(bid == b), (bid == b)
    ptr, ntr = np.sort(fv[tr & yv]), np.sort(fv[tr & ~yv])
    js = [(np.searchsorted(ptr, t, 'left') / len(ptr)) -
          (np.searchsorted(ntr, t, 'left') / len(ntr)) for t in THRESHOLDS]
    th = THRESHOLDS[int(np.nanargmax(js))]
    pte, nte = fv[te & yv], fv[te & ~yv]
    h = np.mean(pte < th); fa = np.mean(nte < th)
    oos.append((th, h, fa, h - fa))
oos = np.array(oos)

print(f"    out-of-sample theta*: median {np.median(oos[:,0]):.2f} "
      f"[{oos[:,0].min():.2f}, {oos[:,0].max():.2f}]")
print(f"    out-of-sample hit   : {100*oos[:,1].mean():.1f}% "
      f"+/- {100*oos[:,1].std():.1f}%")
print(f"    out-of-sample FAR   : {100*oos[:,2].mean():.1f}% "
      f"+/- {100*oos[:,2].std():.1f}%")
print(f"    out-of-sample J     : {oos[:,3].mean():.3f} +/- {oos[:,3].std():.3f}"
      f"   (in-sample {best_r['best_youden']:.3f})")
