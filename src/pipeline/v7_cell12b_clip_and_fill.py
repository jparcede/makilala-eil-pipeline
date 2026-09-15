# ============================================================================
# INSERT AFTER CELL 11 (LSI load), BEFORE CELL 13 (sweep)
#
# Two corrections:
#   A. LSI_North Cotabato.shp is PROVINCE-wide. Its extent strictly contains the
#      geotechnical footprint, and 1,306 of 3,140 polygons fall entirely outside
#      the raster. Validation must be reported against the Makilala subset.
#   B. The SoilGrids depth raster covers 64.3% of the footprint. Its gaps are the
#      only binding constraint on the scored domain (they cost 684 polygons and
#      45.5% of steep pixels; gamma, phi, c and m cost nothing). Build an
#      optional gap-filled depth so the coverage effect can be isolated.
# ============================================================================

PIXEL_M2 = abs(ref_profile['transform'].a * ref_profile['transform'].e)
print(f"Pixel: {abs(ref_profile['transform'].a):.1f} m  ({PIXEL_M2:.0f} m^2)")

# ---- A. Clip the inventory to the geotechnical footprint -------------------
covered = np.isfinite(gamma) & np.isfinite(alpha)
poly_id = rasterize(
    [(mapping(g), i + 1) for i, g in enumerate(lsi.geometry) if g is not None],
    out_shape=ref_shape, transform=ref_profile['transform'],
    fill=0, dtype=np.int32, all_touched=True)

n_cov_poly = np.bincount(poly_id.ravel(), weights=covered.ravel(),
                         minlength=len(lsi) + 1)[1:]
in_mak = n_cov_poly > 0
lsi_mak = lsi.loc[in_mak].copy()

print(f"\nInventory:")
print(f"  province file            : {len(lsi):,} polygons")
print(f"  within Makilala footprint: {len(lsi_mak):,} polygons  <- validation set")
print(f"  outside raster extent    : {int((np.bincount(poly_id.ravel(), minlength=len(lsi)+1)[1:] == 0).sum()):,}")
print(f"  touching but uncovered   : {int((~in_mak).sum()) - int((np.bincount(poly_id.ravel(), minlength=len(lsi)+1)[1:] == 0).sum()):,}")
print(f"  ESI levels (Makilala)    : {sorted(lsi_mak['ESI_local'].unique())}")
print(f"  mapped area (Makilala)   : {lsi_mak['area'].sum()/1e6:.2f} km^2 "
      f"of {lsi['area'].sum()/1e6:.2f} km^2 province-wide")

# Rebuild observed / ESI rasters from the Makilala subset only
shapes = [(mapping(g), 1) for g in lsi_mak.geometry if g is not None]
observed = rasterize(shapes, out_shape=ref_shape, transform=ref_profile['transform'],
                     fill=0, dtype=np.uint8, all_touched=True)
esi_rasters = {}
for esi in sorted(lsi_mak['ESI_local'].unique()):
    sub = lsi_mak[lsi_mak['ESI_local'] == esi]
    esi_rasters[int(esi)] = rasterize(
        [(mapping(g), 1) for g in sub.geometry if g is not None],
        out_shape=ref_shape, transform=ref_profile['transform'],
        fill=0, dtype=np.uint8, all_touched=True)
print(f"  rasterized               : {int(np.sum(observed == 1)):,} landslide pixels")

# ---- B. Gap-filled soil depth ---------------------------------------------
# Fill value at the thin end of the 1-5 m translational range. This is the
# CONSERVATIVE direction: through c'/(gamma z cos^2 alpha), thinner soil raises
# FoS, so the fill biases AGAINST predicting failure on the recovered pixels.
Z_FILL_M = 1.2

z_missing = covered & (~np.isfinite(z_raw) | (z_raw <= 0))
z_filled = np.where(z_missing, Z_FILL_M, z_raw)

n_cov = int(covered.sum())
print(f"\nSoil depth:")
print(f"  footprint                : {n_cov:,} px  ({n_cov*PIXEL_M2/1e6:.1f} km^2)")
print(f"  SoilGrids coverage       : {100*np.sum(covered & np.isfinite(z_raw) & (z_raw > 0))/n_cov:.1f}%")
print(f"  gaps filled at {Z_FILL_M} m     : {int(z_missing.sum()):,} px "
      f"({100*z_missing.sum()/n_cov:.1f}% of footprint)")

# Is the gap spatially biased toward steep terrain? This is the question that
# determines whether the coverage loss is benign or concentrated on the hazard.
for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 90)]:
    band = covered & (alpha >= lo) & (alpha < hi)
    if band.sum():
        print(f"    alpha {lo:>2}-{hi:<2} deg        : "
              f"{100*np.sum(band & z_missing)/band.sum():>5.1f}% missing "
              f"({int(band.sum()):>10,} px)")

ls_band = covered & (observed == 1)
print(f"    on landslide pixels    : {100*np.sum(ls_band & z_missing)/max(ls_band.sum(),1):>5.1f}% missing")

# Expose both depth families to the sweep. Rerun cell 13 with USE_FILLED_DEPTH
# True and False to separate the coverage effect from everything else.
USE_FILLED_DEPTH = True
z_base = z_filled if USE_FILLED_DEPTH else z_raw
print(f"\nUSE_FILLED_DEPTH = {USE_FILLED_DEPTH}")
