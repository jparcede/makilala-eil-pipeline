"""
SUPERSEDED -- retained for the record, not used for any published number.

This script blocks the cross-validation by BARANGAY. The results reported in the
paper block by 2 km SQUARE TILES instead (400 px x 5 m), computed in cell 8 of
notebooks/Makilala_FoS_Pipeline_v9.ipynb, which is authoritative. Tiles were
preferred because they need no administrative raster, the block size is tunable,
and they are the standard construction in the spatial cross-validation literature.

Kept here because barangay blocking was tried first and the comparison is
informative; do not cite numbers produced by this file.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

# ----------------------------------------------------------------------
# COLUMN NAMES -- edit to match your dataframe
# ----------------------------------------------------------------------
COL_FOS   = "FoS"            # pseudo-static FoS at the pixel / unit
COL_LABEL = "Landslide"      # 1 = inside a documented polygon, 0 = outside
COL_BLOCK = "Brgy"           # barangay -- the spatial blocking unit
COL_SLOPE = "Slope Angle"    # degrees, for the Keefer filter

SLOPE_MIN     = 10.0         # alpha >= 10 deg
THETA_GRID    = np.arange(0.60, 2.01, 0.05)
MIN_BLOCK_POS = 30           # skip blocks with too few landslide pixels to score


def prepare(df):
    """Apply the slope filter and drop rows with missing values."""
    d = df[[COL_FOS, COL_LABEL, COL_BLOCK, COL_SLOPE]].dropna()
    d = d[d[COL_SLOPE] >= SLOPE_MIN]
    # score must increase with hazard; FoS decreases with hazard
    d = d.assign(score=-d[COL_FOS].to_numpy())
    return d


def pooled_auc(d):
    """AUC over all pixels. Optimistic under spatial autocorrelation -- report
    alongside the blocked estimate, never alone."""
    return roc_auc_score(d[COL_LABEL], d["score"])


def per_block_auc(d):
    """AUC computed within each barangay, then aggregated. Removes the
    between-barangay contrast that inflates the pooled figure."""
    rows = []
    for blk, g in d.groupby(COL_BLOCK):
        pos = int(g[COL_LABEL].sum())
        neg = int((1 - g[COL_LABEL]).sum())
        if pos < MIN_BLOCK_POS or neg < MIN_BLOCK_POS:
            continue
        rows.append({"block": blk, "n": len(g), "pos": pos,
                     "auc": roc_auc_score(g[COL_LABEL], g["score"])})
    return pd.DataFrame(rows)


def youden(y, fos, theta):
    pred = fos < theta
    pos, neg = y == 1, y == 0
    hit = pred[pos].mean() if pos.any() else np.nan
    far = pred[neg].mean() if neg.any() else np.nan
    return hit, far, hit - far


def loo_block_threshold(d):
    """Leave-one-barangay-out: pick theta* on the training barangays, score it
    on the held-out one. The reported metrics are then out-of-sample with
    respect to the only fitted quantity in the pipeline."""
    blocks = sorted(d[COL_BLOCK].unique())
    rows = []
    for blk in blocks:
        tr = d[d[COL_BLOCK] != blk]
        te = d[d[COL_BLOCK] == blk]
        if te[COL_LABEL].sum() < MIN_BLOCK_POS:
            continue
        js = [youden(tr[COL_LABEL].to_numpy(), tr[COL_FOS].to_numpy(), t)[2]
              for t in THETA_GRID]
        theta_star = THETA_GRID[int(np.nanargmax(js))]
        hit, far, j = youden(te[COL_LABEL].to_numpy(),
                             te[COL_FOS].to_numpy(), theta_star)
        rows.append({"held_out": blk, "theta_star": theta_star,
                     "hit": hit, "far": far, "youden": j})
    return pd.DataFrame(rows)


def report(df):
    d = prepare(df)
    print(f"pixels after alpha >= {SLOPE_MIN} deg : {len(d):,}")
    print(f"landslide pixels                     : {int(d[COL_LABEL].sum()):,}")
    print(f"barangays                            : {d[COL_BLOCK].nunique()}\n")

    print(f"Pooled AUC                : {pooled_auc(d):.4f}")

    pb = per_block_auc(d)
    print(f"Per-barangay AUC          : {pb['auc'].mean():.4f} "
          f"+/- {pb['auc'].std():.4f}  (n = {len(pb)} barangays)")
    print(f"  range                   : {pb['auc'].min():.4f} - {pb['auc'].max():.4f}")

    lo = loo_block_threshold(d)
    print(f"\nLeave-one-barangay-out (out-of-sample threshold):")
    print(f"  theta* mode             : {lo['theta_star'].mode().iloc[0]:.2f}"
          f"   (range {lo['theta_star'].min():.2f}-{lo['theta_star'].max():.2f})")
    print(f"  hit rate                : {lo['hit'].mean():.3f} +/- {lo['hit'].std():.3f}")
    print(f"  false alarm rate        : {lo['far'].mean():.3f} +/- {lo['far'].std():.3f}")
    print(f"  Youden J                : {lo['youden'].mean():.3f} +/- {lo['youden'].std():.3f}")

    return {"per_block_auc": pb, "loo": lo,
            "roc": roc_curve(d[COL_LABEL], d["score"])}


# Run each z-cap and each cohesion scenario through report() and tabulate.
# A single table of (scenario, pooled AUC, blocked AUC, out-of-sample Youden)
# answers Reviewer 2 point 3 and the threshold-bias caveat in one figure.
#
# out = report(df)
