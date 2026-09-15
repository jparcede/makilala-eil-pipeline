# ============================================================================
# TRIPWIRE for Makilala_FoS_Pipeline_v9.ipynb
#
# Localises where `scored` stops being the Table-3 scored domain.
#
# Symptom: at the end of a Run All, alpha[scored].min() == 0.10 and
# alpha[scored].mean() == 20.763, when the scored mask enforces
# alpha >= ALPHA_MIN (= 10) and the published mean is 23.00. The count is still
# exactly 7,104,440, so `scored` has the right size and the wrong membership.
#
# HOW TO USE
#   1. Paste this cell ONCE, early (right after cell 18, where `scored` is
#      first bound at top level). Running it defines check_scored().
#   2. Then put the single line
#          check_scored("after cell NN")
#      at the BOTTOM of each cell you want to bracket. Good first set, being
#      the cells that read `scored`: 18, 30, 34, 37, 40, 45.
#   3. Run All. The first label that prints DRIFTED is the cell that did it.
#
# Nothing is mutated; this only reads.
# ============================================================================
import numpy as np

_REF = {}


def check_scored(label="", alpha_min=10.0, expect_n=7_104_440, expect_mean=23.00):
    """Report whether `scored` is still the alpha >= alpha_min evaluation domain."""
    G = globals()
    if 'scored' not in G:
        print(f"[tripwire] {label:<24} scored NOT BOUND")
        return
    s = np.asarray(G['scored'])
    n = int(s.sum())

    line = f"[tripwire] {label:<24} n={n:>10,} dtype={s.dtype}"

    if 'alpha' in G:
        a = np.asarray(G['alpha'], dtype=np.float64)
        if a.shape == s.shape:
            av = a[s.astype(bool) & np.isfinite(a)]
            amin, amean = float(av.min()), float(av.mean())
            line += f"  alpha[scored] min={amin:6.2f} mean={amean:6.2f}"
            ok = (amin >= alpha_min - 1e-9) and (n == expect_n) \
                 and abs(amean - expect_mean) < 0.05
            line += "   OK" if ok else "   <<< DRIFTED"
            if amin < alpha_min - 1e-9:
                below = int((a < alpha_min) [s.astype(bool)].sum()) \
                    if s.dtype == bool else -1
                line += f"  ({below:,} px below {alpha_min:g} deg in the mask)"
        else:
            line += f"  alpha shape {a.shape} != scored shape {s.shape}  <<< DRIFTED"
    else:
        line += "  (alpha not bound -- cannot test membership)"

    # identity fingerprint: has the OBJECT changed, or only its contents?
    key = (n, int(s.dtype == bool), int(np.asarray(s).ravel()[:4096].sum()))
    if _REF and key != _REF.get('key'):
        line += f"   [fingerprint changed since {_REF.get('label','first call')!r}]"
    if not _REF:
        _REF['key'], _REF['label'] = key, label
    print(line)


check_scored("at definition")

# ---------------------------------------------------------------------------
# Independent ground truth, if the FoS map is on disk. The published scored
# domain is exactly the finite set of output_maps_v9/FoS_map.tif; this compares
# the in-memory mask against it without trusting any notebook state.
# ---------------------------------------------------------------------------
def check_scored_against_disk(path="output_maps_v9/FoS_map.tif"):
    import os
    import rasterio
    if not os.path.exists(path):
        print(f"[tripwire] {path} not found; skipping disk comparison")
        return
    with rasterio.open(path) as src:
        f = src.read(1).astype(np.float64)
        if src.nodata is not None:
            f[f == src.nodata] = np.nan
    truth = np.isfinite(f)
    s = np.asarray(globals()['scored']).astype(bool)
    if s.shape != truth.shape:
        print(f"[tripwire] shape mismatch {s.shape} vs {truth.shape}")
        return
    same = int((s == truth).sum())
    print(f"[tripwire] vs FoS_map.tif: agree on {same:,} / {s.size:,} px"
          f"  ({100*same/s.size:.4f}%)   in-memory-only={int((s & ~truth).sum()):,}"
          f"  disk-only={int((truth & ~s).sum()):,}")
