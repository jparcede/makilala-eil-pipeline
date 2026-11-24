# ============================================================================
#  §22  THRESHOLD TRANSFER: IS THE "ZERO SPREAD" REAL, OR IS IT THE GRID?
#
#  Paste as a new cell in Makilala_FoS_Pipeline_v10.ipynb, AFTER cell §10
#  (spatially blocked validation). Uses only objects that cell already defines:
#  SCORED, OBSERVED, REF, THRESHOLDS, ref_shape, ref_transform, pos, neg.
#  Mutates nothing.
#
#  WHY THIS EXISTS
#  ---------------
#  Section 6.4 claims theta* is "identical on every one of the forty-five
#  held-out tiles, with zero spread", and calls that evidence "as strong as
#  this design can produce". Section 4.8 says of the same quantity that the
#  Youden maximum is flat, that every threshold in 0.883-0.911 is within 0.001
#  of the optimum, and that thresholds "are best read as representatives of
#  that plateau rather than as sharply determined values".
#
#  Both cannot stand. THRESHOLDS has a step of 0.05 and the plateau is ~0.028
#  wide -- narrower than the step. Any tile whose true optimum lies anywhere in
#  roughly 0.875-0.925 reports exactly 0.90. So "zero spread" is consistent
#  with genuine stability AND with a coarse grid over a flat maximum.
#
#  There is a second, independent weakness. Leave-one-block-out as implemented
#  selects theta* on every scored pixel outside the held-out tile. One 2 km
#  tile is a small share of 7.1 M pixels, so the training pool is almost
#  unchanged from fold to fold. A statistic that does not move when 2% of the
#  data is removed has not been tested very hard.
#
#  This cell answers both. Three diagnostics:
#    A  each tile's own optimum at CONTINUOUS resolution, and its plateau width
#    B  spatial K-fold selection, where the training pool really does change
#    C  the grid-vs-continuous comparison stated explicitly
#
#  HOW TO READ THE RESULT
#  ----------------------
#  If the 45 continuous maximisers cluster tightly (say IQR < 0.05) and the
#  K-fold thresholds hold at larger K, the claim is REAL and currently
#  understated: report the distribution and the title's first clause is earned.
#  If they scatter across the plateau, the claim must weaken to "no tile
#  selects a different grid point", and the title needs revisiting.
#  Either outcome is publishable. The present ambiguity is not.
# ============================================================================
import numpy as np

guard()

_TILE_PX   = int(round(2000 / abs(ref_transform.a)))
_MIN_BLOCK = 200
_fos = REF['fos']


def _exact_youden(p, q, tol=(0.001, 0.01)):
    """Exact Youden optimum over all real thresholds, from the sorted values.

    Mirrors the pooled calculation in Section 4.8: J changes only where theta
    crosses an observed FoS value, so evaluating immediately above each
    distinct value is exhaustive, not an approximation.

    Returns (theta_hat, J_max, {tol: (lo, hi, width)}).
    """
    cand = np.unique(np.concatenate([p, q]))
    hr = np.searchsorted(p, cand, 'right') / len(p)
    far = np.searchsorted(q, cand, 'right') / len(q)
    J = hr - far
    k = int(np.argmax(J))
    out = {}
    for t in tol:
        m = J >= J[k] - t
        out[t] = (float(cand[m].min()), float(cand[m].max()),
                  float(cand[m].max() - cand[m].min()))
    return float(cand[k]), float(J[k]), out


# ---------------------------------------------------------------- collect ---
_tiles = []                                   # (r0, c0, p, q)
for r0 in range(0, ref_shape[0], _TILE_PX):
    for c0 in range(0, ref_shape[1], _TILE_PX):
        sl = (slice(r0, r0 + _TILE_PX), slice(c0, c0 + _TILE_PX))
        s = SCORED[sl]
        if not s.any():
            continue
        f, o = _fos[sl], OBSERVED[sl]
        p = np.sort(f[s & o]); q = np.sort(f[s & ~o])
        if len(p) < _MIN_BLOCK or len(q) < _MIN_BLOCK:
            continue
        _tiles.append((r0, c0, p, q))

N = len(_tiles)
print("=" * 78)
print("A.  PER-TILE OPTIMUM AT CONTINUOUS RESOLUTION")
print("=" * 78)
print(f"scored tiles: {N}   (grid step of THRESHOLDS = "
      f"{THRESHOLDS[1]-THRESHOLDS[0]:.2f})\n")

_exact, _w001, _w010, _grid = [], [], [], []
for r0, c0, p, q in _tiles:
    th, Jm, wid = _exact_youden(p, q)
    _exact.append(th); _w001.append(wid[0.001][2]); _w010.append(wid[0.01][2])
    hr = np.searchsorted(p, THRESHOLDS, 'left') / len(p)
    far = np.searchsorted(q, THRESHOLDS, 'left') / len(q)
    _grid.append(THRESHOLDS[int(np.argmax(hr - far))])
_exact = np.array(_exact); _grid = np.array(_grid)
_w001 = np.array(_w001); _w010 = np.array(_w010)

_th = REF['theta']
_half = (THRESHOLDS[1] - THRESHOLDS[0]) / 2
print(f"  exact maximiser : median {np.median(_exact):.4f}   "
      f"IQR {np.percentile(_exact,25):.4f}-{np.percentile(_exact,75):.4f}")
print(f"                    range  {_exact.min():.4f}-{_exact.max():.4f}   "
      f"sd {_exact.std(ddof=1):.4f}")
print(f"  plateau width   : J within 0.001 of max -> median "
      f"{np.median(_w001):.4f}, max {_w001.max():.4f}")
print(f"                    J within 0.01  of max -> median "
      f"{np.median(_w010):.4f}, max {_w010.max():.4f}")
print(f"\n  tiles whose GRID optimum is {_th:.2f}            : "
      f"{int((_grid == _th).sum())}/{N}")
print(f"  tiles whose EXACT optimum is within +/-{_half:.3f} of {_th:.2f}: "
      f"{int((np.abs(_exact - _th) <= _half).sum())}/{N}")
print(f"  distinct grid optima across tiles           : "
      f"{sorted(set(np.round(_grid, 2)))}")
print("\n  READ: if the second count is much lower than the first, the zero")
print("  spread is the grid, not the physics.")

# ------------------------------------------------------------- K-fold ------
print("\n" + "=" * 78)
print("B.  SPATIAL K-FOLD SELECTION  (the training pool actually changes)")
print("=" * 78)
print("Tiles are sorted west to east and cut into K contiguous bands. theta*")
print("is selected on all scored pixels OUTSIDE a band and applied to it.")
print("K = 45 is leave-one-tile-out, i.e. what Section 6.4 currently reports.\n")

_order = np.argsort([c0 for _, c0, _, _ in _tiles])
_P = [_tiles[i][2] for i in _order]
_Q = [_tiles[i][3] for i in _order]
_gp = np.searchsorted(pos, THRESHOLDS, 'left')
_gq = np.searchsorted(neg, THRESHOLDS, 'left')

print(f"{'K':>4}  {'left out':>9}  {'theta* per fold':<34}  {'moved':>6}")
print("-" * 62)
for K in (2, 3, 5, 9, 15, N):
    edges = np.linspace(0, N, K + 1).astype(int)
    ths, frac = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        hp = np.concatenate(_P[a:b]); hq = np.concatenate(_Q[a:b])
        cp = np.searchsorted(np.sort(hp), THRESHOLDS, 'left')
        cq = np.searchsorted(np.sort(hq), THRESHOLDS, 'left')
        HR = (_gp - cp) / (len(pos) - len(hp))
        FAR = (_gq - cq) / (len(neg) - len(hq))
        ths.append(THRESHOLDS[int(np.argmax(HR - FAR))])
        frac.append((len(hp) + len(hq)) / (len(pos) + len(neg)))
    ths = np.array(ths)
    shown = sorted(set(np.round(ths, 2)))
    label = f"{np.mean(frac)*100:.0f}%"
    print(f"{K:>4}  {label:>9}  {str(shown):<34}  "
          f"{int((ths != _th).sum())}/{len(ths)}")

print("\n  READ: theta* holding at K = 2 or 3, where a third to a half of the")
print("  domain is withheld, is a far stronger statement than it holding at")
print("  K = 45, where each fold withholds about 2%.")

# ------------------------------------------------------------ verdict ------
print("\n" + "=" * 78)
print("C.  VERDICT")
print("=" * 78)
_tight = float(np.percentile(_exact, 75) - np.percentile(_exact, 25)) < 0.05
_gridbound = _w001.max() > (THRESHOLDS[1] - THRESHOLDS[0])
print(f"  exact-optimum IQR < one grid step : {_tight}")
print(f"  some tile's 0.001-plateau exceeds a grid step: {_gridbound}")
print("""
  If the first is True, report the exact-maximiser distribution in Section 6.4
  and state the claim in terms of it. The grid then becomes a reporting
  resolution, exactly as Section 4.8 says, and the transfer claim is earned.

  If the first is False, replace "identical on every one of the forty-five
  held-out tiles, with zero spread" with a statement about grid points, move
  the Section 4.8 plateau caveat into 6.4 where the claim is made, and
  reconsider the first clause of the title.
""")
