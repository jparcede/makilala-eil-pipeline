# ============================================================================
#  §25  BLOCK SIZE: WHERE 2 km COMES FROM   ->  Section 6.4
#
#  Paste as a new cell in Makilala_FoS_Pipeline_v10.ipynb, AFTER cell §10.
#  Uses only SCORED, OBSERVED, REF, THRESHOLDS, ref_shape, ref_transform,
#  pos, neg. Mutates nothing. Runtime is dominated by part A (~1-2 min).
#
#  WHY THIS EXISTS
#  ---------------
#  Section 6.4 says "we partitioned the scored domain into 2 km square tiles"
#  and never says why 2 km. As written, a referee is entitled to read it as the
#  number that produced 45 tiles. Standard practice in spatial cross-validation
#  sets the block size from the distance over which the quantity being
#  predicted -- or better, the model's own error -- stays correlated: blocks
#  smaller than that range leak information across the partition, blocks much
#  larger than it throw away data for nothing.
#
#  So the defensible procedure is: measure the range, then show the result over
#  a sweep of sizes around it.
#
#    A  empirical variograms, by random pair sampling, of
#         (i)  the landslide indicator  -- how far landsliding itself clusters
#         (ii) the model residual       -- how far the model's ERRORS cluster,
#              residual = 1{landslide} - 1{FoS < theta*}, which is the
#              quantity spatial CV is actually trying to decorrelate
#       Reported as binned semivariance against separation, with the practical
#       range: the first distance at which semivariance reaches 95% of its sill.
#
#    B  block-size sweep at 0.5, 1, 2, 4 and 8 km: tiles, scored tiles, share
#       of landslide pixels retained, within-block AUC, and pooled AUC on the
#       union of the scored tiles
#
#    C  does theta* still transfer at every block size? (two-fold spatial
#       split at each size, the hard version of the test from cell 27)
#
#  HOW TO READ IT
#  --------------
#  The number to quote in the paper is the RESIDUAL range from A(ii). If 2 km
#  is at or above it, the choice is justified and should be stated that way.
#  If the range is materially larger than 2 km, say so and report the sweep:
#  the honest position is then that 2 km is a lower bound on the block size and
#  the blocked estimate is, if anything, optimistic.
#
#  In B, watch whether within-block AUC falls monotonically with block size.
#  It should: larger blocks retain more internal regional contrast to exploit.
#  A monotone decay toward an asymptote is the strongest possible form of the
#  paper's argument -- it shows the pooled-to-blocked loss is a continuous
#  function of support, not an artifact of one arbitrary choice.
# ============================================================================
import numpy as np

guard()

PX_M = abs(ref_transform.a)                 # metres per pixel (5 m)
_fos = REF['fos']
_th = REF['theta']
rng = np.random.default_rng(20261022)


def _auc(p, q):
    if len(p) == 0 or len(q) == 0:
        return np.nan
    lt = np.searchsorted(q, p, 'left'); le = np.searchsorted(q, p, 'right')
    return ((len(q) - le).sum() + 0.5 * (le - lt).sum()) / (len(p) * len(q))


# ============================== A. VARIOGRAMS ==============================
print("=" * 78)
print("A.  HOW FAR DOES THE SIGNAL -- AND THE ERROR -- STAY CORRELATED?")
print("=" * 78)

R, C = np.nonzero(SCORED)
N = len(R)
M = min(60000, N)
sel = rng.choice(N, M, replace=False)
rs, cs = R[sel], C[sel]
ind = OBSERVED[rs, cs].astype(float)                  # 1 = landslide
pred = (_fos[rs, cs] < _th).astype(float)             # the model's call
resid = ind - pred                                    # -1, 0 or +1

MAXLAG_M = 12000.0
EDGES = np.concatenate([np.arange(0, 2000, 100), np.arange(2000, MAXLAG_M + 1, 500)])
NPAIR = 12_000_000

a = rng.integers(0, M, NPAIR); b = rng.integers(0, M, NPAIR)
ok = a != b
a, b = a[ok], b[ok]
d = np.hypot(rs[a].astype(np.int64) - rs[b], cs[a].astype(np.int64) - cs[b]) * PX_M
keep = d <= MAXLAG_M
a, b, d = a[keep], b[keep], d[keep]
bin_id = np.digitize(d, EDGES) - 1
nb = len(EDGES) - 1
cnt = np.bincount(bin_id, minlength=nb)[:nb]


def _vario(z):
    num = np.bincount(bin_id, weights=0.5 * (z[a] - z[b]) ** 2, minlength=nb)[:nb]
    with np.errstate(invalid='ignore', divide='ignore'):
        return num / cnt


def _range95(g, ctr, tail_from=8000.0):
    good = np.isfinite(g) & (cnt >= 500)
    if good.sum() < 5:
        return np.nan, np.nan
    sill = np.nanmedian(g[good & (ctr >= tail_from)])
    if not np.isfinite(sill):
        sill = np.nanmax(g[good])
    hit = np.where(good & (g >= 0.95 * sill))[0]
    return (ctr[hit[0]] if len(hit) else np.nan), sill


ctr = 0.5 * (EDGES[:-1] + EDGES[1:])
print(f"  sample: {M:,} scored pixels, {len(d):,} pairs within {MAXLAG_M/1000:.0f} km\n")
print(f"  {'lag (m)':>9} {'n pairs':>10} {'gamma indicator':>17} {'gamma residual':>16}")
print("  " + "-" * 56)
g_ind, g_res = _vario(ind), _vario(resid)
for i in range(nb):
    if cnt[i] < 500:
        continue
    if ctr[i] <= 2500 or i % 4 == 0:
        print(f"  {ctr[i]:>9.0f} {cnt[i]:>10,} {g_ind[i]:>17.5f} {g_res[i]:>16.5f}")

r_ind, s_ind = _range95(g_ind, ctr)
r_res, s_res = _range95(g_res, ctr)
print(f"\n  landslide indicator : sill {s_ind:.5f}, practical range "
      f"{r_ind:,.0f} m  ({r_ind/1000:.2f} km)")
print(f"  model residual      : sill {s_res:.5f}, practical range "
      f"{r_res:,.0f} m  ({r_res/1000:.2f} km)")
print(f"\n  block size in use   : 2,000 m")
print(f"  2 km is {'AT OR ABOVE' if r_res <= 2000 else 'BELOW'} the residual range.")
print("""
  READ: the residual range is the number that justifies the block size, because
  decorrelating the model's ERRORS is what blocked validation is for. If 2 km
  is at or above it, say so in Section 6.4 and cite this variogram. If it is
  below, 2 km is a lower bound and the blocked estimate is optimistic -- which
  is still reportable, and more honest than silence.""")

# ============================ B. BLOCK-SIZE SWEEP ==========================
print("\n" + "=" * 78)
print("B.  BLOCK-SIZE SWEEP   (inclusion rule held at 200 of each class)")
print("=" * 78)
print(f"{'size':>7} {'tiles':>6} {'scored':>7} {'% LS px kept':>13} "
      f"{'within-block AUC':>22} {'pooled on union':>16}")
print("-" * 76)
SWEEP = {}
for km in (0.5, 1.0, 2.0, 4.0, 8.0):
    tp = int(round(km * 1000 / PX_M))
    tot = sc = 0; aucs = []; Ps = []; Qs = []; kept = 0
    for r0 in range(0, ref_shape[0], tp):
        for c0 in range(0, ref_shape[1], tp):
            sl = (slice(r0, r0 + tp), slice(c0, c0 + tp))
            s = SCORED[sl]
            if not s.any():
                continue
            tot += 1
            f, o = _fos[sl], OBSERVED[sl]
            p = np.sort(f[s & o]); q = np.sort(f[s & ~o])
            if len(p) < 200 or len(q) < 200:
                continue
            sc += 1; kept += len(p)
            aucs.append(_auc(p, q)); Ps.append(p); Qs.append(q)
    aucs = np.array([x for x in aucs if np.isfinite(x)])
    if len(aucs) < 2:
        print(f"{km:>6.1f}k {tot:>6} {sc:>7}   (too few scored tiles)"); continue
    u = _auc(np.sort(np.concatenate(Ps)), np.sort(np.concatenate(Qs)))
    SWEEP[km] = (tot, sc, aucs, u)
    print(f"{km:>6.1f}k {tot:>6} {sc:>7} {100*kept/len(pos):>12.1f}% "
          f"  {aucs.mean():.3f} +/- {aucs.std(ddof=1):.3f} ({aucs.min():.3f}-{aucs.max():.3f})"
          f"   {u:.4f}")
print(f"\n  whole-domain pooled AUC for reference: {REF['auc']:.4f}")
print("""
  READ: within-block AUC should FALL as blocks shrink -- a smaller tile holds
  less regional contrast to exploit, so more of the ranking has to be earned
  slope by slope. A monotone decay is the strongest form of the paper's claim:
  it makes the pooled-to-blocked loss a continuous function of support rather
  than the consequence of one arbitrary tile size.""")

# ===================== C. DOES theta* SURVIVE AT EVERY SIZE? ===============
print("\n" + "=" * 78)
print("C.  THRESHOLD TRANSFER AT EACH BLOCK SIZE   (two-fold spatial split)")
print("=" * 78)
_gp = np.searchsorted(pos, THRESHOLDS, 'left')
_gq = np.searchsorted(neg, THRESHOLDS, 'left')
print(f"{'size':>7} {'theta* per half':<24} {'withheld':>9}")
print("-" * 44)
for km in sorted(SWEEP):
    tp = int(round(km * 1000 / PX_M))
    T = []
    for r0 in range(0, ref_shape[0], tp):
        for c0 in range(0, ref_shape[1], tp):
            sl = (slice(r0, r0 + tp), slice(c0, c0 + tp))
            s = SCORED[sl]
            if not s.any():
                continue
            f, o = _fos[sl], OBSERVED[sl]
            p = np.sort(f[s & o]); q = np.sort(f[s & ~o])
            if len(p) < 200 or len(q) < 200:
                continue
            T.append((c0, p, q))
    T.sort(key=lambda t: t[0])
    half = len(T) // 2
    ths, fr = [], []
    for band in (T[:half], T[half:]):
        if not band:
            continue
        hp = np.sort(np.concatenate([t[1] for t in band]))
        hq = np.sort(np.concatenate([t[2] for t in band]))
        HR = (_gp - np.searchsorted(hp, THRESHOLDS, 'left')) / (len(pos) - len(hp))
        FAR = (_gq - np.searchsorted(hq, THRESHOLDS, 'left')) / (len(neg) - len(hq))
        ths.append(THRESHOLDS[int(np.argmax(HR - FAR))])
        fr.append((len(hp) + len(hq)) / (len(pos) + len(neg)))
    print(f"{km:>6.1f}k {str(sorted(set(np.round(ths, 2)))):<24} {100*np.mean(fr):>8.0f}%")
print("""
  READ: theta* holding at 0.90 across every block size, while the within-block
  AUC falls steadily, is the cleanest statement of the paper's finding -- the
  calibration is indifferent to the support it is estimated on, the
  discrimination is not.
""")
