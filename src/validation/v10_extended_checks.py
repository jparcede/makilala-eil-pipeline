# ============================================================================
#  v10 EXTENDED CHECKS  (v2)  -- tab:kcvar, tab:polydetect, tab:esi
#
#  Append as ONE cell after a full v10 Run All. Prints, and also writes
#  output_maps_v10/extended_checks.txt.
#
#  v2 CORRECTS THE UNCERTAINTY SECTION. v1 got four things wrong, all mine:
#
#   1. It invented a band-edge derivative for c' and m. The manuscript states
#      the real method in the paragraph BEFORE the table: c' and m "are not
#      differentiable functions of their inputs -- both are step functions of an
#      interpolated variable -- so instead the underlying variable is treated as
#      N(mu, sigma^2) from the EBK output and the class probabilities computed
#      exactly". That is what v2 implements. The "unstated convention" v1 said
#      it had reverse-engineered was stated all along, one paragraph up.
#   2. It ran over the whole scored domain. The manuscript restricts this
#      analysis to where the SE rasters carry values: 4,754,653 px, 66.9%.
#   3. It compared the deterministic map's mean and SPATIAL standard deviation
#      against E[k_c] and a PROPAGATED per-pixel uncertainty. Different things.
#   4. Its AUC for P(k_c<k_h) used the FoS orientation (low = hazardous), but
#      for a failure probability high = hazardous -- hence 0.2517 instead of
#      the complement.
# ============================================================================
guard()
import numpy as np, os, rasterio, io, sys
from math import erf
from rasterio.warp import reproject, Resampling

class _Tee(io.TextIOBase):
    def __init__(self, *s): self.s = s
    def write(self, x):
        for st in self.s: st.write(x)
        return len(x)
    def flush(self):
        for st in self.s:
            try: st.flush()
            except Exception: pass

_BUF, _REAL = io.StringIO(), sys.stdout
sys.stdout = _Tee(_REAL, _BUF)
print("=" * 78); print("EXTENDED CHECKS  (v2)"); print("=" * 78)

_PHI = lambda x: 0.5 * (1.0 + np.vectorize(erf)(np.clip(x, -8, 8) / np.sqrt(2)))
SE_DIR = os.path.join(INPUT_DIR, "standard_error")
_SE = {'gamma': 'buk_v1_standarderror.tif', 'water': 'watercontent_v11standarderror.tif',
       'fines': 'siltclay_v1_standarderror.tif', 'pi': 'PI_v1_standarderror.tif'}


def _align(p):
    with rasterio.open(p) as src:
        a = src.read(1).astype(np.float64)
        if src.nodata is not None: a[a == src.nodata] = np.nan
        if a.shape == ref_shape and src.transform.almost_equals(ref_transform, precision=1e-6):
            return a                     # already on the reference grid
        d = np.full(ref_shape, np.nan)
        reproject(a, d, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref_transform, dst_crs=ref_crs,
                  resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
        return d


se = {k: _align(os.path.join(SE_DIR, f)) for k, f in _SE.items()
      if os.path.exists(os.path.join(SE_DIR, f))}
_FINES = _align(os.path.join(INPUT_DIR, FILE_FINES))   # P dropped this to save RAM
_WATER = P.water

print("\n-- EBK standard errors: tab:kcvar's stated values vs the rasters --")
for k, claim, lab in [('fines', 4.63, "fines (pp)"), ('water', 4.40, "water (pp)"),
                      ('gamma', 1.10, "unit weight (kN/m3)"), ('pi', 6.36, "PI")]:
    if k in se:
        v = se[k][SCORED]; v = v[np.isfinite(v)]
        ok(f"EBK SE, {lab}", float(v.mean()), claim, 0.35)

if len(se) == 4:
    UDOM = SCORED & np.isfinite(se['fines']) & np.isfinite(se['water']) \
                  & np.isfinite(se['gamma']) & np.isfinite(se['pi'])
    n_u = int(UDOM.sum())
    print("\n-- uncertainty domain (where the SE rasters carry values) --")
    ok("uncertainty-domain px", n_u, 4_754_653, 5000)
    ok("fraction of the scored domain", n_u/int(SCORED.sum()), 0.669, 0.005)

    def class_moments(mu, sd, edges, vals):
        """Exact E[] and Var[] of a step function of a N(mu, sd^2) variable."""
        sd = np.maximum(sd, 1e-9)
        cuts = [_PHI((e - mu)/sd) for e in edges]
        p = [cuts[0]] + [cuts[i+1]-cuts[i] for i in range(len(cuts)-1)] + [1.0-cuts[-1]]
        E = sum(pi*vi for pi, vi in zip(p, vals))
        E2 = sum(pi*vi*vi for pi, vi in zip(p, vals))
        return E, np.maximum(E2 - E*E, 0.0), p

    E_c, V_c, p_c = class_moments(_FINES, se['fines'], [5., 12., 20.], [1., 3., 6., 10.])
    E_m, V_m, p_m = class_moments(_WATER, se['water'], [20., 35., 50.], [.2, .4, .6, .8])
    _pmax = np.max(np.stack(p_c), axis=0)

    print("\n-- class-probability treatment of c' and m --")
    ok("E[c'] with the +10 kPa offset", float(np.nanmean((E_c + 10.0)[UDOM])), 13.53, 0.10)
    ok("E[m]", float(np.nanmean(E_m[UDOM])), 0.580, 0.010)
    ok("mean prob. of most likely c' class", float(np.nanmean(_pmax[UDOM])), 0.55, 0.02)
    # 11.1%, corrected 2026-09-15. The published 7.4% was this figure diluted:
    # v9's kc_uncertainty_propagation.py sets se=NaN where EBK wrote 0, then
    # reports np.nanmean(p_assigned[scored] < 0.5). `nan < 0.5` is False, not
    # NaN, so the 2.35M no-data pixels entered the denominator while the
    # numerator could only draw from the 4.75M SE-valid ones:
    #     0.1106 x (4,754,653 / 7,104,440) = 0.07402
    # NOTE: this target now comes from v10 itself, so this one line is a
    # REGRESSION test, not an independent check. Every other target in the
    # harness is the manuscript's own published value.
    ok("px where that class is below even odds",
       float(np.nanmean(_pmax[UDOM] < 0.5)), 0.111, 0.010)

    _a = np.radians(P.alpha); _ph = np.radians(P.phi)
    _z = np.clip(P.z_raw, 0.1, 5.0); _ca = np.cos(_a)
    _cE = E_c + 10.0
    _d = P.gamma * _z * _ca**2
    REL_SE_Z = 0.30                      # tab:kcvar: "assumed, 30% relative"

    terms = {
        'z':        (-_cE/(P.gamma*_z*_z*_ca**2))**2 * (REL_SE_Z*_z)**2,
        'c(fines)': (1.0/_d)**2 * V_c,
        'm(water)': (-(GAMMA_W/P.gamma)*np.tan(_ph))**2 * V_m,
        'alpha':    ((2*_cE*np.sin(_a)/(P.gamma*_z*_ca**3)) - 1.0/_ca**2)**2 * np.radians(2.0)**2,
        'gamma':    (-_cE/(P.gamma**2*_z*_ca**2) + (E_m*GAMMA_W/P.gamma**2)*np.tan(_ph))**2 * se['gamma']**2,
        'phi(PI)':  ((1 - E_m*GAMMA_W/P.gamma)/np.cos(_ph)**2)**2 * (np.radians(0.1)*se['pi'])**2,
    }
    VARMAP = sum(terms.values())
    _den = np.nanmean(VARMAP[UDOM])
    SHARES = {k: 100*np.nanmean(v[UDOM])/_den for k, v in terms.items()}
    print(f"\n-- tab:kcvar variance shares (REL_SE_Z = {REL_SE_Z:.0%}) --")
    # Published Table 11 values. Three of these were stale until 25 Nov 2026,
    # when kc_variance_common_mask.py was run against the archived rasters and
    # the manuscript was corrected to match: z 71.5 -> 71.4, c(fines)
    # 13.4 -> 13.1, alpha 5.8 -> 6.2, and the assumed z+alpha share
    # 77.3 -> 77.6. Checking the code against a hardcoded copy of the paper is
    # exactly how a stale literal gets mistaken for the paper's claim, so if
    # Table 11 changes again, change it here in the same commit.
    for k, claim in [('z', 71.4), ('c(fines)', 13.1), ('m(water)', 6.1),
                     ('alpha', 6.2), ('gamma', 2.9), ('phi(PI)', 0.3)]:
        ok(f"Var(k_c) share, {k}", SHARES[k]/100, claim/100, 0.025)
    ok("assumed z + alpha share", (SHARES['z']+SHARES['alpha'])/100, 0.776, 0.025)
    del terms

    KC_E = compute_kc(_cE, P.phi, P.alpha, E_m, P.gamma, _z)
    SD = np.sqrt(VARMAP)
    print("\n-- k_c: expectation and PROPAGATED (not spatial) uncertainty --")
    ok("E[k_c]", float(np.nanmean(KC_E[UDOM])), 0.213, 0.010)
    ok("mean propagated sd(k_c)", float(np.nanmean(SD[UDOM])), 0.110, 0.012)
    _r = SD/np.abs(KC_E)
    ok("median sd/|k_c|", float(np.nanmedian(_r[UDOM])), 0.44, 0.05)
    ok("px where sd exceeds |k_c|", float(np.nanmean(_r[UDOM] > 1.0)), 0.209, 0.025)
    print(f"  (the deterministic k_c map's SPATIAL sd is "
          f"{float(np.nanstd(REF['kc'][SCORED])):.3f} -- a different quantity, and")
    print(f"   what v1 wrongly compared against the published 0.110)")

    PF = _PHI((P.k_h - KC_E)/np.maximum(SD, 1e-12))
    print("\n-- P(k_c < k_h).  HIGH = hazardous, so the ranking reverses --")
    ok("P on landslide px", float(np.nanmean(PF[UDOM & OBSERVED])), 0.604, 0.02)
    ok("P on stable px",    float(np.nanmean(PF[UDOM & ~OBSERVED])), 0.319, 0.02)
    ok("P over the domain", float(np.nanmean(PF[UDOM])), 0.326, 0.02)
    _pp = np.sort(PF[UDOM & OBSERVED]); _pq = np.sort(PF[UDOM & ~OBSERVED])
    _lt = np.searchsorted(_pq, _pp, 'left'); _le = np.searchsorted(_pq, _pp, 'right')
    ok("AUC of P(k_c<k_h)", float((_lt.sum() + 0.5*(_le-_lt).sum())/(len(_pp)*len(_pq))),
       0.717, 0.010)
    del PF, KC_E, SD, VARMAP

print("\n-- tab:polydetect --")
from rasterio.features import rasterize as _rz
from shapely.geometry import mapping as _mp
_pid_c = _rz([(_mp(g), i+1) for i, g in enumerate(lsi.geometry) if g is not None],
             out_shape=ref_shape, transform=ref_transform, fill=0,
             dtype=np.int32, all_touched=False)
_flag = SCORED & (REF['fos'] < REF['theta'])
for label, pid, nc, claims in [
        ("touched-cell", poly_id, 1682, {0.01: 1253, 0.10: 1177, 0.25: 1066, 0.50: 916}),
        ("centre-only",  _pid_c, 1672, {0.01: 1212, 0.10: 1143, 0.25: 1056, 0.50: 919})]:
    _ps = np.where(SCORED, pid, 0)
    _nt = np.bincount(_ps.ravel(), minlength=len(lsi)+1)[1:]
    _nh = np.bincount(_ps.ravel(), weights=_flag.ravel(), minlength=len(lsi)+1)[1:]
    _has = (_nt > 0) & in_mak
    _fr = np.zeros(len(lsi)); _fr[_has] = _nh[_has]/_nt[_has]
    ok(f"polydetect n, {label}", int(_has.sum()), nc, 0)
    for t, claim in claims.items():
        ok(f"polydetect >={t:.0%}, {label}", int((_fr[_has] >= t).sum()), claim, 3)

print("\n-- tab:esi --")
_ps = np.where(SCORED, poly_id, 0)
_nt = np.bincount(_ps.ravel(), minlength=len(lsi)+1)[1:]
_has = (_nt > 0) & in_mak
_col = lsi['ESI_local'].to_numpy()
for e, (pt, ps_, pxa, pxs, hrc) in {6: (656, 646, 13238, 12463, 0.491),
                                    7: (867, 857, 56483, 53578, 0.590),
                                    8: (158, 157, 56249, 54225, 0.726),
                                    9: (19, 19, 30703, 30219, 0.774),
                                    10: (3, 3, 20916, 20062, 0.451)}.items():
    _ras = ESI_RASTERS.get(e)
    if _ras is None: continue
    _pa = int((_ras & FOOTPRINT).sum()); _pxs = int((_ras & SCORED).sum())
    ok(f"ESI {e}: polygons total", int((in_mak & (_col == e)).sum()), pt, 0)
    ok(f"ESI {e}: polygons scored", int((_has & (_col == e)).sum()), ps_, 0)
    ok(f"ESI {e}: LS px all", _pa, pxa, 40)
    ok(f"ESI {e}: LS px scored", _pxs, pxs, 40)
    ok(f"ESI {e}: hit rate", float(_flag[_ras & SCORED].sum()/max(_pxs, 1)), hrc, 0.01)

print("\n" + "=" * 78)
_p = sum(1 for _, g, _, _ in CHECKS if g)
print(f"CUMULATIVE CHECKS: {_p} of {len(CHECKS)} reproduce the published value")
for lab, g, got, want in CHECKS:
    if not g: print(f"  FAIL  {lab:<44} got {got:>14}  published {want}")
print("=" * 78)
print("\nNot covered: tab:ebk (settings, not output); fig:sobol (3-site Monte")
print("Carlo, needs the site list and a sampler); Appendices A-C; supplementary.")

sys.stdout = _REAL
_o = os.path.join(OUTPUT_DIR, "extended_checks.txt")
open(_o, "w").write(_BUF.getvalue())
print(f"\n(written to {_o})")
