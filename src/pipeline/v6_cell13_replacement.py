# ============================================================================
# REPLACES CELL 13 of Makilala_FoS_Pipeline_v6.ipynb
#
# Changes vs the original cell:
#   1. Threshold grid extended DOWN to 0.40 (was 0.80 -> theta* was censored
#      at the left endpoint, not an interior optimum)
#   2. Sweeps all 4 cohesion scenarios x 3 z-caps (was baseline cohesion only)
#   3. Reports exact AUC per configuration -- threshold-free, immune to the
#      "threshold fitted on the validation set" objection
#   4. Flags any configuration whose optimum still sits on a grid endpoint
#   5. Sweep done by sorting once + searchsorted, not per-threshold reductions
#
# Downstream cells 15/17/19 are unaffected: `all_results` keeps its original
# structure and keys, populated from the BASELINE cohesion scenario.
# ============================================================================

THRESHOLDS = np.arange(0.40, 1.55, 0.05)          # was np.arange(0.80, 1.55, 0.05)

z_scenarios = {
    'z_original': np.clip(z_raw, 0.1, 20.0),
    'z_cap_5m':   np.clip(z_raw, 0.1, 5.0),
    'z_cap_3m':   np.clip(z_raw, 0.1, 3.0),
}

c_scenarios = {
    'c_baseline': c,
    'c_plus5':    c + 5.0,
    'c_times2':   c * 2.0,
    'c_plus10':   c + 10.0,
}


def sweep_config(c_arr, z_arr, observed, alpha, gamma, phi, m, k_h):
    """Compute FoS for one (cohesion, depth) configuration and sweep thresholds.

    Returns a dict with the per-threshold table, the AUC, and diagnostics.
    """
    v = (np.isfinite(alpha) & (alpha >= ALPHA_MIN) &
         np.isfinite(z_arr)  & (z_arr > 0) &
         np.isfinite(gamma)  & (gamma > GAMMA_W) &
         np.isfinite(phi)    & (phi > 0) & (phi < 90.0) &
         np.isfinite(c_arr)  & (c_arr > 0) &
         np.isfinite(m)      & (m > 0) & (m < 1))

    fos = np.full(ref_shape, np.nan)
    fos[v] = compute_FoS(c_arr[v], phi[v], alpha[v], m[v], gamma[v], z_arr[v], k_h[v])

    fos_static = np.full(ref_shape, np.nan)
    fos_static[v] = compute_FoS(c_arr[v], phi[v], alpha[v], m[v], gamma[v], z_arr[v], 0.0)

    kc = np.full(ref_shape, np.nan)
    kc[v] = compute_kc(c_arr[v], phi[v], alpha[v], m[v], gamma[v], z_arr[v])

    # scored domain: valid AND FoS finite (drops pixels with missing PGA)
    scored = v & np.isfinite(fos)
    n_dropped = int(v.sum() - scored.sum())

    pos = np.sort(fos[scored & (observed == 1)])   # landslide pixels
    neg = np.sort(fos[scored & (observed == 0)])   # stable pixels
    n_pos, n_neg = len(pos), len(neg)

    # ---- exact AUC (Mann-Whitney U on the score -FoS, ties at 0.5) ---------
    lt = np.searchsorted(neg, pos, side='left')    # neg strictly below each pos
    le = np.searchsorted(neg, pos, side='right')
    concordant = (n_neg - le).sum()                # neg above pos -> correct
    ties = (le - lt).sum()
    auc = (concordant + 0.5 * ties) / (n_pos * n_neg)

    # ---- threshold sweep via searchsorted ---------------------------------
    rows = []
    for t in THRESHOLDS:
        tp = int(np.searchsorted(pos, t, side='left'))   # FoS < t among positives
        fp = int(np.searchsorted(neg, t, side='left'))
        fn, tn = n_pos - tp, n_neg - fp
        hr  = tp / n_pos if n_pos else 0.0
        far = fp / n_neg if n_neg else 0.0
        rows.append({'thresh': float(t), 'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
                     'hr': hr, 'far': far,
                     'prec': tp / (tp + fp) if (tp + fp) else 0.0,
                     'acc': (tp + tn) / (n_pos + n_neg),
                     'youden': hr - far,
                     'pct_fail': 100.0 * (tp + fp) / (n_pos + n_neg),
                     'fos': fos, 'fos_static': fos_static, 'kc': kc})

    js = np.array([r['youden'] for r in rows])
    i_best = int(np.nanargmax(js))
    censored = (i_best == 0) or (i_best == len(rows) - 1)

    return {'sweep': rows, 'best_thresh': rows[i_best]['thresh'],
            'best_youden': rows[i_best]['youden'], 'auc': auc,
            'censored': censored, 'valid': v, 'scored': scored,
            'n_valid': int(v.sum()), 'n_scored': int(scored.sum()),
            'n_dropped': n_dropped, 'n_pos': n_pos, 'n_neg': n_neg,
            'fos': fos, 'fos_static': fos_static, 'kc': kc,
            'c_mean': float(np.nanmean(c_arr[v])),
            'c_min': float(np.nanmin(c_arr[v])), 'c_max': float(np.nanmax(c_arr[v])),
            'z_mean': float(np.nanmean(z_arr[v])),
            'fos_mean': float(np.nanmean(fos[scored]))}


assert observed is not None, "LSI inventory required -- run cell 11 first"

print("=" * 104)
print(f"COHESION x DEPTH x THRESHOLD SWEEP   (alpha >= {ALPHA_MIN} deg, "
      f"theta in [{THRESHOLDS[0]:.2f}, {THRESHOLDS[-1]:.2f}])")
print("=" * 104)

grid = {}
for cname, c_arr in c_scenarios.items():
    for zname, z_arr in z_scenarios.items():
        grid[(cname, zname)] = sweep_config(c_arr, z_arr, observed,
                                            alpha, gamma, phi, m, k_h)

# ---- headline table -------------------------------------------------------
print(f"\n{'cohesion':<12} {'z-cap':<12} {'c mean':>7} {'z mean':>7} {'FoS mn':>7} "
      f"{'theta*':>7} {'Youden':>7} {'HitR':>7} {'FAR':>7} {'Acc':>7} {'AUC':>7}  flag")
print("-" * 104)
for (cname, zname), r in grid.items():
    b = [s for s in r['sweep'] if s['thresh'] == r['best_thresh']][0]
    flag = "CENSORED" if r['censored'] else ""
    print(f"{cname:<12} {zname:<12} {r['c_mean']:>7.2f} {r['z_mean']:>7.2f} "
          f"{r['fos_mean']:>7.3f} {r['best_thresh']:>7.2f} {r['best_youden']:>7.3f} "
          f"{100*b['hr']:>6.1f}% {100*b['far']:>6.1f}% {100*b['acc']:>6.1f}% "
          f"{r['auc']:>7.4f}  {flag}")

# ---- diagnostics ----------------------------------------------------------
ref = grid[('c_baseline', 'z_cap_3m')]
print("\n" + "=" * 104)
print("DIAGNOSTICS  (baseline cohesion, z-cap 3 m)")
print("-" * 104)
print(f"  valid pixels (mask v)          : {ref['n_valid']:,}")
print(f"  scored pixels (v & FoS finite) : {ref['n_scored']:,}")
print(f"  dropped (non-finite k_h/FoS)   : {ref['n_dropped']:,}"
      f"  ({100*ref['n_dropped']/max(ref['n_valid'],1):.2f}% of valid)")
print(f"  landslide / stable pixels      : {ref['n_pos']:,} / {ref['n_neg']:,}")

m_fallback = int(np.sum(ref['scored'] & (m == 0.5)))
print(f"  m == 0.50 fallback pixels      : {m_fallback:,}"
      f"  ({100*m_fallback/max(ref['n_scored'],1):.2f}% of scored)")

fs = ref['fos_static']
n_stable = int(np.sum(ref['scored'] & np.isfinite(fs) & (fs > 1)))
print(f"  Form B defined (FoS_static > 1): {n_stable:,}"
      f"  ({100*n_stable/max(ref['n_scored'],1):.2f}% of scored)"
      f"   -> NaN elsewhere")

n_cens = sum(1 for r in grid.values() if r['censored'])
if n_cens:
    print(f"\n  !! {n_cens}/{len(grid)} configurations still peak at a grid endpoint."
          f"\n     Widen THRESHOLDS further before quoting theta*.")
else:
    print(f"\n  All {len(grid)} configurations have an interior optimum.")

# ---- backward compatibility for cells 15 / 17 / 19 ------------------------
all_results = {zn: grid[('c_baseline', zn)] for zn in z_scenarios}
cohesion_results = grid
print("\n'all_results' populated from baseline cohesion; full grid in 'cohesion_results'.")
