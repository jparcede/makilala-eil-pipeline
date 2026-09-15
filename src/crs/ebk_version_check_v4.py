"""
EBK run-to-run variability check, version 4.

Filenames are now given EXPLICITLY per version rather than pattern-matched.
The naming varies enough between folders (buk_v1, Silt_Clay__v2, water-content_v3)
that regex matching kept failing silently, which is worse than failing loudly.

Also skips " - Copy.tif" and "(1).tif" duplicates, which exist in rasters_v3
and rasters_v2 respectively.

Statistics are PAIRWISE. Where two rasters are bit-identical this is stated
rather than averaged away: a 3-run standard deviation over [a, b, b] is
|a-b|/sqrt(3) and understates the difference by construction.

Usage:
    python ebk_version_check_v4.py
"""

import os
import glob
import itertools
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

BASE = os.environ.get("MAKILALA_DIR", "data")

DIRS = {v: os.path.join(BASE, f"rasters_{v}") for v in ('v1', 'v2', 'v3')}
SE_FALLBACK = os.path.join(BASE, "standard_error")

# Base filenames WITHOUT the version suffix or .tif. The version tag is
# inserted per folder. Corrected against the actual directory listings.
STEMS = {
    'gamma':  'buk',
    'fines':  'Silt_Clay_',        # note the trailing underscore: Silt_Clay__v3.tif
    'water':  'water-content',
    'pi':     'PI',
    'sand':   'Sand',
    'gravel': 'gravel',
    'LL':     'LL',
    'PL':     'PL',
    'dryuw':  'dryunitweight',
}

# v1 used different capitalisation for some layers; try these too.
ALIASES = {
    'fines':  ['Silt_Clay_', 'siltclay', 'Silt_Clay'],
    'water':  ['water-content', 'watercontent'],
    'sand':   ['Sand', 'sand'],
    'pi':     ['PI', 'pi'],
}

BAD = (' - copy', '(1)', '(2)', '(3)')


def candidates(folder, stem, version, want_se):
    """All plausible paths for one parameter in one folder."""
    tag = f"_{version}"
    suffix = "_standarderror" if want_se else ""
    names = [f"{stem}{tag}{suffix}.tif"]
    # some files carry a doubled underscore or an extra digit (watercontent_v11)
    names += [f"{stem}{tag}1{suffix}.tif", f"{stem}{tag}{suffix}1.tif"]
    out = []
    for n in names:
        p = os.path.join(folder, n)
        if os.path.exists(p):
            out.append(p)
    return out


def find(folder, key, version, want_se=False):
    """Locate a raster, first by exact construction then by loose scan."""
    for stem in ALIASES.get(key, [STEMS[key]]):
        hits = candidates(folder, stem, version, want_se)
        if hits:
            return hits[0]

    # loose fallback: scan the folder
    target = STEMS[key].lower().replace('-', '').replace('_', '')
    for p in sorted(glob.glob(os.path.join(folder, "*.tif"))):
        raw = os.path.basename(p)
        low = raw.lower()
        if any(b in low for b in BAD):
            continue
        n = low.replace('-', '').replace('_', '').replace(' ', '')
        if ('standarderror' in n) != want_se:
            continue
        core = n.replace('standarderror', '').replace('.tif', '')
        core = core.rstrip('0123456789v')
        if core == target.rstrip('0123456789v'):
            return p
    return None


def grid_of(path):
    with rasterio.open(path) as s:
        return {'crs': s.crs, 'transform': s.transform,
                'shape': (s.height, s.width), 'res': s.res}


def align(path, ref):
    with rasterio.open(path) as src:
        out = np.full(ref['shape'], np.nan, dtype='float64')
        reproject(source=rasterio.band(src, 1), destination=out,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref['transform'], dst_crs=ref['crs'],
                  dst_nodata=np.nan, resampling=Resampling.bilinear)
        if src.nodata is not None:
            out = np.where(out == src.nodata, np.nan, out)
    return out


def same_grid(a, b):
    return (a['shape'] == b['shape'] and a['crs'] == b['crs']
            and np.allclose(a['res'], b['res'], rtol=1e-9))


# ---------------------------------------------------------------------------
print("=" * 88)
print("FILE DISCOVERY")
print("=" * 88)

found = {}
for key in STEMS:
    found[key] = {}
    for v, d in DIRS.items():
        found[key][v] = find(d, key, v)
        se = find(d, key, v, want_se=True)
        if se is None:
            for vv in ('v1', 'v2', 'v3'):
                se = find(SE_FALLBACK, key, vv, want_se=True)
                if se:
                    break
        found[key][v + '_se'] = se
    line = " ".join(f"{v}:{'OK' if found[key][v] else '--'}"
                    f"/{'OK' if found[key][v + '_se'] else '--'}"
                    for v in DIRS)
    print(f"  {key:<8} {line}      (pred/SE)")

print("\n  resolved paths:")
for key in STEMS:
    for v in DIRS:
        for sfx in ('', '_se'):
            p = found[key][v + sfx]
            if p:
                print(f"    {key:<7} {v}{sfx:<4} {os.path.basename(p)}")

usable = [k for k in STEMS if sum(found[k][v] is not None for v in DIRS) >= 2]
print(f"\n  comparable (>=2 versions): {', '.join(usable) or 'NONE'}")
if not usable:
    raise SystemExit("Nothing to compare.")


# ---------------------------------------------------------------------------
print("\n" + "=" * 88)
print("PAIRWISE COMPARISON")
print("=" * 88)

results = {}
for key in usable:
    present = [v for v in DIRS if found[key][v]]
    grids = {v: grid_of(found[key][v]) for v in present}
    ref_v = present[0]
    ref = grids[ref_v]

    arr = {v: align(found[key][v], ref) for v in present}
    se_path = found[key][ref_v + '_se']
    se = align(se_path, ref) if se_path else None
    if se is not None:
        se = np.where(se > 0, se, np.nan)

    valid = np.all(np.isfinite(np.stack([arr[v] for v in present])), axis=0)
    if se is not None:
        valid &= np.isfinite(se)
    if valid.sum() == 0:
        print(f"\n  {key}: no overlapping valid pixels")
        continue

    lo, hi = np.nanmin(arr[ref_v][valid]), np.nanmax(arr[ref_v][valid])
    spread = hi - lo
    se_mean = float(np.nanmean(se[valid])) if se is not None else np.nan

    print(f"\n  {key}   (n = {int(valid.sum()):,}, reference {ref_v}, "
          f"versions {'+'.join(present)})")
    print(f"    range      : [{lo:.3f}, {hi:.3f}]   spread {spread:.3f}")
    if se is not None:
        print(f"    EBK SE     : {se_mean:.4f}   "
              f"({100*se_mean/spread:.1f}% of spread)")

    worst = 0.0
    for a, b in itertools.combinations(present, 2):
        d = (arr[b] - arr[a])[valid]
        rms = float(np.sqrt(np.mean(d ** 2)))
        tag = "same grid" if same_grid(grids[a], grids[b]) else "diff grid"
        if rms == 0.0:
            tag += ", IDENTICAL"
        ratio = rms / se_mean if se_mean and np.isfinite(se_mean) else np.nan
        if np.isfinite(ratio):
            worst = max(worst, ratio)
        print(f"    {a}-{b}      : mean {np.nanmean(d):+.4f}  RMS {rms:.4f}  "
              f"max|d| {np.nanmax(np.abs(d)):.4f}  RMS/SE {ratio:.4f}  [{tag}]")

    results[key] = {'worst': worst,
                    'se_pct': 100 * se_mean / spread if se is not None else np.nan}


# ---------------------------------------------------------------------------
print("\n" + "=" * 88)
print("SUMMARY")
print("=" * 88)
print(f"  {'param':<8} {'worst RMS/SE':>13} {'SE % of spread':>17}")
print("  " + "-" * 42)
for k, r in sorted(results.items(), key=lambda kv: -kv[1]['worst']):
    print(f"  {k:<8} {r['worst']:>13.4f} {r['se_pct']:>16.1f}%")

worst_all = max(r['worst'] for r in results.values())
pcts = [r['se_pct'] for r in results.values() if np.isfinite(r['se_pct'])]
print(f"\n  worst pairwise RMS/SE : {worst_all:.3f}")
print(f"  SE as % of spread     : {min(pcts):.1f}%-{max(pcts):.1f}%  "
      f"(mean {np.mean(pcts):.1f}%)")
