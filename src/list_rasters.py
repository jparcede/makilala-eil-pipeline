"""
Diagnostic: list every .tif in each version folder and show how the filename
normalises. Run this first, paste the output, and the matcher can be fixed
against reality rather than against guesses.
"""

import os
import glob

BASE = os.environ.get("MAKILALA_DIR", "data")

FOLDERS = {
    'rasters_v1':    os.path.join(BASE, "rasters_v1"),
    'rasters_v2':    os.path.join(BASE, "rasters_v2"),
    'rasters_v3':    os.path.join(BASE, "rasters_v3"),
    'standard_error': os.path.join(BASE, "standard_error"),
}


def norm(s):
    return s.lower().replace('-', '').replace('_', '').replace(' ', '')


for label, folder in FOLDERS.items():
    print("=" * 78)
    print(f"{label}   ->   {folder}")
    print("=" * 78)
    if not os.path.isdir(folder):
        print("  FOLDER DOES NOT EXIST\n")
        continue

    tifs = sorted(glob.glob(os.path.join(folder, "*.tif")))
    if not tifs:
        print("  no .tif files found")
        print("  directory contains:")
        for f in sorted(os.listdir(folder))[:20]:
            print(f"    {f}")
        print()
        continue

    print(f"  {len(tifs)} .tif files\n")
    print(f"  {'filename':<45} {'normalised':<38} SE?")
    print("  " + "-" * 90)
    for p in tifs:
        raw = os.path.basename(p)
        n = norm(raw)
        is_se = 'standarderror' in n
        print(f"  {raw:<45} {n:<38} {'yes' if is_se else ''}")
    print()
