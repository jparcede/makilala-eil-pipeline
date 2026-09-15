"""Corruption test for the v10 anti-drift machinery.

v9 bound its arrays and its evaluation mask as loose module globals, and by the
end of a Run All the mask no longer selected the domain it was built for --
alpha[scored].min() came back 0.10 where the mask enforces alpha >= 10. Nothing
raised; the numbers were simply wrong. v10 answers that with a frozen namespace
`P`, the module constant `SCORED`, and `guard()` at the top of every cell.

This test deliberately corrupts that state three ways and asserts that guard()
raises each time, plus that it stays quiet on an intact namespace. Run it after
building a fixture:

    python tests/make_fixture.py
    python tests/guard_test.py

Set MAKILALA_DIR to point at real rasters instead if you would rather test
against those; the corruption checks behave identically either way.
"""
import json
import os
import sys
import tempfile
import runpy

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NOTEBOOK = os.path.join(ROOT, "notebooks", "Makilala_FoS_Pipeline_v10.ipynb")

os.environ.setdefault("MAKILALA_DIR", os.path.join(HERE, "fixture"))
if not os.path.isdir(os.environ["MAKILALA_DIR"]):
    sys.exit(f"no input directory at {os.environ['MAKILALA_DIR']}\n"
             f"run `python {os.path.join('tests', 'make_fixture.py')}` first, "
             f"or set MAKILALA_DIR")


def flatten(nb_path):
    """Concatenate a notebook's code cells into a runnable script.

    Flattening on the fly rather than keeping a checked-in copy: a duplicated
    script is one more thing that can drift away from the notebook it mirrors.
    """
    nb = json.load(open(nb_path))
    src = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    fd, path = tempfile.mkstemp(suffix="_v10_flat.py")
    with os.fdopen(fd, "w") as f:
        f.write("\n\n".join(src))
    return path


flat = flatten(NOTEBOOK)
try:
    G = runpy.run_path(flat)
finally:
    os.unlink(flat)

guard = G["guard"]
P = G["P"]
mod = guard.__globals__          # guard() closes over the run's globals

results = []


def expect_raise(label, mutate, restore):
    saved = mutate()
    try:
        guard()
    except RuntimeError as exc:
        print(f"  [CAUGHT ] {label}\n            {exc}")
        restore(saved)
        return True
    print(f"  [MISSED ] {label}  <<< guard did not fire")
    restore(saved)
    return False


print("\n" + "=" * 70)
print("ANTI-DRIFT GUARD TEST")
print("=" * 70)


def _same_count_wrong_pixels():
    """The v9 symptom exactly: right cardinality, wrong membership."""
    old = mod["SCORED"]
    flat_mask = np.zeros(old.size, bool)
    flat_mask[: int(old.sum())] = True
    new = flat_mask.reshape(old.shape)
    new.flags.writeable = False
    mod["SCORED"] = new
    return old


results.append(expect_raise(
    "SCORED rebound, same count / wrong membership",
    _same_count_wrong_pixels,
    lambda old: mod.__setitem__("SCORED", old)))


def _shrink():
    old = mod["SCORED"]
    new = old.copy()
    new[np.unravel_index(np.flatnonzero(new)[:1000], new.shape)] = False
    new.flags.writeable = False
    mod["SCORED"] = new
    return old


results.append(expect_raise(
    "SCORED shrunk by 1,000 px",
    _shrink,
    lambda old: mod.__setitem__("SCORED", old)))


def _mutate_alpha():
    a = P.alpha
    a.flags.writeable = True
    old = a[0, 0].copy()
    a[0, 0] = a[0, 0] + 137.0
    a.flags.writeable = False
    return old


def _restore_alpha(old):
    P.alpha.flags.writeable = True
    P.alpha[0, 0] = old
    P.alpha.flags.writeable = False


results.append(expect_raise("P.alpha mutated in place", _mutate_alpha, _restore_alpha))

try:
    guard()
    print("  [OK     ] clean namespace passes")
    results.append(True)
except RuntimeError as exc:
    print(f"  [BAD    ] guard fired on a clean namespace: {exc}")
    results.append(False)

print("=" * 70)
print(f"{sum(results)}/4 behaved correctly")
sys.exit(0 if all(results) else 1)
