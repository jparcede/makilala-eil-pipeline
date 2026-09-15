"""
Fig. 8 -- global Sobol sensitivity, with the soil-depth range made consistent with
the primary configuration.

ONE change from Sobol_Global_Sensitivity.ipynb: the soil-depth bound moves from
[1, 10] m to [1, 5] m. The published run sampled z over a range twice as deep as the
5 m cap the mapping pipeline actually applies, which inflated z's variance share.
Every other bound, the Saltelli scheme, N = 4096 and the seed are unchanged, so the
comparison is clean.
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol

GAMMA_W = 9.81
N_BASE = 4096
Z_BOUND = [1.0, 5.0]          # was [1.0, 10.0]

NAMES = ["c'", "phi'", "alpha", "m", "gamma", "z", "k_h"]
LABELS = [r"$c'$", r"$\phi'$", r"$\alpha$", r"$m$", r"$\gamma$", r"$z$", r"$k_h$"]
COLORS = ["#C0392B", "#E67E22", "#7D3C98", "#1E8449", "#2E86C1", "#B7950B", "#566573"]

SITES = {
    "Rodero 2\n(steep, unstable)":
        [[2.0, 12.0], [27.0, 36.0], [34.0, 42.0], [0.3, 0.9], [11.0, 16.0], Z_BOUND, [0.03, 0.16]],
    "Villaflores\n(gentle, stable)":
        [[3.0, 13.0], [26.0, 36.0], [16.0, 24.0], [0.1, 0.7], [9.5, 14.0], Z_BOUND, [0.10, 0.28]],
    "Sto. Niño\n(moderate slope)":
        [[1.0, 6.0], [28.0, 37.0], [24.0, 34.0], [0.3, 0.9], [9.5, 14.0], Z_BOUND, [0.07, 0.22]],
}


def compute_FoS_vec(X):
    c, phi, al, m, gam, z, kh = X.T
    a = np.radians(al); ca = np.cos(a); sa = np.sin(a)
    tp = np.tan(np.radians(phi))
    R = c + (gam - m * GAMMA_W) * z * ca * ca * tp
    D = gam * z * sa * ca + kh * gam * z * ca * ca
    return R / D


results = {}
print("=" * 90)
print(f"GLOBAL SOBOL SENSITIVITY   z bounded by the {Z_BOUND[1]:.0f} m cap")
print(f"Saltelli N = {N_BASE}, {N_BASE * 9:,} evaluations per site")
print("=" * 90)

for site, bounds in SITES.items():
    np.random.seed(42)
    prob = {"num_vars": 7, "names": NAMES, "bounds": bounds}
    X = saltelli.sample(prob, N_BASE, calc_second_order=False)
    Y = compute_FoS_vec(X)
    Y[~np.isfinite(Y)] = np.nanmedian(Y)
    Si = sobol.analyze(prob, Y, calc_second_order=False, print_to_console=False)
    results[site] = Si
    flat = site.replace("\n", " ")
    print(f"\n{flat}   FoS over samples: "
          f"[{np.min(Y):.3f}, {np.max(Y):.3f}], mean {np.mean(Y):.3f}")
    print(f"  {'param':<8}{'S1':>9}{'+/-':>8}{'ST':>9}{'+/-':>8}")
    for j in range(7):
        print(f"  {NAMES[j]:<8}{Si['S1'][j]:>9.1%}{Si['S1_conf'][j]:>8.1%}"
              f"{Si['ST'][j]:>9.1%}{Si['ST_conf'][j]:>8.1%}")
    rk = sorted(range(7), key=lambda j: Si["ST"][j], reverse=True)
    print(f"  top 3: {NAMES[rk[0]]} {Si['ST'][rk[0]]:.1%}, "
          f"{NAMES[rk[1]]} {Si['ST'][rk[1]]:.1%}, {NAMES[rk[2]]} {Si['ST'][rk[2]]:.1%}")
    print(f"  z + m = {Si['ST'][5] + Si['ST'][3]:.1%}     k_h = {Si['ST'][6]:.1%}")

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
for ax, (site, Si) in zip(axes, results.items()):
    st = np.array(Si["ST"]); conf = np.array(Si["ST_conf"])
    idx = np.argsort(st)[::-1]
    xs = np.arange(7)
    ax.bar(xs, st[idx], yerr=conf[idx], capsize=3,
           color=[COLORS[i] for i in idx], edgecolor="black", linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[i] for i in idx], fontsize=12)
    ax.set_title(site, fontsize=10)
    ax.grid(axis="y", alpha=0.3, lw=0.6)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Total-order Sobol index $S_T$", fontsize=11)
fig.tight_layout()
fig.savefig("companion_figures/fig5_sobol_global.pdf", bbox_inches="tight")
fig.savefig("companion_figures/fig5_sobol_global.png", dpi=200, bbox_inches="tight")
print("\nwrote companion_figures/fig5_sobol_global.pdf / .png")

print("\n" + "=" * 90)
print("SUMMARY FOR SECTION 5.11")
print("=" * 90)
for j, n in enumerate(NAMES):
    v = [results[s]["ST"][j] for s in SITES]
    print(f"  {n:<8} S_T  {100*min(v):5.1f}% -- {100*max(v):5.1f}%")
zm = [results[s]["ST"][5] + results[s]["ST"][3] for s in SITES]
print(f"  z + m    combined  {100*min(zm):.1f}% -- {100*max(zm):.1f}%")
for s in SITES:
    j = int(np.argmax(results[s]["ST"]))
    print(f"  leading at {s.splitlines()[0]:<12} {NAMES[j]} ({results[s]['ST'][j]:.1%})")
print("\ndone.")
