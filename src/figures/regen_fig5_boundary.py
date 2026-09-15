"""
Fig. 5 -- stability-boundary projections, regenerated at the re-derived reference point.

Follows Stability_Boundary_6Panel.ipynb exactly in panel layout, ranges and colour
scheme. Two changes:

  1. Reference point moves from the old hand-assembled set (c'=8.0 at the c'+5
     scenario, m=0.40 assumed) to the pixel selected by the stated rule in
     ref_points_registered.py, with all seven values read at the PRIMARY
     configuration (c'+10 kPa, z-cap 5 m) on the registered rasters.

  2. The dashed sub-unity contour moves from the representative 0.80 to the actual
     Youden-optimal threshold of the primary configuration, theta* = 0.90, so the
     figure labels a quantity the paper elsewhere derives rather than a round number.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GAMMA_W = 9.81
THETA_STAR = 0.90          # Youden-optimal threshold, primary configuration

# re-derived reference point: Villaflores, median-slope pixel within 2 km of the
# field samples, lon 125.07555 / lat 6.88169, primary configuration
ref = {
    "c":     13.00,   # kPa   (c_base 3 + 10)
    "phi":   31.47,   # deg   (PI 35.3)
    "alpha": 18.15,   # deg
    "m":      0.60,   #       (water 37.2%)
    "gamma": 11.55,   # kN/m3
    "z":      5.00,   # m     (raw 5.74, capped)
    "kh":     0.1453, #       (PGA 0.2906 g)
}
REF_NAME = "Makilala--Villaflores"


def compute_FoS(c, phi_deg, alpha_deg, m, gamma, z, kh):
    phi = np.radians(phi_deg)
    alpha = np.radians(alpha_deg)
    R = c + (gamma - m * GAMMA_W) * z * np.cos(alpha) ** 2 * np.tan(phi)
    D = gamma * z * np.sin(alpha) * np.cos(alpha) + kh * gamma * z * np.cos(alpha) ** 2
    return np.where(D > 0, R / D, np.nan)


def compute_kc(c, phi_deg, alpha_deg, m, gamma, z):
    phi = np.radians(phi_deg)
    alpha = np.radians(alpha_deg)
    d = gamma * z * np.cos(alpha) ** 2
    return c / d + (1 - m * GAMMA_W / gamma) * np.tan(phi) - np.tan(alpha)


fos_ref = float(compute_FoS(**{**ref, "phi_deg": ref["phi"], "alpha_deg": ref["alpha"]}
                            if False else dict(c=ref["c"], phi_deg=ref["phi"],
                                               alpha_deg=ref["alpha"], m=ref["m"],
                                               gamma=ref["gamma"], z=ref["z"],
                                               kh=ref["kh"])))
fos_static = float(compute_FoS(c=ref["c"], phi_deg=ref["phi"], alpha_deg=ref["alpha"],
                               m=ref["m"], gamma=ref["gamma"], z=ref["z"], kh=0.0))
kc_ref = float(compute_kc(ref["c"], ref["phi"], ref["alpha"], ref["m"],
                          ref["gamma"], ref["z"]))
formB = 1 - 1 / fos_static if fos_static > 1 else float("nan")

print("=" * 78)
print(f"REFERENCE POINT  {REF_NAME}  (primary configuration)")
print("=" * 78)
for k, v in ref.items():
    print(f"  {k:<7} {v}")
print(f"\n  FoS            {fos_ref:.4f}")
print(f"  FoS_static     {fos_static:.4f}")
print(f"  k_c            {kc_ref:.4f}")
print(f"  |S_kh|max      {formB:.4f}")
print(f"  k_c - k_h      {kc_ref - ref['kh']:+.4f}   "
      f"({'stable' if kc_ref > ref['kh'] else 'unstable'} margin)")
print(f"  theta* contour {THETA_STAR}")

panels = [
    ("c",     "kh",    (0.5, 20),  (0, 0.6),    r"$c'$ (kPa)",     r"$k_h$",
     "Cohesion vs earthquake:\nmore cohesion tolerates stronger shaking"),
    ("alpha", "kh",    (5, 60),    (0, 0.6),    r"$\alpha$ ($^\circ$)", r"$k_h$",
     "Slope angle vs earthquake:\nsteeper slopes fail with less shaking"),
    ("phi",   "kh",    (20, 45),   (0, 0.6),    r"$\phi'$ ($^\circ$)",  r"$k_h$",
     "Friction angle vs earthquake:\nhigher friction tolerates more shaking"),
    ("z",     "kh",    (0.5, 15),  (0, 0.6),    r"$z$ (m)",        r"$k_h$",
     "Soil depth vs earthquake:\nnon-monotonic --- thin soils protected by cohesion"),
    ("m",     "kh",    (0, 0.95),  (0, 0.6),    r"$m$",            r"$k_h$",
     "Saturation vs earthquake:\nwetter soils fail with less shaking"),
    ("alpha", "c",     (5, 60),    (0.5, 25),   r"$\alpha$ ($^\circ$)", r"$c'$ (kPa)",
     "Slope angle vs cohesion (no earthquake):\nhow much $c'$ stabilises a slope"),
]

N = 400
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

for idx, (xp, yp, xr, yr, xlabel, ylabel, title) in enumerate(panels):
    ax = axes.flat[idx]
    x_vals = np.linspace(xr[0], xr[1], N)
    y_vals = np.linspace(yr[0], yr[1], N)
    X, Y = np.meshgrid(x_vals, y_vals)

    params = {k: np.full_like(X, v) for k, v in ref.items()}
    params[xp] = X
    params[yp] = Y
    FoS = compute_FoS(params["c"], params["phi"], params["alpha"],
                      params["m"], params["gamma"], params["z"], params["kh"])

    levels = np.array([0, 0.5, THETA_STAR, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0])
    cf = ax.contourf(X, Y, FoS, levels=levels, cmap="RdYlGn", alpha=0.8, extend="both")

    cs1 = ax.contour(X, Y, FoS, levels=[1.0], colors="black", linewidths=2.5)
    ax.clabel(cs1, fmt="FoS=1", fontsize=8, inline=True)

    cs_t = ax.contour(X, Y, FoS, levels=[THETA_STAR], colors="darkred",
                      linewidths=1.5, linestyles="--")
    ax.clabel(cs_t, fmt=f"{THETA_STAR:.2f}", fontsize=7, inline=True)

    cs125 = ax.contour(X, Y, FoS, levels=[1.25], colors="darkgreen",
                       linewidths=1.5, linestyles="--")
    ax.clabel(cs125, fmt="1.25", fontsize=7, inline=True)

    rx, ry = ref[xp], ref[yp]
    if xr[0] <= rx <= xr[1] and yr[0] <= ry <= yr[1]:
        ax.plot(rx, ry, "w*", markersize=15, markeredgecolor="black", markeredgewidth=1.1)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"({chr(97+idx)}) {title}", fontsize=9, fontweight="bold")
    if idx % 3 == 2:
        plt.colorbar(cf, ax=ax, shrink=0.8, label="FoS")
    ax.grid(True, alpha=0.15)

    sym = {"c": "c'", "phi": "φ'", "alpha": "α", "m": "m",
           "gamma": "γ", "z": "z", "kh": "k_h"}
    fixed = ", ".join(f"{sym[k]}={v}" for k, v in ref.items() if k not in (xp, yp))
    ax.text(0.02, 0.02, f"Fixed: {fixed}", transform=ax.transAxes, fontsize=5.5,
            color="gray", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

fig.tight_layout()
fig.savefig("companion_figures/stability_boundary_6panel.png", dpi=160,
            bbox_inches="tight")
print("\nwrote companion_figures/stability_boundary_6panel.png")

# ---------------------------------------------------------------- panel-f check
print("\npanel (f) static cohesion requirement at the reference alpha:")
for a in (ref["alpha"], 25, 30, 35):
    lo, hi = 0.0, 60.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f = compute_FoS(mid, ref["phi"], a, ref["m"], ref["gamma"], ref["z"], 0.0)
        if f < 1.0:
            lo = mid
        else:
            hi = mid
    print(f"  alpha {a:5.2f} deg -> c' needed for static FoS = 1: {hi:6.2f} kPa")
print("\ndone.")
