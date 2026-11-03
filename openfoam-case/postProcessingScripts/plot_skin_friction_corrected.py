from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT = Path("surfaceResults/wallShear/wall_shear_results.csv")
OUTPUT = Path("surfaceResults/wallShear")

df = pd.read_csv(INPUT)

# Downstream-positive convention for this case
df["Cf_downstream"] = -df["Cf_tangent_raw"]

# Normal and spanwise components relative to shear magnitude
df["normal_fraction"] = (
    df["tau_normal_Pa"].abs()
    / df["tau_magnitude_Pa"].replace(0, np.nan)
)

df["span_fraction"] = (
    df["tau_y_Pa"].abs()
    / df["tau_magnitude_Pa"].replace(0, np.nan)
)

print("========== SIGN-CORRECTED SKIN FRICTION ==========")

for side in ["upper", "lower"]:

    subset = df[df["side"] == side].sort_values("x_over_c")

    print(f"\n{side.upper()}")

    print(
        f"Cf downstream-positive: "
        f"{subset['Cf_downstream'].min():.6f} to "
        f"{subset['Cf_downstream'].max():.6f}"
    )

    print(
        f"Negative Cf values: "
        f"{(subset['Cf_downstream'] < 0).sum()}"
    )

print("\n========== VECTOR PROJECTION CHECK ==========")

print(
    "Maximum local in-plane normal/shear ratio:",
    f"{df['normal_fraction'].max():.4%}"
)

print(
    "Maximum local spanwise/shear ratio:",
    f"{df['span_fraction'].max():.4%}"
)

# Plot with different line styles so overlapping
# upper and lower surfaces remain distinguishable.
fig, axes = plt.subplots(2, 1, figsize=(11, 9))

styles = {
    "upper": ("--", "Upper surface"),
    "lower": ("-", "Lower surface"),
}

for side, (style, label) in styles.items():

    subset = df[df["side"] == side].sort_values("x_over_c")

    axes[0].plot(
        subset["x_over_c"],
        subset["Cf_magnitude"],
        linestyle=style,
        marker="o",
        markersize=3,
        label=label
    )

    axes[1].plot(
        subset["x_over_c"],
        subset["Cf_downstream"],
        linestyle=style,
        marker="o",
        markersize=3,
        label=label
    )

axes[0].set_title("Magnitude-based skin-friction coefficient")
axes[0].set_ylabel(r"$|\tau_w|/q_\infty$")
axes[0].set_xlabel("x/c")

axes[1].set_title("Downstream-positive skin-friction coefficient")
axes[1].set_ylabel(r"$C_{f,t,\mathrm{downstream}}$")
axes[1].set_xlabel("x/c")
axes[1].axhline(0, linestyle=":", linewidth=1, label="Zero reference")

for ax in axes:
    ax.grid(True, alpha=0.35)
    ax.legend()

fig.suptitle("Cusped Airfoil — Skin Friction")
fig.tight_layout()

OUTPUT.mkdir(parents=True, exist_ok=True)

filename = OUTPUT / "skin_friction_corrected.png"
fig.savefig(filename, dpi=220, bbox_inches="tight")

print("\nSaved:", filename)

plt.show()
