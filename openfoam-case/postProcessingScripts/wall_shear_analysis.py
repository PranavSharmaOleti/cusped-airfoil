from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt

# ==================================================
# 1. SETTINGS
# ==================================================

VTK_FILE = Path("VTK/aerofoil/aerofoil_200.vtk")
OUTPUT_DIR = Path("surfaceResults/wallShear")

P_INF = 1775.544
T_INF = 300.0
U_INF = 433.9
R_GAS = 287.0421397

# ==================================================
# 2. FREESTREAM REFERENCE
# ==================================================

rho_inf = P_INF / (R_GAS * T_INF)
q_inf = 0.5 * rho_inf * U_INF**2

print("========== REFERENCE CONDITIONS ==========")
print(f"Freestream density: {rho_inf:.8f} kg/m3")
print(f"Dynamic pressure:   {q_inf:.4f} Pa")

# ==================================================
# 3. READ AIRFOIL SURFACE
# ==================================================

if not VTK_FILE.exists():
    raise FileNotFoundError(VTK_FILE)

mesh = pv.read(VTK_FILE)

if "wallShearStress" not in mesh.cell_data:
    raise KeyError("wallShearStress is missing from cell data.")

tau = np.asarray(
    mesh.cell_data["wallShearStress"],
    dtype=float
)

centres = mesh.cell_centers().points

x = centres[:, 0]
y = centres[:, 1]
z = centres[:, 2]

print("\n========== SURFACE ==========")
print("Cells:", mesh.n_cells)
print("Points:", mesh.n_points)
print("Shear field shape:", tau.shape)

# ==================================================
# 4. RECONSTRUCT LOCAL TANGENT
# ==================================================

# Each wall face is expected to be a quad formed by
# two airfoil-profile endpoints and two span endpoints.

faces = mesh.faces.reshape(-1, 5)

if len(faces) != mesh.n_cells:
    raise RuntimeError("Unexpected VTK face connectivity.")

if not np.all(faces[:, 0] == 4):
    raise RuntimeError("Expected quadrilateral wall faces.")

tangents = []

for face in faces:

    point_ids = face[1:]
    face_points = mesh.points[point_ids]

    # Remove the extrusion coordinate.
    xz = np.unique(
        np.round(face_points[:, [0, 2]], decimals=10),
        axis=0
    )

    if len(xz) != 2:
        raise RuntimeError(
            "Could not identify two profile endpoints "
            "for a wall face."
        )

    # Orient tangent in downstream (+X) direction.
    xz = xz[np.argsort(xz[:, 0])]

    dx = xz[1, 0] - xz[0, 0]
    dz = xz[1, 1] - xz[0, 1]

    length = np.sqrt(dx**2 + dz**2)

    if length == 0:
        raise RuntimeError("Zero-length surface segment found.")

    tangents.append([dx / length, dz / length])

tangents = np.asarray(tangents)

tx = tangents[:, 0]
tz = tangents[:, 1]

# ==================================================
# 5. WALL SHEAR COMPONENTS
# ==================================================

# Full shear magnitude
tau_magnitude = np.linalg.norm(tau, axis=1)

# Tangential projection in X-Z plane
tau_tangent = tau[:, 0] * tx + tau[:, 2] * tz

# In-plane normal projection.
# A perfect tangential wall-shear vector should have
# little component in this direction.
tau_normal = -tau[:, 0] * tz + tau[:, 2] * tx

# Extrusion-direction component
tau_span = tau[:, 1]

# ==================================================
# 6. SKIN-FRICTION COEFFICIENTS
# ==================================================

# Magnitude-based Cf: always non-negative.
Cf_magnitude = tau_magnitude / q_inf

# Signed projection using OpenFOAM's original vector sign.
Cf_tangent_raw = tau_tangent / q_inf

# ==================================================
# 7. SEPARATE UPPER AND LOWER SURFACES
# ==================================================

upper_mask = z > 0
lower_mask = z < 0

if np.count_nonzero(upper_mask) + np.count_nonzero(lower_mask) != mesh.n_cells:
    raise RuntimeError(
        "Some wall faces have zero z-centre and were not classified."
    )

x_over_c = x  # Chord = 1 m

# ==================================================
# 8. BUILD RESULTS TABLE
# ==================================================

df = pd.DataFrame({
    "x_over_c": x_over_c,
    "x_m": x,
    "y_m": y,
    "z_m": z,

    "tx": tx,
    "tz": tz,

    "tau_x_Pa": tau[:, 0],
    "tau_y_Pa": tau[:, 1],
    "tau_z_Pa": tau[:, 2],

    "tau_magnitude_Pa": tau_magnitude,
    "tau_tangent_raw_Pa": tau_tangent,
    "tau_normal_Pa": tau_normal,

    "Cf_magnitude": Cf_magnitude,
    "Cf_tangent_raw": Cf_tangent_raw,

    "side": np.where(upper_mask, "upper", "lower")
})

df = df.sort_values(["side", "x_over_c"])

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

csv_file = OUTPUT_DIR / "wall_shear_results.csv"
df.to_csv(csv_file, index=False)

print("\nSaved:", csv_file)

# ==================================================
# 9. NUMERICAL SUMMARY
# ==================================================

print("\n========== WALL SHEAR SUMMARY ==========")

for side in ["upper", "lower"]:

    subset = df[df["side"] == side]

    print(f"\n{side.upper()} SURFACE")
    print("Face count:", len(subset))

    print(
        "Shear magnitude:",
        f"min={subset['tau_magnitude_Pa'].min():.4f} Pa,",
        f"max={subset['tau_magnitude_Pa'].max():.4f} Pa"
    )

    print(
        "Magnitude-based Cf:",
        f"min={subset['Cf_magnitude'].min():.6f},",
        f"max={subset['Cf_magnitude'].max():.6f}"
    )

    print(
        "Tangential shear range:",
        f"min={subset['tau_tangent_raw_Pa'].min():.4f} Pa,",
        f"max={subset['tau_tangent_raw_Pa'].max():.4f} Pa"
    )

    print(
        "Negative tangential projections:",
        f"{(subset['tau_tangent_raw_Pa'] < 0).sum()}/{len(subset)}"
    )

print("\nMaximum shear faces:")

print(
    df.nlargest(5, "tau_magnitude_Pa")[
        ["side", "x_over_c", "z_m",
         "tau_magnitude_Pa", "tau_tangent_raw_Pa"]
    ].to_string(index=False)
)

print("\nMaximum in-plane normal component:")
print(f"{np.max(np.abs(tau_normal)):.6f} Pa")

print("Maximum spanwise component:")
print(f"{np.max(np.abs(tau_span)):.6f} Pa")

# ==================================================
# 10. PLOT RESULTS
# ==================================================

fig, axes = plt.subplots(2, 1, figsize=(11, 10))

for side, mask in [
    ("Upper surface", upper_mask),
    ("Lower surface", lower_mask)
]:

    order = np.argsort(x_over_c[mask])
    xs = x_over_c[mask][order]

    # Magnitude-based Cf
    axes[0].plot(
        xs,
        Cf_magnitude[mask][order],
        marker="o",
        markersize=3,
        label=side
    )

    # Signed tangential coefficient
    axes[1].plot(
        xs,
        Cf_tangent_raw[mask][order],
        marker="o",
        markersize=3,
        label=side
    )

axes[0].set_title("Skin-Friction Coefficient — Magnitude")
axes[0].set_ylabel(r"$|\tau_w|/q_\infty$")
axes[0].set_xlabel("x/c")
axes[0].grid(True, alpha=0.35)
axes[0].legend()

axes[1].set_title("Skin-Friction Coefficient — Signed Tangential Projection")
axes[1].set_ylabel(r"$\tau_{w,t}/q_\infty$")
axes[1].set_xlabel("x/c")
axes[1].axhline(0, linewidth=0.8)
axes[1].grid(True, alpha=0.35)
axes[1].legend()

fig.suptitle(
    "Cusped Airfoil — Wall Skin-Friction Analysis",
    fontsize=14
)

fig.tight_layout()

plot_file = OUTPUT_DIR / "wall_shear_analysis.png"

fig.savefig(plot_file, dpi=220, bbox_inches="tight")
plt.show()

print("\nPlot saved:", plot_file)
print("\nAnalysis completed.")
