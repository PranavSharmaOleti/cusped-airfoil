from pathlib import Path

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import csv

# ==================================================
# 1. USER SETTINGS
# ==================================================

VTK_FILE = Path("VTK/aerofoil/aerofoil_200.vtk")
OUTPUT_DIR = Path("surfaceResults")

P_INF = 1775.544       # Pa
T_INF = 300.0          # K
U_INF = 433.9          # m/s
R_GAS = 287.0421397    # J/(kg K)

CHORD = 1.0             # m

# ==================================================
# 2. READ VTK FILE
# ==================================================

if not VTK_FILE.exists():
    raise FileNotFoundError(f"VTK file not found: {VTK_FILE}")

mesh = pv.read(VTK_FILE)

print("\n========== MESH INFORMATION ==========")
print("Points:", mesh.n_points)
print("Cells:", mesh.n_cells)
print("Bounds:", mesh.bounds)
print("Available fields:", list(mesh.cell_data.keys()))

required_fields = ["p", "T", "Ma", "U"]

for field in required_fields:
    if field not in mesh.cell_data:
        raise KeyError(f"Required field '{field}' is missing.")

p = np.asarray(mesh.cell_data["p"]).reshape(-1)
T = np.asarray(mesh.cell_data["T"]).reshape(-1)
Ma = np.asarray(mesh.cell_data["Ma"]).reshape(-1)
U = np.asarray(mesh.cell_data["U"])

if mesh.n_cells != 220:
    print("WARNING: Expected 220 cells, found", mesh.n_cells)

if not (len(p) == len(T) == len(Ma) == len(U) == mesh.n_cells):
    raise ValueError("Field sizes do not match the number of cells.")

# ==================================================
# 3. GET FACE-CENTRE COORDINATES
# ==================================================

centres = mesh.cell_centers().points

x = centres[:, 0]
y = centres[:, 1]
z = centres[:, 2]

x_over_c = x / CHORD

# ==================================================
# 4. CALCULATE PRESSURE COEFFICIENT
# ==================================================

rho_inf = P_INF / (R_GAS * T_INF)
q_inf = 0.5 * rho_inf * U_INF**2

Cp = (p - P_INF) / q_inf

print("\n========== FREESTREAM ==========")
print(f"Density: {rho_inf:.6f} kg/m3")
print(f"Dynamic pressure: {q_inf:.3f} Pa")

# ==================================================
# 5. SEPARATE UPPER AND LOWER SURFACES
# ==================================================

upper_mask = z > 0
lower_mask = z < 0

def get_surface(mask):
    indices = np.where(mask)[0]
    indices = indices[np.argsort(x[indices])]

    return {
        "x_c": x_over_c[indices],
        "x": x[indices],
        "z": z[indices],
        "p": p[indices],
        "T": T[indices],
        "Ma": Ma[indices],
        "Cp": Cp[indices],
        "Ux": U[indices, 0],
        "Uy": U[indices, 1],
        "Uz": U[indices, 2],
    }

upper = get_surface(upper_mask)
lower = get_surface(lower_mask)

print("\n========== SURFACE SPLIT ==========")
print("Upper surface faces:", len(upper["x"]))
print("Lower surface faces:", len(lower["x"]))

if len(upper["x"]) + len(lower["x"]) != mesh.n_cells:
    raise ValueError(
        "Some faces have z = 0 and were not classified. "
        "Inspect their coordinates before proceeding."
    )

# ==================================================
# 6. EXPORT CSV FILES
# ==================================================

OUTPUT_DIR.mkdir(exist_ok=True)

columns = [
    "x/c", "x_m", "z_m", "p_Pa", "T_K",
    "Ma", "Cp", "Ux_m/s", "Uy_m/s", "Uz_m/s"
]

def export_csv(filename, data):
    with open(OUTPUT_DIR / filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for i in range(len(data["x"])):
            writer.writerow([
                data["x_c"][i],
                data["x"][i],
                data["z"][i],
                data["p"][i],
                data["T"][i],
                data["Ma"][i],
                data["Cp"][i],
                data["Ux"][i],
                data["Uy"][i],
                data["Uz"][i],
            ])

export_csv("upper_surface.csv", upper)
export_csv("lower_surface.csv", lower)

print("\nCSV files exported to:", OUTPUT_DIR)

# ==================================================
# 7. PLOT SURFACE DISTRIBUTIONS
# ==================================================

quantities = [
    ("p", "Surface pressure", "Pressure (Pa)"),
    ("Cp", "Pressure coefficient", "Cp"),
    ("T", "Surface temperature", "Temperature (K)"),
    ("Ma", "Surface Mach number", "Mach number"),
]

fig, axes = plt.subplots(4, 1, figsize=(11, 13))

for ax, (key, title, ylabel) in zip(axes, quantities):

    ax.plot(
        upper["x_c"], upper[key],
        marker="o", markersize=3,
        label="Upper surface"
    )

    ax.plot(
        lower["x_c"], lower[key],
        marker="o", markersize=3,
        label="Lower surface"
    )

    ax.set_title(title)
    ax.set_xlabel("x/c")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    ax.legend()

# Conventional aerodynamic Cp presentation
axes[1].invert_yaxis()

fig.suptitle(
    "Cusped airfoil surface distributions — t = 2e-05 s",
    fontsize=14
)

fig.tight_layout()

plot_file = OUTPUT_DIR / "surface_distributions.png"
fig.savefig(plot_file, dpi=200, bbox_inches="tight")

print("Plot saved to:", plot_file)

plt.show()

# ==================================================
# 8. BASIC NUMERICAL SUMMARY
# ==================================================

print("\n========== FIELD STATISTICS ==========")

for name, values in [
    ("Pressure (Pa)", p),
    ("Temperature (K)", T),
    ("Mach number", Ma),
    ("Cp", Cp),
]:
    print(
        f"{name:20s} "
        f"min={np.min(values):10.4f}  "
        f"max={np.max(values):10.4f}  "
        f"mean={np.mean(values):10.4f}"
    )

print("\nPost-processing completed.")
