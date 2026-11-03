from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv

# ==================================================
# 1. SETTINGS
# ==================================================

VTK_FILE = Path("VTK/aerofoil/aerofoil_200.vtk")
OUTPUT_DIR = Path("surfaceResults/viscousForces")

P_INF = 1775.544
T_INF = 300.0
U_INF = 433.9

R_GAS = 287.0421397

CHORD = 1.0

# ==================================================
# 2. FREESTREAM REFERENCE
# ==================================================

rho_inf = P_INF / (R_GAS * T_INF)
q_inf = 0.5 * rho_inf * U_INF**2

# ==================================================
# 3. LOAD WALL MESH
# ==================================================

mesh = pv.read(VTK_FILE)

tau = np.asarray(
    mesh.cell_data["wallShearStress"],
    dtype=float
)

centres = mesh.cell_centers().points

x = centres[:, 0]
y = centres[:, 1]
z = centres[:, 2]

# ==================================================
# 4. CALCULATE ACTUAL FACE AREAS
# ==================================================

sized_mesh = mesh.compute_cell_sizes(
    length=False,
    area=True,
    volume=False
)

areas = np.asarray(
    sized_mesh.cell_data["Area"],
    dtype=float
)

if np.any(areas <= 0):
    raise ValueError("Non-positive surface face area found.")

span = mesh.bounds[3] - mesh.bounds[2]
A_ref = CHORD * span

print("========== GEOMETRY ==========")
print(f"Number of faces: {mesh.n_cells}")
print(f"Extrusion span: {span:.6f} m")
print(f"Reference area: {A_ref:.6f} m2")
print(f"Total wall area: {areas.sum():.6f} m2")

# ==================================================
# 5. CALCULATE FORCE ON THE BODY
# ==================================================

# OpenFOAM wallShearStress is wall-on-fluid.
# Reverse it to obtain fluid-on-body traction.

traction_on_body = -tau

face_forces = traction_on_body * areas[:, None]

# ==================================================
# 6. SEPARATE UPPER AND LOWER SURFACES
# ==================================================

upper = z > 0
lower = z < 0

if np.count_nonzero(upper) + np.count_nonzero(lower) != mesh.n_cells:
    raise ValueError("Some faces could not be classified.")

def integrate(mask):

    return {
        "area": areas[mask].sum(),
        "force": face_forces[mask].sum(axis=0),
        "face_count": np.count_nonzero(mask)
    }

upper_result = integrate(upper)
lower_result = integrate(lower)
total_result = integrate(np.ones(mesh.n_cells, dtype=bool))

# ==================================================
# 7. PRINT FORCES
# ==================================================

print("\n========== VISCOUS FORCE COMPONENTS ==========")

for name, result in [
    ("UPPER", upper_result),
    ("LOWER", lower_result),
    ("TOTAL", total_result)
]:

    Fx, Fy, Fz = result["force"]

    print(f"\n{name} SURFACE")
    print(f"Faces: {result['face_count']}")
    print(f"Area: {result['area']:.8f} m2")
    print(f"Drag force: {Fx:.8f} N")
    print(f"Spanwise force: {Fy:.8f} N")
    print(f"Lift force: {Fz:.8f} N")

# ==================================================
# 8. VISCOUS DRAG COEFFICIENT
# ==================================================

D_viscous = total_result["force"][0]
L_viscous = total_result["force"][2]

Cd_viscous = D_viscous / (q_inf * A_ref)
Cl_viscous = L_viscous / (q_inf * A_ref)

print("\n========== VISCOUS COEFFICIENTS ==========")
print(f"Freestream density: {rho_inf:.8f} kg/m3")
print(f"Dynamic pressure: {q_inf:.4f} Pa")
print(f"Reference area: {A_ref:.6f} m2")

print(f"Viscous drag: {D_viscous:.8f} N")
print(f"Viscous lift: {L_viscous:.8f} N")

print(f"Viscous Cd: {Cd_viscous:.8f}")
print(f"Viscous Cl: {Cl_viscous:.8f}")

if D_viscous < 0:
    print("\nWARNING: Integrated viscous drag is negative.")
    print("Check the stress sign convention and face mapping.")

# ==================================================
# 9. EXPORT FACE-WISE FORCE DATA
# ==================================================

df = pd.DataFrame({
    "x_m": x,
    "z_m": z,
    "area_m2": areas,
    "tau_x_Pa": tau[:, 0],
    "tau_y_Pa": tau[:, 1],
    "tau_z_Pa": tau[:, 2],
    "force_x_N": face_forces[:, 0],
    "force_y_N": face_forces[:, 1],
    "force_z_N": face_forces[:, 2],
    "side": np.where(upper, "upper", "lower")
})

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

csv_file = OUTPUT_DIR / "viscous_face_forces.csv"

df.to_csv(csv_file, index=False)

print("\nFace-wise results saved to:", csv_file)
print("\nIntegration completed.")
