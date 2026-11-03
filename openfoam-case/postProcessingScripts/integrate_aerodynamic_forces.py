from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv

# ==================================================
# 1. REFERENCE CONDITIONS
# ==================================================

VTK_FILE = Path("VTK/aerofoil/aerofoil_200.vtk")
OUTPUT_DIR = Path("surfaceResults/aerodynamicForces")

P_INF = 1775.544
T_INF = 300.0
U_INF = 433.9
R_GAS = 287.0421397

CHORD = 1.0

rho_inf = P_INF / (R_GAS * T_INF)
q_inf = 0.5 * rho_inf * U_INF**2

# ==================================================
# 2. LOAD SURFACE DATA
# ==================================================

mesh = pv.read(VTK_FILE)

required = ["p", "wallShearStress"]

for field in required:
    if field not in mesh.cell_data:
        raise KeyError(f"Missing field: {field}")

p = np.asarray(mesh.cell_data["p"], dtype=float)
tau = np.asarray(
    mesh.cell_data["wallShearStress"],
    dtype=float
)

centres = mesh.cell_centers().points

x = centres[:, 0]
y = centres[:, 1]
z = centres[:, 2]

# ==================================================
# 3. CALCULATE FACE AREAS AND NORMALS
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

normal_mesh = mesh.compute_normals(
    cell_normals=True,
    point_normals=False,
    consistent_normals=False,
    auto_orient_normals=False,
    inplace=False
)

normals = np.asarray(
    normal_mesh.cell_data["Normals"],
    dtype=float
)

if not (
    len(p) == len(tau) == len(areas) == len(normals)
):
    raise ValueError("Surface data arrays do not match.")

# Outward-from-fluid area vectors
area_vectors = normals * areas[:, None]

# ==================================================
# 4. PRESSURE FORCE ON THE AIRFOIL
# ==================================================

p_gauge = p - P_INF

# Pressure force exerted by fluid on body
pressure_force_faces = (
    p_gauge[:, None] * area_vectors
)

# ==================================================
# 5. VISCOUS FORCE ON THE AIRFOIL
# ==================================================

# OpenFOAM wallShearStress sign convention:
# reverse it to obtain fluid-on-body viscous traction

viscous_force_faces = -tau * areas[:, None]

# ==================================================
# 6. TOTAL FACE FORCE
# ==================================================

total_force_faces = (
    pressure_force_faces + viscous_force_faces
)

# ==================================================
# 7. SEPARATE UPPER AND LOWER SURFACES
# ==================================================

upper = z > 0
lower = z < 0

if np.count_nonzero(upper) + np.count_nonzero(lower) != mesh.n_cells:
    raise ValueError("Some surface faces were not classified.")

def integrate(mask, force_array):

    return np.sum(force_array[mask], axis=0)

results = {}

for side, mask in [
    ("Upper", upper),
    ("Lower", lower),
    ("Total", np.ones(mesh.n_cells, dtype=bool))
]:

    results[side] = {
        "pressure": integrate(mask, pressure_force_faces),
        "viscous": integrate(mask, viscous_force_faces),
        "total": integrate(mask, total_force_faces),
    }

# ==================================================
# 8. PRINT FORCE COMPONENTS
# ==================================================

print("\n========== REFERENCE CONDITIONS ==========")
print(f"Density: {rho_inf:.8f} kg/m3")
print(f"Dynamic pressure: {q_inf:.5f} Pa")
print(f"Reference area: {CHORD * 0.1:.5f} m2")

print("\n========== AERODYNAMIC FORCE COMPONENTS ==========")

for side, result in results.items():

    print(f"\n{side.upper()} SURFACE")

    for force_type in ["pressure", "viscous", "total"]:

        Fx, Fy, Fz = result[force_type]

        print(
            f"{force_type:10s}: "
            f"Fx={Fx: .9f} N, "
            f"Fy={Fy: .9f} N, "
            f"Fz={Fz: .9f} N"
        )

# ==================================================
# 9. CALCULATE AERODYNAMIC COEFFICIENTS
# ==================================================

span = mesh.bounds[3] - mesh.bounds[2]
A_ref = CHORD * span

denominator = q_inf * A_ref

print("\n========== AERODYNAMIC COEFFICIENTS ==========")

total = results["Total"]

for force_type in ["pressure", "viscous", "total"]:

    Fx, Fy, Fz = total[force_type]

    Cd = Fx / denominator
    Cl = Fz / denominator

    print(f"\n{force_type.upper()}")
    print(f"Drag coefficient: {Cd:.9f}")
    print(f"Lift coefficient: {Cl:.9f}")

# ==================================================
# 10. FACE-WISE CSV EXPORT
# ==================================================

df = pd.DataFrame({
    "x_m": x,
    "z_m": z,
    "area_m2": areas,

    "normal_x": normals[:, 0],
    "normal_y": normals[:, 1],
    "normal_z": normals[:, 2],

    "pressure_Pa": p,
    "gauge_pressure_Pa": p_gauge,

    "pressure_force_x_N": pressure_force_faces[:, 0],
    "pressure_force_z_N": pressure_force_faces[:, 2],

    "viscous_force_x_N": viscous_force_faces[:, 0],
    "viscous_force_z_N": viscous_force_faces[:, 2],

    "total_force_x_N": total_force_faces[:, 0],
    "total_force_z_N": total_force_faces[:, 2],

    "side": np.where(upper, "upper", "lower")
})

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

csv_file = OUTPUT_DIR / "aerodynamic_face_forces.csv"
df.to_csv(csv_file, index=False)

print("\nFace-wise data saved to:", csv_file)

# ==================================================
# 11. SAVE SUMMARY
# ==================================================

summary = []

for side, result in results.items():

    for force_type in ["pressure", "viscous", "total"]:

        Fx, Fy, Fz = result[force_type]

        summary.append({
            "surface": side,
            "force_type": force_type,
            "Fx_N": Fx,
            "Fy_N": Fy,
            "Fz_N": Fz,
            "Cd": Fx / denominator,
            "Cl": Fz / denominator
        })

summary_file = OUTPUT_DIR / "aerodynamic_force_summary.csv"

pd.DataFrame(summary).to_csv(
    summary_file,
    index=False
)

print("Summary saved to:", summary_file)
print("\nIntegration completed.")
