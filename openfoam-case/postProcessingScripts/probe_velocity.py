from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv

# -----------------------------------------------
# SETTINGS
# -----------------------------------------------

VOLUME_FILE = "VTK/openfoam-case_200.vtk"
PROBE_FILE = "surfaceResults/expansionProbe/expansion_probe.csv"

OUTPUT_FILE = Path(
    "surfaceResults/expansionProbe/expansion_probe_velocity.csv"
)

R = 287.0421397
GAMMA = 1.4
CV = 719.4
CP = CV + R

# Stations to inspect
x_stations = [0.74, 0.76, 0.80, 0.90, 0.98]

# -----------------------------------------------
# LOAD DATA
# -----------------------------------------------

grid = pv.read(VOLUME_FILE)
df = pd.read_csv(PROBE_FILE)

U_field = np.asarray(grid.cell_data["U"])
T_field = np.asarray(grid.cell_data["T"])
p_field = np.asarray(grid.cell_data["p"])
rho_field = np.asarray(grid.cell_data["rho"])
Ma_field = np.asarray(grid.cell_data["Ma"])

# -----------------------------------------------
# EXTRACT VELOCITY AND DERIVED QUANTITIES
# -----------------------------------------------

results = []

for _, row in df.iterrows():

    cid = int(row["cell_id"])

    if cid < 0:
        continue

    U = U_field[cid]

    Ux, Uy, Uz = U

    speed = np.linalg.norm(U)

    T = float(T_field[cid])
    p = float(p_field[cid])
    rho = float(rho_field[cid])
    Ma = float(Ma_field[cid])

    sound_speed = np.sqrt(GAMMA * R * T)

    Ma_check = speed / sound_speed

    flow_angle = np.degrees(np.arctan2(Uz, Ux))

    # Stagnation temperature, assuming ideal gas
    T0 = T + speed**2 / (2 * CP)

    # Ideal-gas equation-of-state consistency
    rho_check = p / (R * T)

    eos_ratio = rho / rho_check

    record = row.to_dict()

    record.update({
        "Ux_m_s": Ux,
        "Uy_m_s": Uy,
        "Uz_m_s": Uz,
        "speed_m_s": speed,
        "flow_angle_deg": flow_angle,
        "Ma_calculated": Ma_check,
        "Ma_difference": Ma - Ma_check,
        "T0_K": T0,
        "rho_kg_m3": rho,
        "EOS_density_ratio": eos_ratio,
    })

    results.append(record)

out = pd.DataFrame(results)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUT_FILE, index=False)

# -----------------------------------------------
# PRINT SELECTED RESULTS
# -----------------------------------------------

print("\n========== VELOCITY AND FLOW DIRECTION ==========")

for x in x_stations:

    subset = out[
        np.isclose(out["x_over_c"], x)
        & out["offset_m"].isin([0.002, 0.005, 0.010, 0.020])
    ]

    print(f"\n========== x/c = {x:.2f} ==========")

    cols = [
        "offset_m",
        "Ux_m_s",
        "Uz_m_s",
        "speed_m_s",
        "flow_angle_deg",
        "Ma",
        "Ma_calculated",
        "T0_K",
        "EOS_density_ratio",
    ]

    print(
        subset[cols].to_string(
            index=False,
            float_format=lambda v: f"{v:.4f}"
        )
    )

print("\nFull results saved to:", OUTPUT_FILE)
