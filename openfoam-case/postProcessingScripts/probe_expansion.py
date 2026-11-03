from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv

# ==================================================
# 1. INPUTS
# ==================================================

VOLUME_FILE = "VTK/openfoam-case_200.vtk"
WALL_FILE = "VTK/aerofoil/aerofoil_200.vtk"

OUTPUT_DIR = Path("surfaceResults/expansionProbe")

# Chordwise sampling locations
x_stations = [0.72, 0.74, 0.76, 0.80, 0.85, 0.90, 0.98]

# Distance measured outward from upper airfoil surface
offsets = [
    0.002,
    0.005,
    0.010,
    0.020,
    0.040,
    0.080,
    0.120,
]

Y_MID = -0.05

# ==================================================
# 2. READ THE MESHES
# ==================================================

print("Loading volume and wall meshes...")

grid = pv.read(VOLUME_FILE)
wall = pv.read(WALL_FILE)

# Confirm required cell fields exist
required = ["Ma", "p", "T"]

for field in required:
    if field not in grid.cell_data:
        raise KeyError(f"Missing cell field: {field}")

print("Volume cells:", grid.n_cells)

# ==================================================
# 3. RECONSTRUCT UPPER AIRFOIL PROFILE
# ==================================================

profile = np.unique(
    np.round(wall.points[:, [0, 2]], decimals=8),
    axis=0
)

upper = profile[profile[:, 1] > 0]
upper = upper[np.argsort(upper[:, 0])]

# ==================================================
# 4. GENERATE PROBE COORDINATES
# ==================================================

probe_points = []
metadata = []

for x in x_stations:

    z_wall = np.interp(
        x,
        upper[:, 0],
        upper[:, 1]
    )

    for offset in offsets:

        z_probe = z_wall + offset

        probe_points.append([
            x,
            Y_MID,
            z_probe
        ])

        metadata.append({
            "x_over_c": x,
            "z_wall_m": z_wall,
            "offset_m": offset,
            "probe_z_m": z_probe,
        })

probe_points = np.asarray(probe_points)

# ==================================================
# 5. FIND THE CELL CONTAINING EACH PROBE
# ==================================================

print("\nLocating containing cells...")

cell_ids = grid.find_containing_cell(probe_points)

# Cell-centre coordinates, for diagnostic purposes
centres = grid.cell_centers().points

valid_count = np.count_nonzero(cell_ids >= 0)

print(f"Valid probes: {valid_count}/{len(cell_ids)}")

if valid_count == 0:
    raise RuntimeError("No probe points found inside the mesh.")

# ==================================================
# 6. EXTRACT CELL-CENTRED VALUES
# ==================================================

records = []

for i, item in enumerate(metadata):

    cell_id = int(cell_ids[i])

    record = item.copy()

    record["cell_id"] = cell_id
    record["valid"] = cell_id >= 0

    if cell_id >= 0:

        cell_centre = centres[cell_id]

        record["cell_centre_x"] = cell_centre[0]
        record["cell_centre_z"] = cell_centre[2]

        record["distance_to_cell_centre"] = np.linalg.norm(
            probe_points[i] - cell_centre
        )

        # Direct lookup of OpenFOAM cell-centred fields
        for field in required:
            record[field] = float(
                grid.cell_data[field][cell_id]
            )

    else:

        for field in required:
            record[field] = np.nan

    records.append(record)

df = pd.DataFrame(records)

# ==================================================
# 7. EXPORT
# ==================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_file = OUTPUT_DIR / "expansion_probe.csv"

df.to_csv(output_file, index=False)

print("\nSaved:", output_file)

# ==================================================
# 8. PRINT RESULTS
# ==================================================

for x in x_stations:

    subset = df[df["x_over_c"] == x]

    print(f"\n========== x/c = {x:.3f} ==========")

    print(
        subset[
            [
                "offset_m",
                "cell_id",
                "Ma",
                "p",
                "T",
                "distance_to_cell_centre",
                "valid"
            ]
        ].to_string(index=False)
    )

print("\n========== SUMMARY ==========")
print("Total probes:", len(df))
print("Valid probes:", valid_count)
print("Invalid probes:", len(df) - valid_count)

print("\nProbe analysis completed.")
