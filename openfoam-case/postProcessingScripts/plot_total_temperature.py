from pathlib import Path

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# Settings
VOLUME = "VTK/openfoam-case_200.vtk"
WALL = "VTK/aerofoil/aerofoil_200.vtk"

OUTPUT = Path("surfaceResults/flowContours")
Y_MID = -0.05

R = 287.0421397
CV = 719.4
CP = CV + R

T_INF = 300.0
U_INF = 433.9

# Read volume
grid = pv.read(VOLUME)

# Extract midplane
section = grid.slice(
    normal=(0, 1, 0),
    origin=(0, Y_MID, 0),
    generate_triangles=True
)

# Read cell-centred values
T = np.asarray(section.cell_data["T"])
U = np.asarray(section.cell_data["U"])

# Calculate total temperature
speed_squared = np.sum(U**2, axis=1)
T0 = T + speed_squared / (2 * CP)

# Freestream reference
T0_inf = T_INF + U_INF**2 / (2 * CP)

print("\n========== TOTAL TEMPERATURE ==========")
print(f"Cp:       {CP:.3f} J/(kg K)")
print(f"T0_inf:   {T0_inf:.3f} K")
print(f"Minimum:  {T0.min():.3f} K")
print(f"Maximum:  {T0.max():.3f} K")
print(f"Mean:     {T0.mean():.3f} K")
print(f"Median:   {np.median(T0):.3f} K")

# Build triangulation
points = section.points
faces = section.faces.reshape(-1, 4)

if not np.all(faces[:, 0] == 3):
    raise RuntimeError("Expected triangular slice cells.")

triangles = faces[:, 1:]

triangulation = mtri.Triangulation(
    points[:, 0],
    points[:, 2],
    triangles=triangles
)

# Read airfoil geometry
wall = pv.read(WALL)

profile = np.unique(
    np.round(wall.points[:, [0, 2]], decimals=8),
    axis=0
)

upper = profile[profile[:, 1] > 0]
lower = profile[profile[:, 1] < 0]

upper = upper[np.argsort(upper[:, 0])]
lower = lower[np.argsort(lower[:, 0])]

airfoil_x = np.concatenate([
    upper[:, 0],
    lower[::-1, 0]
])

airfoil_z = np.concatenate([
    upper[:, 1],
    lower[::-1, 1]
])

# Plot
fig, ax = plt.subplots(figsize=(13, 6))

contour = ax.tripcolor(
    triangulation,
    facecolors=T0,
    cmap="turbo",
    shading="flat"
)

ax.fill(
    airfoil_x,
    airfoil_z,
    facecolor="white",
    edgecolor="black",
    linewidth=1.3,
    zorder=5
)

ax.set_xlim(-0.2, 1.5)
ax.set_ylim(-0.35, 0.35)
ax.set_aspect("equal", adjustable="box")

ax.set_xlabel("x (m)")
ax.set_ylabel("z (m)")
ax.set_title("Stagnation Temperature Distribution")

cb = fig.colorbar(contour, ax=ax)
cb.set_label("Total temperature (K)")

fig.tight_layout()

OUTPUT.mkdir(parents=True, exist_ok=True)

filename = OUTPUT / "total_temperature.png"
fig.savefig(filename, dpi=220, bbox_inches="tight")

print("\nSaved:", filename)

plt.show()
