from pathlib import Path

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# ==================================================
# 1. SETTINGS
# ==================================================

VTK_FILE = Path("VTK/openfoam-case_200.vtk")
WALL_FILE = Path("VTK/aerofoil/aerofoil_200.vtk")

OUTPUT_DIR = Path("surfaceResults/flowContours")

# Midplane through the 2D computational domain
Y_MID = -0.05

# Near-airfoil viewing window
X_LIMITS = (-0.20, 1.50)
Z_LIMITS = (-0.35, 0.35)

# ==================================================
# 2. LOAD VOLUME AND EXTRACT MIDPLANE
# ==================================================

if not VTK_FILE.exists():
    raise FileNotFoundError(VTK_FILE)

grid = pv.read(VTK_FILE)

print("Loaded:", VTK_FILE)
print("Volume cells:", grid.n_cells)

section = grid.slice(
    normal=(0, 1, 0),
    origin=(0, Y_MID, 0),
    generate_triangles=True
)

print("\nMidplane:")
print("Points:", section.n_points)
print("Cells:", section.n_cells)

# ==================================================
# 3. EXTRACT TRIANGLE CONNECTIVITY
# ==================================================

# generate_triangles=True should give triangular polygons.
faces = section.faces.reshape(-1, 4)

if not np.all(faces[:, 0] == 3):
    raise RuntimeError(
        "Slice contains non-triangular cells. "
        "Inspect section.faces before plotting."
    )

triangles = faces[:, 1:]

points = section.points

# Coordinates in the physical X-Z plane
x = points[:, 0]
z = points[:, 2]

triangulation = mtri.Triangulation(
    x,
    z,
    triangles=triangles
)

print("Triangles:", len(triangles))

# ==================================================
# 4. LOAD AIRFOIL OUTLINE
# ==================================================

wall = pv.read(WALL_FILE)

# Remove duplicated points from the two extrusion sides.
profile_points = np.unique(
    np.round(wall.points[:, [0, 2]], decimals=8),
    axis=0
)

upper = profile_points[profile_points[:, 1] > 0]
lower = profile_points[profile_points[:, 1] < 0]

upper = upper[np.argsort(upper[:, 0])]
lower = lower[np.argsort(lower[:, 0])]

# Closed airfoil outline
airfoil_x = np.concatenate([
    upper[:, 0],
    lower[::-1, 0]
])

airfoil_z = np.concatenate([
    upper[:, 1],
    lower[::-1, 1]
])

# ==================================================
# 5. PLOTTING FUNCTION
# ==================================================

def plot_field(field, title, colorbar_label, filename):

    if field not in section.cell_data:
        raise KeyError(f"{field} not found in slice cell data.")

    values = np.asarray(section.cell_data[field])

    if len(values) != len(triangles):
        raise ValueError(
            f"{field}: {len(values)} values for "
            f"{len(triangles)} triangles."
        )

    fig, ax = plt.subplots(figsize=(13, 6))

    contour = ax.tripcolor(
        triangulation,
        facecolors=values,
        cmap="turbo",
        shading="flat"
    )

    # Draw the actual airfoil profile over the fluid field.
    ax.fill(
        airfoil_x,
        airfoil_z,
        facecolor="white",
        edgecolor="black",
        linewidth=1.3,
        zorder=5
    )

    ax.set_xlim(X_LIMITS)
    ax.set_ylim(Z_LIMITS)

    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(title)

    ax.grid(False)

    colorbar = fig.colorbar(contour, ax=ax)
    colorbar.set_label(colorbar_label)

    fig.tight_layout()

    output_file = OUTPUT_DIR / filename

    fig.savefig(
        output_file,
        dpi=220,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", output_file)

# ==================================================
# 6. GENERATE THREE FLOW CONTOURS
# ==================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n========== GENERATING CONTOURS ==========")

plot_field(
    "Ma",
    "Mach Number Distribution",
    "Mach number",
    "mach_contour.png"
)

plot_field(
    "p",
    "Static Pressure Distribution",
    "Pressure (Pa)",
    "pressure_contour.png"
)

plot_field(
    "T",
    "Static Temperature Distribution",
    "Temperature (K)",
    "temperature_contour.png"
)

print("\nAll contour plots generated successfully.")

