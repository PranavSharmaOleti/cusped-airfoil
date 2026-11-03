import numpy as np
import pyvista as pv

FILE = "VTK/aerofoil/aerofoil_200.vtk"

mesh = pv.read(FILE)

# Calculate normals while retaining the original
# polygon orientation.
normal_mesh = mesh.compute_normals(
    cell_normals=True,
    point_normals=False,
    consistent_normals=False,
    auto_orient_normals=False,
    inplace=False
)

normals = np.asarray(
    normal_mesh.cell_data["Normals"]
)

centres = mesh.cell_centers().points
areas = np.asarray(
    mesh.compute_cell_sizes(
        length=False,
        area=True,
        volume=False
    ).cell_data["Area"]
)

x = centres[:, 0]
z = centres[:, 2]

upper = z > 0
lower = z < 0

print("========== NORMAL ORIENTATION ==========")

for name, mask in [
    ("UPPER", upper),
    ("LOWER", lower)
]:
    n = normals[mask]

    print(f"\n{name} SURFACE")
    print("Faces:", len(n))
    print("Mean normal X:", np.mean(n[:, 0]))
    print("Mean normal Y:", np.mean(n[:, 1]))
    print("Mean normal Z:", np.mean(n[:, 2]))

    print("Normal Z range:", np.min(n[:, 2]), np.max(n[:, 2]))

# Area-vector closure check
area_vectors = normals * areas[:, None]
resultant_area = np.sum(area_vectors, axis=0)

print("\n========== CLOSED-SURFACE CHECK ==========")
print("Sum of area vectors:", resultant_area)
print("Sum X:", resultant_area[0])
print("Sum Y:", resultant_area[1])
print("Sum Z:", resultant_area[2])

print("\n========== EXPECTED ORIENTATION ==========")
print("Upper surface normal Z should be predominantly negative.")
print("Lower surface normal Z should be predominantly positive.")
print("The total area vector should be approximately zero.")
