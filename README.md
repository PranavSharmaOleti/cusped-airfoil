# Cusped Airfoil — Coaxial Jet CFD

CFD study of a cusped (Coanda-type) airfoil using OpenFOAM. Includes geometry generation (DAT → STL), meshing, and RANS/LES simulation with post-processing.

## Structure

```
├── geometry/          # Airfoil geometry: DAT data, STL, conversion script
├── openfoam-case/     # OpenFOAM case: 0/, constant/, system/, postProcessing
├── docs/              # Guides, reports, reference materials
├── src/               # Python package (cusped_airfoil)
├── pyproject.toml     # Python project config
└── uv.lock            # Dependency lock
```

## Geometry

- `geometry/cusped_le_akram_hasan.dat` — 2D airfoil coordinate data (Akram & Hasan)
- `geometry/cusped_airfoil.stl` — 3D extruded STL for OpenFOAM
- `geometry/dat_to_stl.py` — converts DAT → STL

## OpenFOAM case

Standard OpenFOAM directory layout under `openfoam-case/`:
- `0/` — initial/boundary conditions (U, p, T)
- `constant/` — physical properties, triSurface geometry, polyMesh
- `system/` — controlDict, fvSchemes, fvSolution, blockMeshDict, etc.
- `postProcessing/` — extracted profiles, forces, surface data

## Running

```bash
cd openfoam-case
blockMesh
snappyHexMesh -overwrite
foamRun  # or simpleFoam / pimpleFoam
```

## Python

```bash
uv sync
uv run cusped-airfoil
```
