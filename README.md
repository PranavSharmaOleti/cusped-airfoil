# Cusped Airfoil — OpenFOAM Reconstruction

Research prototype for compressible flow over a cusped (Coanda-type) leading-edge airfoil. The geometry is taken from Akram & Hasan's published airfoil coordinates and the case is set up in OpenFOAM Foundation v13. This repository contains a **short laminar startup run** and post-processing tooling; it is **not a reproduction of the published results**.

---

## Status

| Item | State |
|---|---|
| Airfoil coordinate data → STL conversion | Implemented (`geometry/dat_to_stl.py`) |
| 2D structured mesh + 1-cell extrusion | Implemented (`blockMeshDict`, `extrudeMeshDict`) |
| Compressible laminar solve | Implemented (`shockFluid`, 200 time steps) |
| Post-processing (forces, coefficients, contours, wall shear) | Implemented; outputs committed |
| Turbulence model (RANS/LES) | **Not implemented** — proposed future work |
| Mesh independence / domain / time-step studies | **Not performed** |
| Comparison against the published paper | **Not performed** |
| Automated `Allrun`-style driver | **Not present** |

The current force coefficients are transient outputs of a 2×10⁻⁵ s run. They are **not validated aerodynamic data**.

---

## Source paper and operating condition

**Paper:** Saif Akram and Nadeem Hasan, "The dynamics of supersonic flow past a new cusped leading edge airfoil", *Progress in Computational Fluid Dynamics*, Vol. 24, No. 1, 2024. DOI 10.1504/PCFD.2022.10052925. A local copy is `docs/Project_Official/Resources/PCFD_Full.pdf`.

Conditions stated in the paper:

| Quantity | Paper value |
|---|---|
| Reynolds number (chord-based) | 5 × 10⁵ |
| Mach number sweep | 1.25–2.13 |
| Angles of attack | 0°, 10°, 20° |
| Chord | 1 m |
| Maximum thickness | ≈ 0.05 c at x/c ≈ 0.75 |
| Far field | 80-chord diameter |
| Mesh | O-grid, 425 × 315 ≈ 134,000 cells, minimum spacing 1 × 10⁻⁴ |
| Wall | Adiabatic, no-slip |
| Gas model | γ = 1.4, Sutherland viscosity, Pr = 0.743 |

The paper itself computes **laminar compressible** flow at the stated Reynolds number; it does not present RANS or LES results. Any RANS/LES work is therefore a proposed extension, not a reproduction of the paper.

---

## Current case implementation

### Geometry

- `geometry/cusped_le_akram_hasan.dat` — two-dimensional airfoil coordinates.
- `geometry/dat_to_stl.py` — converts the coordinate file to an STL surface.
- `geometry/cusped_airfoil.stl` — generated extruded geometry (0.1 m span).
- `openfoam-case/constant/triSurface/` and `constant/geometry/` contain working copies used by the case dictionaries.

### Flow conditions (`openfoam-case/0/`)

| Field | Inlet value | Wall / outlet |
|---|---|---|
| `U` | (433.9, 0, 0) m/s | `noSlip` on `aerofoil`, `zeroGradient` at outlet |
| `p` | 1775.544 Pa | `zeroGradient` on `aerofoil` and outlet |
| `T` | 300 K | `zeroGradient` on `aerofoil` and outlet |

With γ = 1.4 and R = 287 J/(kg·K), these give a freestream Mach number of approximately 1.25 at zero angle of attack.

### Physical models and solver

| File | Setting |
|---|---|
| `system/controlDict` | `solver shockFluid` (density-based compressible solver) |
| `constant/momentumTransport` | `simulationType laminar` |
| `constant/physicalProperties` | Sutherland viscosity, perfect gas, sensible enthalpy |
| `system/fvSchemes`, `system/fvSolution` | Numerical schemes and linear solvers as committed |

There is **no `snappyHexMeshDict`** in this case; meshing uses `blockMesh` followed by `extrudeMesh`. There is no RANS or LES turbulence model.

### Mesh

`system/blockMeshDict` defines a structured O-grid over x ∈ [−4, 4] m, z ∈ [−2, 2] m with 80 upstream, 30 middle, 40 downstream and 80 vertical cells per half, giving roughly 24,000 hex cells. `system/extrudeMeshDict` extrudes the `back` patch by 0.1 m in y with one cell through the span. Relative to the 1 m chord, the domain is about 4 chords upstream, 3 chords downstream and 4 chords vertically — not the paper's 80-chord far field.

Saved `checkMesh` diagnostics are in `openfoam-case/postProcessing/checkMesh/constant/`; the case has documented high non-orthogonality and skewness near the leading edge. No mesh-independence study has been run.

### Time controls and meaning of the results

```
startTime   0
endTime     2e-05 s
deltaT      1e-07 s
writeInterval 5e-06 s
```

That is 200 time steps and four written fields (5×10⁻⁶, 1×10⁻⁵, 1.5×10⁻⁵, 2×10⁻⁵ s). The chord flow-through time is c/U∞ = 1/433.9 ≈ 2.3×10⁻³ s, so the run covers **less than 1 %** of one flow-through time. The case has not reached a statistically steady or converged aerodynamic state.

At the last written time, `openfoam-case/postProcessing/forceCoeffsCompressible/2e-05/forceCoeffs.dat` reports `Cd ≈ 0.0571`, `Cl ≈ 0`, `Cm ≈ −1.7×10⁻⁷` with `Aref = 0.1 m²` (chord × 0.1 m span). These are instantaneous transient values at t = 2×10⁻⁵ s, with a span-based reference area, and are **not** comparable to per-unit-span published coefficients.

---

## Running the case

The case files are complete for the implemented laminar solve. From `openfoam-case/`:

```bash
blockMesh
extrudeMesh
checkMesh
shockFluid
```

The run writes time directories and the function-object outputs configured in `system/functions` (`forceCoeffsCompressible`, `forcesCompressible`). No `Allrun`/`Allclean` script is committed yet; commands must be issued manually. OpenFOAM Foundation v13 is implied by the dictionaries but has not been pinned in a build file.

Python post-processing scripts are in `openfoam-case/postProcessingScripts/`; they expect the VTK surfaces in `openfoam-case/VTK/` and write to `openfoam-case/surfaceResults/`. Several scripts hardcode the `*_200.vtk` filenames, so re-running `foamToVTK` at the final time is required after a new solve.

The `cusped-airfoil` Python package is a placeholder only; there is no working console-script entry point. Use the scripts directly with a Python environment containing `numpy`, `pandas`, `matplotlib` and `pyvista`.

---

## Repository layout

| Path | Contents |
|---|---|
| `geometry/` | Airfoil coordinates, converter script and generated STL |
| `openfoam-case/0/`, `constant/`, `system/` | Complete case: fields, physical properties, mesh and solver dictionaries |
| `openfoam-case/archive/` | Earlier mesh/log files from development runs |
| `openfoam-case/postProcessing*/` | Function-object outputs and Python analysis scripts |
| `openfoam-case/surfaceResults/` | Exported forces, contours, surface distributions and plots |
| `openfoam-case/VTK/` | Surface and volume VTK exports used by the scripts |
| `docs/Project_Official/` | Source paper, guides, chat export and reference resources |
| `src/cusped_airfoil/` | Placeholder Python package (no CLI) |

Some files under `docs/Project_Official/Resources/` are byte-identical copies of `geometry/` files; they are retained as the original project material and are not used by the OpenFOAM case.

---

## What remains to be done

1. Improve leading-edge mesh quality (high non-orthogonality/skewness) and run a mesh-independence study.
2. Extend the far-field domain toward the paper's 80-chord boundary and test the effect.
3. Run for a physically meaningful time (at least several chord flow-through times) and monitor force convergence.
4. Decide whether a turbulence model is required at the target Reynolds number; implement it explicitly if so, as separate work.
5. Re-run at Mach 1.25, α = 0° and compare with the paper before claiming reproduction.
6. Add an `Allrun`/`Allclean` driver and replace hardcoded VTK paths in post-processing scripts.

## Reference

Akram, S., Hasan, N., "The dynamics of supersonic flow past a new cusped leading edge airfoil", *Progress in Computational Fluid Dynamics*, 24(1), 2024, DOI: 10.1504/PCFD.2022.10052925. Local copy: `docs/Project_Official/Resources/PCFD_Full.pdf`.
