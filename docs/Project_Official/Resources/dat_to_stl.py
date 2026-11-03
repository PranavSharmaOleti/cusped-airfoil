#!/usr/bin/env python3
"""Extrude a 2-D airfoil .dat coordinate loop into a watertight ASCII STL.

Dependencies: Python standard library only.

Usage:
  python dat_to_stl.py input.dat output.stl --thickness 0.1

Coordinates are used as-is. If the .dat chord is normalized to 1.0, a thickness
of 0.1 creates a span of 0.1 in the same units (10% of chord).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

Point2 = tuple[float, float]
Point3 = tuple[float, float, float]
Triangle = tuple[Point3, Point3, Point3]


def read_dat(path: Path) -> list[Point2]:
    """Read x/y coordinates, skipping titles, comments, and non-numeric lines."""
    points: list[Point2] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.split("#", 1)[0].split("//", 1)[0].strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if math.isfinite(x) and math.isfinite(y):
                points.append((x, y))

    if len(points) < 3:
        raise ValueError(
            f"Only {len(points)} valid coordinate pairs found in {path}. "
            "Expected an airfoil outline with at least 3 points."
        )

    # Remove consecutive duplicate points and a repeated final copy of the first point.
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    tol = 1e-12 * scale
    cleaned: list[Point2] = []
    for p in points:
        if not cleaned or math.hypot(p[0] - cleaned[-1][0], p[1] - cleaned[-1][1]) > tol:
            cleaned.append(p)
    if len(cleaned) > 1 and math.hypot(
        cleaned[0][0] - cleaned[-1][0], cleaned[0][1] - cleaned[-1][1]
    ) <= tol:
        cleaned.pop()

    if len(cleaned) < 3:
        raise ValueError("The coordinate file collapses to fewer than 3 unique points.")

    # Airfoil files can contain upper/lower leading-edge samples that are
    # mathematically distinct but closer than common STL vertex-welding tolerances.
    # If the two points immediately surrounding the minimum-x nose are this close,
    # discard those two samples and keep the nose point. This changes only a tiny
    # near-zero-thickness region and helps avoid degenerate facets during meshing.
    chord = max(p[0] for p in cleaned) - min(p[0] for p in cleaned)
    tip_tol = 1e-8 * max(chord, 1e-12)
    nose_i = min(range(len(cleaned)), key=lambda i: cleaned[i][0])
    if len(cleaned) >= 5:
        prev_i = (nose_i - 1) % len(cleaned)
        next_i = (nose_i + 1) % len(cleaned)
        p_prev, p_next = cleaned[prev_i], cleaned[next_i]
        if math.hypot(p_prev[0] - p_next[0], p_prev[1] - p_next[1]) <= tip_tol:
            # Remove by index in descending order so positions do not shift.
            for idx in sorted((prev_i, next_i), reverse=True):
                cleaned.pop(idx)
            print(
                f"Note: removed two near-coincident leading-edge samples "
                f"(separation <= {tip_tol:.3g}, coordinate units)."
            )

    if len(cleaned) < 3:
        raise ValueError("The coordinate file collapses to fewer than 3 unique points.")
    return cleaned


def signed_area(poly: list[Point2]) -> float:
    return 0.5 * sum(
        poly[i][0] * poly[(i + 1) % len(poly)][1]
        - poly[(i + 1) % len(poly)][0] * poly[i][1]
        for i in range(len(poly))
    )


def cross2(a: Point2, b: Point2, c: Point2) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def point_in_triangle(p: Point2, a: Point2, b: Point2, c: Point2, eps: float) -> bool:
    # Inclusive test is intentional: it avoids making an invalid ear across another vertex.
    c1 = cross2(a, b, p)
    c2 = cross2(b, c, p)
    c3 = cross2(c, a, p)
    return c1 >= -eps and c2 >= -eps and c3 >= -eps


def triangulate_ccw(poly: list[Point2]) -> list[tuple[int, int, int]]:
    """Ear-clip a simple, counter-clockwise polygon; return CCW index triangles."""
    n = len(poly)
    scale = max(
        max(p[0] for p in poly) - min(p[0] for p in poly),
        max(p[1] for p in poly) - min(p[1] for p in poly),
        1.0,
    )
    eps = 1e-14 * scale * scale
    remaining = list(range(n))
    triangles: list[tuple[int, int, int]] = []
    max_iterations = n * n
    iterations = 0

    while len(remaining) > 3:
        ear_found = False
        m = len(remaining)
        for j in range(m):
            i_prev = remaining[(j - 1) % m]
            i_curr = remaining[j]
            i_next = remaining[(j + 1) % m]
            a, b, c = poly[i_prev], poly[i_curr], poly[i_next]

            # A valid ear must be strictly convex for a CCW polygon.
            if cross2(a, b, c) <= eps:
                continue

            blocked = False
            for k in remaining:
                if k in (i_prev, i_curr, i_next):
                    continue
                if point_in_triangle(poly[k], a, b, c, eps):
                    blocked = True
                    break
            if blocked:
                continue

            triangles.append((i_prev, i_curr, i_next))
            del remaining[j]
            ear_found = True
            break

        iterations += 1
        if not ear_found or iterations > max_iterations:
            raise ValueError(
                "Could not triangulate the airfoil outline. Check that the .dat "
                "coordinates form one simple, non-self-intersecting closed loop "
                "in perimeter order (upper surface around the nose to lower surface)."
            )

    if len(remaining) == 3:
        a, b, c = (poly[i] for i in remaining)
        if cross2(a, b, c) > eps:
            triangles.append((remaining[0], remaining[1], remaining[2]))
        else:
            raise ValueError("The final cap triangle is degenerate; check the input coordinates.")

    return triangles


def normal(a: Point3, b: Point3, c: Point3) -> Point3:
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    nx = ab[1] * ac[2] - ab[2] * ac[1]
    ny = ab[2] * ac[0] - ab[0] * ac[2]
    nz = ab[0] * ac[1] - ab[1] * ac[0]
    mag = math.sqrt(nx * nx + ny * ny + nz * nz)
    if mag <= 0.0:
        return (0.0, 0.0, 0.0)
    return (nx / mag, ny / mag, nz / mag)


def write_ascii_stl(triangles: list[Triangle], output_path: Path, solid_name: str = "airfoil") -> None:
    with output_path.open("w", encoding="ascii", newline="\n") as f:
        f.write(f"solid {solid_name}\n")
        for a, b, c in triangles:
            nx, ny, nz = normal(a, b, c)
            f.write(f"  facet normal {nx:.9e} {ny:.9e} {nz:.9e}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {a[0]:.9e} {a[1]:.9e} {a[2]:.9e}\n")
            f.write(f"      vertex {b[0]:.9e} {b[1]:.9e} {b[2]:.9e}\n")
            f.write(f"      vertex {c[0]:.9e} {c[1]:.9e} {c[2]:.9e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {solid_name}\n")


def convert(dat_path: Path, stl_path: Path, thickness: float) -> int:
    if not dat_path.is_file():
        raise FileNotFoundError(f"Input file not found: {dat_path}")
    if thickness <= 0 or not math.isfinite(thickness):
        raise ValueError("Thickness must be a finite number greater than zero.")

    poly = read_dat(dat_path)
    area = signed_area(poly)
    if abs(area) < 1e-14:
        raise ValueError("The outline has approximately zero area; check the coordinate file.")

    # Use CCW perimeter winding. This makes cap and side-face normals consistent.
    if area < 0:
        poly.reverse()

    cap_tris = triangulate_ccw(poly)
    z0 = 0.0
    z1 = thickness
    bottom: list[Point3] = [(x, y, z0) for x, y in poly]
    top: list[Point3] = [(x, y, z1) for x, y in poly]
    faces: list[Triangle] = []
    n = len(poly)

    # Extruded perimeter walls. Polygon is CCW, so these faces point outward.
    for i in range(n):
        j = (i + 1) % n
        faces.append((bottom[i], bottom[j], top[j]))
        faces.append((bottom[i], top[j], top[i]))

    # Closed end caps: bottom normal points -Z; top normal points +Z.
    for i, j, k in cap_tris:
        faces.append((bottom[k], bottom[j], bottom[i]))
        faces.append((top[i], top[j], top[k]))

    stl_path.parent.mkdir(parents=True, exist_ok=True)
    write_ascii_stl(faces, stl_path, solid_name="extruded_airfoil")

    print(f"Input points after duplicate cleanup : {len(poly)}")
    print(f"Surface triangles written            : {len(faces)}")
    print(f"Extrusion thickness                  : {thickness:g} (same units as .dat)")
    print(f"Output STL                           : {stl_path.resolve()}")
    print("Note: STL is a closed extruded solid; STL files do not store units.")
    return len(faces)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert an airfoil .dat outline into a closed, extruded ASCII STL."
    )
    parser.add_argument("input_dat", type=Path, help="Airfoil coordinate file (.dat)")
    parser.add_argument("output_stl", type=Path, help="Output STL filename")
    parser.add_argument(
        "--thickness", type=float, default=0.1,
        help="Extrusion along Z, in the same units as the coordinates (default: 0.1)",
    )
    args = parser.parse_args()

    try:
        convert(args.input_dat, args.output_stl, args.thickness)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
