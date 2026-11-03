from pathlib import Path
from math import sqrt

src = Path("postProcessing/graphUniform/2e-05/line.xy")
out = src.with_name("line_profiles.svg")

# Read x, pressure, temperature and velocity components
rows = []
for line in src.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    v = [float(x) for x in line.split()]
    if len(v) >= 6:
        rows.append(v[:6])

if not rows:
    raise SystemExit(f"No usable data found in {src}")

x, p, T, Ux, Uy, Uz = zip(*rows)

# Calculate Mach number from velocity and ideal-gas properties
R = 8314.462618 / 28.966
Cv = 719.4
gamma = (Cv + R) / Cv

speed = [sqrt(a*a + b*b + c*c) for a, b, c in zip(Ux, Uy, Uz)]
Ma = [u / sqrt(gamma * R * t) for u, t in zip(speed, T)]

plots = [
    ("Pressure", "p (Pa)", p, 1775.544),
    ("Temperature", "T (K)", T, 300.0),
    ("Mach number", "Ma (-)", Ma, 1.25),
    ("Vertical velocity", "Uz (m/s)", Uz, 0.0),
]

W, H = 1000, 880
left, right = 105, 960
pw = right - left
panel_h, gap, top0 = 145, 60, 45

s = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    '<rect width="100%" height="100%" fill="white"/>',
    '<style>text{font-family:Arial,sans-serif;fill:#222}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#333;stroke-width:1}.data{fill:none;stroke:#1769aa;stroke-width:2}.ref{stroke:#d24b40;stroke-width:1.5;stroke-dasharray:6 4}</style>',
    '<text x="500" y="25" font-size="19" text-anchor="middle">Airfoil line profiles at t = 2e-05 s</text>'
]

xmin, xmax = min(x), max(x)
if xmax == xmin:
    raise SystemExit("All sample x coordinates are identical.")

for i, (title, ylabel, vals, ref) in enumerate(plots):
    ytop = top0 + i * (panel_h + gap)
    ymin = min(min(vals), ref)
    ymax = max(max(vals), ref)
    pad = (ymax - ymin) * 0.08 or max(abs(ymax) * 0.03, 1.0)
    ymin -= pad
    ymax += pad

    def X(a):
        return left + (a - xmin) / (xmax - xmin) * pw

    def Y(a):
        return ytop + panel_h - (a - ymin) / (ymax - ymin) * panel_h

    s.append(f'<text x="{left}" y="{ytop-10}" font-size="15" font-weight="bold">{title}</text>')

    for j in range(5):
        val = ymin + (ymax - ymin) * j / 4
        yy = Y(val)
        s.append(f'<line class="grid" x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}"/>')
        s.append(f'<text x="{left-10}" y="{yy+4:.2f}" font-size="11" text-anchor="end">{val:.5g}</text>')

    s.append(f'<line class="axis" x1="{left}" y1="{ytop}" x2="{left}" y2="{ytop+panel_h}"/>')
    s.append(f'<line class="axis" x1="{left}" y1="{ytop+panel_h}" x2="{right}" y2="{ytop+panel_h}"/>')

    yr = Y(ref)
    s.append(f'<line class="ref" x1="{left}" y1="{yr:.2f}" x2="{right}" y2="{yr:.2f}"/>')
    s.append(f'<text x="{right-3}" y="{yr-5:.2f}" font-size="10" text-anchor="end">reference: {ref:g}</text>')

    pts = " ".join(f"{X(a):.2f},{Y(b):.2f}" for a, b in zip(x, vals))
    s.append(f'<polyline class="data" points="{pts}"/>')

    s.append(f'<text x="20" y="{ytop+panel_h/2}" font-size="12" text-anchor="middle" transform="rotate(-90 20 {ytop+panel_h/2})">{ylabel}</text>')

    for a in (xmin, (xmin+xmax)/2, xmax):
        s.append(f'<text x="{X(a):.2f}" y="{ytop+panel_h+18}" font-size="11" text-anchor="middle">{a:.3g}</text>')

s.append(f'<text x="{(left+right)/2}" y="{H-8}" font-size="13" text-anchor="middle">Position x (m)</text>')
s.append('</svg>')

out.write_text("\n".join(s), encoding="utf-8")
print(f"Read {len(rows)} sample points.")
print(f"Saved plot: {out}")
print("Mach range:", f"{min(Ma):.5f} to {max(Ma):.5f}")
