import numpy as np
from pathlib import Path
# import cairosvg

# --- Lorenz system and integration settings requested by the user ---
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0
T = 2.0
dt = 0.001  # shorter time step
n = int(T / dt)

# Three nearby initial conditions -> fewer trajectories, each of length 2 time units
initial_conditions = [
    np.array([ 0.20,  8.00, 20.20], dtype=float),
    np.array([ 0.00,  1.10, 20.00], dtype=float),
    np.array([ 0.22, -8.92, 19.85], dtype=float),
]


def rhs(s):
    x, y, z = s
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z,
    ], dtype=float)


def rk4_step(s, h):
    k1 = rhs(s)
    k2 = rhs(s + 0.5 * h * k1)
    k3 = rhs(s + 0.5 * h * k2)
    k4 = rhs(s + h * k3)
    return s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# Integrate trajectories
trajectories = []
for s0 in initial_conditions:
    s = s0.copy()
    pts = np.empty((n + 1, 3), dtype=float)
    pts[0] = s
    for i in range(n):
        s = rk4_step(s, dt)
        pts[i + 1] = s
    trajectories.append(pts)

# Project onto x-z plane to get the classic butterfly / Lorenz-attractor appearance
projected = [tr[:, [0, 2]] for tr in trajectories]
all_pts = np.vstack(projected)
mins = all_pts.min(axis=0)
maxs = all_pts.max(axis=0)

# Map to the left icon box while preserving aspect ratio.
# This box is tuned to preserve the feel and scale of the original uploaded SVG.
box_x, box_y = 6.0, 10.0
box_w, box_h = 255.0, 182.0
pad = 5.0
sx = (box_w - 2 * pad) / (maxs[0] - mins[0])
sy = (box_h - 2 * pad) / (maxs[1] - mins[1])
scale = min(sx, sy)

cx_data, cy_data = 0.5 * (mins[0] + maxs[0]), 0.5 * (mins[1] + maxs[1])
cx_svg, cy_svg = box_x + 0.5 * box_w, box_y + 0.5 * box_h


def map_point(p):
    x = cx_svg + scale * (p[0] - cx_data)
    y = cy_svg - scale * (p[1] - cy_data)
    return x, y


def make_path_data(points, step=3):
    xy = np.array([map_point(p) for p in points])
    keep = np.arange(0, len(xy), step)
    if keep[-1] != len(xy) - 1:
        keep = np.append(keep, len(xy) - 1)
    xy = xy[keep]
    d = [f"M {xy[0,0]:.2f} {xy[0,1]:.2f}"]
    for x, y in xy[1:]:
        d.append(f"L {x:.2f} {y:.2f}")
    return " ".join(d), np.array([map_point(p) for p in points])


# Keep style close to the original SVG: one light grey trajectory + two coloured highlighted trajectories.
base_path, base_xy = make_path_data(projected[0], step=3)
left_path, left_xy = make_path_data(projected[1], step=3)
right_path, right_xy = make_path_data(projected[2], step=3)

# Particle markers in Julia colours, placed along real trajectories.
# Fractions selected to visually echo the original logo style.
marker_specs = [
    (base_xy,  0.08, '#389826', 6.0),
    (base_xy,  0.18, '#9558b2', 4.5),
    (base_xy,  0.31, '#4063d8', 5.5),
    (base_xy,  0.47, '#cb3c33', 7.0),
    (base_xy,  0.65, '#389826', 5.5),
    (left_xy,  0.30, '#9558b2', 7.0),
    (left_xy,  0.55, '#4063d8', 6.5),
    (left_xy,  0.84, '#cb3c33', 7.0),
    (right_xy, 0.14, '#cb3c33', 4.5),
    (right_xy, 0.60, '#389826', 5.0),
    (right_xy, 0.87, '#9558b2', 6.0),
]
markers = []
for xy, frac, color, radius in marker_specs:
    idx = min(len(xy) - 1, max(0, int(round(frac * (len(xy) - 1)))))
    x, y = xy[idx]
    markers.append((x, y, color, radius))

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 280" role="img" aria-labelledby="title desc">
  <title id="title">Flows.jl logo</title>
  <desc id="desc">Flows.jl logo using real Lorenz-equation trajectories. Three open trajectories are shown, each integrated for {T:g} time units with time step {dt:g}, while preserving the style of the original logo.</desc>
  <defs>
    <linearGradient id="panel" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0" stop-color="#ffffff"/>
      <stop offset="1" stop-color="#f6f8fb"/>
    </linearGradient>
    <linearGradient id="left-trail" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0" stop-color="#4063d8"/>
      <stop offset="0.5" stop-color="#9558b2"/>
      <stop offset="1" stop-color="#cb3c33"/>
    </linearGradient>
    <linearGradient id="right-trail" x1="0" x2="1" y1="1" y2="0">
      <stop offset="0" stop-color="#389826"/>
      <stop offset="0.5" stop-color="#4063d8"/>
      <stop offset="1" stop-color="#9558b2"/>
    </linearGradient>
    <filter id="shadow" x="-10%" y="-20%" width="120%" height="140%">
      <feDropShadow dx="0" dy="8" stdDeviation="10" flood-color="#172033" flood-opacity="0.14"/>
    </filter>
  </defs>

  <rect x="24" y="24" width="872" height="232" rx="28" fill="url(#panel)" filter="url(#shadow)"/>

  <g transform="translate(58 42)">
    <path d="{base_path}"
          fill="none" stroke="#c9ced8" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>

    <path d="{left_path}"
          fill="none" stroke="url(#left-trail)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" opacity="0.78"/>
    <path d="{right_path}"
          fill="none" stroke="url(#right-trail)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" opacity="0.78"/>

    <g stroke="#ffffff" stroke-width="2.5">
'''
for x, y, fill, r in markers:
    svg += f'      <circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.1f}" fill="{fill}"/>\n'
svg += '''    </g>
  </g>

  <g transform="translate(360 76)">
    <text x="0" y="58" font-family="Inter, Avenir, Helvetica, Arial, sans-serif" font-size="64" font-weight="800" fill="#242936" letter-spacing="0">Flows.jl</text>
    <text x="3" y="105" font-family="Inter, Avenir, Helvetica, Arial, sans-serif" font-size="24" font-weight="600" fill="#5c6475" letter-spacing="0">flow maps for dynamical systems</text>
  </g>
</svg>
'''

svg_path = Path('logo_test.svg')
png_path = Path('logo_test.png')
svg_path.write_text(svg)
# cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=920, output_height=280)
print(svg_path)
print(png_path)
