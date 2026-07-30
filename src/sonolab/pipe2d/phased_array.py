"""Phased-array beamforming inside a 2D circular pipe (NDT-style practice).

Ported into SonoLab from ``wavelab-3d/practice/phased_array.py`` so pipe
dent / wall reconstruction lives next to SPECFEM job packing. Headless
``reconstruct_pipe`` / ``calibrate_c_known_radius`` need only NumPy.
Live OpenGL UI (``run_live``) needs optional glfw + PyOpenGL.

Goal: a tool *inside* the pipe fires a look-aligned
linear array (sector scan). Classical (no-ML) multi-angle A-scans reconstruct
the inner-wall outline via standoff → free-center circle fit → residual.

Device assumption (this pass): perfect electronics — no noise, dead elements,
digitizer error, crosstalk, calibration drift, or clock error. Only
environment / geometry / device–environment interaction is modeled:
  • tool pose offset relative to pipe (eccentricity)
  • pipe geometry (circle + optional inward dent)
  • wave physics in the medium + wall interaction
  • fluid wave-speed *assumption* used for ranging (c_assumed) vs true
    propagation speed C_PROP in the FDTD grid

Physics guidance from wave.py — 2D leapfrog scalar wave:

    u[n+1] = 2*u[n] - u[n-1] + C(x,y)^2 * Laplacian5(u[n])

Circular pipe: fluid disk, hard Dirichlet annulus (clean ToF), exterior
outside OD also Dirichlet (no square-box ghost echoes). Dent = inward
cosine radial bump on the ID.

Array: linear aperture in the *tool frame* (perfect device), perpendicular to
look û, shifted slightly along û (backlobe suppression). Soft-source drive
with broadside delays. Eccentricity moves the tool/array center off the
geometric pipe center — delays/receive stay ideal relative to that center.

Wave-speed calibration (press `k`) — joint known-R / free-center:
  In UT, c is not free: you need a trusted physical scale. Here the operator
  may trust nominal ID = R_known (tape/spec), but must *not* assume the tool
  is centered. Joint cal:
    1. Measure t(θ) over looks (prefer dent OFF; math still allows ecc).
    2. For candidate c, map echoes → tool-frame points (tool at origin).
    3. Fit free-center circle (x_c, y_c, R_fit); unknown ecc → (x_c, y_c).
    4. 1D search on c until R_fit ≈ R_known (known ID fixes scale → c).
  Forbidden (removed): centered-only c = 2(R_known − offset)/〈Δt〉.
  This sim does not retune CFL/physics. Propagation stays at C_PROP (truth).
  Ranging/reconstruction use c_assumed (starts at 1.1·C_PROP until `k`).

Classical reconstruction (press `m`):
  1. Sweep θ = 0..359° at 1° — fire one gated pulse per look.
  2. Gate/pick from operator priors: cold-start center from c_assumed + R_known
     (half-width covers plausible ecc); then track previous t_echo — no true
     pose/dent/god-geometry in the pick path.
  3. Standoff r(θ) = c_assumed·(t_echo − t_tx)/2 + array_offset  (from tool)
  4. Map picks in *tool frame* (origin = tool): p = r·û; fit free circle.
  5. Residuals = R_fit − |p − c_fit|  (operational dent call).
     Pedagogy contrast: residualsForcedOrigin assumes pipe center = tool
     (WRONG under ecc) and fakes once-per-rev ovality; not used operationally.
     Wrong c_assumed biases mean R; after joint `k`, fit R ≈ R_known.

Deps: numpy, glfw, PyOpenGL  (same venv as wave.py)

Keys:
  [ ]     rotate look left / right (degrees, wraps 0..359)
  - =     fewer / more elements
  1 / 2   short pulse / continuous-wave
  0 / 5   no dent / inward wall dent; 0/5 also arm auto baseline/now
  e       toggle eccentricity on/off (demo offset along +X when on)
  E / , . decrease / increase eccentricity magnitude (cells)
  b       freeze baseline snapshot (pulse, prefer dent off)
  n       optional re-freeze now/compare snapshot
  k       joint cal: known R_known + free-center fit → c_assumed (prefer dent off; ecc OK)
  K       reset c_assumed to uncalibrated default (1.1·C_PROP)
  m       run 1° multi-angle classical reconstruction (no ML)
  v       toggle bottom pane: A-scan compare ↔ reconstruction plot
  a       A-scan axis: time (steps) ↔ distance (d = c·t/2)
  p       toggle A-scan pick overlay (gate / thresh / t_echo); default ON
  ← →     after m: scrub recon look; A-scan shows that angle's pick
  r       reset
  q/Esc   quit

Markers: yellow = transducers + look ray; cyan = wall; orange = dent; dark = exterior
A-scan pick overlay: cyan = gate, yellow = pick (t_echo), gray = 0.5·max in gate
Bottom: A-scan or reconstruction (true / measured / fit circle; pipe vs tool center)
Headless: --reconstruct [--no-dent] [--ecc N] [--ecc-phi DEG] [--c-assumed C]
         --calibrate [--no-dent] [--ecc N]  (joint known-R / free-center → c)
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import time
from typing import Any

import numpy as np

# glfw / OpenGL imported lazily inside run_live (optional for headless).

# --- resolution / physics ---
# HIRES≈1.6× linear vs prior ~120-tall channel: wall several cells thick,
# dent spans multiple cells, gate/discretization error visible vs true outline.
HIRES = True
if HIRES:
    NX = NY = 192
    WALL_THICK = 12
    PIPE_R = 68.0  # nominal inner radius (cells) to wall interface
    DENT_DEPTH = 8.0  # peak inward radial bump (cells)
    DENT_HALF_DEG = 22.0  # cosine half-width in degrees
    # Linear aperture (sector-scan): pitch·N ≳ λ for look directivity.
    # λ ≈ 2π/ω ≈ 2π/0.28 ≈ 22 cells; 20 elems × pitch 2 ≈ 38-cell aperture.
    ELEM_PITCH = 2
    STEPS_PER_SECOND = 48.0
else:
    NX = NY = 128
    WALL_THICK = 8
    PIPE_R = 46.0
    DENT_DEPTH = 5.0
    DENT_HALF_DEG = 20.0
    ELEM_PITCH = 2
    STEPS_PER_SECOND = 60.0

CFL = 0.45
C0 = 1.0
C_WALL = 0.55

CX = (NX - 1) * 0.5
CY = (NY - 1) * 0.5
DENT_THETA0_DEG = 0.0  # dent centered at +Y (θ=0 look hits dent first)

N_ELEM_DEFAULT = 20
FREQ = 0.28  # rad/step (ω)
PULSE_WIDTH = 22.0
AMP = 0.40
# Shift aperture toward the look wall so the forward echo arrives before the
# bidirectional backlobe (linear arrays radiate both ±û).
ARRAY_LOOK_OFFSET = 6.0

# Tool eccentricity (pose offset): tool center ≠ pipe geometric center.
# Polar: (ecc_mag, ecc_phi) with φ=0 → +Y, φ=90 → +X (same as look_unit).
ECC_DEMO = 10.0  # cells; used when toggling eccentricity on
ECC_MAX = 18.0
ECC_STEP = 2.0
ECC_PHI_DEFAULT_DEG = 90.0  # demo offset along +X

# Propagation speed in grid cells per timestep (FDTD truth — never retuned by cal)
C_PROP = CFL * C0
# Ranging / reconstruction assumption: starts deliberately wrong until `k`.
# Pedagogy: keep physics at C_PROP; only the estimator lies until calibrated.
C_ASSUMED_DEFAULT = 1.1 * C_PROP
# Acoustic one-way from look-offset array to nominal ID (operator R_known prior,
# *centered* path length — used only as gate *center* prior, never as "e=0 truth")
WALL_ONE_WAY_AC = float(PIPE_R - ARRAY_LOOK_OFFSET)
# Nominal center→ID (labels / true geometry / physics buffer sizing)
WALL_ONE_WAY = float(PIPE_R)
# Physics/sim buffer time scale (FDTD truth C_PROP) — NOT the estimator gate.
WALL_GATE_T = PULSE_WIDTH + 2.0 * WALL_ONE_WAY_AC / C_PROP
ASCAN_MARGIN = 55.0
# Cover farthest eccentric look: R + e − array_offset (sim buffer, not pick)
_ASCAN_FAR = float(PIPE_R + ECC_MAX - ARRAY_LOOK_OFFSET)
ASCAN_LEN = int(
    math.ceil(PULSE_WIDTH + 2.0 * _ASCAN_FAR / C_PROP + ASCAN_MARGIN)
)
PULSE_REFIRE_GAP = 80

# Estimator gate priors (operator-allowed): R_known, array offset, ecc/dent
# uncertainty half-width — never true pose, true dent state, or C_PROP.
TX_EXCLUDE = int(PULSE_WIDTH * 1.8)
GATE_ECC_PRIOR = float(ECC_MAX)  # plausible |e| for gate width
GATE_DENT_PRIOR = float(DENT_DEPTH)  # metal-loss / dent uncertainty (prior)
GATE_R_MARGIN = 8.0

RECON_STEP_DEG = 1  # full 1° sweep
RECON_DISPLAY_STEP = 2  # downsample for canvas / dense polyline draw
CAL_STEP_DEG = 10  # known-radius calibration look spacing

VERT = """
#version 330 core
layout(location=0) in vec2 in_pos;
layout(location=1) in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAG = """
#version 330 core
in vec2 v_uv;
uniform sampler2D tex;
out vec4 f_color;
void main() { f_color = texture(tex, v_uv); }
"""

LINE_VERT = """
#version 330 core
layout(location=0) in vec2 in_pos;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

LINE_FRAG = """
#version 330 core
uniform vec4 color;
out vec4 f_color;
void main() { f_color = color; }
"""


def coolwarm_rgba(field: np.ndarray, peak: float | None = None) -> np.ndarray:
    peak = float(peak) if peak is not None else float(np.max(np.abs(field)))
    peak = max(peak, 1e-6)
    t = np.clip(0.5 + 0.5 * (field / peak), 0.0, 1.0)
    r = np.clip(np.where(t < 0.5, 0.23 + 1.54 * t, 0.40 + 0.60 * t), 0, 1)
    g = np.clip(np.where(t < 0.5, 0.30 + 1.00 * t, 1.40 - 1.00 * t), 0, 1)
    b = np.clip(np.where(t < 0.5, 0.75 + 0.50 * t, 1.50 - 1.50 * t), 0, 1)
    a = np.ones_like(r)
    return (np.stack((r, g, b, a), axis=-1) * 255).astype(np.uint8)


def wrap_deg(theta: float) -> float:
    t = float(theta) % 360.0
    return t + 360.0 if t < 0 else t


def angle_diff_deg(a: float, b: float) -> float:
    """Signed shortest difference a−b in (−180, 180]."""
    d = (float(a) - float(b) + 180.0) % 360.0 - 180.0
    return d


def dent_inward_at_theta(theta_deg: float) -> float:
    """Inward radial bump (cells) at polar angle θ (0=+Y, 90=+X)."""
    dth = angle_diff_deg(theta_deg, DENT_THETA0_DEG) / float(DENT_HALF_DEG)
    if abs(dth) >= 1.0:
        return 0.0
    return float(DENT_DEPTH * 0.5 * (1.0 + math.cos(math.pi * dth)))


def true_inner_radius(theta_deg: float, *, dent: bool) -> float:
    """Analytic ground-truth inner wall radius at look angle θ (pipe-centered)."""
    bump = dent_inward_at_theta(theta_deg) if dent else 0.0
    return float(PIPE_R - bump)


def look_unit(theta_deg: float) -> tuple[float, float]:
    """û = (sin θ, cos θ): θ=0 → +Y, θ=90 → +X."""
    th = math.radians(theta_deg)
    return math.sin(th), math.cos(th)


def tool_offset_xy(ecc_mag: float, ecc_phi_deg: float = ECC_PHI_DEFAULT_DEG) -> tuple[float, float]:
    """Pipe-frame offset of tool center from geometric pipe center (cells).

    Eccentricity = tool not centered in the pipe (pose interaction only).
    φ uses the same convention as look_unit: 0 → +Y, 90 → +X.
    """
    e = max(0.0, float(ecc_mag))
    ux, uy = look_unit(ecc_phi_deg)
    return e * ux, e * uy


def tool_center_grid(
    ecc_mag: float = 0.0,
    ecc_phi_deg: float = ECC_PHI_DEFAULT_DEG,
) -> tuple[float, float]:
    """Absolute grid coordinates of the tool/array center."""
    ex, ey = tool_offset_xy(ecc_mag, ecc_phi_deg)
    return CX + ex, CY + ey


def circle_ray_standoff(
    ecc_x: float,
    ecc_y: float,
    theta_deg: float,
    radius: float,
) -> float:
    """One-way range from eccentric origin to a circle of given radius at pipe origin.

    Classic eccentric-caliper root: r = −e·û + √(R² − |e×û|²).
    """
    ux, uy = look_unit(theta_deg)
    edot = float(ecc_x) * ux + float(ecc_y) * uy
    e2 = float(ecc_x) * float(ecc_x) + float(ecc_y) * float(ecc_y)
    R = float(radius)
    disc = edot * edot - e2 + R * R
    return float(-edot + math.sqrt(max(disc, 0.0)))


def true_standoff_from_tool(
    theta_deg: float,
    *,
    dent: bool,
    ecc_x: float = 0.0,
    ecc_y: float = 0.0,
) -> float:
    """Analytic tool→wall standoff along look û (pipe ID may have a dent).

    Iterates the circular eccentric formula with local ID radius at the hit
    angle so dented walls stay accurate under eccentricity.
    """
    ux, uy = look_unit(theta_deg)
    r = circle_ray_standoff(ecc_x, ecc_y, theta_deg, PIPE_R)
    for _ in range(10):
        hx = float(ecc_x) + r * ux
        hy = float(ecc_y) + r * uy
        th_hit = math.degrees(math.atan2(hx, hy))
        R_loc = true_inner_radius(th_hit, dent=dent)
        r_new = circle_ray_standoff(ecc_x, ecc_y, theta_deg, R_loc)
        if abs(r_new - r) < 1e-4:
            return float(r_new)
        r = r_new
    return float(r)


def estimator_gate_center(
    *,
    c_assumed: float | None = None,
    R_known: float = PIPE_R,
) -> float:
    """Expected echo time from operator priors only (no pose/dent truth).

    Uses nominal centered path L = R_known − ARRAY_LOOK_OFFSET as the *gate
    center* prior, with c_assumed. Eccentricity and dent are handled by making
    the gate wide enough (see estimator_gate_half) — not by injecting true e.
    """
    c = float(C_ASSUMED_DEFAULT if c_assumed is None else c_assumed)
    c = max(c, 1e-9)
    r_ac = max(4.0, float(R_known) - ARRAY_LOOK_OFFSET)
    return float(PULSE_WIDTH + 2.0 * r_ac / c)


def estimator_gate_half(
    *,
    c_assumed: float | None = None,
    ecc_prior: float = GATE_ECC_PRIOR,
    dent_prior: float = GATE_DENT_PRIOR,
) -> float:
    """Gate half-width covering plausible R±e and dent prior uncertainty.

    Priors are design/operator uncertainty bounds, not measured truth. Time
    scale uses c_assumed (estimator), never C_PROP.
    """
    c = float(C_ASSUMED_DEFAULT if c_assumed is None else c_assumed)
    c = max(c, 1e-9)
    # One-way path from aperture ≈ R − offset ± e (−dent); Δr ≈ e + dent + margin
    delta_r = float(ecc_prior) + float(dent_prior) + float(GATE_R_MARGIN)
    return float(
        max(
            2.0 * delta_r / c,
            0.18 * WALL_ONE_WAY_AC / c,
        )
    )


def gate_center_for_look(
    theta_deg: float,
    *,
    dent: bool,
    ecc_x: float = 0.0,
    ecc_y: float = 0.0,
    c_assumed: float | None = None,
) -> float:
    """DEPRECATED truth-geometry gate — do not use in the estimator path.

    Kept only for pedagogy/HUD contrast. Operational picks use
    estimator_gate_center + estimator_gate_half.
    """
    c = float(C_ASSUMED_DEFAULT if c_assumed is None else c_assumed)
    c = max(c, 1e-9)
    r = true_standoff_from_tool(
        theta_deg, dent=dent, ecc_x=ecc_x, ecc_y=ecc_y
    )
    r_ac = max(4.0, r - ARRAY_LOOK_OFFSET)
    return float(PULSE_WIDTH + 2.0 * r_ac / c)


def _polar_r_theta() -> tuple[np.ndarray, np.ndarray]:
    """Cell-centered radius and polar angle (0=+Y, +toward +X)."""
    yy, xx = np.mgrid[0:NY, 0:NX]
    dx = xx.astype(np.float64) - CX
    dy = yy.astype(np.float64) - CY
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.degrees(np.arctan2(dx, dy))  # (−180, 180]
    return r, theta


def _inner_outer_radii(*, dent: bool) -> tuple[np.ndarray, np.ndarray]:
    r, theta = _polar_r_theta()
    if dent:
        dth = (theta - DENT_THETA0_DEG + 180.0) % 360.0 - 180.0
        u = dth / float(DENT_HALF_DEG)
        bump = np.where(
            np.abs(u) < 1.0,
            DENT_DEPTH * 0.5 * (1.0 + np.cos(np.pi * u)),
            0.0,
        )
    else:
        bump = 0.0
    r_inner = PIPE_R - bump
    r_outer = r_inner + float(WALL_THICK)
    return r_inner, r_outer


def pipe_wall_mask(*, dent: bool = False) -> np.ndarray:
    """True on soft annular wall band [r_inner, r_outer).

    Inner interface: r = PIPE_R − dent(θ). Soft wall occupies
    [r_inner, r_inner + WALL_THICK). Domain cells with r ≥ r_outer are
    *exterior* (Dirichlet) — not fluid — so waves cannot transmit past the
    pipe into the square-domain corners and fake a late echo.
    """
    r, _ = _polar_r_theta()
    r_inner, r_outer = _inner_outer_radii(dent=dent)
    mask = (r >= r_inner) & (r < r_outer)
    mask[0, :] = mask[-1, :] = False
    mask[:, 0] = mask[:, -1] = False
    return mask


def pipe_exterior_mask(*, dent: bool = False) -> np.ndarray:
    """True outside the pipe outer radius (and domain frame) — Dirichlet sink."""
    r, _ = _polar_r_theta()
    _, r_outer = _inner_outer_radii(dent=dent)
    ext = r >= r_outer
    ext[0, :] = ext[-1, :] = True
    ext[:, 0] = ext[:, -1] = True
    return ext


def dent_bulge_mask() -> np.ndarray:
    """Wall cells present only because of the inward dent (tint overlay)."""
    intact = pipe_wall_mask(dent=False)
    dented = pipe_wall_mask(dent=True)
    return dented & ~intact


def make_c_map(wall: np.ndarray) -> np.ndarray:
    c = np.full((NY, NX), C0, dtype=np.float64)
    c[wall] = C_WALL
    return c

def aperture_span_cells(n_elem: int, *, elem_pitch: float | None = None) -> float:
    """Linear aperture span in grid cells: (N-1)·pitch."""
    n = max(int(n_elem), 2)
    pitch = float(ELEM_PITCH if elem_pitch is None else elem_pitch)
    return float(n - 1) * pitch


def array_positions(
    n_elem: int,
    theta_deg: float = 0.0,
    *,
    ecc_mag: float = 0.0,
    ecc_phi_deg: float = ECC_PHI_DEFAULT_DEG,
) -> np.ndarray:
    """Linear array in the tool frame, perpendicular to look û (sector scan).

    Aperture is shifted ARRAY_LOOK_OFFSET along û from the *tool* center so the
    forward wall is the nearest reflector (suppresses ±û backlobe ambiguity).
    Tool center may be eccentric relative to the pipe geometric center.
    """
    n = max(int(n_elem), 2)
    tx, ty = tool_center_grid(ecc_mag, ecc_phi_deg)
    ux, uy = look_unit(theta_deg)
    px, py = uy, -ux
    span = aperture_span_cells(n)
    offsets = np.linspace(-0.5 * span, 0.5 * span, n)
    xs = tx + ARRAY_LOOK_OFFSET * ux + offsets * px
    ys = ty + ARRAY_LOOK_OFFSET * uy + offsets * py
    return np.column_stack([xs, ys]).astype(np.float64)


def steering_delays(positions: np.ndarray, theta_deg: float) -> np.ndarray:
    """Focus delays along look û (array already aimed; mild focusing)."""
    ux, uy = look_unit(theta_deg)
    # Plane-wave / broadside along û: relative delay from projection on û
    proj = positions[:, 0] * ux + positions[:, 1] * uy
    return (proj - float(np.mean(proj))) / C0


def _interp_rx_hist(hist: list[float], t_src: float) -> float:
    """Linear interpolate a per-element RX history at fractional time index."""
    if not hist:
        return 0.0
    if t_src <= 0.0:
        return float(hist[0])
    last = len(hist) - 1
    if t_src >= last:
        return float(hist[last])
    i0 = int(math.floor(t_src))
    i1 = i0 + 1
    frac = float(t_src) - float(i0)
    return float(hist[i0]) * (1.0 - frac) + float(hist[i1]) * frac


def pick_echo_result(
    series: np.ndarray,
    *,
    gate_center: float | None = None,
    gate_half: float | None = None,
    c_assumed: float | None = None,
    R_known: float = PIPE_R,
    gate_half_extra: float = 0.0,
    prefer_nearest: bool = True,
) -> dict[str, float | int]:
    """Pick wall echo and expose gate/threshold used (for ranging + A-scan viz).

    Cold start (wide R_known prior): prefer_nearest=True picks the peak closest
    to the nominal center (rejects late ringing). Tracking: prefer_nearest=False
    takes the strongest peak near the previous pick (avoids RF-cycle lag).

    Gate center/half come from operator priors or previous-look continuity —
    never true pose, true dent state, or C_PROP.

    Returns keys: t_echo, idx, lo, hi, center, half, peak, thresh.
    """
    n = int(series.size)
    c_est = float(C_ASSUMED_DEFAULT if c_assumed is None else c_assumed)
    if n < 4:
        t0 = float(PULSE_WIDTH)
        i0 = int(PULSE_WIDTH)
        return {
            "t_echo": t0,
            "idx": i0,
            "lo": i0,
            "hi": min(n, i0 + 1),
            "center": t0,
            "half": 0.0,
            "peak": 0.0,
            "thresh": 0.0,
        }
    center = float(
        estimator_gate_center(c_assumed=c_est, R_known=R_known)
        if gate_center is None
        else gate_center
    )
    half = float(
        estimator_gate_half(c_assumed=c_est)
        if gate_half is None
        else gate_half
    )
    if gate_half_extra > 0.0:
        half += 2.0 * float(gate_half_extra) / max(c_est, 1e-9)
    lo = max(TX_EXCLUDE, int(round(center - half)))
    hi = min(n, int(round(center + half)) + 1)
    if hi <= lo + 2:
        lo, hi = TX_EXCLUDE, n
    window = np.abs(series[lo:hi])
    if window.size == 0:
        return {
            "t_echo": float(lo),
            "idx": lo,
            "lo": lo,
            "hi": hi,
            "center": center,
            "half": float(half),
            "peak": 0.0,
            "thresh": 0.0,
        }
    peak = float(np.max(window))
    if peak < 1e-12:
        return {
            "t_echo": float(lo),
            "idx": lo,
            "lo": lo,
            "hi": hi,
            "center": center,
            "half": float(half),
            "peak": peak,
            "thresh": 0.0,
        }
    thresh = 0.50 * peak
    candidates: list[int] = []
    for i in range(1, window.size - 1):
        if (
            window[i] >= thresh
            and window[i] >= window[i - 1]
            and window[i] >= window[i + 1]
        ):
            candidates.append(lo + i)
    if not candidates:
        idx = lo + int(np.argmax(window))
    elif prefer_nearest:
        idx = min(candidates, key=lambda t: abs(t - center))
    else:
        idx = max(candidates, key=lambda t: float(np.abs(series[t])))
    return {
        "t_echo": float(idx),
        "idx": int(idx),
        "lo": int(lo),
        "hi": int(hi),
        "center": float(center),
        "half": float(half),
        "peak": float(peak),
        "thresh": float(thresh),
    }


def pick_echo_time(
    series: np.ndarray,
    *,
    gate_center: float | None = None,
    gate_half: float | None = None,
    c_assumed: float | None = None,
    R_known: float = PIPE_R,
    gate_half_extra: float = 0.0,
    prefer_nearest: bool = True,
) -> tuple[float, int]:
    """Pick wall echo time (steps). Thin wrapper over pick_echo_result."""
    d = pick_echo_result(
        series,
        gate_center=gate_center,
        gate_half=gate_half,
        c_assumed=c_assumed,
        R_known=R_known,
        gate_half_extra=gate_half_extra,
        prefer_nearest=prefer_nearest,
    )
    return float(d["t_echo"]), int(d["idx"])


def estimator_track_gate_half(*, c_assumed: float | None = None) -> float:
    """Half-width for look-to-look continuity (no true ecc).

    Sized for one angular step at plausible ecc (+ small margin), *not* the
    full cold-start ±ECC_MAX window — a too-wide track gate re-admits late
    ringing/second-bounce lobes that can outrank the wall echo.
    """
    c = float(C_ASSUMED_DEFAULT if c_assumed is None else c_assumed)
    c = max(c, 1e-9)
    # Δr per ~CAL_STEP at max ecc; +margin for dent edge / discretization
    step_r = float(GATE_ECC_PRIOR) * math.sin(
        math.radians(max(float(CAL_STEP_DEG), 5.0))
    )
    delta_r = step_r + 6.0
    # Also keep at least ~1 RF half-cycle so multi-cycle structure stays inside
    half_cycle = math.pi / max(float(FREQ), 1e-6)
    return float(max(2.0 * delta_r / c, half_cycle))


def standoff_from_echo(
    t_echo: float,
    *,
    c_assumed: float | None = None,
) -> float:
    """One-way standoff from the *tool* center (cells).

    Echo time is measured from the look-offset aperture; add ARRAY_LOOK_OFFSET
    so reported r is tool-center→wall (reconstruction maps into tool frame).
    Uses c_assumed (estimator), never C_PROP — wrong c biases range until cal.
    """
    c = float(C_ASSUMED_DEFAULT if c_assumed is None else c_assumed)
    r_from_array = c * (float(t_echo) - PULSE_WIDTH) / 2.0
    return float(r_from_array + ARRAY_LOOK_OFFSET)


def fit_circle_xy(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    """Algebraic least-squares circle fit (Kåsa): x²+y² + D x + E y + F = 0.

    Returns (cx, cy, R). Falls back to centroid + mean radius if singular.
    """
    x = np.asarray(xs, dtype=np.float64).ravel()
    y = np.asarray(ys, dtype=np.float64).ravel()
    if x.size < 3:
        return float(np.mean(x)), float(np.mean(y)), float(np.mean(np.hypot(x, y)))
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)
    try:
        sol, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        if rank < 3:
            raise np.linalg.LinAlgError("rank deficient")
        D, E, F = sol
        cx = -0.5 * D
        cy = -0.5 * E
        r2 = cx * cx + cy * cy - F
        R = math.sqrt(max(r2, 1e-12))
        return float(cx), float(cy), float(R)
    except np.linalg.LinAlgError:
        cx = float(np.mean(x))
        cy = float(np.mean(y))
        R = float(np.mean(np.hypot(x - cx, y - cy)))
        return cx, cy, max(R, 1e-6)


class AScanBuffer:
    """Amplitude-vs-time at the receive aperture.

    Pulse mode (rolling=False): one-shot fill, then freeze until clear().
    CW mode (rolling=True): classic ring buffer (baseline compare disabled).
    """

    def __init__(self, length: int = ASCAN_LEN):
        self.length = int(length)
        self.buf = np.zeros(self.length, dtype=np.float64)
        self.i = 0
        self.filled = 0
        self.rolling = False

    def set_rolling(self, rolling: bool) -> None:
        self.rolling = bool(rolling)
        self.clear()

    def clear(self):
        self.buf.fill(0.0)
        self.i = 0
        self.filled = 0

    @property
    def complete(self) -> bool:
        return self.filled >= self.length

    def push(self, value: float) -> bool:
        v = float(value)
        if self.rolling:
            self.buf[self.i] = v
            self.i = (self.i + 1) % self.length
            self.filled = min(self.filled + 1, self.length)
            return True
        if self.filled >= self.length:
            return False
        self.buf[self.filled] = v
        self.filled += 1
        return True

    def series(self) -> np.ndarray:
        if self.filled == 0:
            return np.zeros(0, dtype=np.float64)
        if self.rolling and self.filled >= self.length:
            return np.concatenate([self.buf[self.i :], self.buf[: self.i]])
        return self.buf[: self.filled].copy()


class PhasedPipe2D:
    def __init__(
        self,
        n_elem: int = N_ELEM_DEFAULT,
        theta_deg: float = 0.0,
        continuous: bool = False,
        dent: bool = False,
        ecc_mag: float = 0.0,
        ecc_phi_deg: float = ECC_PHI_DEFAULT_DEG,
    ):
        self.n_elem = int(n_elem)
        self.theta_deg = wrap_deg(theta_deg)
        self.continuous = bool(continuous)
        self.dent_on = bool(dent)
        self.ecc_mag = float(np.clip(ecc_mag, 0.0, ECC_MAX))
        self.ecc_phi_deg = wrap_deg(ecc_phi_deg)
        self.wall = pipe_wall_mask(dent=self.dent_on)
        self.exterior = pipe_exterior_mask(dent=self.dent_on)
        self.dent = dent_bulge_mask() if self.dent_on else np.zeros((NY, NX), dtype=bool)
        self.c_map = make_c_map(self.wall)
        self.u = np.zeros((NY, NX), dtype=np.float64)
        self.up = np.zeros((NY, NX), dtype=np.float64)
        self.un = np.zeros((NY, NX), dtype=np.float64)
        self.time = 0
        self._rebuild_array()

    @property
    def tool_offset(self) -> tuple[float, float]:
        return tool_offset_xy(self.ecc_mag, self.ecc_phi_deg)

    @property
    def tool_xy(self) -> tuple[float, float]:
        return tool_center_grid(self.ecc_mag, self.ecc_phi_deg)

    def _rebuild_array(self):
        self.positions = array_positions(
            self.n_elem,
            self.theta_deg,
            ecc_mag=self.ecc_mag,
            ecc_phi_deg=self.ecc_phi_deg,
        )
        self.delays = steering_delays(self.positions, self.theta_deg)
        # Per-element RX history for coherent delay-and-sum (same τ as TX).
        self._rx_hist: list[list[float]] = [[] for _ in range(len(self.positions))]

    def reset(
        self,
        n_elem: int | None = None,
        theta_deg: float | None = None,
        continuous: bool | None = None,
        dent: bool | None = None,
        ecc_mag: float | None = None,
        ecc_phi_deg: float | None = None,
    ):
        if n_elem is not None:
            self.n_elem = int(np.clip(n_elem, 2, 48))
        if theta_deg is not None:
            self.theta_deg = wrap_deg(theta_deg)
        if continuous is not None:
            self.continuous = bool(continuous)
        if ecc_mag is not None:
            self.ecc_mag = float(np.clip(ecc_mag, 0.0, ECC_MAX))
        if ecc_phi_deg is not None:
            self.ecc_phi_deg = wrap_deg(ecc_phi_deg)
        if dent is not None:
            self.dent_on = bool(dent)
            self.wall = pipe_wall_mask(dent=self.dent_on)
            self.exterior = pipe_exterior_mask(dent=self.dent_on)
            self.dent = (
                dent_bulge_mask() if self.dent_on else np.zeros((NY, NX), dtype=bool)
            )
            self.c_map = make_c_map(self.wall)
        self._rebuild_array()
        self.u.fill(0.0)
        self.up.fill(0.0)
        self.un.fill(0.0)
        self.time = 0

    def _drive(self, t: float) -> None:
        for (x, y), tau in zip(self.positions, self.delays):
            ix = int(round(x))
            iy = int(round(y))
            if ix < 2 or ix >= NX - 2 or iy < 2 or iy >= NY - 2:
                continue
            if self.exterior[iy, ix] or self.wall[iy, ix]:
                continue
            td = t - tau
            if self.continuous:
                sig = AMP * math.sin(FREQ * td)
            else:
                env = math.exp(-0.5 * ((td - PULSE_WIDTH) / (PULSE_WIDTH * 0.35)) ** 2)
                sig = AMP * env * math.sin(FREQ * td)
            self.u[iy, ix] += sig

    def step(self):
        self._drive(float(self.time))
        u, up, un = self.u, self.up, self.un
        cmap = (CFL * self.c_map) ** 2
        un.fill(0.0)
        lap = (
            u[1:-1, 2:]
            + u[1:-1, :-2]
            + u[2:, 1:-1]
            + u[:-2, 1:-1]
            - 4.0 * u[1:-1, 1:-1]
        )
        un[1:-1, 1:-1] = 2.0 * u[1:-1, 1:-1] - up[1:-1, 1:-1] + cmap[1:-1, 1:-1] * lap
        # Hard inner-wall reflection (Dirichlet on annulus) so classical ToF
        # tracks the geometric ID — soft C_WALL alone was too weak for dent
        # standoff shifts. Exterior OD also Dirichlet (no square-box ghosts).
        un[self.wall] = 0.0
        un[self.exterior] = 0.0
        un[0, :] = un[-1, :] = 0.0
        un[:, 0] = un[:, -1] = 0.0
        self.up, self.u, self.un = u, un, up
        self.time += 1

    def sample_receive(self) -> float:
        """Coherent DAS receive: delay-aligned mean (same steering τ as TX).

        Instantaneous samples are taken 1 cell along look (TX bang reduce),
        appended to a per-element history, then summed at t − τ_i.
        """
        ux, uy = look_unit(self.theta_deg)
        inst: list[float] = []
        for x, y in self.positions:
            ix = int(round(x + ux))
            iy = int(round(y + uy))
            if 1 <= ix < NX - 1 and 1 <= iy < NY - 1:
                if not (self.exterior[iy, ix] or self.wall[iy, ix]):
                    inst.append(float(self.u[iy, ix]))
                    continue
            inst.append(0.0)
        if not any(abs(v) > 0.0 for v in inst):
            tx, ty = self.tool_xy
            ix = int(np.clip(round(tx + 2 * ux), 1, NX - 2))
            iy = int(np.clip(round(ty + 2 * uy), 1, NY - 2))
            fallback = float(self.u[iy, ix])
            inst = [fallback for _ in self.positions]

        if len(self._rx_hist) != len(inst):
            self._rx_hist = [[] for _ in range(len(inst))]
        for i, v in enumerate(inst):
            self._rx_hist[i].append(float(v))

        # After step(), self.time already advanced; last written index is time-1.
        t_rec = float(max(self.time - 1, 0))
        aligned: list[float] = []
        for i, tau in enumerate(self.delays):
            aligned.append(_interp_rx_hist(self._rx_hist[i], t_rec - float(tau)))
        if not aligned:
            return 0.0
        return float(np.mean(aligned))


def field_image(sim: PhasedPipe2D) -> np.ndarray:
    img = coolwarm_rgba(sim.u)
    img = img.copy()
    wall = sim.wall
    img[wall, 0] = (0.45 * img[wall, 0] + 0.55 * 50).astype(np.uint8)
    img[wall, 1] = (0.45 * img[wall, 1] + 0.55 * 160).astype(np.uint8)
    img[wall, 2] = (0.45 * img[wall, 2] + 0.55 * 190).astype(np.uint8)
    # exterior outside pipe OD — dark so the circular bore reads clearly
    ext = sim.exterior & ~wall
    img[ext] = (18, 18, 22, 255)
    if sim.dent_on:
        dn = sim.dent
        img[dn] = (230, 90, 40, 255)
    for x, y in sim.positions:
        ix = int(round(x))
        iy = int(round(y))
        if 1 <= ix < NX - 1 and 1 <= iy < NY - 1:
            img[iy, ix] = (255, 220, 40, 255)
    # pipe geometric center (cyan) vs tool center (magenta) when eccentric
    pcx, pcy = int(round(CX)), int(round(CY))
    if 1 <= pcx < NX - 1 and 1 <= pcy < NY - 1:
        img[pcy, pcx] = (80, 220, 255, 255)
    tx, ty = sim.tool_xy
    tcx, tcy = int(round(tx)), int(round(ty))
    if 1 <= tcx < NX - 1 and 1 <= tcy < NY - 1:
        img[tcy, tcx] = (255, 80, 220, 255)
    return np.ascontiguousarray(img)


def true_wall_xy(*, dent: bool, step_deg: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Analytic true inner-wall polyline in pipe-centered coords."""
    thetas = np.arange(0.0, 360.0, step_deg)
    rs = np.array([true_inner_radius(t, dent=dent) for t in thetas])
    th = np.radians(thetas)
    xs = rs * np.sin(th)
    ys = rs * np.cos(th)
    return xs, ys


def record_pulse_window(
    *,
    theta_deg: float,
    dent: bool,
    n_elem: int = N_ELEM_DEFAULT,
    ecc_mag: float = 0.0,
    ecc_phi_deg: float = ECC_PHI_DEFAULT_DEG,
) -> np.ndarray:
    """Headless one-shot gated A-scan (same receive model as the live demo)."""
    sim = PhasedPipe2D(
        n_elem=n_elem,
        theta_deg=theta_deg,
        continuous=False,
        dent=dent,
        ecc_mag=ecc_mag,
        ecc_phi_deg=ecc_phi_deg,
    )
    ascan = AScanBuffer(ASCAN_LEN)
    while not ascan.complete:
        sim.step()
        ascan.push(sim.sample_receive())
    return ascan.series().copy()


def _tool_frame_circle_from_times(
    t_echo: np.ndarray,
    thetas: np.ndarray,
    c: float,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Map echo times → tool-frame picks at assumed c; free-center circle fit.

    Tool at origin. Returns (cx, cy, R, mx, my).
    """
    c = max(float(c), 1e-9)
    th_rad = np.radians(np.asarray(thetas, dtype=np.float64))
    ux = np.sin(th_rad)
    uy = np.cos(th_rad)
    r = np.array(
        [standoff_from_echo(float(t), c_assumed=c) for t in t_echo],
        dtype=np.float64,
    )
    mx = r * ux
    my = r * uy
    cx, cy, R = fit_circle_xy(mx, my)
    return cx, cy, R, mx, my


def calibrate_c_known_radius(
    *,
    n_elem: int = N_ELEM_DEFAULT,
    step_deg: float = CAL_STEP_DEG,
    dent: bool = False,
    ecc_mag: float = 0.0,
    ecc_phi_deg: float = ECC_PHI_DEFAULT_DEG,
    R_known: float = PIPE_R,
    c_before: float | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Joint wave-speed cal: known R + free-center fit (unknown eccentricity OK).

    Field analogue: trust nominal ID = R_known (tape/spec). Do *not* assume the
    tool is centered. Measure t(θ), then 1D-search c such that the free-center
    circle fit in the tool frame has R_fit ≈ R_known. Unknown ecc is absorbed
    into (x_c, y_c); known ID fixes scale → c.

    Prefer dent OFF: metal loss / dent breaks “round ID = R_known” and can bias
    the prior. Math still does not assume centering when dent is on.

    Demoted (not used): centered-only c = 2(R_known − offset)/mean(Δt).
    """
    ecc_mag = float(np.clip(ecc_mag, 0.0, ECC_MAX))
    ecc_phi_deg = wrap_deg(ecc_phi_deg)
    c0 = float(C_ASSUMED_DEFAULT if c_before is None else c_before)
    R_known = float(R_known)
    L_centered = max(1.0, R_known - ARRAY_LOOK_OFFSET)  # demoted diagnostic only

    warnings: list[str] = []
    if dent:
        warnings.append(
            "dent ON — R_known prior may be biased (metal loss/dent breaks "
            "round ID = R_known); prefer dent OFF (key 0) for clean cal"
        )

    thetas = np.arange(0.0, 360.0, float(step_deg))
    t_echo = np.zeros(thetas.size, dtype=np.float64)
    g0 = estimator_gate_center(c_assumed=c0, R_known=R_known)
    h_wide = estimator_gate_half(c_assumed=c0)
    h_track = estimator_track_gate_half(c_assumed=c0)
    prev_t: float | None = None
    t0 = time.perf_counter()
    for i, th in enumerate(thetas):
        series = record_pulse_window(
            theta_deg=float(th),
            dent=dent,
            n_elem=n_elem,
            ecc_mag=ecc_mag,
            ecc_phi_deg=ecc_phi_deg,
        )
        if prev_t is None:
            # Cold start: peak nearest R_known prior (rejects late ringing).
            te, _ = pick_echo_time(
                series,
                gate_center=g0,
                gate_half=h_wide,
                c_assumed=c0,
                R_known=R_known,
                prefer_nearest=True,
            )
        else:
            # Track: strongest peak near previous pick (avoid RF-cycle lag).
            te, _ = pick_echo_time(
                series,
                gate_center=prev_t,
                gate_half=h_track,
                c_assumed=c0,
                R_known=R_known,
                prefer_nearest=False,
            )
        t_echo[i] = te
        prev_t = te
        if progress and (i % 6 == 0 or i == thetas.size - 1):
            elapsed = time.perf_counter() - t0
            print(
                f"  cal {i + 1}/{thetas.size}  θ={th:.0f}°  "
                f"t_echo={te:.1f}  ({elapsed:.1f}s)",
                flush=True,
            )

    dt = t_echo - float(PULSE_WIDTH)
    dt_pos = dt[dt > 1.0]
    if dt_pos.size < max(3, thetas.size // 4):
        raise RuntimeError(
            "calibration failed: too few valid echo times "
            f"(got {dt_pos.size}/{thetas.size})"
        )
    dt_mean = float(np.mean(dt_pos))
    # Demoted centered-only formula (WRONG under ecc) — diagnostic only.
    c_centered_only = float(2.0 * L_centered / dt_mean)

    def _R_at(c: float) -> tuple[float, float, float]:
        cx, cy, R, _, _ = _tool_frame_circle_from_times(t_echo, thetas, c)
        return cx, cy, R

    # Bracket c so R_fit(c_lo) ≤ R_known ≤ R_fit(c_hi) (R ∝ c, roughly).
    c_lo = max(1e-6, 0.5 * c0)
    c_hi = max(c_lo * 1.01, 2.0 * c0)
    _, _, R_lo = _R_at(c_lo)
    _, _, R_hi = _R_at(c_hi)
    for _ in range(24):
        if R_lo <= R_known <= R_hi:
            break
        if R_known < R_lo:
            c_hi, R_hi = c_lo, R_lo
            c_lo = max(1e-6, 0.5 * c_lo)
            _, _, R_lo = _R_at(c_lo)
        else:
            c_lo, R_lo = c_hi, R_hi
            c_hi = 2.0 * c_hi
            _, _, R_hi = _R_at(c_hi)
    else:
        if not (R_lo <= R_known <= R_hi):
            raise RuntimeError(
                "calibration failed: could not bracket R_known with c search "
                f"(R_lo={R_lo:.3f} R_hi={R_hi:.3f} R_known={R_known:.3f})"
            )

    fit_cx = fit_cy = fit_R = 0.0
    c_after = 0.5 * (c_lo + c_hi)
    for _ in range(48):
        c_mid = 0.5 * (c_lo + c_hi)
        fit_cx, fit_cy, fit_R = _R_at(c_mid)
        if fit_R < R_known:
            c_lo = c_mid
        else:
            c_hi = c_mid
        c_after = c_mid
    fit_cx, fit_cy, fit_R = _R_at(c_after)
    ecc_est = float(math.hypot(fit_cx, fit_cy))  # |pipe ctr| in tool frame

    return {
        "c_true": float(C_PROP),
        "c_before": c0,
        "c_after": float(c_after),
        "R_known": R_known,
        "R_fit": round(fit_R, 6),
        "fitCenter": {"x": round(fit_cx, 6), "y": round(fit_cy, 6)},
        "eccEstimate": round(ecc_est, 6),
        "arrayLookOffset": float(ARRAY_LOOK_OFFSET),
        "dt_mean": dt_mean,
        "nLooks": int(thetas.size),
        "nValid": int(dt_pos.size),
        "stepDeg": float(step_deg),
        "dent": bool(dent),
        "eccMag": round(ecc_mag, 6),
        "eccPhiDeg": round(float(ecc_phi_deg), 6),
        "thetaDeg": thetas.tolist(),
        "tEcho": [round(float(v), 4) for v in t_echo],
        "warnings": warnings,
        "method": (
            "joint known-R / free-center: search c so R_fit(c)≈R_known; "
            "unknown ecc absorbed into (x_c,y_c); tool-frame picks"
        ),
        # Demoted: centered-only formula — do not use when ecc is possible.
        "c_centeredOnly_WRONG": round(c_centered_only, 8),
        "L_centered_WRONG": round(L_centered, 6),
    }


def reconstruct_pipe(
    *,
    dent: bool = True,
    n_elem: int = N_ELEM_DEFAULT,
    step_deg: float = RECON_STEP_DEG,
    progress: bool = True,
    ecc_mag: float = 0.0,
    ecc_phi_deg: float = ECC_PHI_DEFAULT_DEG,
    store_ascans: bool = False,
    c_assumed: float | None = None,
) -> dict[str, Any]:
    """Classical no-ML pipe reconstruction from multi-angle A-scans.

    Pipeline: steer → pulse → estimator gate pick → r = c_assumed·(t−t_tx)/2
    (tool frame) → free-center algebraic circle fit → residuals vs *fit*.

    Device is assumed perfect; eccentricity is a tool–pipe pose offset only.
    Estimator never uses true ecc / true center / C_PROP / true dent for
    ranging or residuals (truth fields are HUD / verification only).
    c_assumed is the ranging speed (defaults to C_ASSUMED_DEFAULT until calibrated).
    If store_ascans, attach raw pulse windows under key "ascans" (live scrub only;
    not JSON-serializable — omit for headless dumps).
    """
    ecc_mag = float(np.clip(ecc_mag, 0.0, ECC_MAX))
    ecc_phi_deg = wrap_deg(ecc_phi_deg)
    # Truth pose for physics sim placement + HUD overlays only (not estimator).
    ex, ey = tool_offset_xy(ecc_mag, ecc_phi_deg)
    c_est = float(C_ASSUMED_DEFAULT if c_assumed is None else c_assumed)
    g0 = estimator_gate_center(c_assumed=c_est, R_known=PIPE_R)
    h_wide = estimator_gate_half(c_assumed=c_est)
    h_track = estimator_track_gate_half(c_assumed=c_est)

    thetas = np.arange(0.0, 360.0, float(step_deg))
    r_meas = np.zeros(thetas.size, dtype=np.float64)
    r_true = np.zeros(thetas.size, dtype=np.float64)
    t_echo = np.zeros(thetas.size, dtype=np.float64)
    ascan_store: list[np.ndarray] | None = [] if store_ascans else None
    prev_t: float | None = None
    t0 = time.perf_counter()
    for i, th in enumerate(thetas):
        series = record_pulse_window(
            theta_deg=float(th),
            dent=dent,
            n_elem=n_elem,
            ecc_mag=ecc_mag,
            ecc_phi_deg=ecc_phi_deg,
        )
        if ascan_store is not None:
            ascan_store.append(series.copy())
        if prev_t is None:
            te, _ = pick_echo_time(
                series,
                gate_center=g0,
                gate_half=h_wide,
                c_assumed=c_est,
                R_known=PIPE_R,
                prefer_nearest=True,
            )
        else:
            te, _ = pick_echo_time(
                series,
                gate_center=prev_t,
                gate_half=h_track,
                c_assumed=c_est,
                R_known=PIPE_R,
                prefer_nearest=False,
            )
        # Standoff from tool center: round-trip minus TX envelope → one-way.
        r_meas[i] = standoff_from_echo(te, c_assumed=c_est)
        # Truth standoff for verification prints / HUD only.
        r_true[i] = true_standoff_from_tool(
            float(th), dent=dent, ecc_x=ex, ecc_y=ey
        )
        t_echo[i] = te
        prev_t = te
        if progress and (i % 30 == 0 or i == thetas.size - 1):
            elapsed = time.perf_counter() - t0
            print(
                f"  recon {i + 1}/{thetas.size}  θ={th:.0f}°  "
                f"r_meas={r_meas[i]:.2f}  r_true={r_true[i]:.2f}  "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    th_rad = np.radians(thetas)
    ux = np.sin(th_rad)
    uy = np.cos(th_rad)
    # Tool-frame picks (estimator): tool at origin — do NOT inject true ecc.
    mx = r_meas * ux
    my = r_meas * uy
    fit_cx, fit_cy, fit_R = fit_circle_xy(mx, my)
    # Free-center residual: positive when measured point is inside fitted circle
    r_from_fit = np.hypot(mx - fit_cx, my - fit_cy)
    residuals = fit_R - r_from_fit
    # WRONG: assumes centered (pipe center = tool origin) — pedagogy contrast only
    residuals_forced = float(np.mean(r_meas)) - r_meas
    # Truth wall in tool frame for HUD overlay (translate by −true ecc).
    true_xs_pipe, true_ys_pipe = true_wall_xy(dent=dent, step_deg=float(step_deg))
    true_xs = true_xs_pipe - ex
    true_ys = true_ys_pipe - ey

    dent_mask = np.array(
        [abs(angle_diff_deg(t, DENT_THETA0_DEG)) < DENT_HALF_DEG for t in thetas]
    )
    peak_res_i = int(np.argmax(residuals)) if residuals.size else 0
    peak_forced_i = int(np.argmax(residuals_forced)) if residuals_forced.size else 0

    out: dict[str, Any] = {
        "meta": {
            "nx": NX,
            "ny": NY,
            "hires": HIRES,
            "pipeR": PIPE_R,
            "wallThick": WALL_THICK,
            "dentDepth": DENT_DEPTH,
            "dentHalfDeg": DENT_HALF_DEG,
            "dentTheta0": DENT_THETA0_DEG,
            "cProp": C_PROP,
            "cAssumed": round(c_est, 8),
            "cAssumedDefault": round(float(C_ASSUMED_DEFAULT), 8),
            "ascanLen": ASCAN_LEN,
            "estimatorGateCenter": round(g0, 4),
            "estimatorGateHalfWide": round(h_wide, 4),
            "estimatorGateHalfTrack": round(h_track, 4),
            "gateMode": "cold-start wide prior then look-to-look continuity",
            "nElem": n_elem,
            "stepDeg": float(step_deg),
            "dent": bool(dent),
            "eccMag": round(ecc_mag, 6),
            "eccPhiDeg": round(float(ecc_phi_deg), 6),
            "trueToolOffsetX": round(ex, 6),
            "trueToolOffsetY": round(ey, 6),
            "devicePerfect": True,
            "frame": "tool",
            "method": (
                "standoff r=c_assumed*(t−t_tx)/2 (tool) → tool-frame XY → "
                "free-center algebraic circle fit → residual R_fit−|p−c_fit|"
            ),
            "noML": True,
        },
        "thetaDeg": thetas.tolist(),
        "rMeasured": [round(float(v), 6) for v in r_meas],
        "rTrue": [round(float(v), 6) for v in r_true],
        "tEcho": [round(float(v), 4) for v in t_echo],
        "measX": [round(float(v), 6) for v in mx],
        "measY": [round(float(v), 6) for v in my],
        "trueX": [round(float(v), 6) for v in true_xs],
        "trueY": [round(float(v), 6) for v in true_ys],
        "fit": {
            "cx": round(fit_cx, 6),
            "cy": round(fit_cy, 6),
            "R": round(fit_R, 6),
        },
        "toolCenter": {"x": 0.0, "y": 0.0},
        "trueToolOffset": {"x": round(ex, 6), "y": round(ey, 6)},
        "pipeCenterTrue": {"x": round(-ex, 6), "y": round(-ey, 6)},
        "residuals": [round(float(v), 6) for v in residuals],
        "residualsForcedOrigin": [round(float(v), 6) for v in residuals_forced],
        "residualsForcedOriginNote": (
            "WRONG: assumes centered (pipe center = tool); pedagogy contrast only"
        ),
        "dentSector": dent_mask.tolist(),
        "peakResidualDeg": float(thetas[peak_res_i]),
        "peakResidual": round(float(residuals[peak_res_i]), 6),
        "peakForcedResidualDeg": float(thetas[peak_forced_i]),
        "peakForcedResidual": round(float(residuals_forced[peak_forced_i]), 6),
        "rmse": round(float(np.sqrt(np.mean((r_meas - r_true) ** 2))), 6),
        "meanResidual": round(float(np.mean(residuals)), 6),
        "rMeasSpan": round(float(np.max(r_meas) - np.min(r_meas)), 6),
    }
    if ascan_store is not None:
        out["ascans"] = ascan_store
    return out


def _recon_polyline_ndc(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    """Map pipe-centered (x,y) cells → NDC for reconstruction pane (y flipped)."""
    ndc_x = xs / scale
    ndc_y = -ys / scale  # match field flip (array y-up → GL y-up via flip)
    return np.column_stack([ndc_x, ndc_y]).astype(np.float32)


def run_live(
    theta: float = 0.0,
    n_elem: int = N_ELEM_DEFAULT,
    *,
    ecc_mag: float = 0.0,
    ecc_phi_deg: float = ECC_PHI_DEFAULT_DEG,
    dent: bool = False,
):
    try:
        import glfw
        from OpenGL import GL
        from OpenGL.GL import shaders
    except ImportError as e:
        raise ImportError(
            "Live UI needs optional deps: pip install glfw PyOpenGL. "
            "For headless dent/wall work use --reconstruct / --calibrate "
            "or: sonolab pipe-reconstruct"
        ) from e

    if not glfw.init():
        raise RuntimeError("glfw.init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

    win = glfw.create_window(
        1200, 780, "Phased array · circular pipe + reconstruction", None, None
    )
    if not win:
        glfw.terminate()
        raise RuntimeError("window failed")
    glfw.make_context_current(win)
    glfw.swap_interval(1)

    prog = shaders.compileProgram(
        shaders.compileShader(VERT, GL.GL_VERTEX_SHADER),
        shaders.compileShader(FRAG, GL.GL_FRAGMENT_SHADER),
    )
    u_tex = GL.glGetUniformLocation(prog, "tex")
    line_prog = shaders.compileProgram(
        shaders.compileShader(LINE_VERT, GL.GL_VERTEX_SHADER),
        shaders.compileShader(LINE_FRAG, GL.GL_FRAGMENT_SHADER),
    )
    u_col = GL.glGetUniformLocation(line_prog, "color")

    quad = np.array(
        [
            -1, -1, 0, 0,
            1, -1, 1, 0,
            1, 1, 1, 1,
            -1, -1, 0, 0,
            1, 1, 1, 1,
            -1, 1, 0, 1,
        ],
        dtype=np.float32,
    )
    vao = GL.glGenVertexArrays(1)
    vbo = GL.glGenBuffers(1)
    GL.glBindVertexArray(vao)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, quad.nbytes, quad, GL.GL_STATIC_DRAW)
    GL.glEnableVertexAttribArray(0)
    GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, False, 16, ctypes.c_void_p(0))
    GL.glEnableVertexAttribArray(1)
    GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, False, 16, ctypes.c_void_p(8))
    GL.glBindVertexArray(0)

    tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)

    line_vao = GL.glGenVertexArrays(1)
    line_vbo = GL.glGenBuffers(1)
    GL.glBindVertexArray(line_vao)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, line_vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, 2 * 2 * 4, None, GL.GL_DYNAMIC_DRAW)
    GL.glEnableVertexAttribArray(0)
    GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, False, 0, None)
    GL.glBindVertexArray(0)

    ascan_vao = GL.glGenVertexArrays(1)
    ascan_vbo = GL.glGenBuffers(1)
    GL.glBindVertexArray(ascan_vao)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, ascan_vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, ASCAN_LEN * 2 * 4, None, GL.GL_DYNAMIC_DRAW)
    GL.glEnableVertexAttribArray(0)
    GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, False, 0, None)
    GL.glBindVertexArray(0)

    recon_vao = GL.glGenVertexArrays(1)
    recon_vbo = GL.glGenBuffers(1)
    GL.glBindVertexArray(recon_vao)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, recon_vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, 4096 * 2 * 4, None, GL.GL_DYNAMIC_DRAW)
    GL.glEnableVertexAttribArray(0)
    GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, False, 0, None)
    GL.glBindVertexArray(0)

    ecc0 = float(np.clip(ecc_mag, 0.0, ECC_MAX))
    sim = PhasedPipe2D(
        n_elem=n_elem,
        theta_deg=theta,
        continuous=False,
        dent=dent,
        ecc_mag=ecc0,
        ecc_phi_deg=ecc_phi_deg,
    )
    ascan = AScanBuffer(ASCAN_LEN)
    state: dict[str, Any] = {
        "n_elem": n_elem,
        "theta": wrap_deg(theta),
        "continuous": False,
        "dent": bool(dent),
        "ecc_mag": ecc0,
        "ecc_phi_deg": wrap_deg(ecc_phi_deg),
        "ascan_distance": True,
        "baseline": None,
        "now_shot": None,
        "auto_baseline_pending": not bool(dent),
        "auto_now_pending": False,
        "pulse_hold": 0,
        "fps": 0.0,
        "show_recon": False,  # False → A-scan pane; True → reconstruction
        "recon": None,
        "show_pick": True,  # A-scan gate / thresh / t_echo overlay
        "recon_inspect_i": 0,  # scrub index into recon ascans / looks
        "inspect_recon_ascan": False,  # Left/Right after m → show that look's A-scan
        # Ranging assumption (wrong until k); FDTD truth stays C_PROP.
        "c_assumed": float(C_ASSUMED_DEFAULT),
        "c_calibrated": False,
    }

    def clear_snapshots(reason: str | None = None):
        state["baseline"] = None
        state["now_shot"] = None
        state["auto_baseline_pending"] = False
        state["auto_now_pending"] = False
        if reason:
            print(f"A-scan snapshots cleared ({reason})")

    def capture_baseline(*, auto: bool = False) -> bool:
        if state["continuous"]:
            print("baseline: pulse mode only (press 1)")
            return False
        if not ascan.complete:
            print("baseline: wait for full pulse window, then press b")
            return False
        series = ascan.series().copy()
        if series.size < 2 or not np.any(np.abs(series) > 0.0):
            print("baseline: empty/zero window — skip")
            return False
        state["baseline"] = series
        state["auto_baseline_pending"] = False
        how = "auto" if auto else "manual"
        print(f"baseline frozen [{how}] n={series.size}")
        return True

    def capture_now(*, auto: bool = False) -> bool:
        if state["continuous"]:
            print("now-shot: pulse mode only (press 1)")
            return False
        if not ascan.complete:
            print("now-shot: wait for full pulse window, then press n")
            return False
        series = ascan.series().copy()
        if series.size < 2 or not np.any(np.abs(series) > 0.0):
            print("now-shot: empty/zero window — skip")
            return False
        state["now_shot"] = series
        state["auto_now_pending"] = False
        how = "auto" if auto else "manual"
        print(f"now-shot frozen [{how}] n={series.size}")
        return True

    def do_reset(**kwargs):
        sim.reset(**kwargs)
        ascan.clear()
        state["pulse_hold"] = 0

    def run_calibration():
        """Joint known-R / free-center wave-speed cal → set state['c_assumed']."""
        print(
            f"Wave-speed calibration (joint known-R / free-center): "
            f"R_known={PIPE_R:.1f}  step={CAL_STEP_DEG}°  "
            f"dent={'ON' if state['dent'] else 'OFF'}  "
            f"ecc={state['ecc_mag']:.0f}  c_before={state['c_assumed']:.5f}  "
            f"(truth C_PROP={C_PROP:.5f}) …",
            flush=True,
        )
        if state["dent"]:
            print(
                "  tip: prefer dent OFF (0) — R_known prior assumes round ID; "
                "ecc is OK (joint cal absorbs pose into free center)",
                flush=True,
            )
        try:
            result = calibrate_c_known_radius(
                n_elem=int(state["n_elem"]),
                step_deg=float(CAL_STEP_DEG),
                dent=bool(state["dent"]),
                ecc_mag=float(state["ecc_mag"]),
                ecc_phi_deg=float(state["ecc_phi_deg"]),
                R_known=float(PIPE_R),
                c_before=float(state["c_assumed"]),
                progress=True,
            )
        except RuntimeError as exc:
            print(f"calibration failed: {exc}", flush=True)
            return
        for w in result.get("warnings") or []:
            print(f"  warn: {w}", flush=True)
        state["c_assumed"] = float(result["c_after"])
        state["c_calibrated"] = True
        fc = result.get("fitCenter") or {"x": 0.0, "y": 0.0}
        print(
            f"cal done.  method=joint (known R, free center)  "
            f"c_true={result['c_true']:.5f}  "
            f"c_before={result['c_before']:.5f}  "
            f"c_after={result['c_after']:.5f}  "
            f"R_known={result['R_known']:.1f}  "
            f"R_fit={result['R_fit']:.2f}  "
            f"fit_center=({fc['x']:.2f},{fc['y']:.2f})  "
            f"ecc_est={result['eccEstimate']:.2f}  "
            f"(true ecc={result['eccMag']:.1f})  "
            f"looks={result['nValid']}/{result['nLooks']}  "
            f"| demoted centered-only c={result['c_centeredOnly_WRONG']:.5f}  "
            f"| press m to reconstruct; K resets to uncalibrated",
            flush=True,
        )

    def run_reconstruction():
        print(
            f"Classical reconstruction (no ML, perfect device): 1° sweep, dent="
            f"{'ON' if state['dent'] else 'OFF'}, "
            f"ecc={state['ecc_mag']:.0f} cells @ φ={state['ecc_phi_deg']:.0f}°, "
            f"c_assumed={state['c_assumed']:.5f} "
            f"(truth C_PROP={C_PROP:.5f}"
            f"{'; calibrated' if state['c_calibrated'] else ', uncalibrated'}) …",
            flush=True,
        )
        data = reconstruct_pipe(
            dent=bool(state["dent"]),
            n_elem=int(state["n_elem"]),
            step_deg=RECON_STEP_DEG,
            progress=True,
            ecc_mag=float(state["ecc_mag"]),
            ecc_phi_deg=float(state["ecc_phi_deg"]),
            store_ascans=True,
            c_assumed=float(state["c_assumed"]),
        )
        state["recon"] = data
        state["show_recon"] = True
        state["recon_inspect_i"] = 0
        state["inspect_recon_ascan"] = False
        fit = data["fit"]
        true_off = data.get("trueToolOffset") or {"x": 0.0, "y": 0.0}
        print(
            f"done. fit R={fit['R']:.2f}  center=({fit['cx']:.2f},{fit['cy']:.2f})  "
            f"(tool-frame: tool@0,0; fit≈pipe ctr; true tool offset="
            f"({true_off['x']:.1f},{true_off['y']:.1f}))  "
            f"peak residual={data['peakResidual']:.2f} @ θ={data['peakResidualDeg']:.0f}°  "
            f"WRONG-centered peak={data['peakForcedResidual']:.2f}  "
            f"RMSE(meas−true)={data['rmse']:.2f}  r_span={data['rMeasSpan']:.2f}  "
            f"| press v to toggle pane; ←→ scrub look A-scan pick",
            flush=True,
        )
        if float(state["ecc_mag"]) > 0.5 and not state["dent"]:
            print(
                "  note: dent OFF + ecc ON → r(θ) varies once/rev (eccentric caliper). "
                "Free-center fit absorbs pose; residualsForcedOrigin "
                "(WRONG: assumes centered) fakes ovality.",
                flush=True,
            )

    def _print_recon_inspect():
        data = state["recon"]
        if data is None:
            return
        thetas = data["thetaDeg"]
        n = len(thetas)
        if n < 1:
            return
        i = int(state["recon_inspect_i"]) % n
        state["recon_inspect_i"] = i
        th = float(thetas[i])
        te = float(data["tEcho"][i])
        rm = float(data["rMeasured"][i])
        rt = float(data["rTrue"][i])
        print(
            f"recon look [{i}/{n - 1}] θ={th:.0f}°  t_echo={te:.1f}  "
            f"r_meas={rm:.2f}  r_true={rt:.2f}  | v → A-scan pick for this look",
            flush=True,
        )

    def on_key(window, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_LEFT_BRACKET:
            state["theta"] = wrap_deg(state["theta"] - 5.0)
            state["inspect_recon_ascan"] = False
            clear_snapshots("steer changed")
            do_reset(theta_deg=state["theta"])
            print(f"look → {state['theta']:.0f}°")
        elif key == glfw.KEY_RIGHT_BRACKET:
            state["theta"] = wrap_deg(state["theta"] + 5.0)
            state["inspect_recon_ascan"] = False
            clear_snapshots("steer changed")
            do_reset(theta_deg=state["theta"])
            print(f"look → {state['theta']:.0f}°")
        elif key in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
            data = state["recon"]
            if data is None or not data.get("ascans"):
                return
            n = len(data["thetaDeg"])
            step = -1 if key == glfw.KEY_LEFT else 1
            # Fine scrub 1°; Shift = 10° jumps
            if mods & glfw.MOD_SHIFT:
                step *= 10
            state["recon_inspect_i"] = (int(state["recon_inspect_i"]) + step) % n
            state["inspect_recon_ascan"] = True
            # Prefer A-scan pane so the pick overlay is visible
            if state["show_recon"]:
                state["show_recon"] = False
                print("bottom pane → A-scan (recon look inspect)", flush=True)
            _print_recon_inspect()
        elif key == glfw.KEY_MINUS:
            state["n_elem"] = max(2, state["n_elem"] - 2)
            clear_snapshots("elements changed")
            do_reset(n_elem=state["n_elem"])
            print(f"elements → {state['n_elem']}")
        elif key == glfw.KEY_EQUAL:
            state["n_elem"] = min(48, state["n_elem"] + 2)
            clear_snapshots("elements changed")
            do_reset(n_elem=state["n_elem"])
            print(f"elements → {state['n_elem']}")
        elif key == glfw.KEY_1:
            state["continuous"] = False
            ascan.set_rolling(False)
            clear_snapshots("drive changed")
            do_reset(continuous=False)
            state["auto_baseline_pending"] = True
            state["auto_now_pending"] = False
            print("drive → pulsed")
        elif key == glfw.KEY_2:
            state["continuous"] = True
            ascan.set_rolling(True)
            clear_snapshots("drive changed")
            do_reset(continuous=True)
            print("drive → continuous")
        elif key == glfw.KEY_0:
            state["dent"] = False
            state["now_shot"] = None
            state["auto_now_pending"] = False
            do_reset(dent=False)
            state["auto_baseline_pending"] = True
            print("dent → off")
        elif key == glfw.KEY_5:
            state["dent"] = True
            state["now_shot"] = None
            state["auto_baseline_pending"] = False
            state["auto_now_pending"] = not state["continuous"]
            do_reset(dent=True)
            print("dent → on (inward cosine bump at θ=0 / +Y)")
        elif key == glfw.KEY_E:
            # Shift+E increases; plain e toggles on/off with demo offset.
            shift = bool(mods & glfw.MOD_SHIFT)
            if shift:
                state["ecc_mag"] = float(
                    min(ECC_MAX, state["ecc_mag"] + ECC_STEP)
                )
            else:
                if state["ecc_mag"] > 0.0:
                    state["ecc_mag"] = 0.0
                else:
                    state["ecc_mag"] = float(ECC_DEMO)
            clear_snapshots("eccentricity changed")
            do_reset(
                ecc_mag=state["ecc_mag"],
                ecc_phi_deg=state["ecc_phi_deg"],
            )
            state["recon"] = None
            state["inspect_recon_ascan"] = False
            print(
                f"eccentricity → {state['ecc_mag']:.0f} cells "
                f"(φ={state['ecc_phi_deg']:.0f}°, tool vs pipe center; perfect device)"
            )
        elif key == glfw.KEY_COMMA:
            state["ecc_mag"] = float(max(0.0, state["ecc_mag"] - ECC_STEP))
            clear_snapshots("eccentricity changed")
            do_reset(
                ecc_mag=state["ecc_mag"],
                ecc_phi_deg=state["ecc_phi_deg"],
            )
            state["recon"] = None
            state["inspect_recon_ascan"] = False
            print(f"eccentricity → {state['ecc_mag']:.0f} cells")
        elif key == glfw.KEY_PERIOD:
            state["ecc_mag"] = float(min(ECC_MAX, state["ecc_mag"] + ECC_STEP))
            clear_snapshots("eccentricity changed")
            do_reset(
                ecc_mag=state["ecc_mag"],
                ecc_phi_deg=state["ecc_phi_deg"],
            )
            state["recon"] = None
            state["inspect_recon_ascan"] = False
            print(f"eccentricity → {state['ecc_mag']:.0f} cells")
        elif key == glfw.KEY_B:
            capture_baseline(auto=False)
        elif key == glfw.KEY_N:
            capture_now(auto=False)
        elif key == glfw.KEY_K:
            if mods & glfw.MOD_SHIFT:
                state["c_assumed"] = float(C_ASSUMED_DEFAULT)
                state["c_calibrated"] = False
                print(
                    f"c_assumed reset → {state['c_assumed']:.5f} "
                    f"(uncalibrated default = 1.1·C_PROP; truth C_PROP={C_PROP:.5f})",
                    flush=True,
                )
            else:
                run_calibration()
        elif key == glfw.KEY_M:
            run_reconstruction()
        elif key == glfw.KEY_V:
            state["show_recon"] = not state["show_recon"]
            pane = "reconstruction" if state["show_recon"] else "A-scan"
            print(f"bottom pane → {pane}")
        elif key == glfw.KEY_A:
            state["ascan_distance"] = not state["ascan_distance"]
            axis = "distance d=c·t/2" if state["ascan_distance"] else "time (steps)"
            print(f"A-scan axis → {axis}")
        elif key == glfw.KEY_P:
            state["show_pick"] = not state["show_pick"]
            print(
                f"A-scan pick overlay → {'ON' if state['show_pick'] else 'OFF'} "
                f"(cyan=gate yellow=pick gray=0.5·max)"
            )
        elif key == glfw.KEY_R:
            do_reset(
                n_elem=state["n_elem"],
                theta_deg=state["theta"],
                continuous=state["continuous"],
                dent=state["dent"],
                ecc_mag=state["ecc_mag"],
                ecc_phi_deg=state["ecc_phi_deg"],
            )
            if not state["continuous"]:
                if not state["dent"]:
                    state["auto_baseline_pending"] = True
                    state["auto_now_pending"] = False
                elif state["now_shot"] is None:
                    state["auto_now_pending"] = True
            print("reset")
        elif key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
            glfw.set_window_should_close(window, True)

    glfw.set_key_callback(win, on_key)

    print(
        f"Circular pipe phased array  {NX}x{NY}  R={PIPE_R:.0f}  "
        f"wall={WALL_THICK}  dent_depth={DENT_DEPTH:.0f}  HIRES={HIRES}\n"
        f"elements={n_elem} (linear sector)  θ={wrap_deg(theta):.0f}°  "
        f"C_PROP={C_PROP:.5f} (truth)  c_assumed={C_ASSUMED_DEFAULT:.5f} "
        f"(uncalibrated)  ASCAN_LEN={ASCAN_LEN}\n"
        f"Device: perfect (no electronics error). Eccentricity = tool–pipe pose only.\n"
        f"Cal: known pipe ID R={PIPE_R:.0f} stands in for physical length prior "
        f"(field: block / fixture / tape ID).\n"
        "[ ] look | -/= elements | 1 pulse 2 CW | 0/5 dent | "
        "e toggle ecc | E/,/. ecc mag | "
        "b/n baseline | k cal-c | K uncal | m reconstruct | v pane | a axis | p pick | "
        "←→ recon look | r reset | q quit\n"
        "Reconstruction: 1° sweep → r=c_assumed·t/2 (tool) → free-center circle fit → residual\n"
        "A-scan pick: cyan=gate  yellow=t_echo  gray=0.5·max (toggle p)"
    )

    prev = time.perf_counter()
    accum = 0.0
    dt = 1.0 / STEPS_PER_SECOND
    fps_t0 = prev
    frames = 0
    FIELD_FRAC = 0.70

    def _draw_line_strip(verts: np.ndarray, rgba: tuple[float, float, float, float]):
        if verts.shape[0] < 2:
            return
        GL.glUniform4f(u_col, *rgba)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_DYNAMIC_DRAW)
        GL.glDrawArrays(GL.GL_LINE_STRIP, 0, verts.shape[0])

    def _draw_lines(verts: np.ndarray, rgba: tuple[float, float, float, float]):
        if verts.shape[0] < 2:
            return
        GL.glUniform4f(u_col, *rgba)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_DYNAMIC_DRAW)
        GL.glDrawArrays(GL.GL_LINES, 0, verts.shape[0])

    try:
        while not glfw.window_should_close(win):
            now = time.perf_counter()
            accum += now - prev
            prev = now
            steps = 0
            while accum >= dt and steps < 8:
                if state["continuous"]:
                    sim.step()
                    ascan.push(sim.sample_receive())
                elif not ascan.complete:
                    sim.step()
                    ascan.push(sim.sample_receive())
                    if ascan.complete:
                        if state["auto_baseline_pending"] and not state["dent"]:
                            capture_baseline(auto=True)
                        elif state["auto_now_pending"] and state["dent"]:
                            capture_now(auto=True)
                        state["pulse_hold"] = PULSE_REFIRE_GAP
                elif state["pulse_hold"] > 0:
                    sim.step()
                    state["pulse_hold"] -= 1
                else:
                    sim.reset(
                        n_elem=state["n_elem"],
                        theta_deg=state["theta"],
                        continuous=False,
                        dent=state["dent"],
                        ecc_mag=state["ecc_mag"],
                        ecc_phi_deg=state["ecc_phi_deg"],
                    )
                    ascan.clear()
                    sim.step()
                    ascan.push(sim.sample_receive())
                accum -= dt
                steps += 1
            if steps == 8:
                accum = 0.0

            img = field_image(sim)
            img_gl = np.flipud(img)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_RGBA,
                NX,
                NY,
                0,
                GL.GL_RGBA,
                GL.GL_UNSIGNED_BYTE,
                img_gl,
            )

            w, h = glfw.get_framebuffer_size(win)
            GL.glClearColor(0.07, 0.07, 0.09, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            field_h = max(1, int(h * FIELD_FRAC))
            bot_h = max(1, h - field_h)

            # --- top: wave field + look ray ---
            GL.glViewport(0, bot_h, w, field_h)
            GL.glUseProgram(prog)
            GL.glUniform1i(u_tex, 0)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            GL.glBindVertexArray(vao)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)

            ux, uy = look_unit(state["theta"])
            tx, ty = tool_center_grid(state["ecc_mag"], state["ecc_phi_deg"])

            def g2n(ix, iy):
                return (
                    2.0 * (ix + 0.5) / NX - 1.0,
                    2.0 * (iy + 0.5) / NY - 1.0,
                )

            length = (PIPE_R - ARRAY_LOOK_OFFSET) * 0.9
            x0 = tx + ARRAY_LOOK_OFFSET * ux
            y0 = ty + ARRAY_LOOK_OFFSET * uy
            x1 = x0 + length * ux
            y1 = y0 + length * uy
            p0 = g2n(x0, y0)
            p1 = g2n(x1, y1)
            line = np.array([[p0[0], -p0[1]], [p1[0], -p1[1]]], dtype=np.float32)
            GL.glUseProgram(line_prog)
            GL.glUniform4f(u_col, 1.0, 0.9, 0.2, 0.9)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, line_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, line.nbytes, line, GL.GL_DYNAMIC_DRAW)
            GL.glBindVertexArray(line_vao)
            GL.glDrawArrays(GL.GL_LINES, 0, 2)
            GL.glBindVertexArray(0)

            # --- bottom pane ---
            GL.glViewport(0, 0, w, bot_h)
            GL.glUseProgram(line_prog)

            if state["show_recon"] and state["recon"] is not None:
                # Reconstruction XY plot: true / measured / fit circle
                GL.glBindVertexArray(recon_vao)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, recon_vbo)
                data = state["recon"]
                scale = max(PIPE_R + DENT_DEPTH + 8.0, 1.0) / 0.88

                frame = np.array(
                    [
                        [-0.96, -0.90],
                        [0.96, -0.90],
                        [0.96, 0.90],
                        [-0.96, 0.90],
                        [-0.96, -0.90],
                    ],
                    dtype=np.float32,
                )
                _draw_line_strip(frame, (0.40, 0.42, 0.48, 0.95))

                # axes
                _draw_lines(
                    np.array([[-0.92, 0.0], [0.92, 0.0]], dtype=np.float32),
                    (0.30, 0.32, 0.36, 0.8),
                )
                _draw_lines(
                    np.array([[0.0, -0.86], [0.0, 0.86]], dtype=np.float32),
                    (0.30, 0.32, 0.36, 0.8),
                )

                # fitted circle
                fit = data["fit"]
                circ_t = np.linspace(0, 2 * math.pi, 181)
                fcx, fcy, fR = fit["cx"], fit["cy"], fit["R"]
                circ = _recon_polyline_ndc(
                    fcx + fR * np.sin(circ_t),
                    fcy + fR * np.cos(circ_t),
                    scale=scale,
                )
                _draw_line_strip(circ, (0.45, 0.75, 0.95, 0.85))

                # true wall
                tx = np.asarray(data["trueX"], dtype=np.float64)
                ty = np.asarray(data["trueY"], dtype=np.float64)
                true_poly = _recon_polyline_ndc(tx, ty, scale=scale)
                # close loop
                true_poly = np.vstack([true_poly, true_poly[:1]])
                _draw_line_strip(true_poly, (0.35, 0.90, 0.55, 0.95))

                # measured polyline (downsample for draw)
                mx = np.asarray(data["measX"], dtype=np.float64)
                my = np.asarray(data["measY"], dtype=np.float64)
                step = max(1, RECON_DISPLAY_STEP)
                meas_poly = _recon_polyline_ndc(mx[::step], my[::step], scale=scale)
                meas_poly = np.vstack([meas_poly, meas_poly[:1]])
                _draw_line_strip(meas_poly, (0.95, 0.88, 0.25, 1.0))

                # highlight dent-sector measured points (short ticks)
                dent_seg = np.asarray(data["dentSector"], dtype=bool)
                if dent_seg.any():
                    hx = mx[dent_seg]
                    hy = my[dent_seg]
                    ticks = []
                    for px, py in zip(hx[::2], hy[::2]):
                        p = _recon_polyline_ndc(
                            np.array([px]), np.array([py]), scale=scale
                        )[0]
                        ticks.append([p[0] - 0.015, p[1]])
                        ticks.append([p[0] + 0.015, p[1]])
                    if ticks:
                        _draw_lines(
                            np.asarray(ticks, dtype=np.float32),
                            (0.95, 0.40, 0.20, 0.95),
                        )

                # true pipe center (cyan) vs tool origin (magenta) vs fit center
                def _cross(x, y, half, rgba):
                    p = _recon_polyline_ndc(
                        np.array([x]), np.array([y]), scale=scale
                    )[0]
                    _draw_lines(
                        np.array(
                            [
                                [p[0] - half, p[1]],
                                [p[0] + half, p[1]],
                                [p[0], p[1] - half],
                                [p[0], p[1] + half],
                            ],
                            dtype=np.float32,
                        ),
                        rgba,
                    )

                # Tool-frame plot: tool at origin; true pipe ctr = −trueToolOffset
                pc = data.get("pipeCenterTrue") or {"x": 0.0, "y": 0.0}
                _cross(pc["x"], pc["y"], 0.04, (0.35, 0.85, 1.0, 0.95))  # true pipe
                tc = data.get("toolCenter", {"x": 0.0, "y": 0.0})
                _cross(tc["x"], tc["y"], 0.04, (1.0, 0.35, 0.85, 0.95))  # tool
                _cross(fcx, fcy, 0.03, (0.45, 0.75, 0.95, 0.95))  # fit

                # scrubbed recon look (←→) — ray from tool through measured pick
                if state["inspect_recon_ascan"] and data.get("measX"):
                    ii = int(state["recon_inspect_i"]) % len(data["measX"])
                    _cross(
                        float(data["measX"][ii]),
                        float(data["measY"][ii]),
                        0.05,
                        (1.0, 0.95, 0.20, 1.0),
                    )
                    ray = _recon_polyline_ndc(
                        np.array([tc["x"], data["measX"][ii]], dtype=np.float64),
                        np.array([tc["y"], data["measY"][ii]], dtype=np.float64),
                        scale=scale,
                    )
                    _draw_lines(ray, (1.0, 0.95, 0.20, 0.85))

                # legend color keys
                legend = [
                    ((-0.92, 0.78), (-0.78, 0.78), (0.35, 0.90, 0.55, 0.95)),  # true
                    ((-0.92, 0.64), (-0.78, 0.64), (0.95, 0.88, 0.25, 1.0)),  # meas
                    ((-0.92, 0.50), (-0.78, 0.50), (0.45, 0.75, 0.95, 0.85)),  # fit
                    ((-0.92, 0.36), (-0.78, 0.36), (0.95, 0.40, 0.20, 0.95)),  # dent
                ]
                for (x0l, y0l), (x1l, y1l), rgba in legend:
                    _draw_lines(
                        np.array([[x0l, y0l], [x1l, y1l]], dtype=np.float32), rgba
                    )
                GL.glBindVertexArray(0)
            else:
                # A-scan strip (compare + classical pick overlay)
                GL.glBindVertexArray(ascan_vao)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, ascan_vbo)

                frame = np.array(
                    [
                        [-0.96, -0.82],
                        [0.96, -0.82],
                        [0.96, 0.82],
                        [-0.96, 0.82],
                        [-0.96, -0.82],
                    ],
                    dtype=np.float32,
                )
                GL.glUniform4f(u_col, 0.40, 0.42, 0.48, 0.95)
                GL.glBufferData(GL.GL_ARRAY_BUFFER, frame.nbytes, frame, GL.GL_DYNAMIC_DRAW)
                GL.glDrawArrays(GL.GL_LINE_STRIP, 0, 5)

                y_mid = 0.0
                zero = np.array([[-0.94, y_mid], [0.94, y_mid]], dtype=np.float32)
                GL.glUniform4f(u_col, 0.32, 0.34, 0.38, 0.85)
                GL.glBufferData(GL.GL_ARRAY_BUFFER, zero.nbytes, zero, GL.GL_DYNAMIC_DRAW)
                GL.glDrawArrays(GL.GL_LINES, 0, 2)

                preview = ascan.series()
                baseline = state["baseline"]
                now_shot = state["now_shot"]

                # Series used for pick: recon inspect (after ←→) else live pulse window
                pick_series: np.ndarray | None = None
                recon = state["recon"]
                if (
                    state["inspect_recon_ascan"]
                    and recon is not None
                    and recon.get("ascans")
                ):
                    ii = int(state["recon_inspect_i"]) % len(recon["ascans"])
                    pick_series = np.asarray(recon["ascans"][ii], dtype=np.float64)
                elif (not state["continuous"]) and preview.size >= 4:
                    pick_series = preview

                def _axis_xs(n: int) -> np.ndarray:
                    return -0.94 + (
                        np.arange(n, dtype=np.float64) / max(ASCAN_LEN - 1, 1)
                    ) * 1.88

                def _t_to_x(t: float) -> float:
                    return float(
                        -0.94 + (float(t) / max(ASCAN_LEN - 1, 1)) * 1.88
                    )

                def _peak(s: np.ndarray) -> float:
                    if (not state["continuous"]) and s.size > 100:
                        return max(float(np.max(np.abs(s[80:]))), 1e-6)
                    return max(float(np.max(np.abs(s))), 1e-6)

                peak_a = 1e-6
                for s in (preview, baseline, now_shot, pick_series):
                    if s is not None and getattr(s, "size", 0) >= 2:
                        peak_a = max(peak_a, _peak(s))

                def _ys(amp: np.ndarray, peak: float) -> np.ndarray:
                    return y_mid + np.clip(amp / peak, -1.15, 1.15) * 0.72

                def _draw_trace(amp, rgba, peak):
                    if amp.size < 2:
                        return
                    xs = _axis_xs(amp.size)
                    tr = np.column_stack([xs, _ys(amp, peak)]).astype(np.float32)
                    GL.glUniform4f(u_col, *rgba)
                    GL.glBufferData(GL.GL_ARRAY_BUFFER, tr.nbytes, tr, GL.GL_DYNAMIC_DRAW)
                    GL.glDrawArrays(GL.GL_LINE_STRIP, 0, amp.size)

                # Compute pick once; shade under traces, markers on top
                pick: dict[str, float | int] | None = None
                if (
                    state["show_pick"]
                    and pick_series is not None
                    and pick_series.size >= 4
                ):
                    pick_c = float(state["c_assumed"])
                    if state["inspect_recon_ascan"] and recon is not None:
                        pick_c = float(
                            recon["meta"].get("cAssumed", state["c_assumed"])
                        )
                    # Estimator gate: c_assumed + R_known prior; no true ecc/dent.
                    g_center = estimator_gate_center(
                        c_assumed=pick_c, R_known=PIPE_R
                    )
                    g_half = estimator_gate_half(c_assumed=pick_c)
                    pick = pick_echo_result(
                        pick_series,
                        gate_center=g_center,
                        gate_half=g_half,
                        c_assumed=pick_c,
                        R_known=PIPE_R,
                    )
                    x_lo = _t_to_x(float(pick["lo"]))
                    x_hi = _t_to_x(float(pick["hi"] - 1))
                    # Gate shade under the waveforms
                    GL.glEnable(GL.GL_BLEND)
                    GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
                    shade = np.array(
                        [
                            [x_lo, -0.78],
                            [x_hi, -0.78],
                            [x_lo, 0.78],
                            [x_hi, 0.78],
                        ],
                        dtype=np.float32,
                    )
                    GL.glUniform4f(u_col, 0.25, 0.75, 0.90, 0.18)
                    GL.glBufferData(
                        GL.GL_ARRAY_BUFFER, shade.nbytes, shade, GL.GL_DYNAMIC_DRAW
                    )
                    GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
                    GL.glDisable(GL.GL_BLEND)

                # Legend: baseline / now / |diff|  + pick keys when overlay on
                legend = [
                    ((-0.92, 0.70), (-0.78, 0.70), (0.55, 0.58, 0.62, 0.9)),  # baseline
                    ((-0.92, 0.55), (-0.78, 0.55), (0.95, 0.88, 0.25, 0.9)),  # now
                    ((-0.92, 0.40), (-0.78, 0.40), (0.95, 0.35, 0.75, 0.9)),  # |diff|
                ]
                if state["show_pick"]:
                    legend.extend(
                        [
                            ((0.55, 0.70), (0.72, 0.70), (0.35, 0.85, 0.95, 0.95)),  # gate
                            ((0.55, 0.55), (0.72, 0.55), (1.0, 0.92, 0.15, 0.95)),  # pick
                            ((0.55, 0.40), (0.72, 0.40), (0.55, 0.55, 0.58, 0.9)),  # thresh
                        ]
                    )
                for (x0l, y0l), (x1l, y1l), rgba in legend:
                    seg = np.array([[x0l, y0l], [x1l, y1l]], dtype=np.float32)
                    GL.glUniform4f(u_col, *rgba)
                    GL.glBufferData(GL.GL_ARRAY_BUFFER, seg.nbytes, seg, GL.GL_DYNAMIC_DRAW)
                    GL.glDrawArrays(GL.GL_LINES, 0, 2)

                if state["continuous"]:
                    _draw_trace(preview, (0.75, 0.70, 0.35, 0.55), peak_a)
                else:
                    if (
                        state["inspect_recon_ascan"]
                        and pick_series is not None
                        and pick_series.size >= 2
                    ):
                        # Emphasize the inspected recon A-scan (the one that was picked)
                        _draw_trace(pick_series, (0.70, 0.68, 0.40, 0.85), peak_a)
                    elif preview.size >= 2:
                        _draw_trace(preview, (0.55, 0.52, 0.28, 0.35), peak_a)
                    if baseline is not None and baseline.size >= 2:
                        _draw_trace(baseline, (0.55, 0.58, 0.62, 0.90), peak_a)
                    if now_shot is not None and now_shot.size >= 2:
                        _draw_trace(now_shot, (0.95, 0.88, 0.25, 1.0), peak_a)
                    if (
                        baseline is not None
                        and now_shot is not None
                        and baseline.size >= 2
                        and now_shot.size >= 2
                    ):
                        m = min(baseline.size, now_shot.size)
                        diff = np.abs(now_shot[:m] - baseline[:m])
                        peak_d = max(float(np.max(diff)), 1e-9)
                        _draw_trace(diff, (0.95, 0.35, 0.75, 0.85), peak_d)

                # Gate lines / thresh / t_echo on top of traces
                if pick is not None and pick_series is not None:
                    x_lo = _t_to_x(float(pick["lo"]))
                    x_hi = _t_to_x(float(pick["hi"] - 1))
                    x_pick = _t_to_x(float(pick["t_echo"]))
                    gate_lines = np.array(
                        [
                            [x_lo, -0.78],
                            [x_lo, 0.78],
                            [x_hi, -0.78],
                            [x_hi, 0.78],
                        ],
                        dtype=np.float32,
                    )
                    GL.glUniform4f(u_col, 0.35, 0.85, 0.95, 0.90)
                    GL.glBufferData(
                        GL.GL_ARRAY_BUFFER,
                        gate_lines.nbytes,
                        gate_lines,
                        GL.GL_DYNAMIC_DRAW,
                    )
                    GL.glDrawArrays(GL.GL_LINES, 0, 4)

                    if float(pick["thresh"]) > 0.0 and float(pick["peak"]) > 0.0:
                        y_th = float(_ys(np.array([pick["thresh"]]), peak_a)[0])
                        y_th_n = float(_ys(np.array([-pick["thresh"]]), peak_a)[0])
                        thr = np.array(
                            [
                                [x_lo, y_th],
                                [x_hi, y_th],
                                [x_lo, y_th_n],
                                [x_hi, y_th_n],
                            ],
                            dtype=np.float32,
                        )
                        GL.glUniform4f(u_col, 0.55, 0.55, 0.58, 0.75)
                        GL.glBufferData(
                            GL.GL_ARRAY_BUFFER, thr.nbytes, thr, GL.GL_DYNAMIC_DRAW
                        )
                        GL.glDrawArrays(GL.GL_LINES, 0, 4)

                    pick_line = np.array(
                        [[x_pick, -0.78], [x_pick, 0.78]], dtype=np.float32
                    )
                    GL.glUniform4f(u_col, 1.0, 0.92, 0.15, 0.95)
                    GL.glBufferData(
                        GL.GL_ARRAY_BUFFER,
                        pick_line.nbytes,
                        pick_line,
                        GL.GL_DYNAMIC_DRAW,
                    )
                    GL.glDrawArrays(GL.GL_LINES, 0, 2)
                    idx = int(pick["idx"])
                    if 0 <= idx < pick_series.size:
                        y_dot = float(_ys(pick_series[idx : idx + 1], peak_a)[0])
                        hs = 0.035
                        dot = np.array(
                            [
                                [x_pick - hs, y_dot],
                                [x_pick + hs, y_dot],
                                [x_pick, y_dot - hs],
                                [x_pick, y_dot + hs],
                            ],
                            dtype=np.float32,
                        )
                        GL.glUniform4f(u_col, 1.0, 0.95, 0.25, 1.0)
                        GL.glBufferData(
                            GL.GL_ARRAY_BUFFER, dot.nbytes, dot, GL.GL_DYNAMIC_DRAW
                        )
                        GL.glDrawArrays(GL.GL_LINES, 0, 4)

                GL.glBindVertexArray(0)

            recon_note = ""
            if state["recon"] is not None:
                fit = state["recon"]["fit"]
                recon_note = (
                    f"  recon:R={fit['R']:.1f} peakΔ={state['recon']['peakResidual']:.1f}"
                    f"@θ={state['recon']['peakResidualDeg']:.0f}°"
                )
                if state["inspect_recon_ascan"]:
                    ii = int(state["recon_inspect_i"]) % len(state["recon"]["thetaDeg"])
                    recon_note += (
                        f"  lookθ={state['recon']['thetaDeg'][ii]:.0f}°"
                        f" t_echo={state['recon']['tEcho'][ii]:.0f}"
                    )
            pane = "recon" if state["show_recon"] else "A-scan"
            pick_note = "pick" if state["show_pick"] else "pickOFF"
            ecc_note = (
                f"ecc={state['ecc_mag']:.0f}"
                if state["ecc_mag"] > 0
                else "ecc=0"
            )
            cal_note = "cal" if state["c_calibrated"] else "uncal"
            glfw.set_window_title(
                win,
                f"Pipe  {NX}x{NY}  t={sim.time}  θ={state['theta']:.0f}°  "
                f"N={state['n_elem']}  "
                f"{'CW' if state['continuous'] else 'pulse'}  "
                f"{'dent' if state['dent'] else 'no-dent'}  {ecc_note}  "
                f"perfect-device  pane={pane}  {pick_note}  "
                f"c_assumed={state['c_assumed']:.4f}({cal_note})  "
                f"C_PROP={C_PROP:.4f}  "
                f"fps={state['fps']:.0f}"
                f"{recon_note}  | k=cal K=uncal m=recon p=pick ←→=look",
            )
            glfw.swap_buffers(win)
            glfw.poll_events()

            frames += 1
            if now - fps_t0 >= 0.5:
                state["fps"] = frames / (now - fps_t0)
                frames = 0
                fps_t0 = now
    finally:
        glfw.terminate()


def run_angle_sweep(
    angles: list[float] | None = None,
    *,
    n_elem: int = N_ELEM_DEFAULT,
    downsample_step: int = 3,
    near_gate_half: int = 15,
) -> dict:
    """Pulse OFF/ON dent A-scans over look angles (legacy canvas helper)."""
    if angles is None:
        angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    # Estimator-style nominal gate (not C_PROP-centered truth gate)
    gate_idx = int(
        round(estimator_gate_center(c_assumed=C_ASSUMED_DEFAULT, R_known=PIPE_R))
    )
    by_angle: dict[str, dict] = {}
    for th in angles:
        baseline = record_pulse_window(theta_deg=float(th), dent=False, n_elem=n_elem)
        now = record_pulse_window(theta_deg=float(th), dent=True, n_elem=n_elem)
        m = min(baseline.size, now.size)
        abs_delta = np.abs(now[:m] - baseline[:m])
        idx = list(range(0, m, downsample_step))
        if idx[-1] != m - 1:
            idx.append(m - 1)
        idx_arr = np.asarray(idx, dtype=np.int64)
        dist = (C_PROP * idx_arr.astype(np.float64)) / 2.0
        peak_i = int(np.argmax(abs_delta))
        lo = max(0, gate_idx - near_gate_half)
        hi = min(m, gate_idx + near_gate_half + 1)
        near = abs_delta[lo:hi]
        near_local = int(np.argmax(near)) if near.size else 0
        near_i = lo + near_local
        by_angle[str(int(th) if float(th).is_integer() else th)] = {
            "sampleIdx": idx_arr.tolist(),
            "distanceCells": [round(float(v), 6) for v in dist],
            "baseline": [round(float(v), 6) for v in baseline[idx_arr]],
            "now": [round(float(v), 6) for v in now[idx_arr]],
            "absDelta": [round(float(v), 6) for v in abs_delta[idx_arr]],
            "peakAbsDelta": float(abs_delta[peak_i]),
            "peakNearGate": float(abs_delta[near_i]),
            "peakAbsDeltaIdx": peak_i,
            "peakNearGateIdx": near_i,
            "deltaRms": float(np.sqrt(np.mean(abs_delta**2))),
        }
        print(
            f"θ={th:5.1f}°  peak|Δ|={abs_delta[peak_i]:.5e} @ {peak_i}  "
            f"near-gate={abs_delta[near_i]:.5e} @ {near_i}"
        )
    return {
        "meta": {
            "ascanLen": ASCAN_LEN,
            "cProp": C_PROP,
            "wallGateT": WALL_GATE_T,
            "wallGateIdx": gate_idx,
            "wallOneWay": WALL_ONE_WAY,
            "nElem": n_elem,
            "downsampleStep": downsample_step,
            "pipeR": PIPE_R,
            "dentDepth": DENT_DEPTH,
            "dentHalfDeg": DENT_HALF_DEG,
            "nx": NX,
            "ny": NY,
        },
        "angles": angles,
        "byAngle": by_angle,
    }


def main():
    parser = argparse.ArgumentParser(
        description="2D phased-array beam in a circular pipe + classical reconstruction"
    )
    parser.add_argument("--theta", type=float, default=0.0, help="look angle degrees")
    parser.add_argument("--elements", type=int, default=N_ELEM_DEFAULT)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="headless dent OFF/ON A-scan sweep (JSON to stdout)",
    )
    parser.add_argument(
        "--sweep-json",
        type=str,
        default="",
        help="optional path to write sweep JSON (implies --sweep)",
    )
    parser.add_argument(
        "--reconstruct",
        action="store_true",
        help="headless classical pipe reconstruction (JSON to stdout)",
    )
    parser.add_argument(
        "--reconstruct-json",
        type=str,
        default="",
        help="write reconstruction JSON (implies --reconstruct)",
    )
    parser.add_argument(
        "--no-dent",
        action="store_true",
        help="reconstruction / live start with dent off",
    )
    parser.add_argument(
        "--ecc",
        type=float,
        default=0.0,
        help="tool eccentricity magnitude in cells (pose offset from pipe center)",
    )
    parser.add_argument(
        "--ecc-phi",
        type=float,
        default=ECC_PHI_DEFAULT_DEG,
        help="eccentricity direction degrees (0=+Y, 90=+X; default 90)",
    )
    parser.add_argument(
        "--recon-step",
        type=float,
        default=RECON_STEP_DEG,
        help="angle step degrees for reconstruction (default 1)",
    )
    parser.add_argument(
        "--c-assumed",
        type=float,
        default=None,
        help=(
            "ranging wave speed (default: uncalibrated "
            f"{C_ASSUMED_DEFAULT:.5f} = 1.1·C_PROP; truth C_PROP={C_PROP:.5f})"
        ),
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help=(
            "headless joint known-R / free-center wave-speed calibration "
            f"(R_known=PIPE_R={PIPE_R:.0f}; prefer --no-dent; ecc OK)"
        ),
    )
    args = parser.parse_args()
    if args.calibrate:
        result = calibrate_c_known_radius(
            n_elem=args.elements,
            step_deg=float(CAL_STEP_DEG),
            dent=not args.no_dent,
            ecc_mag=float(args.ecc),
            ecc_phi_deg=float(args.ecc_phi),
            R_known=float(PIPE_R),
            c_before=(
                float(args.c_assumed)
                if args.c_assumed is not None
                else float(C_ASSUMED_DEFAULT)
            ),
            progress=True,
        )
        for w in result.get("warnings") or []:
            print(f"warn: {w}", flush=True)
        fc = result.get("fitCenter") or {"x": 0.0, "y": 0.0}
        print(
            f"method=joint (known R, free center)  "
            f"c_true={result['c_true']:.6f}  "
            f"c_before={result['c_before']:.6f}  "
            f"c_after={result['c_after']:.6f}  "
            f"R_known={result['R_known']:.1f}  "
            f"R_fit={result['R_fit']:.3f}  "
            f"fit_center=({fc['x']:.3f},{fc['y']:.3f})  "
            f"ecc_est={result['eccEstimate']:.3f}  "
            f"true_ecc={result['eccMag']:.1f}  "
            f"centered_only_WRONG={result['c_centeredOnly_WRONG']:.6f}",
            flush=True,
        )
        print(json.dumps(result))
        return
    if args.reconstruct or args.reconstruct_json:
        c_est = (
            float(args.c_assumed)
            if args.c_assumed is not None
            else float(C_ASSUMED_DEFAULT)
        )
        data = reconstruct_pipe(
            dent=not args.no_dent,
            n_elem=args.elements,
            step_deg=args.recon_step,
            progress=True,
            ecc_mag=float(args.ecc),
            ecc_phi_deg=float(args.ecc_phi),
            c_assumed=c_est,
        )
        payload = json.dumps(data)
        if args.reconstruct_json:
            with open(args.reconstruct_json, "w", encoding="utf-8") as f:
                f.write(payload)
            print(f"wrote {args.reconstruct_json}", flush=True)
        else:
            print(payload)
        return
    if args.sweep or args.sweep_json:
        data = run_angle_sweep(n_elem=args.elements)
        payload = json.dumps(data)
        if args.sweep_json:
            with open(args.sweep_json, "w", encoding="utf-8") as f:
                f.write(payload)
            print(f"wrote {args.sweep_json}", flush=True)
        else:
            print(payload)
        return
    run_live(
        theta=args.theta,
        n_elem=args.elements,
        ecc_mag=float(args.ecc),
        ecc_phi_deg=float(args.ecc_phi),
    )


if __name__ == "__main__":
    main()
