"""3D wave FDTD + GPU instanced voxel glyphs (optimized).

Ported into SonoLab from ``wavelab-3d/practice/wave.py``. Headless
``Wave3D`` needs only NumPy. Live OpenGL UI (``run_live``) needs optional
glfw + PyOpenGL.

Bottleneck before: matplotlib rebuilt thousands of CPU polygons every frame.
Now: NumPy FDTD + OpenGL instanced cubes (one draw call).

Keys: 1-4 source | 0 none | 5 soft box | 8 soft pipe | [ ] c_inside | r reset | q quit
      Tab: active slice A/B | arrows: move X/Y | PgUp/Dn: move Z
      J/L yaw | I/K pitch | U/O roll | 6/7 snap through-box / clear
      LMB orbit | wheel zoom
"""

from __future__ import annotations

import argparse
import ctypes
import math
import time

import numpy as np

# glfw / OpenGL imported lazily inside run_live (optional for headless).

N = 48
CFL = 0.45          # keep CFL * max(c_ratio) under ~0.58 in 3D
CUBE_FRAC = 0.25
DRAW_FRAC = 0.10
A = 1.0
SIGMA = 4.5
LAMBDA = 2.5
OMEGA = 0.35
CHIRP_A = 0.002
DEFAULT_C_INSIDE = 0.55   # relative wave speed in the soft box
SLICE_RES = 96
# Fixed physics rate (independent of how fast the GPU draws).
STEPS_PER_SECOND = 10.0
ASCAN_LEN = 600

MODES = ("spherical", "plane", "oscillating", "chirp")
INCLUSIONS = ("none", "soft_box", "soft_pipe")

# Soft pipe (annulus) radii in grid units, centered on mid (x,y), along z
PIPE_R_INNER = 6.0
PIPE_R_OUTER = 9.0

VERT = """
#version 330 core
layout(location=0) in vec3 in_vert;
layout(location=1) in vec3 in_pos;
layout(location=2) in vec4 in_color;
uniform mat4 mvp;
uniform float half_edge;
out vec4 v_color;
void main() {
    vec3 world = in_pos + in_vert * (half_edge * 2.0);
    gl_Position = mvp * vec4(world, 1.0);
    v_color = in_color;
}
"""

FRAG = """
#version 330 core
in vec4 v_color;
out vec4 f_color;
void main() { f_color = v_color; }
"""

SLICE_VERT = """
#version 330 core
layout(location=0) in vec2 in_pos;
layout(location=1) in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

SLICE_FRAG = """
#version 330 core
in vec2 v_uv;
uniform sampler2D tex;
out vec4 f_color;
void main() {
    f_color = texture(tex, v_uv);
}
"""

PLANE_VERT = """
#version 330 core
layout(location=0) in vec3 in_pos;
uniform mat4 mvp;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

PLANE_FRAG = """
#version 330 core
uniform vec4 color;
out vec4 f_color;
void main() { f_color = color; }
"""


class AScanBuffer:
    """Rolling amplitude-vs-time trace at a fixed probe."""

    def __init__(self, length: int = ASCAN_LEN):
        self.length = int(length)
        self.buf = np.zeros(self.length, dtype=np.float64)
        self.i = 0
        self.filled = 0

    def clear(self):
        self.buf.fill(0.0)
        self.i = 0
        self.filled = 0

    def push(self, value: float):
        self.buf[self.i] = float(value)
        self.i = (self.i + 1) % self.length
        self.filled = min(self.filled + 1, self.length)

    def series(self) -> np.ndarray:
        if self.filled == 0:
            return np.zeros(0, dtype=np.float64)
        if self.filled < self.length:
            return self.buf[: self.filled].copy()
        return np.concatenate([self.buf[self.i :], self.buf[: self.i]])


def coolwarm_rgba(field2d: np.ndarray, peak: float | None = None) -> np.ndarray:
    """(H,W) float → (H,W,4) uint8 coolwarm image."""
    peak = float(peak) if peak is not None else float(np.max(np.abs(field2d)))
    peak = max(peak, 1e-6)
    t = np.clip(0.5 + 0.5 * (field2d / peak), 0.0, 1.0)
    r = np.clip(np.where(t < 0.5, 0.23 + 1.54 * t, 0.40 + 0.60 * t), 0, 1)
    g = np.clip(np.where(t < 0.5, 0.30 + 1.00 * t, 1.40 - 1.00 * t), 0, 1)
    b = np.clip(np.where(t < 0.5, 0.75 + 0.50 * t, 1.50 - 1.50 * t), 0, 1)
    a = np.ones_like(r)
    img = np.stack((r, g, b, a), axis=-1)
    return (img * 255).astype(np.uint8)


def slice_x_planes(n: int) -> tuple[int, int]:
    """Default X positions: through soft box, and clear of box."""
    c = (n - 1) / 2.0
    x0, x1 = int(c + 6), int(c + 16)
    x_obst = (x0 + x1) // 2
    x_clear = max(2, x0 - 5)
    return x_obst, x_clear


def _rot_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Rotation matrix: yaw about Z, pitch about Y, roll about X (radians)."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    return rz @ ry @ rx


def trilinear_sample(vol: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Sample vol[z,y,x] at scattered (x,y,z). Out-of-bounds → 0."""
    n = vol.shape[0]
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    out = np.zeros(x.shape, dtype=np.float64)
    valid = (
        (x >= 0) & (x <= n - 1)
        & (y >= 0) & (y <= n - 1)
        & (z >= 0) & (z <= n - 1)
    )
    if not np.any(valid):
        return out
    xv, yv, zv = x[valid], y[valid], z[valid]
    x0 = np.floor(xv).astype(np.int32)
    y0 = np.floor(yv).astype(np.int32)
    z0 = np.floor(zv).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, n - 1)
    y1 = np.clip(y0 + 1, 0, n - 1)
    z1 = np.clip(z0 + 1, 0, n - 1)
    x0 = np.clip(x0, 0, n - 1)
    y0 = np.clip(y0, 0, n - 1)
    z0 = np.clip(z0, 0, n - 1)
    xd, yd, zd = xv - x0, yv - y0, zv - z0
    c000 = vol[z0, y0, x0]
    c100 = vol[z0, y0, x1]
    c010 = vol[z0, y1, x0]
    c110 = vol[z0, y1, x1]
    c001 = vol[z1, y0, x0]
    c101 = vol[z1, y0, x1]
    c011 = vol[z1, y1, x0]
    c111 = vol[z1, y1, x1]
    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd
    out[valid] = c0 * (1 - zd) + c1 * zd
    return out


class SlicePlane:
    """Oriented cutting plane through the grid (movable + rotatable)."""

    def __init__(
        self,
        n: int,
        origin: tuple[float, float, float],
        yaw: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
        res: int = SLICE_RES,
        half_extent: float | None = None,
    ):
        self.n = n
        self.origin = np.array(origin, dtype=np.float64)  # (x, y, z)
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.roll = float(roll)
        self.res = int(res)
        self.half_extent = float(half_extent if half_extent is not None else (n - 1) * 0.55)

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (normal, u_axis, v_axis). Default yaw=0 → normal +X (YZ plane)."""
        r = _rot_zyx(self.yaw, self.pitch, self.roll)
        normal = r @ np.array([1.0, 0.0, 0.0])
        u_ax = r @ np.array([0.0, 1.0, 0.0])
        v_ax = r @ np.array([0.0, 0.0, 1.0])
        return normal, u_ax, v_ax

    def sample_field(self, vol: np.ndarray) -> np.ndarray:
        """Return (res, res) samples; axis0 = v (up), axis1 = u (right)."""
        _, u_ax, v_ax = self.basis()
        half = self.half_extent
        us = np.linspace(-half, half, self.res)
        vs = np.linspace(-half, half, self.res)
        uu, vv = np.meshgrid(us, vs, indexing="xy")
        # world = origin + u*u_ax + v*v_ax
        o = self.origin
        x = o[0] + uu * u_ax[0] + vv * v_ax[0]
        y = o[1] + uu * u_ax[1] + vv * v_ax[1]
        z = o[2] + uu * u_ax[2] + vv * v_ax[2]
        return trilinear_sample(vol, x, y, z)

    def sample_mask(self, mask: np.ndarray) -> np.ndarray:
        """Nearest inclusion flags on the plane (bool)."""
        field = self.sample_field(mask.astype(np.float64))
        return field > 0.5

    def world_corners(self) -> np.ndarray:
        """4 corners for drawing the plane in 3D (float32 Nx3)."""
        _, u_ax, v_ax = self.basis()
        h = self.half_extent
        o = self.origin
        corners = []
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            corners.append(o + su * h * u_ax + sv * h * v_ax)
        return np.asarray(corners, dtype=np.float32)

    def nudge_origin(self, dx=0.0, dy=0.0, dz=0.0):
        self.origin[0] = float(np.clip(self.origin[0] + dx, 0, self.n - 1))
        self.origin[1] = float(np.clip(self.origin[1] + dy, 0, self.n - 1))
        self.origin[2] = float(np.clip(self.origin[2] + dz, 0, self.n - 1))

    def nudge_angles(self, dyaw=0.0, dpitch=0.0, droll=0.0):
        self.yaw += dyaw
        self.pitch = float(np.clip(self.pitch + dpitch, -math.pi / 2 + 0.05, math.pi / 2 - 0.05))
        self.roll += droll

    def label(self) -> str:
        return (
            f"o=({self.origin[0]:.1f},{self.origin[1]:.1f},{self.origin[2]:.1f}) "
            f"yaw={math.degrees(self.yaw):.0f}° "
            f"pitch={math.degrees(self.pitch):.0f}° "
            f"roll={math.degrees(self.roll):.0f}°"
        )


def packet_field(n: int, kind: str) -> np.ndarray:
    """Initial packet (spherical / plane). Driven modes start at zero."""
    zz, yy, xx = np.meshgrid(
        np.arange(n), np.arange(n), np.arange(n), indexing="ij"
    )
    c = (n - 1) / 2.0
    xx = xx.astype(np.float64)
    yy = yy.astype(np.float64)
    zz = zz.astype(np.float64)
    if kind == "spherical":
        d2 = (xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2
    elif kind == "plane":
        d2 = (xx - c) ** 2
    else:
        raise ValueError(kind)
    d = np.sqrt(d2)
    u = A * np.exp(-d2 / (SIGMA**2)) * np.cos(d / LAMBDA)
    u[0] = u[-1] = 0.0
    u[:, 0] = u[:, -1] = 0.0
    u[:, :, 0] = u[:, :, -1] = 0.0
    return u


def drive_value(mode: str, t: float) -> float:
    if mode == "oscillating":
        return A * np.cos(OMEGA * t)
    if mode == "chirp":
        return A * np.cos(OMEGA * (t + CHIRP_A * t * t))
    return 0.0


def soft_box_mask(n: int) -> np.ndarray:
    """Axis-aligned soft inclusion (True inside). Off-center so a packet hits it."""
    zz, yy, xx = np.meshgrid(
        np.arange(n), np.arange(n), np.arange(n), indexing="ij"
    )
    c = (n - 1) / 2.0
    mask = (
        (xx >= c + 6)
        & (xx <= c + 16)
        & (yy >= c - 10)
        & (yy <= c + 10)
        & (zz >= c - 10)
        & (zz <= c + 10)
    )
    mask[0] = mask[-1] = False
    mask[:, 0] = mask[:, -1] = False
    mask[:, :, 0] = mask[:, :, -1] = False
    return mask


def soft_pipe_mask(
    n: int,
    r_inner: float = PIPE_R_INNER,
    r_outer: float = PIPE_R_OUTER,
) -> np.ndarray:
    """Hollow cylinder along X, centered in Y/Z (pipe wall = True).

    Volume layout is [z, y, x]. Radius is in the YZ plane; axis runs along X.
    Interior bore (r < r_inner) and exterior stay background fluid.
    """
    zz, yy, xx = np.meshgrid(
        np.arange(n), np.arange(n), np.arange(n), indexing="ij"
    )
    cy = cz = (n - 1) / 2.0
    r2 = (yy - cy) ** 2 + (zz - cz) ** 2
    mask = (r2 >= r_inner**2) & (r2 <= r_outer**2)
    mask[0] = mask[-1] = False
    mask[:, 0] = mask[:, -1] = False
    mask[:, :, 0] = mask[:, :, -1] = False
    return mask


def inclusion_mask(n: int, kind: str) -> np.ndarray:
    if kind == "soft_box":
        return soft_box_mask(n)
    if kind == "soft_pipe":
        return soft_pipe_mask(n)
    return np.zeros((n, n, n), dtype=bool)


def make_c_ratio(n: int, inclusion: str, c_inside: float) -> np.ndarray:
    ratio = np.ones((n, n, n), dtype=np.float64)
    mask = inclusion_mask(n, inclusion)
    if mask.any():
        ratio[mask] = c_inside
    return ratio


class Wave3D:
    __slots__ = (
        "n",
        "mode",
        "inclusion",
        "c_inside",
        "box",
        "c_ratio",
        "u",
        "up",
        "un",
        "time",
    )

    def __init__(
        self,
        n: int = N,
        mode: str = "spherical",
        inclusion: str = "soft_pipe",
        c_inside: float = DEFAULT_C_INSIDE,
    ):
        self.n = n
        self.mode = mode
        self.inclusion = inclusion
        self.c_inside = float(c_inside)
        self.box = inclusion_mask(n, inclusion)
        self.c_ratio = make_c_ratio(n, inclusion, self.c_inside)
        self.u = np.zeros((n, n, n), dtype=np.float64)
        self.up = np.zeros((n, n, n), dtype=np.float64)
        self.un = np.zeros((n, n, n), dtype=np.float64)
        self.time = 0
        self.reset(mode=mode, inclusion=inclusion, c_inside=c_inside)

    def reset(
        self,
        mode: str | None = None,
        inclusion: str | None = None,
        c_inside: float | None = None,
    ):
        if mode is not None:
            self.mode = mode
        if inclusion is not None:
            self.inclusion = inclusion
        if c_inside is not None:
            self.c_inside = float(c_inside)
        self.box = inclusion_mask(self.n, self.inclusion)
        self.c_ratio = make_c_ratio(self.n, self.inclusion, self.c_inside)
        self.un.fill(0.0)
        self.time = 0
        if self.mode in ("spherical", "plane"):
            u0 = packet_field(self.n, self.mode)
            self.u[:] = u0
            self.up[:] = u0
        else:
            self.u.fill(0.0)
            self.up.fill(0.0)

    def step(self):
        u, up, un = self.u, self.up, self.un
        # Local C(x) = (CFL * c_ratio)^2  — soft inclusion transmits
        cmap = (CFL * self.c_ratio) ** 2
        un.fill(0.0)
        lap = (
            u[1:-1, 1:-1, 2:]
            + u[1:-1, 1:-1, :-2]
            + u[1:-1, 2:, 1:-1]
            + u[1:-1, :-2, 1:-1]
            + u[2:, 1:-1, 1:-1]
            + u[:-2, 1:-1, 1:-1]
            - 6.0 * u[1:-1, 1:-1, 1:-1]
        )
        un[1:-1, 1:-1, 1:-1] = (
            2.0 * u[1:-1, 1:-1, 1:-1]
            - up[1:-1, 1:-1, 1:-1]
            + cmap[1:-1, 1:-1, 1:-1] * lap
        )

        t_next = self.time + 1
        if self.mode in ("oscillating", "chirp"):
            left = drive_value(self.mode, float(t_next))
            un[:, :, 0] = left
            un[0, :, 0] = 0.0
            un[-1, :, 0] = 0.0
            un[:, 0, 0] = 0.0
            un[:, -1, 0] = 0.0
        else:
            un[:, :, 0] = 0.0

        un[:, :, -1] = 0.0
        un[0, :, :] = 0.0
        un[-1, :, :] = 0.0
        un[:, 0, :] = 0.0
        un[:, -1, :] = 0.0

        self.up, self.u, self.un = u, un, up
        self.time = t_next

    def sample(self, x: float, y: float, z: float) -> float:
        """Probe amplitude at (x,y,z) with trilinear sample."""
        return float(
            trilinear_sample(self.u, np.array([x]), np.array([y]), np.array([z]))[0]
        )

    def center(self) -> float:
        c = self.n // 2
        return float(self.u[c, c, c])

    def active_instances(self) -> tuple[np.ndarray, np.ndarray]:
        u = self.u
        peak = float(np.max(np.abs(u)))
        peak = max(peak, 1e-6)
        thr = DRAW_FRAC * peak

        # Field cubes
        zs, ys, xs = np.where(np.abs(u) >= thr)
        parts_pos = []
        parts_col = []
        if xs.size:
            vals = u[zs, ys, xs]
            pos = np.column_stack((xs, ys, zs)).astype(np.float32)
            t = np.clip(0.5 + 0.5 * (vals / peak), 0.0, 1.0)
            r = np.clip(np.where(t < 0.5, 0.23 + 1.54 * t, 0.40 + 0.60 * t), 0, 1)
            g = np.clip(np.where(t < 0.5, 0.30 + 1.00 * t, 1.40 - 1.00 * t), 0, 1)
            b = np.clip(np.where(t < 0.5, 0.75 + 0.50 * t, 1.50 - 1.50 * t), 0, 1)
            a = 0.30 + 0.65 * np.clip(np.abs(vals) / peak, 0, 1)
            # Tint inclusion cells slightly cyan so you see the region
            if self.inclusion != "none":
                inside = self.box[zs, ys, xs]
                r = np.where(inside, 0.55 * r + 0.45 * 0.25, r)
                g = np.where(inside, 0.55 * g + 0.45 * 0.85, g)
                b = np.where(inside, 0.55 * b + 0.45 * 0.90, b)
            col = np.column_stack((r, g, b, a)).astype(np.float32)
            parts_pos.append(pos)
            parts_col.append(col)

        # Ghost surface of inclusion (even where |u| is small)
        if self.inclusion != "none" and self.box.any():
            shell = self.box.copy()
            shell[1:-1, 1:-1, 1:-1] = (
                self.box[1:-1, 1:-1, 1:-1]
                & ~(
                    self.box[:-2, 1:-1, 1:-1]
                    & self.box[2:, 1:-1, 1:-1]
                    & self.box[1:-1, :-2, 1:-1]
                    & self.box[1:-1, 2:, 1:-1]
                    & self.box[1:-1, 1:-1, :-2]
                    & self.box[1:-1, 1:-1, 2:]
                )
            )
            sz, sy, sx = np.where(shell)
            if sx.size:
                gpos = np.column_stack((sx, sy, sz)).astype(np.float32)
                gcol = np.tile(
                    np.array([0.2, 0.75, 0.85, 0.18], dtype=np.float32),
                    (sx.size, 1),
                )
                parts_pos.append(gpos)
                parts_col.append(gcol)

        if not parts_pos:
            return (
                np.zeros((0, 3), np.float32),
                np.zeros((0, 4), np.float32),
            )
        return np.vstack(parts_pos), np.vstack(parts_col)


def unit_cube():
    h = 0.5
    v = np.array(
        [
            [-h, -h, -h],
            [h, -h, -h],
            [h, h, -h],
            [-h, h, -h],
            [-h, -h, h],
            [h, -h, h],
            [h, h, h],
            [-h, h, h],
        ],
        dtype=np.float32,
    )
    idx = np.array(
        [
            0, 1, 2, 0, 2, 3,
            4, 6, 5, 4, 7, 6,
            0, 4, 5, 0, 5, 1,
            2, 6, 7, 2, 7, 3,
            0, 3, 7, 0, 7, 4,
            1, 5, 6, 1, 6, 2,
        ],
        dtype=np.uint32,
    )
    return v, idx


def look_at(eye, target, up):
    eye = np.asarray(eye, np.float64)
    target = np.asarray(target, np.float64)
    up = np.asarray(up, np.float64)
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float64)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -s @ eye
    m[1, 3] = -u @ eye
    m[2, 3] = f @ eye
    return m


def perspective(fovy_deg, aspect, znear, zfar):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (zfar + znear) / (znear - zfar)
    m[2, 3] = (2 * zfar * znear) / (znear - zfar)
    m[3, 2] = -1.0
    return m


def run_live(n: int, mode: str, inclusion: str = "soft_pipe", c_inside: float = DEFAULT_C_INSIDE):
    try:
        import glfw
        from OpenGL import GL
        from OpenGL.GL import shaders
    except ImportError as e:
        raise ImportError(
            "Live 3D UI needs optional deps: pip install glfw PyOpenGL "
            '(or: pip install -e ".[pipe-live]"). '
            "Headless physics: from sonolab.wave3d import Wave3D"
        ) from e
    if not glfw.init():
        raise RuntimeError("glfw.init failed")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

    win = glfw.create_window(1400, 800, "Wave3D GPU glyphs + slices", None, None)
    if not win:
        glfw.terminate()
        raise RuntimeError("window create failed")
    glfw.make_context_current(win)
    glfw.swap_interval(1)

    prog = shaders.compileProgram(
        shaders.compileShader(VERT, GL.GL_VERTEX_SHADER),
        shaders.compileShader(FRAG, GL.GL_FRAGMENT_SHADER),
    )
    u_mvp = GL.glGetUniformLocation(prog, "mvp")
    u_half = GL.glGetUniformLocation(prog, "half_edge")

    slice_prog = shaders.compileProgram(
        shaders.compileShader(SLICE_VERT, GL.GL_VERTEX_SHADER),
        shaders.compileShader(SLICE_FRAG, GL.GL_FRAGMENT_SHADER),
    )
    u_slice_tex = GL.glGetUniformLocation(slice_prog, "tex")

    cube_v, cube_i = unit_cube()
    vao = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)

    vbo = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, cube_v.nbytes, cube_v, GL.GL_STATIC_DRAW)
    GL.glEnableVertexAttribArray(0)
    GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

    ibo = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ibo)
    GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, cube_i.nbytes, cube_i, GL.GL_STATIC_DRAW)

    capacity = max(4096, n * n)
    pos_vbo = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, pos_vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, capacity * 12, None, GL.GL_DYNAMIC_DRAW)
    GL.glEnableVertexAttribArray(1)
    GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
    GL.glVertexAttribDivisor(1, 1)

    col_vbo = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, col_vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, capacity * 16, None, GL.GL_DYNAMIC_DRAW)
    GL.glEnableVertexAttribArray(2)
    GL.glVertexAttribPointer(2, 4, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
    GL.glVertexAttribDivisor(2, 1)

    GL.glBindVertexArray(0)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_BLEND)
    GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

    # --- 2D slice panel quads (NDC): top = through obstacle, bottom = clear ---
    def make_slice_quad(x0, y0, x1, y1):
        # pos.xy, uv
        data = np.array(
            [
                x0, y0, 0, 0,
                x1, y0, 1, 0,
                x1, y1, 1, 1,
                x0, y0, 0, 0,
                x1, y1, 1, 1,
                x0, y1, 0, 1,
            ],
            dtype=np.float32,
        )
        vao = GL.glGenVertexArrays(1)
        vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, ctypes.c_void_p(8))
        GL.glBindVertexArray(0)
        return vao

    # Right-side panels
    # Right panels: slices; bottom: A-scan
    quad_obst = make_slice_quad(0.42, 0.12, 0.98, 0.52)
    quad_clear = make_slice_quad(0.42, -0.28, 0.98, 0.08)

    def make_tex():
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        return tex

    tex_a = make_tex()
    tex_b = make_tex()

    def upload_slice(tex, img_u8):
        h, w, _ = img_u8.shape
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA,
            w,
            h,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            img_u8,
        )

    # World-space slice rectangle (line loop)
    plane_prog = shaders.compileProgram(
        shaders.compileShader(PLANE_VERT, GL.GL_VERTEX_SHADER),
        shaders.compileShader(PLANE_FRAG, GL.GL_FRAGMENT_SHADER),
    )
    u_plane_mvp = GL.glGetUniformLocation(plane_prog, "mvp")
    u_plane_col = GL.glGetUniformLocation(plane_prog, "color")
    plane_vao = GL.glGenVertexArrays(1)
    plane_vbo = GL.glGenBuffers(1)
    GL.glBindVertexArray(plane_vao)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, plane_vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, 4 * 3 * 4, None, GL.GL_DYNAMIC_DRAW)
    GL.glEnableVertexAttribArray(0)
    GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
    GL.glBindVertexArray(0)

    # A-scan line buffer (NDC via identity MVP)
    ascan_vao = GL.glGenVertexArrays(1)
    ascan_vbo = GL.glGenBuffers(1)
    GL.glBindVertexArray(ascan_vao)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, ascan_vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, ASCAN_LEN * 3 * 4, None, GL.GL_DYNAMIC_DRAW)
    GL.glEnableVertexAttribArray(0)
    GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
    GL.glBindVertexArray(0)

    wave = Wave3D(n, mode, inclusion, c_inside)
    x_obst, x_clear = slice_x_planes(n)
    mid = (n - 1) * 0.5
    slice_a = SlicePlane(n, (float(x_obst), mid, mid))  # through box, YZ
    slice_b = SlicePlane(n, (float(x_clear), mid, mid))  # clear, YZ
    slices = [slice_a, slice_b]
    active = {"i": 0}  # 0=A top, 1=B bottom

    # Fixed receiver: +X side, on pipe axis (y=z=center) — hears waves along the pipe
    recv = np.array([n - 5.0, mid, mid], dtype=np.float64)
    ascan = AScanBuffer(ASCAN_LEN)

    print(
        f"Slices A (top)=through box x={x_obst} | B (bottom)=clear x={x_clear}\n"
        f"Receiver fixed at (x,y,z)=({recv[0]:.0f},{recv[1]:.0f},{recv[2]:.0f}) — live A-scan\n"
        "Tab switches active slice; move/rotate with arrows / PgUpDn / JLI KUO"
    )
    center = (n - 1) * 0.5
    orbit = {
        "yaw": 0.7,
        "pitch": 0.4,
        "dist": n * 2.2,
        "dragging": False,
        "last": (0.0, 0.0),
    }
    meta = {"mode": mode, "inclusion": inclusion, "c_inside": c_inside, "fps": 0.0}
    step_pos = 0.5
    step_ang = math.radians(5.0)

    def active_slice() -> SlicePlane:
        return slices[active["i"]]

    def on_key(window, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        key_modes = {
            glfw.KEY_1: "spherical",
            glfw.KEY_2: "plane",
            glfw.KEY_3: "oscillating",
            glfw.KEY_4: "chirp",
        }
        s = active_slice()
        if key in key_modes:
            wave.reset(mode=key_modes[key])
            meta["mode"] = key_modes[key]
            ascan.clear()
            print(f"mode → {meta['mode']}")
        elif key == glfw.KEY_0:
            wave.reset(inclusion="none")
            meta["inclusion"] = "none"
            ascan.clear()
            print("inclusion → none")
        elif key == glfw.KEY_5:
            wave.reset(inclusion="soft_box")
            meta["inclusion"] = "soft_box"
            ascan.clear()
            print(f"inclusion → soft_box  c_inside={wave.c_inside:.2f}")
        elif key == glfw.KEY_8:
            wave.reset(inclusion="soft_pipe")
            meta["inclusion"] = "soft_pipe"
            ascan.clear()
            print(
                f"inclusion → soft_pipe (X-axis, centered YZ)  "
                f"r=[{PIPE_R_INNER:.0f},{PIPE_R_OUTER:.0f}]  c_inside={wave.c_inside:.2f}"
            )
        elif key == glfw.KEY_LEFT_BRACKET:
            wave.reset(c_inside=max(0.2, wave.c_inside - 0.1))
            meta["c_inside"] = wave.c_inside
            ascan.clear()
            print(f"c_inside → {wave.c_inside:.2f}")
        elif key == glfw.KEY_RIGHT_BRACKET:
            wave.reset(c_inside=min(1.6, wave.c_inside + 0.1))
            meta["c_inside"] = wave.c_inside
            ascan.clear()
            print(f"c_inside → {wave.c_inside:.2f}")
        elif key == glfw.KEY_TAB:
            active["i"] = 1 - active["i"]
            print(f"active slice → {'A' if active['i'] == 0 else 'B'}  {active_slice().label()}")
        elif key == glfw.KEY_LEFT:
            s.nudge_origin(dx=-step_pos)
        elif key == glfw.KEY_RIGHT:
            s.nudge_origin(dx=+step_pos)
        elif key == glfw.KEY_UP:
            s.nudge_origin(dy=+step_pos)
        elif key == glfw.KEY_DOWN:
            s.nudge_origin(dy=-step_pos)
        elif key == glfw.KEY_PAGE_UP:
            s.nudge_origin(dz=+step_pos)
        elif key == glfw.KEY_PAGE_DOWN:
            s.nudge_origin(dz=-step_pos)
        elif key == glfw.KEY_J:
            s.nudge_angles(dyaw=-step_ang)
        elif key == glfw.KEY_L:
            s.nudge_angles(dyaw=+step_ang)
        elif key == glfw.KEY_I:
            s.nudge_angles(dpitch=+step_ang)
        elif key == glfw.KEY_K:
            s.nudge_angles(dpitch=-step_ang)
        elif key == glfw.KEY_U:
            s.nudge_angles(droll=-step_ang)
        elif key == glfw.KEY_O:
            s.nudge_angles(droll=+step_ang)
        elif key == glfw.KEY_6:
            # snap active to YZ through box
            s.origin[:] = (float(x_obst), mid, mid)
            s.yaw = s.pitch = s.roll = 0.0
            print(f"snap → through box  {s.label()}")
        elif key == glfw.KEY_7:
            s.origin[:] = (float(x_clear), mid, mid)
            s.yaw = s.pitch = s.roll = 0.0
            print(f"snap → clear  {s.label()}")
        elif key == glfw.KEY_R:
            wave.reset(
                mode=meta["mode"],
                inclusion=meta["inclusion"],
                c_inside=meta["c_inside"],
            )
            ascan.clear()
            print("reset")
        elif key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
            glfw.set_window_should_close(window, True)
        else:
            return
        if key in (
            glfw.KEY_LEFT,
            glfw.KEY_RIGHT,
            glfw.KEY_UP,
            glfw.KEY_DOWN,
            glfw.KEY_PAGE_UP,
            glfw.KEY_PAGE_DOWN,
            glfw.KEY_J,
            glfw.KEY_L,
            glfw.KEY_I,
            glfw.KEY_K,
            glfw.KEY_U,
            glfw.KEY_O,
        ):
            print(f"slice {'A' if active['i'] == 0 else 'B'}: {s.label()}")

    def on_button(window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            orbit["dragging"] = action == glfw.PRESS
            orbit["last"] = glfw.get_cursor_pos(window)

    def on_move(window, x, y):
        if not orbit["dragging"]:
            return
        lx, ly = orbit["last"]
        orbit["last"] = (x, y)
        orbit["yaw"] += (x - lx) * 0.005
        orbit["pitch"] = float(np.clip(orbit["pitch"] + (y - ly) * 0.005, -1.4, 1.4))

    def on_scroll(window, xoff, yoff):
        orbit["dist"] = float(
            np.clip(orbit["dist"] * (0.9 if yoff > 0 else 1.1), n * 0.8, n * 8)
        )

    glfw.set_key_callback(win, on_key)
    glfw.set_mouse_button_callback(win, on_button)
    glfw.set_cursor_pos_callback(win, on_move)
    glfw.set_scroll_callback(win, on_scroll)

    print(
        f"GPU glyphs  N={n}^3  mode={mode}  inclusion={inclusion}  "
        f"c_inside={c_inside:.2f}\n"
        f"Physics fixed at {STEPS_PER_SECOND:.0f} steps/s\n"
        "1-4 source | 0 none | 5 box | 8 pipe | [ ] speed | r reset | q quit\n"
        "Tab A/B | arrows X/Y | PgUp/Dn Z | J/L yaw | I/K pitch | U/O roll | 6/7 snap\n"
        "Right: TOP=slice A  BOTTOM=slice B | Bottom strip: receiver A-scan"
    )

    nonlocal_cap = {"cap": capacity}
    fps_t0 = time.perf_counter()
    frames = 0
    prev_t = time.perf_counter()
    accum = 0.0
    dt_step = 1.0 / max(STEPS_PER_SECOND, 1e-6)
    # Cap catch-up so a hitch doesn't explode the sim
    max_steps_per_frame = 8
    try:
        while not glfw.window_should_close(win):
            now = time.perf_counter()
            accum += now - prev_t
            prev_t = now

            steps = 0
            while accum >= dt_step and steps < max_steps_per_frame:
                wave.step()
                ascan.push(wave.sample(recv[0], recv[1], recv[2]))
                accum -= dt_step
                steps += 1
            if steps == max_steps_per_frame:
                accum = 0.0  # drop leftover time after a stall

            pos, col = wave.active_instances()
            count = int(pos.shape[0])

            if count > nonlocal_cap["cap"]:
                nonlocal_cap["cap"] = int(count * 1.5)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, pos_vbo)
                GL.glBufferData(
                    GL.GL_ARRAY_BUFFER, nonlocal_cap["cap"] * 12, None, GL.GL_DYNAMIC_DRAW
                )
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, col_vbo)
                GL.glBufferData(
                    GL.GL_ARRAY_BUFFER, nonlocal_cap["cap"] * 16, None, GL.GL_DYNAMIC_DRAW
                )

            if count:
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, pos_vbo)
                GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, pos.nbytes, pos)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, col_vbo)
                GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, col.nbytes, col)

            w, h = glfw.get_framebuffer_size(win)
            GL.glViewport(0, 0, w, h)
            GL.glClearColor(0.08, 0.08, 0.10, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # Left ~55%×~78%: 3D glyphs (leave bottom for A-scan)
            vp3d_w = int(w * 0.55)
            vp3d_h = int(h * 0.78)
            GL.glViewport(0, int(h * 0.22), vp3d_w, vp3d_h)
            GL.glEnable(GL.GL_DEPTH_TEST)

            eye = np.array(
                [
                    center
                    + orbit["dist"]
                    * math.cos(orbit["pitch"])
                    * math.cos(orbit["yaw"]),
                    center
                    + orbit["dist"]
                    * math.cos(orbit["pitch"])
                    * math.sin(orbit["yaw"]),
                    center + orbit["dist"] * math.sin(orbit["pitch"]),
                ]
            )
            view = look_at(eye, (center, center, center), (0, 0, 1))
            proj = perspective(50.0, max(vp3d_w / max(vp3d_h, 1), 1e-3), 0.1, n * 20.0)
            mvp = (proj @ view).astype(np.float32)
            mvp_gl = np.ascontiguousarray(mvp.T)

            GL.glUseProgram(prog)
            GL.glUniformMatrix4fv(u_mvp, 1, GL.GL_FALSE, mvp_gl)
            GL.glUniform1f(u_half, 0.5 * CUBE_FRAC)

            if count:
                GL.glBindVertexArray(vao)
                GL.glDrawElementsInstanced(
                    GL.GL_TRIANGLES, 36, GL.GL_UNSIGNED_INT, None, count
                )
                GL.glBindVertexArray(0)

            # Draw both slice rectangles in 3D (active = yellow, other = cyan)
            # Core profile only guarantees line width 1.0 (width>1 → INVALID_VALUE on Mesa).
            GL.glUseProgram(plane_prog)
            GL.glUniformMatrix4fv(u_plane_mvp, 1, GL.GL_FALSE, mvp_gl)
            GL.glLineWidth(1.0)
            for idx, sp in enumerate(slices):
                corners = sp.world_corners()
                # line loop order
                loop = np.vstack([corners, corners[0:1]])
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, plane_vbo)
                GL.glBufferData(GL.GL_ARRAY_BUFFER, loop.nbytes, loop, GL.GL_DYNAMIC_DRAW)
                if idx == active["i"]:
                    GL.glUniform4f(u_plane_col, 1.0, 0.9, 0.2, 0.95)
                else:
                    GL.glUniform4f(u_plane_col, 0.3, 0.8, 0.9, 0.55)
                GL.glBindVertexArray(plane_vao)
                GL.glDrawArrays(GL.GL_LINE_STRIP, 0, 5)
            GL.glBindVertexArray(0)

            # Receiver marker (magenta cross) in 3D
            rx, ry, rz = float(recv[0]), float(recv[1]), float(recv[2])
            cross = np.array(
                [
                    [rx - 1.2, ry, rz],
                    [rx + 1.2, ry, rz],
                    [rx, ry - 1.2, rz],
                    [rx, ry + 1.2, rz],
                    [rx, ry, rz - 1.2],
                    [rx, ry, rz + 1.2],
                ],
                dtype=np.float32,
            )
            GL.glUniform4f(u_plane_col, 1.0, 0.2, 0.85, 1.0)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, plane_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, cross.nbytes, cross, GL.GL_DYNAMIC_DRAW)
            GL.glBindVertexArray(plane_vao)
            GL.glDrawArrays(GL.GL_LINES, 0, 6)
            GL.glBindVertexArray(0)

            # Right panels from oriented plane samples
            peak = float(np.max(np.abs(wave.u)))
            field_a = slice_a.sample_field(wave.u)
            field_b = slice_b.sample_field(wave.u)
            img_a = coolwarm_rgba(field_a, peak)
            img_b = coolwarm_rgba(field_b, peak)
            if wave.inclusion != "none":
                tint_a = slice_a.sample_mask(wave.box)
                tint_b = slice_b.sample_mask(wave.box)
                img_a = img_a.copy()
                img_b = img_b.copy()
                for img, tint in ((img_a, tint_a), (img_b, tint_b)):
                    img[tint, 0] = (0.4 * img[tint, 0] + 0.6 * 40).astype(np.uint8)
                    img[tint, 1] = (0.4 * img[tint, 1] + 0.6 * 200).astype(np.uint8)
                    img[tint, 2] = (0.4 * img[tint, 2] + 0.6 * 220).astype(np.uint8)

            upload_slice(tex_a, np.ascontiguousarray(img_a))
            upload_slice(tex_b, np.ascontiguousarray(img_b))

            GL.glViewport(0, 0, w, h)
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glUseProgram(slice_prog)
            GL.glUniform1i(u_slice_tex, 0)

            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex_a)
            GL.glBindVertexArray(quad_obst)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)

            GL.glBindTexture(GL.GL_TEXTURE_2D, tex_b)
            GL.glBindVertexArray(quad_clear)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
            GL.glBindVertexArray(0)

            # Bottom A-scan: amplitude vs time (NDC, identity MVP)
            series = ascan.series()
            GL.glUseProgram(plane_prog)
            ident = np.eye(4, dtype=np.float32)
            GL.glUniformMatrix4fv(u_plane_mvp, 1, GL.GL_FALSE, ident)

            # frame + zero line
            frame = np.array(
                [
                    [-0.95, -0.95, 0],
                    [0.95, -0.95, 0],
                    [0.95, -0.58, 0],
                    [-0.95, -0.58, 0],
                    [-0.95, -0.95, 0],
                ],
                dtype=np.float32,
            )
            GL.glUniform4f(u_plane_col, 0.45, 0.45, 0.5, 0.9)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, ascan_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, frame.nbytes, frame, GL.GL_DYNAMIC_DRAW)
            GL.glBindVertexArray(ascan_vao)
            GL.glDrawArrays(GL.GL_LINE_STRIP, 0, 5)

            y_mid = -0.765
            zero = np.array([[-0.95, y_mid, 0], [0.95, y_mid, 0]], dtype=np.float32)
            GL.glUniform4f(u_plane_col, 0.35, 0.35, 0.4, 0.8)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, zero.nbytes, zero, GL.GL_DYNAMIC_DRAW)
            GL.glDrawArrays(GL.GL_LINES, 0, 2)

            if series.size >= 2:
                peak_a = max(float(np.max(np.abs(series))), 1e-6)
                xs = np.linspace(-0.93, 0.93, series.size)
                ys = y_mid + (series / peak_a) * 0.16
                line = np.column_stack(
                    [xs, ys, np.zeros(series.size)]
                ).astype(np.float32)
                GL.glUniform4f(u_plane_col, 0.95, 0.85, 0.2, 1.0)
                GL.glBufferData(GL.GL_ARRAY_BUFFER, line.nbytes, line, GL.GL_DYNAMIC_DRAW)
                GL.glDrawArrays(GL.GL_LINE_STRIP, 0, series.size)
            GL.glBindVertexArray(0)

            sa = active_slice()
            glfw.set_window_title(
                win,
                f"Wave3D  t={wave.time}  fps={meta['fps']:.0f}  "
                f"{meta['mode']}/{meta['inclusion']}  "
                f"recv=({recv[0]:.0f},{recv[1]:.0f},{recv[2]:.0f})  "
                f"slice={'A' if active['i']==0 else 'B'}  {sa.label()}",
            )
            glfw.swap_buffers(win)
            glfw.poll_events()

            frames += 1
            if now - fps_t0 >= 0.5:
                meta["fps"] = frames / (now - fps_t0)
                frames = 0
                fps_t0 = now
                print(
                    f"t={wave.time}: center={wave.center():.4f} "
                    f"cubes={count} fps={meta['fps']:.0f} "
                    f"sim={STEPS_PER_SECOND:.0f} steps/s"
                )
    finally:
        glfw.terminate()


def main():
    parser = argparse.ArgumentParser(description="Optimized 3D wave GPU glyphs")
    parser.add_argument("--mode", choices=MODES, default="spherical")
    parser.add_argument("--inclusion", choices=INCLUSIONS, default="soft_pipe")
    parser.add_argument("--c-inside", type=float, default=DEFAULT_C_INSIDE)
    parser.add_argument("--n", type=int, default=N, help="grid size N^3")
    args = parser.parse_args()
    run_live(int(args.n), args.mode, args.inclusion, args.c_inside)


if __name__ == "__main__":
    main()
