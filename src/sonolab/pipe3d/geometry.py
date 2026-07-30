"""3D pipe geometry: axis along z, radial looks in xy, localized 3D dent."""

from __future__ import annotations

import math

import numpy as np

from sonolab.pipe3d import constants as C


def wrap_deg(theta: float) -> float:
    return float(theta % 360.0)


def angle_diff_deg(a: float, b: float) -> float:
    d = (float(a) - float(b) + 180.0) % 360.0 - 180.0
    return float(d)


def look_unit(theta_deg: float) -> tuple[float, float, float]:
    """û = (sin θ, cos θ, 0): θ=0 → +Y, θ=90 → +X (pipe2d convention)."""
    th = math.radians(theta_deg)
    return math.sin(th), math.cos(th), 0.0


def tool_offset_xy(
    ecc_mag: float, ecc_phi_deg: float = 90.0
) -> tuple[float, float]:
    e = max(0.0, float(ecc_mag))
    ux, uy, _ = look_unit(ecc_phi_deg)
    return e * ux, e * uy


def dent_bump_3d(
    theta_deg: float,
    z: float,
    *,
    A: float,
    theta0_deg: float,
    z0: float,
    half_deg: float = C.DENT_HALF_DEG,
    half_z: float = C.DENT_HALF_Z,
) -> float:
    """Separable cosine bump in θ and z (inward on ID)."""
    a = max(0.0, float(A))
    if a <= 0.0:
        return 0.0
    dth = angle_diff_deg(theta_deg, theta0_deg) / max(float(half_deg), 1e-6)
    dz = (float(z) - float(z0)) / max(float(half_z), 1e-6)
    if abs(dth) >= 1.0 or abs(dz) >= 1.0:
        return 0.0
    w_th = 0.5 * (1.0 + math.cos(math.pi * dth))
    w_z = 0.5 * (1.0 + math.cos(math.pi * dz))
    return float(a * w_th * w_z)


def r_inner(
    theta_deg: float,
    z: float,
    *,
    dent: bool,
    pipe_r: float = C.PIPE_R,
    dent_depth: float = C.DENT_DEPTH,
    dent_theta0: float = C.DENT_THETA0_DEG,
    dent_z0: float = C.DENT_Z0,
    half_deg: float = C.DENT_HALF_DEG,
    half_z: float = C.DENT_HALF_Z,
) -> float:
    r = float(pipe_r)
    if dent:
        r -= dent_bump_3d(
            theta_deg,
            z,
            A=dent_depth,
            theta0_deg=dent_theta0,
            z0=dent_z0,
            half_deg=half_deg,
            half_z=half_z,
        )
    return max(r, 1.0)


def circle_ray_standoff(
    ecc_x: float, ecc_y: float, theta_deg: float, radius: float
) -> float:
    ux, uy, _ = look_unit(theta_deg)
    e_dot = float(ecc_x) * ux + float(ecc_y) * uy
    e_cross2 = (float(ecc_x) * uy - float(ecc_y) * ux) ** 2
    disc = max(float(radius) ** 2 - e_cross2, 0.0)
    return float(-e_dot + math.sqrt(disc))


def true_standoff(
    theta_deg: float,
    z: float,
    *,
    dent: bool,
    ecc_mag: float = 0.0,
    ecc_phi_deg: float = 90.0,
    pipe_r: float = C.PIPE_R,
    dent_depth: float = C.DENT_DEPTH,
    dent_theta0: float = C.DENT_THETA0_DEG,
    dent_z0: float = C.DENT_Z0,
) -> float:
    """Analytic tool→ID range along û at this z (iterated local radius)."""
    ecc_x, ecc_y = tool_offset_xy(ecc_mag, ecc_phi_deg)
    ux, uy, _ = look_unit(theta_deg)
    r = circle_ray_standoff(ecc_x, ecc_y, theta_deg, pipe_r)
    for _ in range(12):
        hx = ecc_x + r * ux
        hy = ecc_y + r * uy
        th_hit = math.degrees(math.atan2(hx, hy))
        R_loc = r_inner(
            th_hit,
            z,
            dent=dent,
            pipe_r=pipe_r,
            dent_depth=dent_depth,
            dent_theta0=dent_theta0,
            dent_z0=dent_z0,
        )
        r_new = circle_ray_standoff(ecc_x, ecc_y, theta_deg, R_loc)
        if abs(r_new - r) < 1e-5:
            return float(r_new)
        r = r_new
    return float(r)


def soft_wall_mask(
    nxy: int = C.NXY,
    nz: int = C.NZ,
    *,
    dent: bool = True,
    pipe_r: float = C.PIPE_R,
    wall_thick: float = C.WALL_THICK,
    dent_depth: float = C.DENT_DEPTH,
    dent_theta0: float = C.DENT_THETA0_DEG,
    dent_z0: float = C.DENT_Z0,
) -> np.ndarray:
    """Boolean mask True in soft wall annulus (variable ID). Shape (nz, nxy, nxy)."""
    zz, yy, xx = np.meshgrid(
        np.arange(nz, dtype=np.float64),
        np.arange(nxy, dtype=np.float64),
        np.arange(nxy, dtype=np.float64),
        indexing="ij",
    )
    cx = (nxy - 1) * 0.5
    cy = (nxy - 1) * 0.5
    dx = xx - cx
    dy = yy - cy
    rho = np.sqrt(dx * dx + dy * dy)
    th = np.degrees(np.arctan2(dx, dy))
    # Vectorized dent: separable cosine in θ and z
    if dent and dent_depth > 0:
        dth = ((th - dent_theta0 + 180.0) % 360.0) - 180.0
        dth_n = dth / max(float(C.DENT_HALF_DEG), 1e-6)
        dz_n = (zz - dent_z0) / max(float(C.DENT_HALF_Z), 1e-6)
        inside = (np.abs(dth_n) < 1.0) & (np.abs(dz_n) < 1.0)
        bump = np.zeros_like(rho)
        bump[inside] = (
            dent_depth
            * 0.5
            * (1.0 + np.cos(np.pi * dth_n[inside]))
            * 0.5
            * (1.0 + np.cos(np.pi * dz_n[inside]))
        )
        r_in = np.maximum(pipe_r - bump, 1.0)
    else:
        r_in = np.full_like(rho, float(pipe_r))
    r_out = r_in + float(wall_thick)
    mask = (rho >= r_in) & (rho <= r_out)
    mask[0] = mask[-1] = False
    mask[:, 0] = mask[:, -1] = False
    mask[:, :, 0] = mask[:, :, -1] = False
    return mask


def make_c_field(
    nxy: int = C.NXY,
    nz: int = C.NZ,
    *,
    dent: bool = True,
    **geo_kw,
) -> np.ndarray:
    """Propagation speed field (cells/step): fluid C_PROP, wall C_PROP*C_WALL/C0."""
    c = np.full((nz, nxy, nxy), float(C.C_PROP), dtype=np.float64)
    wall = soft_wall_mask(nxy, nz, dent=dent, **geo_kw)
    c[wall] = float(C.CFL * C.C_WALL)
    return c
