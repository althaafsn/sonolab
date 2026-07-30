"""3D scalar acoustic FDTD - radial pulse-echo look at (θ, z)."""

from __future__ import annotations

import math

import numpy as np

from sonolab.pipe3d import constants as C
from sonolab.pipe3d.geometry import look_unit, make_c_field, tool_offset_xy


def _laplacian(u: np.ndarray) -> np.ndarray:
    return (
        np.roll(u, 1, 0)
        + np.roll(u, -1, 0)
        + np.roll(u, 1, 1)
        + np.roll(u, -1, 1)
        + np.roll(u, 1, 2)
        + np.roll(u, -1, 2)
        - 6.0 * u
    )


def record_pulse_look(
    *,
    theta_deg: float,
    z_tool: float,
    dent: bool = True,
    ecc_mag: float = 0.0,
    ecc_phi_deg: float = 90.0,
    nxy: int = C.NXY,
    nz: int = C.NZ,
    ascan_len: int | None = None,
    dent_depth: float = C.DENT_DEPTH,
    dent_theta0: float = C.DENT_THETA0_DEG,
    dent_z0: float = C.DENT_Z0,
    pipe_r: float = C.PIPE_R,
) -> np.ndarray:
    """One radial PE look: inject at tool+offset along û, record A-scan at TX."""
    ascan_len = int(ascan_len or C.ASCAN_LEN)
    c = make_c_field(
        nxy,
        nz,
        dent=dent,
        pipe_r=pipe_r,
        dent_depth=dent_depth,
        dent_theta0=dent_theta0,
        dent_z0=dent_z0,
    )
    c2 = (c * c).astype(np.float64)

    cx = (nxy - 1) * 0.5
    cy = (nxy - 1) * 0.5
    ex, ey = tool_offset_xy(ecc_mag, ecc_phi_deg)
    ux, uy, _ = look_unit(theta_deg)
    # Array center shifted toward the look wall (pipe2d ARRAY_LOOK_OFFSET).
    tx = cx + ex + float(C.ARRAY_LOOK_OFFSET) * ux
    ty = cy + ey + float(C.ARRAY_LOOK_OFFSET) * uy
    tz = float(z_tool)

    zz, yy, xx = np.meshgrid(
        np.arange(nz, dtype=np.float64),
        np.arange(nxy, dtype=np.float64),
        np.arange(nxy, dtype=np.float64),
        indexing="ij",
    )
    d2 = (xx - tx) ** 2 + (yy - ty) ** 2 + (zz - tz) ** 2
    # Compact aperture; slight look-directed dipole weight for forward preference.
    along = (xx - tx) * ux + (yy - ty) * uy
    aperture = np.exp(-d2 / (C.TX_SIGMA**2)) * (1.0 + 0.35 * np.tanh(along))

    ix = int(np.clip(round(tx), 1, nxy - 2))
    iy = int(np.clip(round(ty), 1, nxy - 2))
    iz = int(np.clip(round(tz), 1, nz - 2))

    u_prev = np.zeros((nz, nxy, nxy), dtype=np.float64)
    u = np.zeros_like(u_prev)
    ascan = np.zeros(ascan_len, dtype=np.float64)

    # Simple sponge near axial faces (reduces early false picks at slab ends).
    sponge = np.ones((nz, 1, 1), dtype=np.float64)
    n_sp = min(6, nz // 6)
    for i in range(n_sp):
        w = 0.65 + 0.35 * (i / max(n_sp - 1, 1))
        sponge[i, 0, 0] = w
        sponge[-(i + 1), 0, 0] = w

    pw = float(C.PULSE_WIDTH)
    for t in range(ascan_len):
        lap = _laplacian(u)
        u_next = 2.0 * u - u_prev + c2 * lap
        if t < pw:
            # Raised-cosine toneburst.
            env = 0.5 * (1.0 - math.cos(2.0 * math.pi * t / max(pw, 1.0)))
            drive = float(C.AMP) * env * math.cos(float(C.FREQ) * t)
            u_next += drive * aperture
        u_next *= sponge
        # Dirichlet boundaries
        u_next[0] = u_next[-1] = 0.0
        u_next[:, 0] = u_next[:, -1] = 0.0
        u_next[:, :, 0] = u_next[:, :, -1] = 0.0
        u_prev, u = u, u_next
        ascan[t] = float(u[iz, iy, ix])

    return ascan


def blind_pick_echo(series: np.ndarray) -> float:
    """Instrument-only first-break pick (no pipe R prior)."""
    n = int(series.size)
    if n < 4:
        return float(C.TX_EXCLUDE)
    lo = int(C.TX_EXCLUDE)
    window = np.abs(series[lo:])
    if window.size < 3:
        return float(lo)
    noise_n = min(24, max(4, window.size // 20))
    noise = float(np.median(window[:noise_n]))
    peak = float(np.max(window))
    thresh = max(4.0 * noise + 1e-6, 0.15 * peak)
    for i in range(1, window.size - 1):
        if (
            window[i] >= thresh
            and window[i] >= window[i - 1]
            and window[i] >= window[i + 1]
        ):
            return float(lo + i)
    above = np.where(window >= thresh)[0]
    if above.size:
        return float(lo + int(above[0]))
    return float(lo + int(np.argmax(window)))
