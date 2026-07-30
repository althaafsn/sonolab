"""Display-grid wall evaluation: fine mesh from acquire-resolution Lab B + truth."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from sonolab.pipe2d.phased_array import look_unit
from sonolab.pipe3d import constants as C
from sonolab.pipe3d.geometry import true_standoff
from sonolab.pipe3d.survey import SurveyTruth

# Block 8 display defaults (visual density; not FDTD acquire density).
DISPLAY_STEP_DEG = 0.25
DISPLAY_N_Z = 64


def r_map_to_positions(
    R_map: np.ndarray,
    z: np.ndarray,
    theta_deg: np.ndarray,
) -> np.ndarray:
    """(n_z, n_th) R → flat xyz positions shape (n_z * n_th, 3)."""
    R_map = np.asarray(R_map, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    theta_deg = np.asarray(theta_deg, dtype=np.float64)
    n_z, n_th = R_map.shape
    pos = np.zeros((n_z * n_th, 3), dtype=np.float64)
    k = 0
    for i in range(n_z):
        for j in range(n_th):
            ux, uy = look_unit(float(theta_deg[j]))
            r = float(R_map[i, j])
            pos[k, 0] = r * ux
            pos[k, 1] = r * uy
            pos[k, 2] = float(z[i])
            k += 1
    return pos


def mesh_indices(n_z: int, n_th: int, *, periodic_theta: bool = True) -> np.ndarray:
    """Triangle indices for a regular (z × θ) grid. Vertex id = i*n_th + j."""
    n_z = int(n_z)
    n_th = int(n_th)
    tris: list[int] = []
    for i in range(n_z - 1):
        for j in range(n_th if periodic_theta else n_th - 1):
            j0 = j
            j1 = (j + 1) % n_th if periodic_theta else j + 1
            a = i * n_th + j0
            b = i * n_th + j1
            c = (i + 1) * n_th + j0
            d = (i + 1) * n_th + j1
            tris.extend([a, c, b, b, c, d])
    return np.asarray(tris, dtype=np.int32)


def _interp_R_map(
    R_map: np.ndarray,
    z_src: np.ndarray,
    th_src: np.ndarray,
    z_dst: np.ndarray,
    th_dst: np.ndarray,
) -> np.ndarray:
    """Bilinear resample of R(θ,z) with periodic θ."""
    R_map = np.asarray(R_map, dtype=np.float64)
    z_src = np.asarray(z_src, dtype=np.float64)
    th_src = np.asarray(th_src, dtype=np.float64)
    if z_src.size < 2 or th_src.size < 2:
        # Nearest / broadcast degenerate grids
        out = np.zeros((z_dst.size, th_dst.size), dtype=np.float64)
        for i, z in enumerate(z_dst):
            iz = int(np.argmin(np.abs(z_src - z)))
            for j, th in enumerate(th_dst):
                d = np.abs(((th_src - th + 180.0) % 360.0) - 180.0)
                jt = int(np.argmin(d))
                out[i, j] = R_map[iz, jt]
        return out

    th_ext = np.concatenate([th_src, [float(th_src[0]) + 360.0]])
    R_ext = np.concatenate([R_map, R_map[:, :1]], axis=1)
    interp = RegularGridInterpolator(
        (z_src, th_ext),
        R_ext,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    ZZ, TT = np.meshgrid(z_dst, th_dst, indexing="ij")
    # wrap θ into [th_ext[0], th_ext[-1]]
    TT_w = np.mod(TT - float(th_ext[0]), 360.0) + float(th_ext[0])
    pts = np.stack([ZZ.ravel(), TT_w.ravel()], axis=-1)
    vals = interp(pts)
    return vals.reshape(z_dst.size, th_dst.size)


def evaluate_truth_display(
    truth: SurveyTruth,
    *,
    z_display: np.ndarray,
    theta_display: np.ndarray,
) -> np.ndarray:
    """Analytic truth R on the display grid (same params as Lab A truth)."""
    R = np.zeros((z_display.size, theta_display.size), dtype=np.float64)
    for i, z in enumerate(z_display):
        for j, th in enumerate(theta_display):
            R[i, j] = true_standoff(
                float(th),
                float(z),
                dent=bool(truth.dent_on),
                ecc_mag=float(truth.ecc_mag),
                ecc_phi_deg=float(truth.ecc_phi_deg),
                pipe_r=float(truth.pipe_r),
                dent_depth=float(truth.dent_depth),
                dent_theta0=float(truth.dent_theta0_deg),
                dent_z0=float(truth.dent_z0),
            )
    return R


def build_display_wall(
    product: dict[str, Any],
    truth: SurveyTruth,
    *,
    display_step_deg: float = DISPLAY_STEP_DEG,
    n_z_display: int = DISPLAY_N_Z,
    acquire_step_deg: float | None = None,
    acquire_n_z: int | None = None,
) -> dict[str, Any]:
    """Resample Lab B wall + analytic truth onto a fine display mesh.

    Free CPU after acquire: this is visual density, not new FDTD looks.
    """
    z_acq = np.asarray(product["z"], dtype=np.float64)
    th_acq = np.asarray(product["theta_deg"], dtype=np.float64)
    R_acq = np.asarray(product["R_map"], dtype=np.float64)

    step = float(display_step_deg)
    if step <= 0:
        raise ValueError("display_step_deg must be positive")
    n_z = max(int(n_z_display), 2)
    th_disp = np.arange(0.0, 360.0, step, dtype=np.float64)
    z_disp = np.linspace(float(z_acq[0]), float(z_acq[-1]), n_z, dtype=np.float64)

    R_recon = _interp_R_map(R_acq, z_acq, th_acq, z_disp, th_disp)
    R_truth = evaluate_truth_display(truth, z_display=z_disp, theta_display=th_disp)

    pos_recon = r_map_to_positions(R_recon, z_disp, th_disp)
    pos_truth = r_map_to_positions(R_truth, z_disp, th_disp)
    indices = mesh_indices(n_z, int(th_disp.size), periodic_theta=True)

    acq_step = (
        float(acquire_step_deg)
        if acquire_step_deg is not None
        else (float(th_acq[1] - th_acq[0]) if th_acq.size > 1 else float("nan"))
    )
    acq_nz = int(acquire_n_z if acquire_n_z is not None else z_acq.size)

    return {
        "z": [float(v) for v in z_disp],
        "theta_deg": [float(v) for v in th_disp],
        "R_map_recon": [[round(float(v), 6) for v in row] for row in R_recon],
        "R_map_truth": [[round(float(v), 6) for v in row] for row in R_truth],
        "reconPositions": [round(float(v), 5) for v in pos_recon.ravel()],
        "truthPositions": [round(float(v), 5) for v in pos_truth.ravel()],
        "indices": [int(v) for v in indices],
        "nZ": int(n_z),
        "nTheta": int(th_disp.size),
        "meta": {
            "acquireStepDeg": acq_step,
            "acquireNz": acq_nz,
            "displayStepDeg": step,
            "displayNz": int(n_z),
            "nVertices": int(pos_recon.shape[0]),
            "nTriangles": int(indices.size // 3),
            "physics": C.PHYSICS_LABEL,
            "honesty": (
                f"Display sampling of fitted wall; acquire was "
                f"{acq_step:g}° × {acq_nz} z"
            ),
        },
    }


def viewer_payload(
    display: dict[str, Any],
    *,
    score: dict[str, Any] | None = None,
    title: str = "SonoLab Block 8 — dense pipe wall",
) -> dict[str, Any]:
    """Compact payload for the Three.js viewer (external sidecar)."""
    meta = dict(display.get("meta") or {})
    sc = score or {}
    return {
        "title": title,
        "meta": meta,
        "score": {
            "rAbsRmseOverR": sc.get("rAbsRmseOverR"),
            "shapeRmse": sc.get("shapeRmse"),
            "dentThetaErrDeg": sc.get("dentThetaErrDeg"),
            "dentZErr": sc.get("dentZErr"),
            "dentThetaEst": sc.get("dentThetaEst"),
            "dentThetaTrue": sc.get("dentThetaTrue"),
            "dentZEst": sc.get("dentZEst"),
            "dentZTrue": sc.get("dentZTrue"),
            "pass": sc.get("pass"),
            "physics": sc.get("physics") or meta.get("physics"),
        },
        "recon": {
            "positions": display["reconPositions"],
            "indices": display["indices"],
            "nZ": display["nZ"],
            "nTheta": display["nTheta"],
        },
        "truth": {
            "positions": display["truthPositions"],
            "indices": display["indices"],
            "nZ": display["nZ"],
            "nTheta": display["nTheta"],
        },
        "dentHighlight": {
            "thetaDeg": sc.get("dentThetaEst"),
            "z": sc.get("dentZEst"),
        },
    }
