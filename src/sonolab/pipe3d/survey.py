"""Blind survey packs, Lab B assembly, and scoring for pipe3d."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from sonolab.pipe2d.blind import AcquisitionPack
from sonolab.pipe2d.joint_geometry import estimate_from_pack
from sonolab.pipe2d.phased_array import angle_diff_deg, look_unit
from sonolab.pipe3d import constants as C


@dataclass
class StationAcquisition:
    z: float
    pack: AcquisitionPack

    def to_dict(self) -> dict[str, Any]:
        return {"z": float(self.z), "pack": self.pack.to_dict()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StationAcquisition:
        return cls(z=float(d["z"]), pack=AcquisitionPack.from_dict(d["pack"]))


@dataclass
class SurveyPack:
    """Multi-station instrument + echoes only (no pipe/dent/c truth)."""

    stations: list[StationAcquisition]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stations": [s.to_dict() for s in self.stations],
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SurveyPack:
        return cls(
            stations=[StationAcquisition.from_dict(s) for s in d["stations"]],
            meta=dict(d.get("meta") or {}),
        )


@dataclass
class SurveyTruth:
    """Lab A sidecar for scoring only."""

    z: list[float]
    theta_deg: list[float]
    r_true: list[list[float]]  # [n_z][n_theta]
    pipe_r: float
    c_prop: float
    dent_on: bool
    dent_theta0_deg: float
    dent_z0: float
    dent_depth: float
    ecc_mag: float
    ecc_phi_deg: float
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SurveyTruth:
        return cls(
            z=[float(v) for v in d["z"]],
            theta_deg=[float(v) for v in d["theta_deg"]],
            r_true=[list(row) for row in d["r_true"]],
            pipe_r=float(d["pipe_r"]),
            c_prop=float(d["c_prop"]),
            dent_on=bool(d["dent_on"]),
            dent_theta0_deg=float(d["dent_theta0_deg"]),
            dent_z0=float(d["dent_z0"]),
            dent_depth=float(d["dent_depth"]),
            ecc_mag=float(d["ecc_mag"]),
            ecc_phi_deg=float(d["ecc_phi_deg"]),
            meta=dict(d.get("meta") or {}),
        )


def estimate_survey(
    survey: SurveyPack,
    *,
    c_cal: float,
    freeze_circle_from_mid: bool = True,
) -> list[dict[str, Any]]:
    """Lab B: per-station estimate with shared labeled fluid c_cal.

    Optionally freeze circle from the mid-station fit (instrument data) so
    dent-ON stations share a stable absolute circle - never Lab A truth.
    """
    c_cal = float(c_cal)
    if c_cal <= 0:
        raise ValueError("c_cal must be positive labeled fluid sound speed")
    n = len(survey.stations)
    mid = n // 2
    # First pass: independent fits
    raw: list[dict[str, Any]] = []
    for st in survey.stations:
        est = estimate_from_pack(st.pack, c_cal=c_cal)
        est["z"] = float(st.z)
        raw.append(est)
    if not freeze_circle_from_mid or n < 2:
        return raw
    ref = {
        "R": float(raw[mid]["R"]),
        "cx": float(raw[mid]["cx"]),
        "cy": float(raw[mid]["cy"]),
        "c": float(raw[mid]["c"]),
    }
    out: list[dict[str, Any]] = []
    for i, st in enumerate(survey.stations):
        if i == mid:
            out.append(raw[i])
            continue
        est = estimate_from_pack(st.pack, c_cal=c_cal, circle_ref=ref)
        est["z"] = float(st.z)
        out.append(est)
    return out


def assemble_wall(estimates: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack station r_pred → R(θ,z) grid and tool-frame XYZ cloud."""
    if not estimates:
        raise ValueError("need ≥1 station estimate")
    z = np.array([float(e["z"]) for e in estimates], dtype=np.float64)
    thetas = np.asarray(estimates[0]["theta_deg"], dtype=np.float64)
    n_z = len(estimates)
    n_th = int(thetas.size)
    R_map = np.zeros((n_z, n_th), dtype=np.float64)
    xyz: list[list[float]] = []
    for k, est in enumerate(estimates):
        r = np.asarray(est["r_pred"], dtype=np.float64)
        if r.size != n_th:
            raise ValueError("station theta grids must match")
        R_map[k] = r
        for th, ri in zip(thetas, r):
            ux, uy = look_unit(float(th))
            xyz.append(
                [round(float(ri * ux), 6), round(float(ri * uy), 6), round(float(z[k]), 6)]
            )
    return {
        "z": [float(v) for v in z],
        "theta_deg": [float(v) for v in thetas],
        "R_map": [[round(float(v), 6) for v in row] for row in R_map],
        "xyz": xyz,
        "A_of_z": [float(e["A"]) for e in estimates],
        "theta0_of_z": [float(e["theta0_deg"]) for e in estimates],
        "e_of_z": [float(e["e"]) for e in estimates],
        "R_of_z": [float(e["R"]) for e in estimates],
        "scaleMode": str(estimates[0].get("scaleMode") or "free_c"),
        "physics": C.PHYSICS_LABEL,
    }


def score_survey(
    product: dict[str, Any],
    truth: SurveyTruth | None,
    *,
    estimates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compare assembled wall to SurveyTruth. Fails closed without truth."""
    if truth is None:
        raise ValueError(
            "score_survey requires SurveyTruth sidecar; cannot score without truth"
        )
    R_est = np.asarray(product["R_map"], dtype=np.float64)
    R_true = np.asarray(truth.r_true, dtype=np.float64)
    if R_est.shape != R_true.shape:
        raise ValueError(
            f"R_map shape {R_est.shape} != truth {R_true.shape}"
        )
    r_abs = float(np.sqrt(np.mean((R_est - R_true) ** 2)))
    # Shape: normalize each z-row
    def _row_norm(m: np.ndarray) -> np.ndarray:
        out = np.zeros_like(m)
        for i in range(m.shape[0]):
            mu = float(np.mean(m[i]))
            out[i] = m[i] / max(mu, 1e-12)
        return out

    shape_rmse = float(
        np.sqrt(np.mean((_row_norm(R_est) - _row_norm(R_true)) ** 2))
    )

    # Dent peak localization
    flat = R_true.ravel()
    k_min = int(np.argmin(flat))
    n_th = R_true.shape[1]
    z_i_true = k_min // n_th
    th_i_true = k_min % n_th
    z_true_peak = float(truth.z[z_i_true])
    th_true_peak = float(truth.theta_deg[th_i_true])

    flat_e = R_est.ravel()
    k_min_e = int(np.argmin(flat_e))
    z_i_est = k_min_e // n_th
    th_i_est = k_min_e % n_th
    z_est_peak = float(product["z"][z_i_est])
    th_est_peak = float(product["theta_deg"][th_i_est])

    z_err = abs(z_est_peak - z_true_peak)
    th_err = abs(angle_diff_deg(th_est_peak, th_true_peak))
    if estimates:
        # Prefer Lab B θ0 at station nearest dent_z0
        zs = np.asarray(product["z"], dtype=np.float64)
        k = int(np.argmin(np.abs(zs - float(truth.dent_z0))))
        th_est_peak = float(estimates[k]["theta0_deg"])
        th_err = abs(angle_diff_deg(th_est_peak, float(truth.dent_theta0_deg)))
        z_est_peak = float(zs[k])
        # z peak from max A(z)
        a_of_z = np.asarray(product.get("A_of_z") or [], dtype=np.float64)
        if a_of_z.size == zs.size and a_of_z.size:
            z_est_peak = float(zs[int(np.argmax(a_of_z))])
            z_err = abs(z_est_peak - float(truth.dent_z0))

    pipe_r = float(truth.pipe_r)
    return {
        "rAbsRmse": round(r_abs, 6),
        "rAbsRmseOverR": round(r_abs / max(pipe_r, 1e-12), 6),
        "shapeRmse": round(shape_rmse, 6),
        "dentThetaEst": round(th_est_peak, 3),
        "dentThetaTrue": round(float(truth.dent_theta0_deg), 3),
        "dentThetaErrDeg": round(th_err, 3),
        "dentZEst": round(z_est_peak, 3),
        "dentZTrue": round(float(truth.dent_z0), 3),
        "dentZErr": round(z_err, 3),
        "pipeR": pipe_r,
        "physics": C.PHYSICS_LABEL,
        "pass": {
            "rAbsRmseOverR": (r_abs / max(pipe_r, 1e-12)) <= 0.08,
            "dentTheta": th_err <= 25.0,
            "dentZ": z_err <= (max(float(C.DENT_HALF_Z), 1.0) * 1.5),
        },
    }
