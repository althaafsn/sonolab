"""Blind Lab B packs and scale-invariant scoring (no pipe-prior cheat).

AcquisitionPack holds only instrument + TX/RX facts. TruthPack is a sidecar
for the scorer. Lab B estimators must accept AcquisitionPack only.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from sonolab.pipe2d.phased_array import angle_diff_deg


@dataclass
class AcquisitionPack:
    """Instrument + echo times only — no pipe R, c, pose, or dent truth."""

    theta_deg: list[float]
    t_echo: list[float]
    array_look_offset: float
    pulse_width: float
    n_elem: int
    step_deg: float
    # Instrument aperture facts (Lab B may know device geometry; not pipe R).
    elem_pitch: float = 2.0
    aperture_span: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = max(int(self.n_elem), 2)
        pitch = float(self.elem_pitch)
        if float(self.aperture_span) <= 0.0:
            self.aperture_span = float(n - 1) * pitch

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AcquisitionPack:
        n_elem = int(d["n_elem"])
        elem_pitch = float(d.get("elem_pitch", 2.0))
        span_raw = d.get("aperture_span")
        aperture_span = (
            float(span_raw)
            if span_raw is not None
            else float(max(n_elem, 2) - 1) * elem_pitch
        )
        return cls(
            theta_deg=list(d["theta_deg"]),
            t_echo=list(d["t_echo"]),
            array_look_offset=float(d["array_look_offset"]),
            pulse_width=float(d["pulse_width"]),
            n_elem=n_elem,
            step_deg=float(d["step_deg"]),
            elem_pitch=elem_pitch,
            aperture_span=aperture_span,
            meta=dict(d.get("meta") or {}),
        )


@dataclass
class TruthPack:
    """Lab A ground truth sidecar — scorer only; never for inference."""

    ecc_mag: float
    ecc_phi_deg: float
    c_prop: float
    pipe_r: float
    dent_on: bool
    dent_theta0_deg: float
    dent_depth: float
    dent_half_deg: float
    r_true: list[float]
    theta_deg: list[float]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TruthPack:
        return cls(
            ecc_mag=float(d["ecc_mag"]),
            ecc_phi_deg=float(d["ecc_phi_deg"]),
            c_prop=float(d["c_prop"]),
            pipe_r=float(d["pipe_r"]),
            dent_on=bool(d["dent_on"]),
            dent_theta0_deg=float(d["dent_theta0_deg"]),
            dent_depth=float(d["dent_depth"]),
            dent_half_deg=float(d["dent_half_deg"]),
            r_true=list(d["r_true"]),
            theta_deg=list(d["theta_deg"]),
            meta=dict(d.get("meta") or {}),
        )


def _shape_norm(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    m = float(np.mean(r))
    if m < 1e-12:
        return np.zeros_like(r)
    return r / m


def score_estimate(
    estimate: dict[str, Any],
    truth: TruthPack | None,
    *,
    a_over_r_tau: float,
    loc_tol_deg: float = 30.0,
) -> dict[str, Any]:
    """Compare Lab B estimate to truth. Fails closed if truth is missing."""
    if truth is None:
        raise ValueError(
            "score_estimate requires TruthPack sidecar; cannot score without truth"
        )

    theta0_est = float(estimate["theta0_deg"])
    a_over_r_est = float(estimate["A_over_R"])
    e_over_r_est = float(estimate["e_over_R"])
    tau_est = float(estimate["tau_R_over_c"])

    R_t = float(truth.pipe_r)
    A_t = float(truth.dent_depth) if truth.dent_on else 0.0
    a_over_r_true = A_t / max(R_t, 1e-12)
    e_over_r_true = float(truth.ecc_mag) / max(R_t, 1e-12)
    tau_true = R_t / max(float(truth.c_prop), 1e-12)

    loc_err = float(angle_diff_deg(theta0_est, truth.dent_theta0_deg))
    dent_call = bool(estimate.get("dentCall", True))
    above = bool(dent_call and a_over_r_est > float(a_over_r_tau))
    localized = bool(abs(loc_err) <= float(loc_tol_deg))
    if truth.dent_on:
        detected = above and localized
    else:
        detected = above  # false alarm if dent off

    r_est = np.asarray(estimate.get("r_pred") or [], dtype=np.float64)
    r_true = np.asarray(truth.r_true, dtype=np.float64)
    if r_est.size == r_true.size and r_est.size > 0:
        shape_rmse = float(
            np.sqrt(np.mean((_shape_norm(r_est) - _shape_norm(r_true)) ** 2))
        )
    else:
        shape_rmse = float("nan")

    out: dict[str, Any] = {
        "aboveTau": above,
        "localized": localized,
        "locErrorDeg": round(loc_err, 3),
        "detected": detected,
        "A_over_R_est": round(a_over_r_est, 6),
        "A_over_R_true": round(a_over_r_true, 6),
        "A_over_R_err": round(a_over_r_est - a_over_r_true, 6),
        "e_over_R_est": round(e_over_r_est, 6),
        "e_over_R_true": round(e_over_r_true, 6),
        "e_over_R_err": round(e_over_r_est - e_over_r_true, 6),
        "tau_est": round(tau_est, 6),
        "tau_true": round(tau_true, 6),
        "tau_err": round(tau_est - tau_true, 6),
        "theta0_est": round(theta0_est, 3),
        "theta0_true": round(float(truth.dent_theta0_deg), 3),
        "shapeRmse": (
            None if math.isnan(shape_rmse) else round(shape_rmse, 6)
        ),
        "a_over_r_tau": float(a_over_r_tau),
        "locTolDeg": float(loc_tol_deg),
        "dentOn": bool(truth.dent_on),
    }

    if str(estimate.get("scaleMode") or "") == "c_cal":
        R_est = float(estimate.get("R", float("nan")))
        e_est = float(estimate.get("e", float("nan")))
        A_est = float(estimate.get("A", 0.0))
        e_true = float(truth.ecc_mag)
        A_true = float(truth.dent_depth) if truth.dent_on else 0.0
        if r_est.size == r_true.size and r_est.size > 0:
            r_abs_rmse = float(np.sqrt(np.mean((r_est - r_true) ** 2)))
        else:
            r_abs_rmse = float("nan")
        out.update(
            {
                "scaleMode": "c_cal",
                "R_est": round(R_est, 6),
                "R_true": round(R_t, 6),
                "R_err": round(R_est - R_t, 6),
                "R_rel_err": round((R_est - R_t) / max(R_t, 1e-12), 6),
                "e_est": round(e_est, 6),
                "e_true": round(e_true, 6),
                "e_err": round(e_est - e_true, 6),
                "A_est": round(A_est, 6),
                "A_true": round(A_true, 6),
                "A_err": round(A_est - A_true, 6),
                "rAbsRmse": (
                    None if math.isnan(r_abs_rmse) else round(r_abs_rmse, 6)
                ),
            }
        )
    return out
