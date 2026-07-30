"""Unit tests for blind packs / scorer and joint geometry (no FDTD)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sonolab.pipe2d.blind import AcquisitionPack, TruthPack, score_estimate
from sonolab.pipe2d.joint_geometry import (
    estimate_from_pack,
    fit_joint_geometry,
    predict_echo_time,
    predict_standoff,
)
from sonolab.pipe2d.phased_array import (
    ARRAY_LOOK_OFFSET,
    PULSE_WIDTH,
    angle_diff_deg,
)


def test_score_fails_closed_without_truth():
    est = {
        "theta0_deg": 0.0,
        "A_over_R": 0.1,
        "e_over_R": 0.0,
        "tau_R_over_c": 100.0,
        "r_pred": [60.0] * 8,
    }
    with pytest.raises(ValueError, match="TruthPack"):
        score_estimate(est, None, a_over_r_tau=0.05)


def test_estimator_accepts_acquisition_pack_only():
    """Synthetic circle+dent times → joint fit via AcquisitionPack API."""
    R, c, A, th0 = 68.0, 0.45, 8.0, 40.0
    cx, cy = -10.0, 3.0  # pipe center in tool frame
    thetas = np.arange(0.0, 360.0, 15.0)
    t_echo = np.array(
        [
            predict_echo_time(
                float(th),
                cx=cx,
                cy=cy,
                R0=R,
                A=A,
                theta0_deg=th0,
                c=c,
            )
            for th in thetas
        ]
    )
    pack = AcquisitionPack(
        theta_deg=[float(v) for v in thetas],
        t_echo=[float(v) for v in t_echo],
        array_look_offset=float(ARRAY_LOOK_OFFSET),
        pulse_width=float(PULSE_WIDTH),
        n_elem=20,
        step_deg=15.0,
    )
    # Must not require truth fields on the pack
    assert "pipe_r" not in pack.to_dict()
    est = estimate_from_pack(pack, c_init=0.5)
    assert est["scaleMode"] == "free_c"
    assert abs(angle_diff_deg(est["theta0_deg"], th0)) < 15.0
    assert abs(est["A_over_R"] - A / R) < 0.04
    assert abs(est["e_over_R"] - math.hypot(cx, cy) / R) < 0.05


def test_c_cal_recovers_absolute_radius():
    """Labeled fluid c_cal freezes scale → absolute R within tol."""
    R, c, A, th0 = 68.0, 0.45, 8.0, 40.0
    cx, cy = -10.0, 3.0
    thetas = np.arange(0.0, 360.0, 15.0)
    t_echo = np.array(
        [
            predict_echo_time(
                float(th),
                cx=cx,
                cy=cy,
                R0=R,
                A=A,
                theta0_deg=th0,
                c=c,
            )
            for th in thetas
        ]
    )
    pack = AcquisitionPack(
        theta_deg=[float(v) for v in thetas],
        t_echo=[float(v) for v in t_echo],
        array_look_offset=float(ARRAY_LOOK_OFFSET),
        pulse_width=float(PULSE_WIDTH),
        n_elem=20,
        step_deg=15.0,
    )
    est = estimate_from_pack(pack, c_cal=c)
    assert est["scaleMode"] == "c_cal"
    assert abs(est["c"] - c) < 1e-4
    assert abs(est["R"] - R) / R < 0.03
    assert abs(est["e"] - math.hypot(cx, cy)) < 2.0
    assert "wall_xy" in est and len(est["wall_xy"]) == len(thetas)

    r_true = [
        predict_standoff(
            float(th), cx=cx, cy=cy, R0=R, A=A, theta0_deg=th0
        )
        for th in thetas
    ]
    truth = TruthPack(
        ecc_mag=math.hypot(cx, cy),
        ecc_phi_deg=math.degrees(math.atan2(cx, cy)),
        c_prop=c,
        pipe_r=R,
        dent_on=True,
        dent_theta0_deg=th0,
        dent_depth=A,
        dent_half_deg=20.0,
        r_true=r_true,
        theta_deg=[float(v) for v in thetas],
    )
    score = score_estimate(est, truth, a_over_r_tau=0.048, loc_tol_deg=15.0)
    assert score["scaleMode"] == "c_cal"
    assert abs(score["R_err"]) / R < 0.03
    assert score["rAbsRmse"] is not None


def test_fit_recovers_centered_dent_scale_invariant():
    R, c, A, th0 = 68.0, 0.45, 8.0, 0.0
    thetas = np.arange(0.0, 360.0, 10.0)
    t_echo = [
        predict_echo_time(
            float(th), cx=0.0, cy=0.0, R0=R, A=A, theta0_deg=th0, c=c
        )
        for th in thetas
    ]
    est = fit_joint_geometry(t_echo, thetas, c_init=0.55)
    assert abs(angle_diff_deg(est["theta0_deg"], th0)) < 5.0
    assert est["A_over_R"] > 0.06
    assert est["e_over_R"] < 0.08


def test_predict_standoff_matches_centered_radius():
    r = predict_standoff(0.0, cx=0.0, cy=0.0, R0=68.0, A=0.0, theta0_deg=0.0)
    assert abs(r - 68.0) < 1e-6
