"""Unit tests for Block 7a uncertainty / refuse (no FDTD)."""

from __future__ import annotations

import numpy as np

from sonolab.pipe2d.blind import AcquisitionPack
from sonolab.pipe2d.joint_geometry import estimate_from_pack, predict_echo_time
from sonolab.pipe2d.phased_array import ARRAY_LOOK_OFFSET, PULSE_WIDTH
from sonolab.pipe2d.uncertainty import ci_covers
from sonolab.pipe2d.uncertainty_gate import evaluate_gate as gate_eval


def _synthetic_pack(
    *,
    R=68.0,
    c=0.45,
    A=8.0,
    th0=40.0,
    cx=-10.0,
    cy=3.0,
    step=15.0,
    noise=0.0,
) -> AcquisitionPack:
    thetas = np.arange(0.0, 360.0, step)
    rng = np.random.default_rng(0)
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
            + float(noise * rng.normal())
            for th in thetas
        ]
    )
    return AcquisitionPack(
        theta_deg=[float(v) for v in thetas],
        t_echo=[float(v) for v in t_echo],
        array_look_offset=float(ARRAY_LOOK_OFFSET),
        pulse_width=float(PULSE_WIDTH),
        n_elem=20,
        step_deg=float(step),
    )


def test_clean_synthetic_has_ci_and_not_refuse():
    pack = _synthetic_pack(noise=0.0)
    est = estimate_from_pack(pack, c_init=0.5)
    assert "uncertainty" in est
    assert est["uncertainty"]["method"] == "nlls_linearization_approx"
    assert "ci95" in est["uncertainty"]
    assert "A_over_R" in est["uncertainty"]["ci95"]
    assert est.get("refuse") is False
    assert est.get("confidence") == "ok"
    half = est["uncertainty"]["ci95"]["A_over_R"]["halfWidth"]
    assert half < 0.15
    # Clean synthetic with clear A should not refuse on dentUncertain
    assert "dentUncertain" not in (est.get("refuseReasons") or [])
    assert est.get("dentCall") is True or est.get("A_over_R", 0) > 0.05


def test_garbage_times_refuse():
    thetas = list(np.arange(0.0, 360.0, 15.0))
    rng = np.random.default_rng(1)
    t_echo = [float(50.0 + 80.0 * rng.random()) for _ in thetas]
    pack = AcquisitionPack(
        theta_deg=thetas,
        t_echo=t_echo,
        array_look_offset=float(ARRAY_LOOK_OFFSET),
        pulse_width=float(PULSE_WIDTH),
        n_elem=20,
        step_deg=15.0,
    )
    est = estimate_from_pack(pack, c_init=0.5)
    assert est.get("refuse") is True
    assert est.get("dentCall") is False
    assert est.get("confidence") == "refuse"
    assert len(est.get("refuseReasons") or []) >= 1


def test_ci_covers_circular():
    ci = {"value": 10.0, "lo": 0.0, "hi": 20.0, "halfWidth": 10.0}
    assert ci_covers(ci, 5.0, circular_deg=True)
    assert not ci_covers(ci, 90.0, circular_deg=True)


def test_gate_eval_logic():
    primary = {
        "coverage": {"theta0": 0.8, "A_over_R": 0.75, "e_over_R": 0.9},
        "refuseRateDentOn": 0.1,
        "refuseRateDentOff": 0.1,
        "falseAlarmRate": 0.0,
    }
    stress = {
        "coverage": {"theta0": 0.5, "A_over_R": 0.5, "e_over_R": 0.5},
        "refuseRateDentOn": 0.5,
        "refuseRateDentOff": 0.5,
        "falseAlarmRate": 0.0,
    }
    g = gate_eval(primary, stress)
    assert g["pass"] is True

    bad = gate_eval(
        {
            "coverage": {"theta0": 0.2, "A_over_R": 0.2, "e_over_R": 0.2},
            "refuseRateDentOn": 0.5,
            "refuseRateDentOff": 0.0,
            "falseAlarmRate": 0.0,
        },
        {
            "refuseRateDentOn": 0.1,
            "refuseRateDentOff": 0.0,
            "falseAlarmRate": 0.0,
            "coverage": {},
        },
    )
    assert bad["pass"] is False
