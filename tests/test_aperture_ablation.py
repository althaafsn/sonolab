"""Block 6 aperture ablation: pack schema, coherent RX SNR, gate logic."""

from __future__ import annotations

import numpy as np
import pytest

from sonolab.pipe2d.aperture_ablation import evaluate_ablation_pass
from sonolab.pipe2d.blind import AcquisitionPack
from sonolab.pipe2d.joint_geometry import estimate_from_pack
from sonolab.pipe2d.phased_array import (
    ARRAY_LOOK_OFFSET,
    PULSE_WIDTH,
    TX_EXCLUDE,
    aperture_span_cells,
    record_pulse_window,
)


def test_acquisition_pack_aperture_fields_backward_compat():
    """Old packs without elem_pitch/aperture_span still load."""
    d = {
        "theta_deg": [0.0, 15.0],
        "t_echo": [100.0, 101.0],
        "array_look_offset": float(ARRAY_LOOK_OFFSET),
        "pulse_width": float(PULSE_WIDTH),
        "n_elem": 20,
        "step_deg": 15.0,
    }
    pack = AcquisitionPack.from_dict(d)
    assert pack.elem_pitch == 2.0
    assert pack.aperture_span == pytest.approx(aperture_span_cells(20))
    assert "pipe_r" not in pack.to_dict()


def test_aperture_span_cells():
    assert aperture_span_cells(2) == 2.0
    assert aperture_span_cells(20) == 38.0


def test_estimator_ignores_aperture_fields_for_geometry():
    """Lab B still runs from picks; aperture fields are instrument metadata."""
    from sonolab.pipe2d.joint_geometry import predict_echo_time

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
        n_elem=8,
        step_deg=15.0,
        elem_pitch=2.0,
        aperture_span=14.0,
    )
    est = estimate_from_pack(pack, c_init=0.5)
    assert est["scaleMode"] == "free_c"
    assert "success" in est


def test_coherent_rx_peak_grows_with_aperture():
    """Centered dent-OFF: denser coherent aperture yields higher post-TX peak."""
    s2 = record_pulse_window(theta_deg=0.0, dent=False, n_elem=2, ecc_mag=0.0)
    s20 = record_pulse_window(theta_deg=0.0, dent=False, n_elem=20, ecc_mag=0.0)
    peak2 = float(np.max(np.abs(s2[TX_EXCLUDE:])))
    peak20 = float(np.max(np.abs(s20[TX_EXCLUDE:])))
    assert peak20 > peak2 * 1.05


def test_evaluate_ablation_pass_logic():
    by_n = {
        2: {
            "locErrorDegMedian": 12.0,
            "A_over_R_rmse": 0.08,
            "falseAlarmRate": 0.125,
            "recall": 0.5,
        },
        20: {
            "locErrorDegMedian": 5.0,
            "A_over_R_rmse": 0.04,
            "falseAlarmRate": 0.0,
            "recall": 0.9,
        },
    }
    gate = evaluate_ablation_pass(by_n)
    assert gate["pass"] is True
    assert gate["checks"]["locMedianImproved"] is True

    # Sparse never detects; dense does → loc improves.
    by_n_null_sparse = {
        2: {
            "locErrorDegMedian": None,
            "A_over_R_rmse": 0.10,
            "falseAlarmRate": 0.0,
            "recall": 0.0,
        },
        20: {
            "locErrorDegMedian": 4.0,
            "A_over_R_rmse": 0.03,
            "falseAlarmRate": 0.0,
            "recall": 0.75,
        },
    }
    assert evaluate_ablation_pass(by_n_null_sparse)["pass"] is True

    by_n_bad = {
        2: {
            "locErrorDegMedian": 5.0,
            "A_over_R_rmse": 0.04,
            "falseAlarmRate": 0.0,
            "recall": 0.9,
        },
        20: {
            "locErrorDegMedian": 12.0,
            "A_over_R_rmse": 0.08,
            "falseAlarmRate": 0.25,
            "recall": 0.5,
        },
    }
    assert evaluate_ablation_pass(by_n_bad)["pass"] is False
