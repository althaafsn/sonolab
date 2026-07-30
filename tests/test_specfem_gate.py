"""Unit tests for Block 5 SPECFEM ToF gate (no full SPECFEM required)."""

from __future__ import annotations

import numpy as np
import pytest

from sonolab.specfem_gate import (
    EX01_SDH_CENTER_Z_M,
    EX01_SDH_RADIUS_M,
    EX01_SURFACE_Z_M,
    EX01_VP_DEFAULT,
    analytic_sdh_tof_s,
    compare_tof,
    pick_echo_tof_s,
)


def test_analytic_sdh_tof_hand_calc():
    L = EX01_SURFACE_Z_M - (EX01_SDH_CENTER_Z_M + EX01_SDH_RADIUS_M)
    assert abs(L - 0.023) < 1e-12
    t = analytic_sdh_tof_s()
    assert abs(t - 2.0 * L / EX01_VP_DEFAULT) < 1e-15
    assert abs(t * 1e6 - 7.73109243697479) < 1e-6


def test_pick_recovers_synthetic_lag():
    dt = 2.0e-8
    n = 1000
    lag_s = 8.0e-6
    lag_i = int(round(lag_s / dt))
    amp = np.zeros(n, dtype=np.float64)
    # Main bang
    amp[5:25] = np.hanning(20)
    # Echo burst
    w = np.hanning(30)
    amp[lag_i : lag_i + 30] = 0.4 * w
    pick = pick_echo_tof_s(amp, dt, tx_exclude_s=3.0e-6)
    assert abs(pick - lag_s) < 5 * dt


def test_compare_tof_band():
    analytic = analytic_sdh_tof_s()
    ok = compare_tof(analytic * 1.05, analytic, rel_tol=0.10)
    assert ok["pass"] is True
    bad = compare_tof(analytic * 1.20, analytic, rel_tol=0.10)
    assert bad["pass"] is False


def test_specfem_root_prefers_nested():
    from sonolab.config import default_specfem_root, sonolab_root

    nested = sonolab_root() / "specfem2d-UT"
    if nested.is_dir():
        assert default_specfem_root().resolve() == nested.resolve()
