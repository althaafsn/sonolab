"""Parity / smoke for Rust Lab A when extension is installed."""

from __future__ import annotations

import numpy as np
import pytest

from sonolab.pipe2d import acquire as acq
from sonolab.pipe2d import phased_array as pa


@pytest.fixture(scope="module")
def rust_eng():
    eng = acq._load_rust()
    if eng is None:
        pytest.skip(f"Rust Lab A not installed: {acq._RUST_ERR}")
    return eng


def test_rust_look_pick_matches_python(rust_eng):
    py = pa.record_pulse_window(theta_deg=0.0, dent=True, n_elem=20, ecc_mag=0.0)
    rs = np.asarray(
        rust_eng.record_look(
            0.0,
            True,
            20,
            0.0,
            90.0,
            float(pa.DENT_THETA0_DEG),
            float(pa.DENT_DEPTH),
            float(pa.DENT_HALF_DEG),
        )
    )
    assert py.shape == rs.shape
    # f32 engine: picks must stay within 1 sample of float64 reference
    assert abs(acq.blind_pick_echo(py) - acq.blind_pick_echo(rs)) <= 1.0


def test_acquire_defaults_to_rust_when_available(rust_eng):
    assert acq.laba_backend() == "rust"
    pack, truth = acq.acquire_sector_scan(
        dent=True, step_deg=15.0, ecc_mag=0.0, progress=False
    )
    assert pack.meta.get("labaBackend") == "rust"
    assert len(pack.theta_deg) == 24
    assert truth.dent_on is True
