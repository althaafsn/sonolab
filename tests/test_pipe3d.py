"""Unit tests for pipe3d geometry / survey assembly (minimal FDTD)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from sonolab.pipe2d.blind import AcquisitionPack
from sonolab.pipe3d import constants as C
from sonolab.pipe3d.fdtd3d import blind_pick_echo, record_pulse_look
from sonolab.pipe3d.geometry import true_standoff
from sonolab.pipe3d.survey import (
    StationAcquisition,
    SurveyPack,
    SurveyTruth,
    assemble_wall,
    estimate_survey,
    score_survey,
)


def test_true_standoff_centered():
    r = true_standoff(0.0, C.CZ, dent=False, ecc_mag=0.0)
    assert abs(r - C.PIPE_R) < 1e-6


def test_dent_reduces_standoff_at_peak():
    r0 = true_standoff(
        C.DENT_THETA0_DEG, C.DENT_Z0, dent=False, ecc_mag=0.0
    )
    r1 = true_standoff(
        C.DENT_THETA0_DEG, C.DENT_Z0, dent=True, ecc_mag=0.0
    )
    assert r1 < r0 - 1.0


def test_one_look_tof_near_analytic():
    """Single 3D PE look: picked ToF ≈ pulse + 2(r−offset)/c."""
    th, z = 0.0, float(C.CZ)
    r = true_standoff(th, z, dent=False, ecc_mag=0.0)
    series = record_pulse_look(
        theta_deg=th, z_tool=z, dent=False, ecc_mag=0.0
    )
    t = blind_pick_echo(series)
    t_exp = C.PULSE_WIDTH + 2.0 * (r - C.ARRAY_LOOK_OFFSET) / C.C_PROP
    # Soft wall + pick threshold: allow generous band on coarse grid
    assert abs(t - t_exp) < 25.0, f"t={t:.1f} t_exp={t_exp:.1f}"


def test_assemble_and_score_synthetic():
    thetas = list(np.arange(0.0, 360.0, 30.0))
    z_list = [20.0, 24.0, 28.0]
    estimates = []
    r_true = []
    for z in z_list:
        r_row = []
        for th in thetas:
            r = 22.0 - (2.0 if abs(th) < 1e-9 and abs(z - 24.0) < 1e-9 else 0.0)
            r_row.append(r)
        r_true.append(r_row)
        estimates.append(
            {
                "z": z,
                "theta_deg": thetas,
                "r_pred": r_row,
                "R": 22.0,
                "e": 0.0,
                "A": 2.0 if abs(z - 24.0) < 1e-9 else 0.1,
                "theta0_deg": 0.0,
                "c": 0.45,
                "scaleMode": "c_cal",
                "cx": 0.0,
                "cy": 0.0,
            }
        )
    product = assemble_wall(estimates)
    assert len(product["xyz"]) == len(z_list) * len(thetas)
    truth = SurveyTruth(
        z=z_list,
        theta_deg=thetas,
        r_true=r_true,
        pipe_r=22.0,
        c_prop=0.45,
        dent_on=True,
        dent_theta0_deg=0.0,
        dent_z0=24.0,
        dent_depth=2.0,
        ecc_mag=0.0,
        ecc_phi_deg=90.0,
    )
    score = score_survey(product, truth, estimates=estimates)
    assert score["rAbsRmse"] < 1e-6
    with pytest.raises(ValueError, match="SurveyTruth"):
        score_survey(product, None)


def test_survey_pack_roundtrip():
    pack = AcquisitionPack(
        theta_deg=[0.0, 90.0, 180.0, 270.0, 45.0, 135.0],
        t_echo=[100.0] * 6,
        array_look_offset=2.0,
        pulse_width=12.0,
        n_elem=1,
        step_deg=90.0,
        meta={"z": 24.0},
    )
    survey = SurveyPack(
        stations=[StationAcquisition(z=24.0, pack=pack)],
        meta={"physics": C.PHYSICS_LABEL},
    )
    d = survey.to_dict()
    s2 = SurveyPack.from_dict(d)
    assert s2.stations[0].z == 24.0
    assert s2.meta["physics"] == C.PHYSICS_LABEL


def test_display_wall_dense_mesh():
    from sonolab.pipe3d.display import build_display_wall, mesh_indices

    thetas = list(np.arange(0.0, 360.0, 30.0))
    z_list = [20.0, 24.0, 28.0]
    estimates = []
    r_true = []
    for z in z_list:
        r_row = [22.0 - (1.5 if abs(th) < 1e-9 else 0.0) for th in thetas]
        r_true.append(r_row)
        estimates.append(
            {
                "z": z,
                "theta_deg": thetas,
                "r_pred": r_row,
                "R": 22.0,
                "e": 0.0,
                "A": 1.5,
                "theta0_deg": 0.0,
                "c": 0.45,
                "scaleMode": "c_cal",
                "cx": 0.0,
                "cy": 0.0,
            }
        )
    product = assemble_wall(estimates)
    truth = SurveyTruth(
        z=z_list,
        theta_deg=thetas,
        r_true=r_true,
        pipe_r=22.0,
        c_prop=0.45,
        dent_on=True,
        dent_theta0_deg=0.0,
        dent_z0=24.0,
        dent_depth=2.0,
        ecc_mag=0.0,
        ecc_phi_deg=90.0,
    )
    display = build_display_wall(
        product,
        truth,
        display_step_deg=0.25,
        n_z_display=64,
        acquire_step_deg=30.0,
        acquire_n_z=3,
    )
    assert display["nZ"] == 64
    assert display["nTheta"] == 1440
    assert len(display["reconPositions"]) == 64 * 1440 * 3
    assert display["meta"]["displayStepDeg"] == 0.25
    assert display["meta"]["acquireStepDeg"] == 30.0
    idx = mesh_indices(64, 1440)
    assert idx.size == len(display["indices"])
    assert idx.size % 3 == 0
    assert idx.size // 3 == 63 * 1440 * 2


def test_write_survey_view_external(tmp_path):
    from sonolab.pipe3d.display import build_display_wall
    from sonolab.pipe3d.plot import write_survey_view

    thetas = list(np.arange(0.0, 360.0, 90.0))
    z_list = [22.0, 26.0]
    estimates = [
        {
            "z": z,
            "theta_deg": thetas,
            "r_pred": [22.0] * len(thetas),
            "R": 22.0,
            "e": 0.0,
            "A": 0.1,
            "theta0_deg": 0.0,
            "c": 0.45,
            "scaleMode": "c_cal",
            "cx": 0.0,
            "cy": 0.0,
        }
        for z in z_list
    ]
    product = assemble_wall(estimates)
    truth = SurveyTruth(
        z=z_list,
        theta_deg=thetas,
        r_true=[[22.0] * len(thetas) for _ in z_list],
        pipe_r=22.0,
        c_prop=0.45,
        dent_on=False,
        dent_theta0_deg=0.0,
        dent_z0=24.0,
        dent_depth=0.0,
        ecc_mag=0.0,
        ecc_phi_deg=90.0,
    )
    display = build_display_wall(
        product, truth, display_step_deg=5.0, n_z_display=8
    )
    paths = write_survey_view(tmp_path, display=display, score={"rAbsRmseOverR": 0.01})
    html = paths["html"].read_text(encoding="utf-8")
    assert "SURVEY3D_DATA" in html or "survey3d_view_data.js" in html
    assert "90k" not in html  # points not inlined as a giant blob of coords
    assert "reconPositions" not in html
    js = paths["view_js"].read_text(encoding="utf-8")
    assert "window.SURVEY3D_DATA" in js
    assert "acquireStepDeg" in js or "displayStepDeg" in js
    data = json.loads(paths["view_data"].read_text(encoding="utf-8"))
    assert data["meta"]["displayNz"] == 8
    assert "recon" in data and "positions" in data["recon"]
