"""Lab A 3D survey acquire: radial PE looks over θ×z → SurveyPack + SurveyTruth."""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np

from sonolab.pipe2d.blind import AcquisitionPack
from sonolab.pipe3d import constants as C
from sonolab.pipe3d.fdtd3d import blind_pick_echo, record_pulse_look
from sonolab.pipe3d.geometry import true_standoff
from sonolab.pipe3d.survey import (
    StationAcquisition,
    SurveyPack,
    SurveyTruth,
)


def _look_worker(job: dict[str, Any]) -> tuple[float, float, float, float]:
    """One Lab A look. Returns (theta_deg, z, t_echo, r_true). Picklable for mp."""
    th = float(job["theta_deg"])
    z_k = float(job["z"])
    series = record_pulse_look(
        theta_deg=th,
        z_tool=z_k,
        dent=bool(job["dent"]),
        ecc_mag=float(job["ecc_mag"]),
        ecc_phi_deg=float(job["ecc_phi_deg"]),
        nxy=int(job["nxy"]),
        nz=int(job["nz"]),
        dent_depth=float(job["dent_depth"]),
        dent_theta0=float(job["dent_theta0"]),
        dent_z0=float(job["dent_z0"]),
    )
    t_echo = float(blind_pick_echo(series))
    r_true = float(
        true_standoff(
            th,
            z_k,
            dent=bool(job["dent"]),
            ecc_mag=float(job["ecc_mag"]),
            ecc_phi_deg=float(job["ecc_phi_deg"]),
            dent_depth=float(job["dent_depth"]),
            dent_theta0=float(job["dent_theta0"]),
            dent_z0=float(job["dent_z0"]),
        )
    )
    return th, z_k, t_echo, r_true


def acquire_pipe_survey_3d(
    *,
    z_list: list[float] | np.ndarray | None = None,
    n_z: int = 6,
    z_span: float | None = None,
    step_deg: float = 15.0,
    dent: bool = True,
    ecc_mag: float = 0.0,
    ecc_phi_deg: float = 90.0,
    dent_theta0: float = C.DENT_THETA0_DEG,
    dent_z0: float | None = None,
    dent_depth: float = C.DENT_DEPTH,
    nxy: int = C.NXY,
    nz: int = C.NZ,
    progress: bool = False,
    workers: int | None = None,
) -> tuple[SurveyPack, SurveyTruth]:
    """True 3D scalar PE survey: fire radial looks at each (θ, z).

    Parallelizes Lab A looks across processes. ``workers=None`` uses CPU
    count; ``workers=1`` stays sequential (handy for smoke / debug).
    """
    dent_z0 = float(C.DENT_Z0 if dent_z0 is None else dent_z0)
    # Keep tool far enough from axial faces that wall echo wins ToF race.
    z_margin = float(C.PIPE_R - C.ARRAY_LOOK_OFFSET + 4.0)
    if z_list is None:
        usable = max(float(nz) - 2.0 * z_margin, 4.0)
        span = float(z_span if z_span is not None else min(18.0, usable))
        span = min(span, usable)
        z0 = float(C.CZ) - 0.5 * span
        z0 = max(z0, z_margin)
        z1 = min(z0 + span, float(nz) - 1.0 - z_margin)
        z_list = [
            z0 + (z1 - z0) * i / max(int(n_z) - 1, 1) for i in range(int(n_z))
        ]
    z_arr = np.asarray(z_list, dtype=np.float64)
    z_arr = np.clip(z_arr, z_margin, float(nz) - 1.0 - z_margin)
    thetas = np.arange(0.0, 360.0, float(step_deg), dtype=np.float64)
    ecc_mag = float(np.clip(ecc_mag, 0.0, C.ECC_MAX))

    base_job = {
        "dent": bool(dent),
        "ecc_mag": float(ecc_mag),
        "ecc_phi_deg": float(ecc_phi_deg),
        "nxy": int(nxy),
        "nz": int(nz),
        "dent_depth": float(dent_depth),
        "dent_theta0": float(dent_theta0),
        "dent_z0": float(dent_z0),
    }
    jobs: list[dict[str, Any]] = []
    for z_k in z_arr:
        for th in thetas:
            jobs.append({**base_job, "theta_deg": float(th), "z": float(z_k)})

    n_workers = int(workers) if workers is not None else int(os.cpu_count() or 1)
    n_workers = max(1, min(n_workers, len(jobs) if jobs else 1))

    t0 = time.perf_counter()
    total = len(jobs)
    # Indexed results: jobs are ordered as (z_i major, theta_j minor).
    t_all = np.zeros(total, dtype=np.float64)
    r_all = np.zeros(total, dtype=np.float64)
    n_th = int(thetas.size)

    if n_workers <= 1 or total <= 1:
        for i, job in enumerate(jobs):
            th, z_k, t_echo, r_true = _look_worker(job)
            t_all[i] = t_echo
            r_all[i] = r_true
            if progress and ((i + 1) % 8 == 0 or i + 1 == total):
                elapsed = time.perf_counter() - t0
                print(
                    f"  pipe3d {i + 1}/{total}  z={z_k:.1f} θ={th:.0f}°  "
                    f"t={t_echo:.1f}  ({elapsed:.1f}s)",
                    flush=True,
                )
    else:
        if progress:
            print(
                f"  pipe3d parallel acquire: {total} looks, workers={n_workers}",
                flush=True,
            )
        done = 0
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = {
                pool.submit(_look_worker, job): idx for idx, job in enumerate(jobs)
            }
            for fut in as_completed(futs):
                idx = futs[fut]
                th, z_k, t_echo, r_true = fut.result()
                t_all[idx] = t_echo
                r_all[idx] = r_true
                done += 1
                if progress and (done % max(8, n_workers) == 0 or done == total):
                    elapsed = time.perf_counter() - t0
                    print(
                        f"  pipe3d {done}/{total}  z={z_k:.1f} θ={th:.0f}°  "
                        f"t={t_echo:.1f}  ({elapsed:.1f}s)",
                        flush=True,
                    )

    stations: list[StationAcquisition] = []
    r_true_rows: list[list[float]] = []
    for iz, z_k in enumerate(z_arr):
        zf = float(z_k)
        sl = slice(iz * n_th, (iz + 1) * n_th)
        t_echo = t_all[sl]
        r_row = r_all[sl]

        pack = AcquisitionPack(
            theta_deg=[float(v) for v in thetas],
            t_echo=[float(v) for v in t_echo],
            array_look_offset=float(C.ARRAY_LOOK_OFFSET),
            pulse_width=float(C.PULSE_WIDTH),
            n_elem=1,
            step_deg=float(step_deg),
            meta={
                "z": zf,
                "ascanLen": int(C.ASCAN_LEN),
                "pickMode": "blind first-break 3d",
                "noPipeRPrior": True,
                "labaBackend": "pipe3d_python",
                "physics": C.PHYSICS_LABEL,
                "nxy": int(nxy),
                "nzGrid": int(nz),
            },
        )
        stations.append(StationAcquisition(z=zf, pack=pack))
        r_true_rows.append([float(v) for v in r_row])

    survey = SurveyPack(
        stations=stations,
        meta={
            "physics": C.PHYSICS_LABEL,
            "nStations": len(stations),
            "stepDeg": float(step_deg),
            "nxy": int(nxy),
            "nzGrid": int(nz),
            "workers": int(n_workers),
            "nLooks": int(total),
            "elapsedSec": round(time.perf_counter() - t0, 2),
        },
    )
    truth = SurveyTruth(
        z=[float(v) for v in z_arr],
        theta_deg=[float(v) for v in thetas],
        r_true=r_true_rows,
        pipe_r=float(C.PIPE_R),
        c_prop=float(C.C_PROP),
        dent_on=bool(dent),
        dent_theta0_deg=float(dent_theta0),
        dent_z0=float(dent_z0),
        dent_depth=float(dent_depth),
        ecc_mag=float(ecc_mag),
        ecc_phi_deg=float(ecc_phi_deg),
        meta={"physics": C.PHYSICS_LABEL},
    )
    return survey, truth
