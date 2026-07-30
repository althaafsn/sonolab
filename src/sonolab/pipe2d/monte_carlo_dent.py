#!/usr/bin/env python3
"""Monte Carlo for joint parametric pipe/dent estimator (no R_nominal cheat).

Protocol
--------
- Randomize field unknowns: eccentricity, wrong c_init scale, dent angle/depth.
- Lab A: acquire_sector_scan → AcquisitionPack + TruthPack.
- Lab B: estimate_from_pack(AcquisitionPack only); never reads truth.
- Score scale-invariant metrics (θ0, A/R, e/R, shape); pre-registered A/R τ.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from sonolab.pipe2d import acquire as acq_mod
from sonolab.pipe2d import phased_array as pa
from sonolab.pipe2d.blind import score_estimate
from sonolab.pipe2d.joint_geometry import estimate_from_pack

STEP_DEG = 1.0
N_POSES = 16
ECC_MAX_STUDY = 8.0  # dense 1° study; high-ecc (>10) still needs better Lab B
C_INIT_SCALE_LO = 0.90
C_INIT_SCALE_HI = 1.20
DENT_DEPTH_SCALE_LO = 0.85
DENT_DEPTH_SCALE_HI = 1.15
LOC_TOL_DEG = 10.0
# Pre-registered relative dent depth threshold.
A_OVER_R_TAU = 0.048
SEED = 20260728
# Loc median target when detected (dense 1° protocol).
LOC_MED_TARGET_DEG = 5.0


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def run_pose(
    *,
    rng: np.random.Generator,
    pose_id: int,
    step_deg: float,
    progress: bool,
) -> dict[str, Any]:
    ecc_mag = float(rng.uniform(0.0, ECC_MAX_STUDY))
    ecc_phi = float(rng.uniform(0.0, 360.0))
    c_scale = float(rng.uniform(C_INIT_SCALE_LO, C_INIT_SCALE_HI))
    c_init = float(pa.C_PROP * c_scale)
    dent_theta0 = float(rng.uniform(0.0, 360.0))
    dent_depth = float(
        pa.DENT_DEPTH * rng.uniform(DENT_DEPTH_SCALE_LO, DENT_DEPTH_SCALE_HI)
    )

    old_theta = pa.DENT_THETA0_DEG
    old_depth = pa.DENT_DEPTH
    pa.DENT_THETA0_DEG = dent_theta0
    pa.DENT_DEPTH = dent_depth
    try:
        t_pose0 = time.perf_counter()
        if progress:
            print(
                f"\n=== pose {pose_id}  ecc={ecc_mag:.2f}@{ecc_phi:.1f}°  "
                f"c_init={c_init:.4f} (×{c_scale:.3f})  "
                f"dentθ={dent_theta0:.1f}° depth={dent_depth:.2f} ===",
                flush=True,
            )

        pack_off, truth_off = acq_mod.acquire_sector_scan(
            dent=False,
            n_elem=pa.N_ELEM_DEFAULT,
            step_deg=step_deg,
            ecc_mag=ecc_mag,
            ecc_phi_deg=ecc_phi,
            progress=progress,
        )
        est_off = estimate_from_pack(pack_off, c_init=c_init)
        score_off = score_estimate(
            est_off,
            truth_off,
            a_over_r_tau=A_OVER_R_TAU,
            loc_tol_deg=LOC_TOL_DEG,
        )

        pack_on, truth_on = acq_mod.acquire_sector_scan(
            dent=True,
            n_elem=pa.N_ELEM_DEFAULT,
            step_deg=step_deg,
            ecc_mag=ecc_mag,
            ecc_phi_deg=ecc_phi,
            progress=progress,
        )
        # Null-reference circle from dent-OFF (instrument data, not pipe prior).
        circle_ref = {
            "R": float(est_off["R"]),
            "cx": float(est_off["cx"]),
            "cy": float(est_off["cy"]),
            "c": float(est_off["c"]),
            "t_echo": list(pack_off.t_echo),
        }
        est_on = estimate_from_pack(pack_on, c_init=c_init, circle_ref=circle_ref)
        score_on = score_estimate(
            est_on,
            truth_on,
            a_over_r_tau=A_OVER_R_TAU,
            loc_tol_deg=LOC_TOL_DEG,
        )

        elapsed = time.perf_counter() - t_pose0
        return {
            "poseId": pose_id,
            "elapsedSec": round(elapsed, 2),
            "truth": {
                "eccMag": round(ecc_mag, 6),
                "eccPhiDeg": round(ecc_phi, 6),
                "cProp": float(pa.C_PROP),
                "cInit": round(c_init, 8),
                "cInitScale": round(c_scale, 6),
                "dentTheta0Deg": round(dent_theta0, 6),
                "dentDepth": round(dent_depth, 6),
                "pipeR": float(pa.PIPE_R),
                "A_over_R_true": round(dent_depth / pa.PIPE_R, 6),
                "e_over_R_true": round(ecc_mag / pa.PIPE_R, 6),
            },
            "estimateDentOff": {
                "A_over_R": est_off["A_over_R"],
                "e_over_R": est_off["e_over_R"],
                "theta0_deg": est_off["theta0_deg"],
                "tau_R_over_c": est_off["tau_R_over_c"],
                "R": est_off["R"],
                "c": est_off["c"],
                "t_rmse": est_off["t_rmse"],
                "success": est_off["success"],
                "detection": score_off,
            },
            "estimateDentOn": {
                "A_over_R": est_on["A_over_R"],
                "e_over_R": est_on["e_over_R"],
                "theta0_deg": est_on["theta0_deg"],
                "tau_R_over_c": est_on["tau_R_over_c"],
                "R": est_on["R"],
                "c": est_on["c"],
                "t_rmse": est_on["t_rmse"],
                "success": est_on["success"],
                "shapeRmse": score_on["shapeRmse"],
                "detection": score_on,
            },
        }
    finally:
        pa.DENT_THETA0_DEG = old_theta
        pa.DENT_DEPTH = old_depth


def compute_metrics(poses: list[dict[str, Any]]) -> dict[str, Any]:
    null_a = [float(p["estimateDentOff"]["A_over_R"]) for p in poses]
    on_a = [float(p["estimateDentOn"]["A_over_R"]) for p in poses]
    null_det = [
        bool(p["estimateDentOff"]["detection"]["detected"]) for p in poses
    ]
    on_det = [bool(p["estimateDentOn"]["detection"]["detected"]) for p in poses]
    loc_errs = [
        abs(float(p["estimateDentOn"]["detection"]["locErrorDeg"]))
        for p in poses
        if p["estimateDentOn"]["detection"]["detected"]
    ]
    e_err = np.array(
        [float(p["estimateDentOn"]["detection"]["e_over_R_err"]) for p in poses]
    )
    a_err = np.array(
        [
            float(p["estimateDentOn"]["detection"]["A_over_R_err"])
            for p in poses
        ]
    )
    shape = np.array(
        [
            float(p["estimateDentOn"]["shapeRmse"] or np.nan)
            for p in poses
        ]
    )

    return {
        "nPoses": len(poses),
        "a_over_r_tau": A_OVER_R_TAU,
        "locTolDeg": LOC_TOL_DEG,
        "falseAlarmRate": round(sum(null_det) / max(1, len(null_det)), 4),
        "recall": round(sum(on_det) / max(1, len(on_det)), 4),
        "nDetected": int(sum(on_det)),
        "nFalseAlarms": int(sum(null_det)),
        "locErrorDegMedian": (
            round(float(np.median(loc_errs)), 3) if loc_errs else None
        ),
        "locErrorDegMean": (
            round(float(np.mean(loc_errs)), 3) if loc_errs else None
        ),
        "e_over_R_rmse": round(float(np.sqrt(np.mean(e_err**2))), 6),
        "A_over_R_rmse": round(float(np.sqrt(np.mean(a_err**2))), 6),
        "shapeRmseMedian": round(float(np.nanmedian(shape)), 6),
        "nullA_over_R_median": round(float(np.median(null_a)), 6),
        "onA_over_R_median": round(float(np.median(on_a)), 6),
        "targets": {
            "recall_ge": 0.70,
            "far_le": 0.20,
            "loc_med_le_deg": LOC_MED_TARGET_DEG,
            "e_over_R_rmse_le": 0.05,
            "A_over_R_rmse_le": 0.05,
        },
        "pass": {
            "recall": (sum(on_det) / max(1, len(on_det))) >= 0.70,
            "far": (sum(null_det) / max(1, len(null_det))) <= 0.20,
            "loc_med": (
                (float(np.median(loc_errs)) <= LOC_MED_TARGET_DEG)
                if loc_errs
                else False
            ),
            "e_over_R_rmse": float(np.sqrt(np.mean(e_err**2))) <= 0.05,
            "A_over_R_rmse": float(np.sqrt(np.mean(a_err**2))) <= 0.05,
        },
        "detectionByPose": [
            {
                "poseId": p["poseId"],
                "eccMag": p["truth"]["eccMag"],
                "detected": p["estimateDentOn"]["detection"]["detected"],
                "far": p["estimateDentOff"]["detection"]["detected"],
                "locErr": p["estimateDentOn"]["detection"]["locErrorDeg"],
                "A_over_R_on": p["estimateDentOn"]["A_over_R"],
                "A_over_R_off": p["estimateDentOff"]["A_over_R"],
                "e_over_R": p["estimateDentOn"]["e_over_R"],
            }
            for p in poses
        ],
    }


def run_sanity(*, step_deg: float = STEP_DEG, progress: bool = True) -> dict[str, Any]:
    """Centered dent-ON sanity (ecc=0); θ0≈0, A/R well above null."""
    old_theta = pa.DENT_THETA0_DEG
    old_depth = pa.DENT_DEPTH
    pa.DENT_THETA0_DEG = 0.0
    pa.DENT_DEPTH = float(pa.DENT_DEPTH)
    try:
        pack_off, truth_off = acq_mod.acquire_sector_scan(
            dent=False, step_deg=step_deg, ecc_mag=0.0, progress=progress
        )
        est_off = estimate_from_pack(pack_off, c_init=float(pa.C_PROP))
        score_off = score_estimate(
            est_off, truth_off, a_over_r_tau=A_OVER_R_TAU, loc_tol_deg=LOC_TOL_DEG
        )
        pack_on, truth_on = acq_mod.acquire_sector_scan(
            dent=True, step_deg=step_deg, ecc_mag=0.0, progress=progress
        )
        circle_ref = {
            "R": float(est_off["R"]),
            "cx": float(est_off["cx"]),
            "cy": float(est_off["cy"]),
            "c": float(est_off["c"]),
            "t_echo": list(pack_off.t_echo),
        }
        est_on = estimate_from_pack(
            pack_on, c_init=float(pa.C_PROP), circle_ref=circle_ref
        )
        score_on = score_estimate(
            est_on, truth_on, a_over_r_tau=A_OVER_R_TAU, loc_tol_deg=LOC_TOL_DEG
        )
        return {
            "estimateDentOff": {**{k: est_off[k] for k in (
                "A_over_R", "e_over_R", "theta0_deg", "tau_R_over_c", "R", "c", "t_rmse"
            )}, "detection": score_off},
            "estimateDentOn": {**{k: est_on[k] for k in (
                "A_over_R", "e_over_R", "theta0_deg", "tau_R_over_c", "R", "c", "t_rmse"
            )}, "detection": score_on},
            "pass": {
                "theta0_near_0": abs(float(est_on["theta0_deg"])) <= 20.0
                or abs(float(est_on["theta0_deg"]) - 360.0) <= 20.0,
                "A_over_R_above_null": float(est_on["A_over_R"])
                > float(est_off["A_over_R"]) + 0.03,
                "detected": bool(score_on["detected"]),
            },
        }
    finally:
        pa.DENT_THETA0_DEG = old_theta
        pa.DENT_DEPTH = old_depth


def run_pose_abs(
    *,
    rng: np.random.Generator,
    pose_id: int,
    step_deg: float,
    progress: bool,
    c_cal: float,
    c_cal_label: str = "truth",
) -> dict[str, Any]:
    """One pose with labeled fluid c_cal (absolute geometry)."""
    ecc_mag = float(rng.uniform(0.0, ECC_MAX_STUDY))
    ecc_phi = float(rng.uniform(0.0, 360.0))
    dent_theta0 = float(rng.uniform(0.0, 360.0))
    dent_depth = float(
        pa.DENT_DEPTH * rng.uniform(DENT_DEPTH_SCALE_LO, DENT_DEPTH_SCALE_HI)
    )

    old_theta = pa.DENT_THETA0_DEG
    old_depth = pa.DENT_DEPTH
    pa.DENT_THETA0_DEG = dent_theta0
    pa.DENT_DEPTH = dent_depth
    try:
        t_pose0 = time.perf_counter()
        if progress:
            print(
                f"\n=== abs pose {pose_id}  ecc={ecc_mag:.2f}@{ecc_phi:.1f}°  "
                f"c_cal={c_cal:.4f} ({c_cal_label})  "
                f"dentθ={dent_theta0:.1f}° ===",
                flush=True,
            )

        pack_off, truth_off = acq_mod.acquire_sector_scan(
            dent=False,
            n_elem=pa.N_ELEM_DEFAULT,
            step_deg=step_deg,
            ecc_mag=ecc_mag,
            ecc_phi_deg=ecc_phi,
            progress=progress,
        )
        est_off = estimate_from_pack(pack_off, c_cal=c_cal)
        score_off = score_estimate(
            est_off,
            truth_off,
            a_over_r_tau=A_OVER_R_TAU,
            loc_tol_deg=LOC_TOL_DEG,
        )

        pack_on, truth_on = acq_mod.acquire_sector_scan(
            dent=True,
            n_elem=pa.N_ELEM_DEFAULT,
            step_deg=step_deg,
            ecc_mag=ecc_mag,
            ecc_phi_deg=ecc_phi,
            progress=progress,
        )
        circle_ref = {
            "R": float(est_off["R"]),
            "cx": float(est_off["cx"]),
            "cy": float(est_off["cy"]),
            "c": float(est_off["c"]),
            "t_echo": list(pack_off.t_echo),
        }
        est_on = estimate_from_pack(
            pack_on, c_cal=c_cal, circle_ref=circle_ref
        )
        score_on = score_estimate(
            est_on,
            truth_on,
            a_over_r_tau=A_OVER_R_TAU,
            loc_tol_deg=LOC_TOL_DEG,
        )

        elapsed = time.perf_counter() - t_pose0
        return {
            "poseId": pose_id,
            "elapsedSec": round(elapsed, 2),
            "cCal": float(c_cal),
            "cCalLabel": c_cal_label,
            "truth": {
                "eccMag": round(ecc_mag, 6),
                "eccPhiDeg": round(ecc_phi, 6),
                "cProp": float(pa.C_PROP),
                "dentTheta0Deg": round(dent_theta0, 6),
                "dentDepth": round(dent_depth, 6),
                "pipeR": float(pa.PIPE_R),
            },
            "estimateDentOff": {
                "scaleMode": est_off.get("scaleMode"),
                "R": est_off["R"],
                "e": est_off["e"],
                "A": est_off["A"],
                "A_over_R": est_off["A_over_R"],
                "e_over_R": est_off["e_over_R"],
                "theta0_deg": est_off["theta0_deg"],
                "c": est_off["c"],
                "detection": score_off,
            },
            "estimateDentOn": {
                "scaleMode": est_on.get("scaleMode"),
                "R": est_on["R"],
                "e": est_on["e"],
                "A": est_on["A"],
                "A_over_R": est_on["A_over_R"],
                "e_over_R": est_on["e_over_R"],
                "theta0_deg": est_on["theta0_deg"],
                "c": est_on["c"],
                "shapeRmse": score_on.get("shapeRmse"),
                "detection": score_on,
            },
        }
    finally:
        pa.DENT_THETA0_DEG = old_theta
        pa.DENT_DEPTH = old_depth


def compute_abs_metrics(poses: list[dict[str, Any]]) -> dict[str, Any]:
    """Absolute + ratio metrics for c_cal poses."""
    base = compute_metrics(poses)
    r_rel = []
    e_err = []
    a_err = []
    r_abs = []
    for p in poses:
        d = p["estimateDentOn"]["detection"]
        if "R_rel_err" in d:
            r_rel.append(abs(float(d["R_rel_err"])))
        if "e_err" in d:
            e_err.append(float(d["e_err"]))
        if "A_err" in d:
            a_err.append(float(d["A_err"]))
        if d.get("rAbsRmse") is not None:
            r_abs.append(float(d["rAbsRmse"]))
    med_r_rel = float(np.median(r_rel)) if r_rel else float("nan")
    e_rmse = float(np.sqrt(np.mean(np.square(e_err)))) if e_err else float("nan")
    a_rmse = float(np.sqrt(np.mean(np.square(a_err)))) if a_err else float("nan")
    base["abs"] = {
        "medianAbsRRelErr": (
            None if np.isnan(med_r_rel) else round(med_r_rel, 6)
        ),
        "e_rmse": None if np.isnan(e_rmse) else round(e_rmse, 6),
        "A_rmse": None if np.isnan(a_rmse) else round(a_rmse, 6),
        "rAbsRmseMedian": (
            None
            if not r_abs
            else round(float(np.median(r_abs)), 6)
        ),
        "targets": {
            "medianAbsRRelErr_le": 0.03,
            "e_rmse_le": 2.5,
            "A_rmse_le": 4.0,
        },
        "pass": {
            "medianAbsRRelErr": (
                (not np.isnan(med_r_rel)) and med_r_rel <= 0.03
            ),
            "e_rmse": (not np.isnan(e_rmse)) and e_rmse <= 2.5,
            "A_rmse": (not np.isnan(a_rmse)) and a_rmse <= 4.0,
        },
    }
    return base


def run_absolute_cal_smoke(
    *,
    n_poses: int = 4,
    step_deg: float = 15.0,
    seed: int = SEED,
    progress: bool = True,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Labeled c_cal=C_PROP absolute recon smoke + wrong-cal scale check."""
    rng = np.random.default_rng(int(seed))
    c_true = float(pa.C_PROP)
    poses: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for i in range(int(n_poses)):
        poses.append(
            run_pose_abs(
                rng=rng,
                pose_id=i,
                step_deg=float(step_deg),
                progress=progress,
                c_cal=c_true,
                c_cal_label="truth",
            )
        )
    metrics = compute_abs_metrics(poses)

    # Wrong-cal documents scale: R should track ~1.2× when c_cal = 1.2·C_PROP.
    wrong = run_pose_abs(
        rng=rng,
        pose_id=900,
        step_deg=float(step_deg),
        progress=progress,
        c_cal=1.2 * c_true,
        c_cal_label="1.2x_wrong",
    )
    r_wrong = float(wrong["estimateDentOn"]["R"])
    r_true = float(wrong["truth"]["pipeR"])
    scale_ratio = r_wrong / max(r_true, 1e-12)
    wrong_check = {
        "cCalScale": 1.2,
        "R_est": r_wrong,
        "R_true": r_true,
        "R_over_Rtrue": round(scale_ratio, 4),
        "A_over_R": wrong["estimateDentOn"]["A_over_R"],
        "e_over_R": wrong["estimateDentOn"]["e_over_R"],
        "pass_scale_tracks_cal": 1.05 <= scale_ratio <= 1.35,
        "pass_ratios_ok": (
            abs(
                float(wrong["estimateDentOn"]["A_over_R"])
                - float(wrong["truth"]["dentDepth"]) / r_true
            )
            < 0.06
        ),
    }

    out = {
        "meta": {
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seed": int(seed),
            "nPoses": int(n_poses),
            "stepDeg": float(step_deg),
            "mode": "absolute_c_cal",
            "cCal": c_true,
            "note": (
                "Lab B c_cal is labeled operator input (=C_PROP here); "
                "estimator never reads TruthPack"
            ),
            "wallSec": round(time.perf_counter() - t0, 2),
        },
        "metrics": metrics,
        "wrongCal": wrong_check,
        "poses": poses,
        "passAll": bool(
            metrics["pass"]["recall"]
            and metrics["pass"]["far"]
            and metrics["abs"]["pass"]["medianAbsRRelErr"]
            and wrong_check["pass_scale_tracks_cal"]
        ),
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        if progress:
            print(f"wrote {out_path}", flush=True)
            print(
                f"abs passAll={out['passAll']}  "
                f"median|R_rel|={metrics['abs']['medianAbsRRelErr']}  "
                f"wrongCal R/Rtrue={wrong_check['R_over_Rtrue']}",
                flush=True,
            )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-poses", type=int, default=N_POSES)
    parser.add_argument("--step-deg", type=float, default=STEP_DEG)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--out",
        type=str,
        default=str(
            Path(__file__).resolve().parents[3]
            / "datasets"
            / "runs"
            / "_pipe2d_joint"
            / "monte_carlo_joint.json"
        ),
    )
    parser.add_argument(
        "--sanity-out",
        type=str,
        default=str(
            Path(__file__).resolve().parents[3]
            / "datasets"
            / "runs"
            / "_pipe2d_joint"
            / "sanity_joint.json"
        ),
    )
    parser.add_argument("--skip-sanity", action="store_true")
    parser.add_argument("--skip-mc", action="store_true")
    parser.add_argument(
        "--absolute-cal",
        action="store_true",
        help="run absolute c_cal smoke instead of free-c MC",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    progress = not bool(args.quiet)
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.absolute_cal:
        abs_out = out_dir / "monte_carlo_abs_c_cal.json"
        # Fast smoke default: 15° when caller left module STEP_DEG (1°).
        step = float(args.step_deg)
        if abs(step - STEP_DEG) < 1e-12:
            step = 15.0
        run_absolute_cal_smoke(
            n_poses=min(int(args.n_poses), 8),
            step_deg=step,
            seed=int(args.seed),
            progress=progress,
            out_path=abs_out,
        )
        return 0

    sanity = None
    if not args.skip_sanity:
        if progress:
            print("=== sanity (ecc=0) ===", flush=True)
        sanity = run_sanity(step_deg=float(args.step_deg), progress=progress)
        Path(args.sanity_out).write_text(
            json.dumps(sanity, indent=2), encoding="utf-8"
        )
        if progress:
            print(f"wrote {args.sanity_out}", flush=True)
            print(f"sanity pass={sanity['pass']}", flush=True)

    if args.skip_mc:
        return 0

    rng = np.random.default_rng(int(args.seed))
    n_poses = int(args.n_poses)
    step_deg = float(args.step_deg)
    t0 = time.perf_counter()
    poses: list[dict[str, Any]] = []
    for i in range(n_poses):
        poses.append(
            run_pose(
                rng=rng,
                pose_id=i,
                step_deg=step_deg,
                progress=progress,
            )
        )
    metrics = compute_metrics(poses)
    wall = time.perf_counter() - t0
    out = {
        "meta": {
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seed": int(args.seed),
            "nPoses": n_poses,
            "stepDeg": step_deg,
            "nLooksPerSweep": int(round(360.0 / step_deg)),
            "a_over_r_tau": A_OVER_R_TAU,
            "locTolDeg": LOC_TOL_DEG,
            "locMedTargetDeg": LOC_MED_TARGET_DEG,
            "eccMaxStudy": ECC_MAX_STUDY,
            "cProp": float(pa.C_PROP),
            "pipeR": float(pa.PIPE_R),
            "labaBackend": acq_mod.laba_backend(),
            "protocol": (
                "Per pose: randomize ecc/c_init/dentθ/depth → acquire dent-OFF "
                "→ joint fit → acquire dent-ON with null-ref circle freeze → joint fit. "
                f"Detection: A/R > {A_OVER_R_TAU} AND |Δθ|≤{LOC_TOL_DEG}° "
                f"(loc median target ≤{LOC_MED_TARGET_DEG}°). "
                f"Primary step 1°; ecc∈[0,{ECC_MAX_STUDY}]; "
                "No R_nominal prior; instrument array offset only."
            ),
            "noCheat": (
                "Lab B sees AcquisitionPack only; TruthPack used in scorer. "
                "No pipe R / C_PROP / pose / dent at inference."
            ),
            "wallTimeSec": round(wall, 1),
        },
        "metrics": metrics,
        "sanity": sanity,
        "poses": poses,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}", flush=True)
    print(
        f"wall={wall:.1f}s  recall={metrics['recall']}  "
        f"FAR={metrics['falseAlarmRate']}  "
        f"loc_med={metrics['locErrorDegMedian']}  "
        f"e/R_rmse={metrics['e_over_R_rmse']}  "
        f"A/R_rmse={metrics['A_over_R_rmse']}  "
        f"pass={metrics['pass']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
