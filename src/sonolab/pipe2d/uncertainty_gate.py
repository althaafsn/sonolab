"""Block 7a uncertainty gate: CI coverage + refuse behavior under MC poses."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from sonolab.pipe2d import acquire as acq_mod
from sonolab.pipe2d import monte_carlo_dent as mc
from sonolab.pipe2d import phased_array as pa
from sonolab.pipe2d.blind import score_estimate
from sonolab.pipe2d.joint_geometry import estimate_from_pack
from sonolab.pipe2d.uncertainty import A_OVER_R_TAU, ci_covers

DEFAULT_OUT = Path("datasets/runs/_pipe2d_joint/uncertainty_gate.json")
COVERAGE_TARGET = 0.70
COVERAGE_TARGET_THETA = 0.50  # θ0 harder; soft floor, still reported


def _run_pose(
    *,
    rng: np.random.Generator,
    pose_id: int,
    step_deg: float,
    n_elem: int,
    progress: bool,
) -> dict[str, Any]:
    ecc_mag = float(rng.uniform(0.0, mc.ECC_MAX_STUDY))
    ecc_phi = float(rng.uniform(0.0, 360.0))
    c_scale = float(rng.uniform(mc.C_INIT_SCALE_LO, mc.C_INIT_SCALE_HI))
    c_init = float(pa.C_PROP * c_scale)
    dent_theta0 = float(rng.uniform(0.0, 360.0))
    dent_depth = float(
        pa.DENT_DEPTH * rng.uniform(mc.DENT_DEPTH_SCALE_LO, mc.DENT_DEPTH_SCALE_HI)
    )

    old_theta = pa.DENT_THETA0_DEG
    old_depth = pa.DENT_DEPTH
    pa.DENT_THETA0_DEG = dent_theta0
    pa.DENT_DEPTH = dent_depth
    try:
        t0 = time.perf_counter()
        if progress:
            print(
                f"\n=== pose {pose_id} ecc={ecc_mag:.2f} N={n_elem} "
                f"dentθ={dent_theta0:.1f}° ===",
                flush=True,
            )
        pack_off, truth_off = acq_mod.acquire_sector_scan(
            dent=False,
            n_elem=n_elem,
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
            loc_tol_deg=mc.LOC_TOL_DEG,
        )

        pack_on, truth_on = acq_mod.acquire_sector_scan(
            dent=True,
            n_elem=n_elem,
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
        est_on = estimate_from_pack(pack_on, c_init=c_init, circle_ref=circle_ref)
        score_on = score_estimate(
            est_on,
            truth_on,
            a_over_r_tau=A_OVER_R_TAU,
            loc_tol_deg=mc.LOC_TOL_DEG,
        )

        unc = est_on.get("uncertainty") or {}
        ci = unc.get("ci95") or {}
        a_true = float(dent_depth / pa.PIPE_R)
        e_true = float(ecc_mag / pa.PIPE_R)
        cover = {
            "theta0": (
                ci_covers(ci["theta0_deg"], dent_theta0, circular_deg=True)
                if "theta0_deg" in ci and not est_on.get("refuse")
                else None
            ),
            "A_over_R": (
                ci_covers(ci["A_over_R"], a_true)
                if "A_over_R" in ci and not est_on.get("refuse")
                else None
            ),
            "e_over_R": (
                ci_covers(ci["e_over_R"], e_true)
                if "e_over_R" in ci and not est_on.get("refuse")
                else None
            ),
        }
        return {
            "poseId": pose_id,
            "nElem": int(n_elem),
            "elapsedSec": round(time.perf_counter() - t0, 2),
            "truth": {
                "eccMag": round(ecc_mag, 6),
                "dentTheta0Deg": round(dent_theta0, 6),
                "dentDepth": round(dent_depth, 6),
                "A_over_R_true": round(a_true, 6),
                "e_over_R_true": round(e_true, 6),
            },
            "estimateDentOff": {
                "refuse": bool(est_off.get("refuse")),
                "refuseReasons": list(est_off.get("refuseReasons") or []),
                "dentCall": bool(est_off.get("dentCall")),
                "A_over_R": est_off.get("A_over_R"),
                "detection": score_off,
            },
            "estimateDentOn": {
                "refuse": bool(est_on.get("refuse")),
                "refuseReasons": list(est_on.get("refuseReasons") or []),
                "dentCall": bool(est_on.get("dentCall")),
                "confidence": est_on.get("confidence"),
                "A_over_R": est_on.get("A_over_R"),
                "e_over_R": est_on.get("e_over_R"),
                "theta0_deg": est_on.get("theta0_deg"),
                "t_rmse": est_on.get("t_rmse"),
                "uncertainty": {
                    "sigma_t": unc.get("sigma_t"),
                    "condJTJ": unc.get("condJTJ"),
                    "ci95": ci,
                },
                "coverage": cover,
                "detection": score_on,
            },
        }
    finally:
        pa.DENT_THETA0_DEG = old_theta
        pa.DENT_DEPTH = old_depth


def _coverage_rate(poses: list[dict[str, Any]], key: str) -> float | None:
    vals = [
        p["estimateDentOn"]["coverage"].get(key)
        for p in poses
        if p["estimateDentOn"]["coverage"].get(key) is not None
    ]
    if not vals:
        return None
    return float(sum(1 for v in vals if v) / len(vals))


def summarize_band(poses: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(1, len(poses))
    refuse_on = sum(1 for p in poses if p["estimateDentOn"]["refuse"])
    refuse_off = sum(1 for p in poses if p["estimateDentOff"]["refuse"])
    far = sum(
        1 for p in poses if p["estimateDentOff"]["detection"]["detected"]
    ) / n
    recall = sum(
        1 for p in poses if p["estimateDentOn"]["detection"]["detected"]
    ) / n
    cov_th = _coverage_rate(poses, "theta0")
    cov_a = _coverage_rate(poses, "A_over_R")
    cov_e = _coverage_rate(poses, "e_over_R")
    n_cov = sum(
        1
        for p in poses
        if p["estimateDentOn"]["coverage"].get("theta0") is not None
    )
    return {
        "nPoses": len(poses),
        "refuseRateDentOn": round(refuse_on / n, 4),
        "refuseRateDentOff": round(refuse_off / n, 4),
        "falseAlarmRate": round(far, 4),
        "recall": round(recall, 4),
        "nNonRefuseForCoverage": n_cov,
        "coverage": {
            "theta0": None if cov_th is None else round(cov_th, 4),
            "A_over_R": None if cov_a is None else round(cov_a, 4),
            "e_over_R": None if cov_e is None else round(cov_e, 4),
        },
    }


def evaluate_gate(
    primary: dict[str, Any],
    stress: dict[str, Any],
) -> dict[str, Any]:
    """A/R + e/R coverage ≥ 0.7; θ0 ≥ 0.5; stress refuse total ≥ primary; FAR ok."""
    cov = primary["coverage"]
    a_ok = cov.get("A_over_R") is not None and float(cov["A_over_R"]) >= COVERAGE_TARGET
    e_ok = cov.get("e_over_R") is None or float(cov["e_over_R"]) >= COVERAGE_TARGET
    th_ok = cov.get("theta0") is None or float(cov["theta0"]) >= COVERAGE_TARGET_THETA
    cov_ok = bool(a_ok and e_ok and th_ok)
    refuse_stress = float(stress["refuseRateDentOn"])
    refuse_primary = float(primary["refuseRateDentOn"])
    refuse_stress_off = float(stress.get("refuseRateDentOff", 0.0))
    refuse_primary_off = float(primary.get("refuseRateDentOff", 0.0))
    refuse_helps = (
        refuse_stress + refuse_stress_off
        >= refuse_primary + refuse_primary_off - 1e-12
    )
    far_ok = float(primary["falseAlarmRate"]) <= 0.25
    overall = bool(cov_ok and refuse_helps and far_ok)
    return {
        "coverageTarget": COVERAGE_TARGET,
        "coverageTargetTheta": COVERAGE_TARGET_THETA,
        "coverageOk": cov_ok,
        "checks": {"A_over_R": a_ok, "e_over_R": e_ok, "theta0": th_ok},
        "stressRefuseNotLower": refuse_helps,
        "farOk": far_ok,
        "pass": overall,
        "primaryRefuseRate": refuse_primary,
        "stressRefuseRate": refuse_stress,
        "stressRefuseRateOff": refuse_stress_off,
    }


def run_uncertainty_gate(
    *,
    n_poses: int = mc.N_POSES,
    step_deg: float = 15.0,
    seed: int = mc.SEED,
    progress: bool = True,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(int(seed))
    primary_poses = [
        _run_pose(
            rng=rng,
            pose_id=i,
            step_deg=float(step_deg),
            n_elem=pa.N_ELEM_DEFAULT,
            progress=progress,
        )
        for i in range(int(n_poses))
    ]
    # Matched stress band: sparse aperture (higher refuse expected).
    rng_s = np.random.default_rng(int(seed))
    stress_poses = [
        _run_pose(
            rng=rng_s,
            pose_id=i,
            step_deg=float(step_deg),
            n_elem=2,
            progress=progress,
        )
        for i in range(int(n_poses))
    ]
    primary = summarize_band(primary_poses)
    stress = summarize_band(stress_poses)
    gate = evaluate_gate(primary, stress)
    return {
        "schema_version": "0.1",
        "block": 7,
        "case": "uncertainty_gate_nlls_linearization",
        "seed": int(seed),
        "stepDeg": float(step_deg),
        "nPoses": int(n_poses),
        "labaBackend": acq_mod.laba_backend(),
        "elapsedSec": round(time.perf_counter() - t0, 2),
        "primaryN20": primary,
        "stressN2": stress,
        "gate": gate,
        "pass": bool(gate["pass"]),
        "note": (
            "CIs are nlls_linearization_approx; coverage target is honest ~70% "
            "band, not claimed exact frequentist 95%. Dual-obs refuse = Block 7b."
        ),
        "posesPrimary": primary_poses,
        "posesStress": stress_poses,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Block 7a uncertainty / refuse gate")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=mc.SEED)
    p.add_argument("--n-poses", type=int, default=mc.N_POSES)
    p.add_argument("--step-deg", type=float, default=15.0)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    result = run_uncertainty_gate(
        n_poses=int(args.n_poses),
        step_deg=float(args.step_deg),
        seed=int(args.seed),
        progress=not args.quiet,
    )
    # Compact write: drop full pose CI blobs if huge — keep summaries + thin poses.
    slim = dict(result)
    for key in ("posesPrimary", "posesStress"):
        slim[key] = [
            {
                "poseId": p["poseId"],
                "nElem": p["nElem"],
                "truth": p["truth"],
                "refuseOn": p["estimateDentOn"]["refuse"],
                "refuseReasons": p["estimateDentOn"]["refuseReasons"],
                "detected": p["estimateDentOn"]["detection"]["detected"],
                "far": p["estimateDentOff"]["detection"]["detected"],
                "coverage": p["estimateDentOn"]["coverage"],
                "ciHalfA": (
                    (p["estimateDentOn"]["uncertainty"]["ci95"] or {})
                    .get("A_over_R", {})
                    .get("halfWidth")
                ),
            }
            for p in result[key]
        ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(json.dumps({"pass": slim["pass"], "gate": slim["gate"], "primaryN20": slim["primaryN20"], "stressN2": slim["stressN2"]}, indent=2))
    print(f"wrote {args.out}", flush=True)
    return 0 if slim["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
