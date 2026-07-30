"""Block 6: aperture ablation — denser aperture vs sparse under joint Lab B.

Protocol
--------
- Same pose RNG as monte_carlo_dent (SEED, ecc ≤ 8) for each n_elem.
- Lab A: acquire_sector_scan(n_elem=…).
- Lab B: estimate_from_pack(AcquisitionPack only); null-circle freeze dent-OFF→ON.
- Pass: N=20 beats N=2 on loc median (when detected) and A/R RMSE; FAR not worse.
"""

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

DEFAULT_ELEMENTS = (2, 8, 20)
DEFAULT_OUT = Path("datasets/runs/_pipe2d_joint/aperture_ablation.json")


def run_pose_for_aperture(
    *,
    rng: np.random.Generator,
    pose_id: int,
    step_deg: float,
    n_elem: int,
    progress: bool,
) -> dict[str, Any]:
    """One MC pose at a fixed aperture (mirrors monte_carlo_dent.run_pose)."""
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
        t_pose0 = time.perf_counter()
        if progress:
            print(
                f"\n=== N={n_elem} pose {pose_id}  ecc={ecc_mag:.2f}@{ecc_phi:.1f}°  "
                f"c_init={c_init:.4f}  dentθ={dent_theta0:.1f}° ===",
                flush=True,
            )

        pack_off, truth_off = acq_mod.acquire_sector_scan(
            dent=False,
            n_elem=int(n_elem),
            step_deg=step_deg,
            ecc_mag=ecc_mag,
            ecc_phi_deg=ecc_phi,
            progress=progress,
        )
        est_off = estimate_from_pack(pack_off, c_init=c_init)
        score_off = score_estimate(
            est_off,
            truth_off,
            a_over_r_tau=mc.A_OVER_R_TAU,
            loc_tol_deg=mc.LOC_TOL_DEG,
        )

        pack_on, truth_on = acq_mod.acquire_sector_scan(
            dent=True,
            n_elem=int(n_elem),
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
            a_over_r_tau=mc.A_OVER_R_TAU,
            loc_tol_deg=mc.LOC_TOL_DEG,
        )

        elapsed = time.perf_counter() - t_pose0
        return {
            "poseId": pose_id,
            "nElem": int(n_elem),
            "elemPitch": float(pa.ELEM_PITCH),
            "apertureSpan": float(pa.aperture_span_cells(n_elem)),
            "elapsedSec": round(elapsed, 2),
            "packMeta": {
                "rxMode": pack_on.meta.get("rxMode"),
                "pickSnrMedian": pack_on.meta.get("pickSnrMedian"),
                "labaBackend": pack_on.meta.get("labaBackend"),
            },
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


def evaluate_ablation_pass(
    by_n: dict[int, dict[str, Any]],
    *,
    sparse_n: int = 2,
    dense_n: int = 20,
) -> dict[str, Any]:
    """N=dense beats N=sparse on loc median + A/R RMSE; FAR not worse.

    Loc rule: dense must detect at least once. If sparse never detects
    (loc median None) while dense does, that counts as an improvement.
    """
    if sparse_n not in by_n or dense_n not in by_n:
        return {
            "sparseN": sparse_n,
            "denseN": dense_n,
            "pass": False,
            "reason": "missing sparse or dense metrics",
        }
    s = by_n[sparse_n]
    d = by_n[dense_n]
    loc_s = s.get("locErrorDegMedian")
    loc_d = d.get("locErrorDegMedian")
    a_s = float(s["A_over_R_rmse"])
    a_d = float(d["A_over_R_rmse"])
    far_s = float(s["falseAlarmRate"])
    far_d = float(d["falseAlarmRate"])
    recall_s = float(s.get("recall", 0.0))
    recall_d = float(d.get("recall", 0.0))

    if loc_d is None:
        loc_ok = False
    elif loc_s is None:
        # Sparse never localized; dense did → aperture helped.
        loc_ok = True
    else:
        loc_ok = float(loc_d) < float(loc_s)

    a_ok = a_d < a_s
    far_ok = far_d <= far_s + 1e-12
    recall_ok = recall_d >= recall_s - 1e-12
    overall = bool(loc_ok and a_ok and far_ok and recall_ok)
    return {
        "sparseN": sparse_n,
        "denseN": dense_n,
        "locMedianSparse": loc_s,
        "locMedianDense": loc_d,
        "A_over_R_rmseSparse": a_s,
        "A_over_R_rmseDense": a_d,
        "farSparse": far_s,
        "farDense": far_d,
        "recallSparse": recall_s,
        "recallDense": recall_d,
        "checks": {
            "locMedianImproved": loc_ok,
            "A_over_R_rmseImproved": a_ok,
            "farNotWorse": far_ok,
            "recallNotWorse": recall_ok,
        },
        "pass": overall,
    }


def run_aperture_ablation(
    *,
    elements: list[int] | tuple[int, ...] = DEFAULT_ELEMENTS,
    n_poses: int = mc.N_POSES,
    step_deg: float = 15.0,
    seed: int = mc.SEED,
    progress: bool = True,
) -> dict[str, Any]:
    elements = [int(n) for n in elements]
    by_n: dict[int, dict[str, Any]] = {}
    poses_by_n: dict[int, list[dict[str, Any]]] = {}
    t0 = time.perf_counter()
    for n_elem in elements:
        # Same seed → matched pose band across apertures.
        rng = np.random.default_rng(int(seed))
        poses: list[dict[str, Any]] = []
        if progress:
            print(
                f"\n######## aperture N={n_elem}  "
                f"span={pa.aperture_span_cells(n_elem):.1f}  "
                f"poses={n_poses} step={step_deg}° ########",
                flush=True,
            )
        for pose_id in range(int(n_poses)):
            poses.append(
                run_pose_for_aperture(
                    rng=rng,
                    pose_id=pose_id,
                    step_deg=float(step_deg),
                    n_elem=n_elem,
                    progress=progress,
                )
            )
        metrics = mc.compute_metrics(poses)
        metrics["nElem"] = int(n_elem)
        metrics["apertureSpan"] = float(pa.aperture_span_cells(n_elem))
        metrics["elemPitch"] = float(pa.ELEM_PITCH)
        by_n[int(n_elem)] = metrics
        poses_by_n[int(n_elem)] = poses

    gate = evaluate_ablation_pass(by_n)
    elapsed = time.perf_counter() - t0
    return {
        "schema_version": "0.1",
        "block": 6,
        "case": "aperture_ablation_joint_lab_b",
        "seed": int(seed),
        "stepDeg": float(step_deg),
        "nPoses": int(n_poses),
        "elements": elements,
        "rxMode": "coherent_das",
        "labaBackend": acq_mod.laba_backend(),
        "elapsedSec": round(elapsed, 2),
        "metricsByN": {str(k): v for k, v in by_n.items()},
        "ablationGate": gate,
        "pass": bool(gate["pass"]),
        "note": (
            "Lab B estimate_from_pack(AcquisitionPack) only; denser aperture "
            "must improve loc median and A/R RMSE vs N=2; FAR must not worsen."
        ),
        # Keep pose detail optional / compact: only detection summaries.
        "detectionSummaryByN": {
            str(n): mc.compute_metrics(poses)["detectionByPose"]
            for n, poses in poses_by_n.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Block 6 aperture ablation (joint Lab B, no R_nominal)"
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="JSON output path",
    )
    p.add_argument("--seed", type=int, default=mc.SEED)
    p.add_argument("--n-poses", type=int, default=mc.N_POSES)
    p.add_argument("--step-deg", type=float, default=15.0)
    p.add_argument(
        "--elements-list",
        type=str,
        default="2,8,20",
        help="Comma-separated n_elem list (default 2,8,20)",
    )
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    elements = [
        int(x.strip())
        for x in str(args.elements_list).split(",")
        if x.strip()
    ]
    if not elements:
        raise SystemExit("elements-list must be non-empty")

    result = run_aperture_ablation(
        elements=elements,
        n_poses=int(args.n_poses),
        step_deg=float(args.step_deg),
        seed=int(args.seed),
        progress=not args.quiet,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    gate = result["ablationGate"]
    print(json.dumps({"pass": result["pass"], "ablationGate": gate}, indent=2))
    print(f"wrote {args.out}", flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
