"""CLI: sonolab run-example01 | pack | pipe-joint-reconstruct | pipe-aperture-ablation …"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sonolab",
        description=(
            "SonoLab — SPECFEM2D-UT datasets + 2D/3D FDTD pipe practice analysis"
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser(
        "run-example01",
        help="Run conventional pulse-echo EXAMPLE 01 into datasets/runs/<job_id>",
    )
    p_run.add_argument(
        "--job-id",
        default=None,
        help="Run folder name under datasets/runs/ (default: from job_spec or pulse_echo_sdh_demo)",
    )
    p_run.add_argument(
        "--job-spec",
        type=Path,
        default=None,
        help="Optional YAML job spec (overlays EXAMPLE 01 params)",
    )
    p_run.add_argument(
        "--gif",
        action="store_true",
        help="Also build wavefield GIF (slow / large)",
    )

    p_pack = sub.add_parser(
        "pack",
        help="Pack an existing EXAMPLE-style job_dir into artifacts/",
    )
    p_pack.add_argument("job_dir", type=Path)

    p_b5 = sub.add_parser(
        "block5-ex01",
        help=(
            "Block 5: SPECFEM EXAMPLE 01 ToF gate "
            "(pick vs analytic SDH round-trip)"
        ),
    )
    p_b5.add_argument(
        "--job-id",
        default="_pack_smoke_from_ex01",
        help="Existing or new run under datasets/runs/ (default: _pack_smoke_from_ex01)",
    )
    p_b5.add_argument(
        "--rerun",
        action="store_true",
        help="Force sonolab run-example01 even if SU outputs exist",
    )
    p_b5.add_argument(
        "--out",
        type=Path,
        default=Path("datasets/runs/_block5_ex01/tof_gate.json"),
    )
    p_b5.add_argument(
        "--rel-tol",
        type=float,
        default=0.10,
        help="Relative ToF tolerance (default 0.10 = 10%%)",
    )

    p_recon = sub.add_parser(
        "pipe-reconstruct",
        help=(
            "LEGACY baseline: free-center circle + residual peak "
            "(fails under random ecc; headless JSON)"
        ),
    )
    p_recon.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON here (default: stdout)",
    )
    p_recon.add_argument("--no-dent", action="store_true")
    p_recon.add_argument("--ecc", type=float, default=0.0, help="eccentricity cells")
    p_recon.add_argument(
        "--ecc-phi",
        type=float,
        default=90.0,
        help="eccentricity direction degrees (0=+Y, 90=+X)",
    )
    p_recon.add_argument(
        "--step",
        type=float,
        default=1.0,
        help="look angle step degrees (default 1; use 10 for a fast smoke)",
    )
    p_recon.add_argument("--elements", type=int, default=20)
    p_recon.add_argument(
        "--c-assumed",
        type=float,
        default=None,
        help="ranging speed (default: uncalibrated 1.1·C_PROP)",
    )
    p_recon.add_argument(
        "--quiet",
        action="store_true",
        help="suppress per-look progress lines",
    )

    p_cal = sub.add_parser(
        "pipe-calibrate",
        help="Joint known-R / free-center wave-speed calibration (headless JSON)",
    )
    p_cal.add_argument("--out", type=Path, default=None)
    p_cal.add_argument("--no-dent", action="store_true")
    p_cal.add_argument("--ecc", type=float, default=0.0)
    p_cal.add_argument("--ecc-phi", type=float, default=90.0)
    p_cal.add_argument("--elements", type=int, default=20)
    p_cal.add_argument("--c-assumed", type=float, default=None)
    p_cal.add_argument("--quiet", action="store_true")

    p_live = sub.add_parser(
        "pipe-live",
        help="2D OpenGL live UI (needs: pip install glfw PyOpenGL)",
    )
    p_live.add_argument("--theta", type=float, default=0.0)
    p_live.add_argument("--elements", type=int, default=20)
    p_live.add_argument("--ecc", type=float, default=0.0)
    p_live.add_argument("--ecc-phi", type=float, default=90.0)
    p_live.add_argument("--no-dent", action="store_true")

    p_joint = sub.add_parser(
        "pipe-joint-reconstruct",
        help=(
            "Primary Lab B path: blind sector acquire + joint parametric "
            "pipe/dent fit (no R_nominal prior; headless JSON)"
        ),
    )
    p_joint.add_argument("--out", type=Path, default=None)
    p_joint.add_argument("--truth-out", type=Path, default=None)
    p_joint.add_argument("--no-dent", action="store_true")
    p_joint.add_argument("--ecc", type=float, default=0.0)
    p_joint.add_argument("--ecc-phi", type=float, default=90.0)
    p_joint.add_argument("--step", type=float, default=1.0)
    p_joint.add_argument("--elements", type=int, default=20)
    p_joint.add_argument(
        "--c-init",
        type=float,
        default=None,
        help="optimizer init for c when free-c (default: uncalibrated 1.1·C_PROP)",
    )
    p_joint.add_argument(
        "--c-cal",
        type=float,
        default=None,
        help=(
            "labeled fluid sound-speed cal; freezes c for absolute R/e/A. "
            "Omit for free-c scale-invariant mode"
        ),
    )
    p_joint.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="write truth-vs-recon PNG (uses truth sidecar for overlay only)",
    )
    p_joint.add_argument("--quiet", action="store_true")

    p_mc = sub.add_parser(
        "pipe-monte-carlo",
        help=(
            "Monte Carlo for joint estimator (scale-invariant metrics); "
            "writes datasets/runs/_pipe2d_joint/"
        ),
    )
    p_mc.add_argument("--out", type=Path, default=Path("datasets/runs/_pipe2d_joint/monte_carlo_joint.json"))
    p_mc.add_argument("--seed", type=int, default=None)
    p_mc.add_argument("--n-poses", type=int, default=None)
    p_mc.add_argument("--step-deg", type=float, default=None, help="Look step (default 1°)")
    p_mc.add_argument("--skip-sanity", action="store_true")
    p_mc.add_argument("--skip-mc", action="store_true")
    p_mc.add_argument(
        "--absolute-cal",
        action="store_true",
        help="run absolute c_cal smoke (writes monte_carlo_abs_c_cal.json)",
    )
    p_mc.add_argument("--quiet", action="store_true")

    p_abl = sub.add_parser(
        "pipe-aperture-ablation",
        help=(
            "Block 6: aperture ablation (N=2/8/20) through joint Lab B; "
            "prove denser aperture improves loc + A/R"
        ),
    )
    p_abl.add_argument(
        "--out",
        type=Path,
        default=Path("datasets/runs/_pipe2d_joint/aperture_ablation.json"),
    )
    p_abl.add_argument("--seed", type=int, default=None)
    p_abl.add_argument("--n-poses", type=int, default=None)
    p_abl.add_argument(
        "--step-deg",
        type=float,
        default=15.0,
        help="Look step (default 15° smoke; use 1° for evidence)",
    )
    p_abl.add_argument(
        "--elements-list",
        type=str,
        default="2,8,20",
        help="Comma-separated n_elem list",
    )
    p_abl.add_argument("--quiet", action="store_true")

    p_unc = sub.add_parser(
        "pipe-uncertainty-gate",
        help=(
            "Block 7a: NLLS CI coverage + refuse gate "
            "(primary N=20 vs stress N=2)"
        ),
    )
    p_unc.add_argument(
        "--out",
        type=Path,
        default=Path("datasets/runs/_pipe2d_joint/uncertainty_gate.json"),
    )
    p_unc.add_argument("--seed", type=int, default=None)
    p_unc.add_argument("--n-poses", type=int, default=None)
    p_unc.add_argument("--step-deg", type=float, default=15.0)
    p_unc.add_argument("--quiet", action="store_true")

    p_bench = sub.add_parser(
        "pipe-lab-a-bench",
        help="Benchmark Lab A acquire (rust vs python) for given --step-deg",
    )
    p_bench.add_argument("--step-deg", type=float, default=1.0)
    p_bench.add_argument(
        "--backend",
        choices=("auto", "rust", "python"),
        default="auto",
    )
    p_bench.add_argument("--dent", action="store_true", default=True)
    p_bench.add_argument("--no-dent", action="store_true")
    p_bench.add_argument("--ecc", type=float, default=0.0)
    p_bench.add_argument(
        "--out",
        type=Path,
        default=Path("datasets/runs/_pipe2d_joint/lab_a_bench.json"),
    )

    p_s3 = sub.add_parser(
        "pipe-survey-3d",
        help=(
            "True 3D scalar PE survey (radial looks θ×z) + "
            "Lab B absolute wall with --c-cal; Block 8 display mesh"
        ),
    )
    p_s3.add_argument(
        "--c-cal",
        type=float,
        required=True,
        help="labeled fluid sound speed (required for absolute 3D product)",
    )
    p_s3.add_argument("--step", type=float, default=None)
    p_s3.add_argument("--n-z", type=int, default=None)
    p_s3.add_argument(
        "--fast",
        action="store_true",
        help="smoke acquire: 15° × 6 z (overrides default Block 8 1° × 16)",
    )
    p_s3.add_argument(
        "--display-step",
        type=float,
        default=0.25,
        help="display mesh circumferential step (deg); default 0.25",
    )
    p_s3.add_argument(
        "--display-n-z",
        type=int,
        default=64,
        help="display mesh axial samples; default 64",
    )
    p_s3.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Lab A look processes (default: CPU count; 1 = sequential)",
    )
    p_s3.add_argument("--z-span", type=float, default=None)
    p_s3.add_argument("--ecc", type=float, default=2.0)
    p_s3.add_argument("--ecc-phi", type=float, default=90.0)
    p_s3.add_argument("--no-dent", action="store_true")
    p_s3.add_argument("--grid", type=int, default=64, help="NXY (=NZ related slab)")
    p_s3.add_argument("--nz-grid", type=int, default=80)
    p_s3.add_argument(
        "--out",
        type=Path,
        default=Path("datasets/runs/_pipe2d_joint/survey3d_demo.json"),
    )
    p_s3.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("datasets/runs/_pipe2d_joint/"),
    )
    p_s3.add_argument(
        "--open-view",
        action="store_true",
        help="open interactive HTML 3D viewer when done",
    )
    p_s3.add_argument("--quiet", action="store_true")

    p_b8 = sub.add_parser(
        "block8-demo",
        help=(
            "Block 8: dense interactive pipe cloud "
            "(acquire 1°×16z parallel; display 0.25°×64z mesh)"
        ),
    )
    p_b8.add_argument(
        "--c-cal",
        type=float,
        required=True,
        help="labeled fluid sound speed",
    )
    p_b8.add_argument(
        "--out",
        type=Path,
        default=Path("datasets/runs/_block8_demo"),
        help="evidence directory",
    )
    p_b8.add_argument(
        "--fast",
        action="store_true",
        help="smoke path: 15° × 6 z acquire (minutes)",
    )
    p_b8.add_argument("--step", type=float, default=None)
    p_b8.add_argument("--n-z", type=int, default=None)
    p_b8.add_argument("--display-step", type=float, default=0.25)
    p_b8.add_argument("--display-n-z", type=int, default=64)
    p_b8.add_argument("--workers", type=int, default=None)
    p_b8.add_argument("--ecc", type=float, default=2.0)
    p_b8.add_argument("--ecc-phi", type=float, default=90.0)
    p_b8.add_argument("--no-dent", action="store_true")
    p_b8.add_argument("--grid", type=int, default=64)
    p_b8.add_argument("--nz-grid", type=int, default=80)
    p_b8.add_argument("--open-view", action="store_true")
    p_b8.add_argument("--quiet", action="store_true")

    p_s3v = sub.add_parser(
        "pipe-survey-3d-view",
        help="Open interactive HTML 3D viewer from survey3d_demo.json",
    )
    p_s3v.add_argument(
        "--from",
        dest="from_json",
        type=Path,
        default=Path("datasets/runs/_pipe2d_joint/survey3d_demo.json"),
    )
    p_s3v.add_argument(
        "--html",
        type=Path,
        default=Path("datasets/runs/_pipe2d_joint/survey3d_view.html"),
    )
    p_s3v.add_argument(
        "--no-open",
        action="store_true",
        help="only write HTML, do not open browser",
    )

    p_w3 = sub.add_parser(
        "wave3d-live",
        help="3D FDTD OpenGL live UI (soft pipe/box; needs glfw PyOpenGL)",
    )
    p_w3.add_argument(
        "--mode",
        choices=("spherical", "plane", "oscillating", "chirp"),
        default="spherical",
    )
    p_w3.add_argument(
        "--inclusion",
        choices=("none", "soft_box", "soft_pipe"),
        default="soft_pipe",
    )
    p_w3.add_argument("--c-inside", type=float, default=0.55)
    p_w3.add_argument("--n", type=int, default=48, help="grid size N^3")

    p_w3s = sub.add_parser(
        "wave3d-smoke",
        help="Headless 3D FDTD smoke: step N times, print energy stats",
    )
    p_w3s.add_argument("--n", type=int, default=32, help="grid size (default 32 for speed)")
    p_w3s.add_argument("--steps", type=int, default=40)
    p_w3s.add_argument(
        "--inclusion",
        choices=("none", "soft_box", "soft_pipe"),
        default="soft_pipe",
    )
    p_w3s.add_argument("--mode", default="spherical")

    args = parser.parse_args(argv)

    if args.cmd == "run-example01":
        from sonolab.runner import run_example01

        job_spec = {}
        if args.job_spec is not None:
            with args.job_spec.open() as f:
                job_spec = yaml.safe_load(f) or {}
        job_id = (
            args.job_id
            or job_spec.get("job_id")
            or "pulse_echo_sdh_demo"
        )
        run_example01(
            str(job_id),
            job_spec=job_spec,
            job_spec_path=args.job_spec,
            skip_gif=not args.gif,
        )
        return 0

    if args.cmd == "pack":
        from sonolab.pack import pack_artifacts

        art = pack_artifacts(args.job_dir.resolve())
        print(art)
        return 0

    if args.cmd == "block5-ex01":
        from sonolab.runner import run_example01
        from sonolab.specfem_gate import resolve_job_dir, run_tof_gate

        job_dir = resolve_job_dir(str(args.job_id))
        su = job_dir / "OUTPUT_FILES" / "Uz_file_single_v.su"
        need_run = bool(args.rerun) or not su.exists()
        if need_run:
            print(f"running EXAMPLE 01 → {job_dir}", flush=True)
            run_example01(str(args.job_id), job_spec={"job_id": str(args.job_id)})
        out_json = Path(args.out)
        out_dir = out_json.parent
        evidence = run_tof_gate(
            job_dir,
            out_dir=out_dir,
            rel_tol=float(args.rel_tol),
            repack=True,
        )
        # Ensure primary evidence path matches --out
        out_json.parent.mkdir(parents=True, exist_ok=True)
        if out_json.resolve() != Path(evidence["artifacts"]["tof_gate_json"]).resolve():
            out_json.write_text(
                Path(evidence["artifacts"]["tof_gate_json"]).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            evidence["artifacts"]["tof_gate_json"] = str(out_json.resolve())
        cmp_ = evidence["comparison"]
        print(f"wrote {evidence['artifacts']['tof_gate_json']}", flush=True)
        print(f"wrote {evidence['artifacts']['annotated_ascan']}", flush=True)
        print(
            f"Block5 Ex01  pick={cmp_['pick_s']*1e6:.3f}µs  "
            f"analytic={cmp_['analytic_s']*1e6:.3f}µs  "
            f"rel_err={cmp_['rel_err']:.4f}  pass={evidence['pass']}",
            flush=True,
        )
        return 0 if evidence["pass"] else 1

    if args.cmd == "pipe-reconstruct":
        from sonolab.pipe2d import reconstruct_pipe

        data = reconstruct_pipe(
            dent=not args.no_dent,
            n_elem=args.elements,
            step_deg=float(args.step),
            progress=not args.quiet,
            ecc_mag=float(args.ecc),
            ecc_phi_deg=float(args.ecc_phi),
            c_assumed=args.c_assumed,
        )
        payload = json.dumps(data)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(payload, encoding="utf-8")
            print(f"wrote {args.out}", flush=True)
            print(
                f"peakResidualDeg={data['peakResidualDeg']}  "
                f"peakResidual={data['peakResidual']}  "
                f"rmse={data['rmse']}  fitR={data['fit']['R']}",
                flush=True,
            )
        else:
            print(payload)
        return 0

    if args.cmd == "pipe-joint-reconstruct":
        from sonolab.pipe2d.acquire import acquire_sector_scan
        from sonolab.pipe2d.blind import score_estimate
        from sonolab.pipe2d.joint_geometry import estimate_from_pack
        from sonolab.pipe2d.phased_array import C_ASSUMED_DEFAULT
        from sonolab.pipe2d.recon_plot import write_recon_overlay

        pack, truth = acquire_sector_scan(
            dent=not args.no_dent,
            n_elem=int(args.elements),
            step_deg=float(args.step),
            ecc_mag=float(args.ecc),
            ecc_phi_deg=float(args.ecc_phi),
            progress=not args.quiet,
        )
        c_cal = float(args.c_cal) if args.c_cal is not None else None
        c_init = (
            float(args.c_init)
            if args.c_init is not None
            else float(C_ASSUMED_DEFAULT)
        )
        if c_cal is not None:
            est = estimate_from_pack(pack, c_cal=c_cal)
        else:
            est = estimate_from_pack(pack, c_init=c_init)
        score = score_estimate(est, truth, a_over_r_tau=0.048, loc_tol_deg=10.0)
        out = {
            "acquisition": pack.to_dict(),
            "estimate": est,
            "score": score,
            "note": (
                "truth used only for score/plot; estimate from AcquisitionPack"
                + ("; c_cal labeled absolute scale" if c_cal is not None else "")
            ),
        }
        payload = json.dumps(out)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(payload, encoding="utf-8")
            print(f"wrote {args.out}", flush=True)
        if args.truth_out is not None:
            args.truth_out.parent.mkdir(parents=True, exist_ok=True)
            args.truth_out.write_text(
                json.dumps(truth.to_dict()), encoding="utf-8"
            )
            print(f"wrote truth sidecar {args.truth_out}", flush=True)
        if args.plot_out is not None:
            write_recon_overlay(args.plot_out, estimate=est, truth=truth)
            print(f"wrote {args.plot_out}", flush=True)
        if c_cal is not None:
            print(
                f"scale=c_cal  R={est['R']:.3f}  e={est['e']:.3f}  "
                f"A={est['A']:.3f}  θ0={est['theta0_deg']:.1f}  "
                f"A/R={est['A_over_R']:.4f}  e/R={est['e_over_R']:.4f}  "
                f"detected={score['detected']}  locErr={score['locErrorDeg']}",
                flush=True,
            )
            if "R_err" in score:
                print(
                    f"  abs: R_err={score['R_err']:.3f}  "
                    f"e_err={score['e_err']:.3f}  A_err={score['A_err']:.3f}  "
                    f"rAbsRmse={score.get('rAbsRmse')}",
                    flush=True,
                )
        else:
            print(
                f"θ0={est['theta0_deg']:.1f}  A/R={est['A_over_R']:.4f}  "
                f"e/R={est['e_over_R']:.4f}  τ=R/c={est['tau_R_over_c']:.2f}  "
                f"detected={score['detected']}  locErr={score['locErrorDeg']}",
                flush=True,
            )
        unc = est.get("uncertainty") or {}
        ci = unc.get("ci95") or {}
        print(
            f"  confidence={est.get('confidence')}  refuse={est.get('refuse')}  "
            f"reasons={est.get('refuseReasons')}  "
            f"σ_t={unc.get('sigma_t')}  cond={unc.get('condJTJ')}",
            flush=True,
        )
        if "A_over_R" in ci:
            a = ci["A_over_R"]
            print(
                f"  CI95 A/R=[{a['lo']:.4f},{a['hi']:.4f}]  "
                f"θ0±{ci.get('theta0_deg', {}).get('halfWidth', float('nan')):.1f}°",
                flush=True,
            )
        if args.out is None:
            print(payload)
        return 0

    if args.cmd == "pipe-calibrate":
        from sonolab.pipe2d import calibrate_c_known_radius
        from sonolab.pipe2d.phased_array import C_ASSUMED_DEFAULT, PIPE_R

        result = calibrate_c_known_radius(
            n_elem=args.elements,
            dent=not args.no_dent,
            ecc_mag=float(args.ecc),
            ecc_phi_deg=float(args.ecc_phi),
            R_known=float(PIPE_R),
            c_before=(
                float(args.c_assumed)
                if args.c_assumed is not None
                else float(C_ASSUMED_DEFAULT)
            ),
            progress=not args.quiet,
        )
        for w in result.get("warnings") or []:
            print(f"warn: {w}", flush=True)
        payload = json.dumps(result)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(payload, encoding="utf-8")
            print(f"wrote {args.out}", flush=True)
        print(
            f"c_after={result['c_after']:.6f}  R_fit={result['R_fit']:.3f}  "
            f"ecc_est={result['eccEstimate']:.3f}",
            flush=True,
        )
        if args.out is None:
            print(payload)
        return 0

    if args.cmd == "pipe-live":
        from sonolab.pipe2d.phased_array import run_live

        run_live(
            theta=args.theta,
            n_elem=args.elements,
            ecc_mag=float(args.ecc),
            ecc_phi_deg=float(args.ecc_phi),
            dent=not args.no_dent,
        )
        return 0

    if args.cmd == "pipe-monte-carlo":
        from sonolab.pipe2d import monte_carlo_dent as mc

        argv_mc = ["--out", str(args.out)]
        if args.seed is not None:
            argv_mc.extend(["--seed", str(args.seed)])
        if args.n_poses is not None:
            argv_mc.extend(["--n-poses", str(args.n_poses)])
        if getattr(args, "step_deg", None) is not None:
            argv_mc.extend(["--step-deg", str(args.step_deg)])
        if args.skip_sanity:
            argv_mc.append("--skip-sanity")
        if args.skip_mc:
            argv_mc.append("--skip-mc")
        if getattr(args, "absolute_cal", False):
            argv_mc.append("--absolute-cal")
        if args.quiet:
            argv_mc.append("--quiet")
        return int(mc.main(argv_mc) or 0)

    if args.cmd == "pipe-aperture-ablation":
        from sonolab.pipe2d import aperture_ablation as abl
        from sonolab.pipe2d import monte_carlo_dent as mc

        argv_abl = [
            "--out",
            str(args.out),
            "--step-deg",
            str(args.step_deg),
            "--elements-list",
            str(args.elements_list),
        ]
        seed = mc.SEED if args.seed is None else int(args.seed)
        n_poses = mc.N_POSES if args.n_poses is None else int(args.n_poses)
        argv_abl.extend(["--seed", str(seed), "--n-poses", str(n_poses)])
        if args.quiet:
            argv_abl.append("--quiet")
        return int(abl.main(argv_abl) or 0)

    if args.cmd == "pipe-uncertainty-gate":
        from sonolab.pipe2d import monte_carlo_dent as mc
        from sonolab.pipe2d import uncertainty_gate as ug

        argv_u = ["--out", str(args.out), "--step-deg", str(args.step_deg)]
        seed = mc.SEED if args.seed is None else int(args.seed)
        n_poses = mc.N_POSES if args.n_poses is None else int(args.n_poses)
        argv_u.extend(["--seed", str(seed), "--n-poses", str(n_poses)])
        if args.quiet:
            argv_u.append("--quiet")
        return int(ug.main(argv_u) or 0)

    if args.cmd == "pipe-survey-3d":
        from sonolab.pipe3d.acquire import acquire_pipe_survey_3d
        from sonolab.pipe3d.block8 import (
            BLOCK8_ACQUIRE_N_Z,
            BLOCK8_ACQUIRE_STEP_DEG,
            BLOCK8_FAST_N_Z,
            BLOCK8_FAST_STEP_DEG,
        )
        from sonolab.pipe3d.display import build_display_wall
        from sonolab.pipe3d.plot import write_survey_plots
        from sonolab.pipe3d.survey import (
            assemble_wall,
            estimate_survey,
            score_survey,
        )

        if args.fast:
            step = float(
                BLOCK8_FAST_STEP_DEG if args.step is None else args.step
            )
            n_z = int(BLOCK8_FAST_N_Z if args.n_z is None else args.n_z)
        else:
            step = float(
                BLOCK8_ACQUIRE_STEP_DEG if args.step is None else args.step
            )
            n_z = int(BLOCK8_ACQUIRE_N_Z if args.n_z is None else args.n_z)

        nxy = int(args.grid)
        survey, truth = acquire_pipe_survey_3d(
            n_z=n_z,
            z_span=args.z_span,
            step_deg=step,
            dent=not args.no_dent,
            ecc_mag=float(args.ecc),
            ecc_phi_deg=float(args.ecc_phi),
            nxy=nxy,
            nz=int(args.nz_grid),
            progress=not args.quiet,
            workers=args.workers,
        )
        estimates = estimate_survey(survey, c_cal=float(args.c_cal))
        product = assemble_wall(estimates)
        score = score_survey(product, truth, estimates=estimates)
        display = build_display_wall(
            product,
            truth,
            display_step_deg=float(args.display_step),
            n_z_display=int(args.display_n_z),
            acquire_step_deg=step,
            acquire_n_z=n_z,
        )
        out = {
            "acquireStepDeg": step,
            "acquireNz": n_z,
            "displayStepDeg": float(args.display_step),
            "displayNz": int(args.display_n_z),
            "survey": survey.to_dict(),
            "truth": truth.to_dict(),
            "product": product,
            "display": {
                "z": display["z"],
                "theta_deg": display["theta_deg"],
                "nZ": display["nZ"],
                "nTheta": display["nTheta"],
                "meta": display["meta"],
                "R_map_recon": display["R_map_recon"],
                "R_map_truth": display["R_map_truth"],
                "reconPositions": display["reconPositions"],
                "truthPositions": display["truthPositions"],
                "indices": display["indices"],
            },
            "score": score,
            "estimates": [
                {
                    k: e[k]
                    for k in (
                        "z",
                        "R",
                        "e",
                        "A",
                        "theta0_deg",
                        "A_over_R",
                        "e_over_R",
                        "c",
                        "scaleMode",
                        "dentCall",
                    )
                    if k in e
                }
                for e in estimates
            ],
            "note": (
                "Lab B from SurveyPack + labeled c_cal only; "
                "display mesh resampled (not extra FDTD); "
                "truth used for score/plot; physics=3D_scalar_acoustic"
            ),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {args.out}", flush=True)
        paths = write_survey_plots(
            args.plot_dir,
            product=product,
            truth=truth,
            score=score,
            display=display,
            display_step_deg=float(args.display_step),
            n_z_display=int(args.display_n_z),
            acquire_step_deg=step,
        )
        for k, p in paths.items():
            print(f"wrote {p}", flush=True)
        print(
            f"physics=3D_scalar  acquire={step:g}°×{n_z}z  "
            f"display={args.display_step:g}°×{args.display_n_z}z  "
            f"rAbs/R={score['rAbsRmseOverR']:.4f}  "
            f"θerr={score['dentThetaErrDeg']:.1f}°  "
            f"zerr={score['dentZErr']:.2f}  pass={score['pass']}",
            flush=True,
        )
        if getattr(args, "open_view", False) and "html" in paths:
            from sonolab.pipe3d.plot import open_survey_html

            open_survey_html(paths["html"])
            print(f"opened {paths['html']}", flush=True)
        return 0

    if args.cmd == "block8-demo":
        from sonolab.pipe3d.block8 import run_block8_demo

        result = run_block8_demo(
            c_cal=float(args.c_cal),
            out_dir=args.out,
            fast=bool(args.fast),
            step_deg=args.step,
            n_z=args.n_z,
            display_step_deg=float(args.display_step),
            n_z_display=int(args.display_n_z),
            ecc_mag=float(args.ecc),
            ecc_phi_deg=float(args.ecc_phi),
            dent=not args.no_dent,
            nxy=int(args.grid),
            nz=int(args.nz_grid),
            workers=args.workers,
            progress=not args.quiet,
            open_view=bool(args.open_view),
        )
        score = result["score"]
        print(f"wrote {result['demo_json']}", flush=True)
        for k, p in result["paths"].items():
            print(f"wrote {p}", flush=True)
        print(
            f"Block 8  acquire={result['acquireStepDeg']:g}°×"
            f"{result['acquireNz']}z  "
            f"display={result['displayStepDeg']:g}°×{result['displayNz']}z  "
            f"workers={result.get('workers')}  "
            f"elapsed={result.get('elapsedSec')}s  "
            f"rAbs/R={score['rAbsRmseOverR']:.4f}  "
            f"θerr={score['dentThetaErrDeg']:.1f}°  "
            f"pass={score['pass']}",
            flush=True,
        )
        return 0

    if args.cmd == "pipe-survey-3d-view":
        from sonolab.pipe3d.plot import open_survey_html, rebuild_view_from_demo_json

        html = rebuild_view_from_demo_json(args.from_json, args.html)
        print(f"wrote {html}", flush=True)
        if not args.no_open:
            open_survey_html(html)
            print(f"opened {html}", flush=True)
        return 0

    if args.cmd == "pipe-lab-a-bench":
        import os
        import time

        from sonolab.pipe2d import acquire as acq

        if args.backend == "python":
            os.environ["SONOLAB_LABA_BACKEND"] = "python"
        elif args.backend == "rust":
            os.environ["SONOLAB_LABA_BACKEND"] = "rust"
        else:
            os.environ.pop("SONOLAB_LABA_BACKEND", None)

        dent = not bool(args.no_dent)
        t0 = time.perf_counter()
        pack, truth = acq.acquire_sector_scan(
            dent=dent,
            step_deg=float(args.step_deg),
            ecc_mag=float(args.ecc),
            progress=True,
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        backend = acq.laba_backend()
        n = len(pack.theta_deg)
        out = {
            "backend": backend,
            "stepDeg": float(args.step_deg),
            "nLooks": n,
            "wallMs": round(wall_ms, 3),
            "msPerLook": round(wall_ms / max(n, 1), 4),
            "elapsedMsMeta": pack.meta.get("elapsedMs"),
            "dent": dent,
            "ecc": float(args.ecc),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2), flush=True)
        print(f"wrote {args.out}", flush=True)
        return 0

    if args.cmd == "wave3d-live":
        from sonolab.wave3d import run_live as wave3d_live

        wave3d_live(
            int(args.n),
            args.mode,
            args.inclusion,
            float(args.c_inside),
        )
        return 0

    if args.cmd == "wave3d-smoke":
        import numpy as np

        from sonolab.wave3d import Wave3D

        sim = Wave3D(
            n=int(args.n),
            mode=str(args.mode),
            inclusion=str(args.inclusion),
        )
        for _ in range(int(args.steps)):
            sim.step()
        e = float(np.sum(sim.u * sim.u))
        peak = float(np.max(np.abs(sim.u)))
        print(
            f"wave3d-smoke n={sim.n} inclusion={sim.inclusion} "
            f"steps={args.steps} energy={e:.6e} peak={peak:.6e}",
            flush=True,
        )
        return 0

    parser.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
