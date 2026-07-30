# SonoLab — agent / project guidelines

## Vision (locked)

**North star:** open, research-grade work on **in-pipe ultrasonic geometry**:

> Emit ultrasonic pulses, record echoes, and recover a **3D inner-wall / casing surface** (point cloud or \(R(\theta,z)\) map) when tool pose, fluid sound speed, materials, and the defect map are largely unknown. **Dents and corrosion are read from that reconstructed geometry** - not from a standalone yes/no detector scorecard.

Full plan and building blocks: `docs/ECHO_TO_GEOMETRY_VISION.md`.

### Product focus (locked 2026-07-29)

| Priority | Goal |
|----------|------|
| **Primary** | Blind echo → wall geometry → **3D cloud / map** you can inspect (truth overlay for demos) |
| **Secondary** | Shape metrics (\(A/R\), \(e/R\), \(\theta_0\), shape RMSE) and uncertainty/refuse as *quality of the reconstruction* |
| **Tertiary** | Dent recall / FAR gates - useful regression checks, **not** the product story |

Next engineering push: improve the **cloud product** (denser acquire when affordable, elastic Lab A later, dual-obs 7b). Do not optimize refuse/recall thresholds as the main workstream unless the cloud path is blocked.

### Lab A vs Lab B (do not collapse)

| Lab | Role |
|-----|------|
| **A — Physics truth** | FDTD (`pipe2d` / `wave3d`) and SPECFEM2D-UT generate raw RF + known ground truth |
| **B — Blind estimator** | Instrument + TX/RX only; must not read Lab A truth, pipe \(R\), fluid \(c\), pose, or dent map |

- Physics engines are the **referee and data factory**, not the field product.
- **Lab A pipe2d engine:** default is the Rust FDTD extension `sonolab_pipe2d_fdtd` (`crates/sonolab_pipe2d_fdtd/`, build with `maturin develop --release`). Force NumPy reference with `SONOLAB_LABA_BACKEND=python`. Bench: `sonolab pipe-lab-a-bench --step-deg 1`.
- **Primary joint study is 1°** full-circle sweeps (360 looks). The old 15° grid was a compute compromise; keep `--step-deg 15` only for smoke.
- **Dense MC pose band:** `ecc ≤ 8` grid units for the 1° acceptance study (high-ecc >10 still needs more Lab B work). Lab B may freeze circle `(R,cx,cy,c)` from a dent-OFF pack (null reference) when scoring dent-ON - still no `R_nominal` prior.
- **No pipe-prior cheat:** Lab B must not use nominal pipe ID \(R_\mathrm{nominal}\) (or true \(c\)) as a *secret* prior. Pulse-echo times alone are scale-invariant; free-\(c\) mode recovers \(\theta_0\), \(A/R\), \(e/R\), shape, \(\tau=R/c\). Absolute \(R\)/\(e\)/\(A\) require an **explicit labeled** fluid \(c_{\mathrm{cal}}\) (`--c-cal` / `estimate_from_pack(..., c_cal=...)`) - never silent mill ID.
- Classical free-center circle + residual dent call is a **failed baseline** (`pipe-reconstruct`). Primary path: joint fit (`pipe-joint-reconstruct`, `pipe2d/joint_geometry.py`) → wall XY / pipe3d cloud; scored via `blind.py` when needed.
- Vision **Blocks 1–8 done** (blind harness + joint fit + Rust Lab A + absolute \(c_{\mathrm{cal}}\) + pipe3d + SPECFEM EXAMPLE 01 ToF gate + aperture ablation + uncertainty/refuse + **Block 8 dense cloud demo**). Evidence: `datasets/runs/_pipe2d_joint/`, `datasets/runs/_block5_ex01/`, `datasets/runs/_block8_demo/`.
- **SPECFEM2D-UT:** default root prefers nested `sonolab/specfem2d-UT` (override `SONOLAB_SPECFEM_ROOT`). Portfolio path: `sonolab run-example01` then `sonolab block5-ex01` (pick vs analytic SDH, rel err ≤ 10%).
- **pipe3d / Block 8:** short-segment 3D scalar FDTD; acquire default **1° × 16 z** (parallel); display mesh **0.25° × 64 z**; CLI `sonolab block8-demo` / `pipe-survey-3d --fast`. Label `physics: 3D_scalar_acoustic`. Do not claim commercial ILI mm imaging density. Elastic SPECFEM3D/k-Wave remains later. **Primary product surface for demos.**
- **Block 6 aperture:** Lab A coherent DAS RX; `AcquisitionPack` carries `elem_pitch` / `aperture_span`; CLI `sonolab pipe-aperture-ablation`. Dual-obs / FMC deferred.
- **Block 7a uncertainty:** CIs + refuse support *when the cloud/fit is untrustworthy*; not a dent-POD product. Soft-threshold recall tune rejected. Dual-obs refuse = 7b.
- Absolute scale = explicit `c_cal`; still no silent `R_nominal`.

### Honesty

Do not copy proprietary beamformers, ASICs, frequencies, acquisition schedules, or closed commercial algorithms. Label assumed vs measured vs calibrated parameters honestly. This package is open research / portfolio evidence, not a field ILI product.

## Home and layout

- **Package home:** this `sonolab/` tree
- **Python:** `src/sonolab/` (`cli`, `runner`, `pack`, `pipe2d`, `pipe3d`, `wave3d`)
- **UT engine:** nested `specfem2d-UT/` (or `SONOLAB_SPECFEM_ROOT` / `SPECFEM2D_DIR`)
- **SPECFEM3D:** nested `specfem3d/` for learning / orientation; stock examples are seismic-scale
- **Vision doc:** `docs/ECHO_TO_GEOMETRY_VISION.md`
- **Orientation:** `datasets/runs/_orientation/`

## Engineering boundaries

- Forward SPECFEM: do **not** rewrite `specfem2d-UT/src/` or `specfem3d/src/` for product features.
- Portfolio path: EXAMPLE 01 pulse-echo / packaged artifacts before custom pipe meshes.
- Outputs: `datasets/runs/<job_id>/artifacts/{ascan.npz,meta.json,a_scan.png}` for SPECFEM jobs; pipe3d clouds under `datasets/runs/` survey artifacts.
- Prefer changes that strengthen **hiring evidence**: runnable **geometry/cloud demos**, honest limits, validated estimators - not detector ROC tuning for its own sake.

## Agent behavior

- Follow `docs/ECHO_TO_GEOMETRY_VISION.md` when scoping echo→geometry work; prefer cloud/map quality over detector-scorecard tuning unless the user asks otherwise.
- Ask before meaningful architecture / scientific method changes (e.g. Block 2a vs 2b).
- Preserve dirty user state; do not commit unless asked.
- Nested `specfem2d-UT/AGENTS.md` still governs solver-checkout work inside that tree.
- **Never** name commercial ILI vendors or product lines in public README, demos, commits, or LinkedIn copy.
