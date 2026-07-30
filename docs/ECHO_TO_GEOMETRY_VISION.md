# Vision: Echo-to-Geometry (in-pipe ultrasonic wall)

**Status:** locked product north star; Blocks 1–8 done; **cloud-first demo packaged (2026-07-29)**  
**Date:** 2026-07-29  
**Home:** this `sonolab/` package  
**North star:** recover **3D pipe/casing wall geometry** from dense ultrasonic TX/RX under real unknowns (pose, fluid \(c\), geometry / defects).  
**Scope:** open research / portfolio replica of the *problem class*; do not copy proprietary beamformers, ASICs, frequencies, or closed commercial algorithms.


---

## 1. One-sentence goal

Build a system that, given **only transmitted waveforms and received echoes** (plus what can be inferred from them), recovers a **3D inner-wall / casing surface** (point cloud or \(R(\theta,z)\)) under unknown tool pose, unknown fluid sound speed, unknown materials, and unknown dent/corrosion maps - then **inspects that surface** for dents - while keeping a **separate, fully correct physics lab** that generates ground truth for validation.

### Product focus (locked)

1. **Primary deliverable:** reconstructed wall geometry you can *see* (cloud / map + truth overlay for demos).  
2. **Secondary:** shape fidelity metrics and uncertainty/refuse (is this cloud trustworthy?).  
3. **Tertiary:** binary dent recall/FAR - regression gates only; **not** the headline product.

Dent “detection” is a *consequence of geometry*, not a separate classifier we optimize in isolation.

---

## 2. What “everything unknown” actually means

### Operator / device knows (minimal) - Lab B instrument facts only

- It fired something (time base, TX schedule, element indices if array).
- It recorded RF / A-scans on receivers; picks \(t(\theta)\).
- Commanded look angles \(\theta\) and tool-frame array geometry (`ARRAY_LOOK_OFFSET`).
- Rough prior: “we are inside a roughly cylindrical metal pipe/casing” (topology prior, not a free-form room).

### Operator / device does **not** know a priori (no pipe-prior cheat)

| Unknown | Why it hurts naive ranging |
|---------|----------------------------|
| Pipe nominal ID \(R\) | **Forbidden as Lab B prior in v1** - not a free cheat |
| Tool center vs pipe center (eccentricity) | Standoffs are not radii from the pipe axis |
| Tool tilt / depth registration | Looks are not a perfect polar grid |
| Fluid sound speed \(c\) | Converts time → distance; wrong \(c\) scales the world |
| Fluid type / temperature / pressure | Changes \(c\) and attenuation |
| Steel properties / coating | Mode conversion, OD echoes, thickness |
| Dent / ovality / corrosion map | Visible on the reconstructed wall |
| Absolute scale if \(c\) and \(R\) both free | Underdetermined from ToF alone |

**Scale invariance:** pulse-echo times alone are invariant under \((R,e,A,c)\rightarrow\lambda(R,e,A,c)\). Lab B therefore recovers **scale-invariant** geometry (\(\theta_0\), \(A/R\), \(e/R\), shape, \(\tau=R/c\)). Absolute \(R\) and \(c\) separately need an **explicit labeled anchor** later - do not sneak in \(R_\mathrm{nominal}\).

**Honest limit (legacy MC):**  
If \(c\), eccentricity, and local wall shape are all free, a single circle-fit + residual peak **does not** recover dent ground truth. Block 2 replaces that with a joint parametric fit (instrument facts only).

---

## 3. Two labs, one product (non-negotiable split)

```text
┌─────────────────────────────────────────────────────────────┐
│  LAB A — Physics truth (simulate the world)                 │
│  FDTD pipe2d/wave3d → SPECFEM2D-UT → (later) 3D full-wave   │
│  Knows: pose, c, materials, dent map                        │
│  Emits: raw RF / A-scans + registered truth artifacts         │
└──────────────────────────┬──────────────────────────────────┘
                           │ synthetic “field tape”
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  LAB B — Blind estimator (device brain)                     │
│  Input: TX/RX + instrument geometry only                    │
│  Must NOT read Lab A truth, pipe R, fluid c, or dent map    │
│  Output (v1): wall geometry (θ0, A/R, e/R, shape, τ=R/c)    │
│            → with c_cal: absolute R, wall XY / 3D cloud       │
└─────────────────────────────────────────────────────────────┘
```

- Lab A answers: “Is our physics good enough?”  
- Lab B answers: “Can the device reconstruct the wall when it doesn’t know the world?”  
- Evaluation compares Lab B **geometry** to Lab A truth (cloud / \(R(\theta,z)\) / shape); binary dent gates are optional regression checks.
- Blind pack API: `AcquisitionPack` (instrument + picks) vs `TruthPack` sidecar for scoring only (`src/sonolab/pipe2d/blind.py`).

**You do need correct physics** for ground truth and for hiring evidence.  
**You do not** run full-wave SEM inside the field estimator on every look. Field path is picks → geometry → joint inversion (ray / beam / reduced models), with full-wave used offline for training data and validation.

This matches common industrial UT practice (semi-analytic / ray models + occasional FEM; dense-array pulse-echo → geometry products; research on joint tool localization + \(c\) + casing imaging).

---

## 4. Public anchors (verified direction)

### Portfolio product signals (target experience)

- High-density arrays; full circumferential coverage where practical.  
- Acoustic imaging through water / brine / mud / production fluids when calibrated.  
- Dense 3D point clouds / maps of ID deformation and wall loss (honest fidelity limits for *this* scalar FDTD stack).  
- Imaging modes: beamforming, steering, multi-pulse; geometry-aware reconstruction.  
- Method emphasis in open literature: recover **inner surface first**, then geometry-aware paths; pose / standoff / eccentricity matter.

SonoLab: **open research on that problem class**, not a commercial ILI product.

### Research that unlocks “unknown \(c\) + unknown pose”

- Dual observation (e.g. pulse-echo + pitch-catch) + iterative joint inversion of **trajectory + casing geometry + fluid \(c\)** (ultrasonic logging SLAM literature).  
- Circumference / closure constraints to update \(c\).  
- Dijkstra / graph methods for tool localization in deformed casing.  
- Our own negative result: free-center residual dent call fails under random ecc → motivates joint / dual-obs / ecc-aware residuals.

### What already exists in this directory

| Asset | Role in the vision |
|-------|--------------------|
| `sonolab/pipe2d/` | Fast 2D truth + joint blind estimator + legacy residual baseline |
| `pipe2d/blind.py`, `joint_geometry.py`, `acquire.py` | Blocks 1–2: blind packs, joint fit, Lab A acquire split |
| `sonolab pipe-joint-reconstruct` / `pipe-monte-carlo` | CLI for joint recon + MC |
| `datasets/runs/_pipe2d_joint/` | Acceptance evidence (sanity + MC) |
| `sonolab/wave3d/` | 3D scalar FDTD sandbox |
| `sonolab` SPECFEM2D-UT path | Higher-fidelity UT truth |
| `well-array-beamforming-sim` | Ray/SAFT → point cloud product shape |
| `ut-array-imaging-lab` | FMC/TFM elastic FDTD |
| `specfem2d-UT/docs/SONOLAB_MASTER_PLAN.md` | Broader architecture notes (solver tree) |
| Legacy MC (`wavelab-3d/.../_monte_carlo_dent_results.json`) | Proof that naive residual detection is insufficient |

---

## 5. Product contract (what “done” looks like)

### Device-facing API (Lab B)

```text
Input (Lab B):
  acquisition: { theta, t_echo, array_look_offset, pulse_width, … }
  # NO pipe R_nominal, NO pose, NO dent truth
  optional labeled: c_cal   # fluid sound-speed cal (absolute scale)

Output (free-c / scale-invariant):
  wall: { theta0, A_over_R, e_over_R, r_pred / mean(r), shape }
  time_scale: { tau = R/c }
  diagnostics: { t_rmse, dentCorrPos, ovalEnergy, dentCall?, refuse, … }
  # dentCall is optional regression metadata - primary product is the wall

Output (+ c_cal absolute):
  absolute: { R, e, A, cx, cy, wall_xy / 3D cloud, scaleMode=c_cal }
  # same ratios as above; scorer adds R_err, e_err, A_err, rAbsRmse, cloud RMSE
```

### Acceptance criteria (staged, no quality compromise)

1. **Known-pose, known-\(c\)** (oracle): wall \(R(\theta)\) / cloud within agreed error vs Lab A.  
2. **[Done - Block 2]** No pipe \(R\) / fluid \(c\) prior: joint fit recovers scale-invariant \(\theta_0\), \(A/R\), \(e/R\) on full-sector MC (see `_pipe2d_joint/`). Absolute \(R\), \(c\) not required.  
3. **[Done - Block 3]** Explicit absolute anchor via labeled fluid \(c_{\mathrm{cal}}\) (never silent \(R_\mathrm{nominal}\)). Absolute wall when `c_cal` is supplied.  
4. **[Done - Block 5]** SPECFEM2D-UT EXAMPLE 01 pulse-echo SDH: pick ToF within 10% of analytic round-trip (`sonolab block5-ex01`).  
5. **[Done - Block 4]** Registered \((\theta,z)\) wall map / point cloud from true 3D scalar PE looks (`pipe3d`); absolute with labeled \(c_{\mathrm{cal}}\).  
6. **[Done - Block 6]** Aperture ablation: coherent DAS Lab A + same Lab B API; denser aperture (\(N=20\)) beats sparse (\(N=2\)) on loc / \(A/R\) (`sonolab pipe-aperture-ablation`).  
7. **[Done - Block 7a]** Uncertainty: NLLS linearization CIs + instrument-only refuse (`sonolab pipe-uncertainty-gate`) - trust flags for the reconstruction. Dual-obs consistency deferred to 7b.  
8. **[Done - Block 8]** Portfolio demo: echoes → **dense interactive 3D wall** (`sonolab block8-demo`); dual acquire/display density; honest HUD.

Non-goals: claiming commercial ILI performance numbers; full 3D SEM every shot; ML before deterministic Lab B beats the MC failure mode; sneaking pipe ID into Lab B; treating dent POD/ROC as the product.

---

## 6. Building blocks (implement one-by-one on request)

Each block has: purpose, inputs/outputs, depends-on, acceptance test.  
**Do not start Block \(n+1\) until Block \(n\) acceptance is green** (unless you explicitly waive).

### Block 0 - Vision & contracts (this document)

- [x] Written  
- Acceptance: you agree north star + Lab A/B split + block order (or edit them).

### Block 1 - Blind evaluation harness - DONE

**Purpose:** Never cheat. Lab B cannot see truth fields.  
**Build:** `AcquisitionPack` / `TruthPack` + `score_estimate` in `src/sonolab/pipe2d/blind.py`; unit tests in `tests/test_joint_blind.py`.  
**Accept:** Estimator takes `AcquisitionPack` only; scorer fails closed without truth. ✓

### Block 2 - Observation model upgrade (fix the MC failure) - DONE (2a)

**Purpose:** Replace “single residual peak = dent” with ecc-aware geometry **without pipe \(R\) prior**.  
**Implemented (2a):** joint parametric wall (circle + cosine dent) + fluid \(c\); full-sector opposite looks; gauge-aware scale via instrument `ARRAY_LOOK_OFFSET`; report \(A/R\), \(e/R\), \(\theta_0\), \(\tau=R/c\).  
**Code:** `joint_geometry.py`, `acquire_sector_scan`, CLI `pipe-joint-reconstruct`, MC → `datasets/runs/_pipe2d_joint/`.  
**Accept (2026-07-29 MC, 16 poses, step 15°, \(\tau_{A/R}=0.048\)):** recall 0.75, FAR 0.125, loc median 7.5°, \(e/R\) RMSE 0.039, \(A/R\) RMSE 0.046; sanity ecc=0 detected. ✓  
**Accept (2026-07-29 dense MC, step 1°, Rust Lab A, ecc≤8, loc tol 10°):** recall 0.9375, FAR 0, loc median 4.3°, \(e/R\) RMSE 0.0095, \(A/R\) RMSE 0.047; sweep ~0.7 s / 360 looks. Evidence: `monte_carlo_joint_1deg.json`, `bench_rust_1deg.json`. ✓  
**Deferred:** (2b) dual pitch-catch hardware; (2c) circumference closure as labeled absolute-\(c\) path.

### Block 3 - Absolute scale via labeled fluid \(c_{\mathrm{cal}}\) - DONE

**Purpose:** Calibrate-then-map: freeze fluid sound speed from an **explicit** operator/mud/cal-cell input; recover absolute \(R\), \(e\), \(A\), wall XY.  
**Implemented:** `c_cal` on `fit_joint_geometry` / `estimate_from_pack`; scorer absolute metrics; CLI `--c-cal` + `--plot-out`; absolute-cal smoke `pipe-monte-carlo --absolute-cal`.  
**Accept:** unit test recovers synthetic absolute \(R\); smoke with `c_cal=C_{\mathrm{PROP}}\) (labeled) passes median \(|R_{\mathrm{rel}}|\le 3\%\); wrong `c_cal=1.2\,c` scales absolute \(R\) ~1.2 while ratios stay OK. Evidence: `monte_carlo_abs_c_cal.json`, `recon_abs_c_cal.png`. ✓  
**Deferred:** (2b) dual pitch-catch; circumference-only closure without labeled \(c\).

### Block 4 - True 3D azimuthal pulse-echo + wall assembly - DONE

**Purpose:** Real **3D scalar** acoustic Lab A: tool fires radial PE looks in \(\theta\) at stations \(z\); Lab B + labeled \(c_{\mathrm{cal}}\) assemble \(R(\theta,z)\) / XYZ cloud.  
**Implemented:** [`src/sonolab/pipe3d/`](src/sonolab/pipe3d/) (`geometry`, `fdtd3d`, `acquire`, `survey`, `display`, `plot`, `block8`); CLI `sonolab pipe-survey-3d` / `block8-demo`. Parallel Lab A looks; dual acquire/display density.  
**Physics label:** `3D_scalar_acoustic` (soft wall; not elastic SPECFEM3D).  
**Accept:** unit ToF sanity + assemble/score; smoke survey evidence; Block 8 dense mesh demo. ✓  
**Deferred:** elastic k-Wave / SPECFEM3D, FMC/TFM.

### Block 5 - Lab A fidelity gate (EXAMPLE 01) - DONE

**Purpose:** SPECFEM2D-UT as referee for pick/ToF on a canned pulse-echo case.  
**Implemented:** nested SPECFEM root default; corrected seismogram `dt` (`DT × NTSTEP_BETWEEN_OUTPUT_SAMPLE`); [`specfem_gate.py`](src/sonolab/specfem_gate.py); CLI `sonolab block5-ex01`.  
**Accept:** EXAMPLE 01 SDH pick vs analytic \(t=2L/v_p\) with relative error ≤ 10%. Evidence: `datasets/runs/_block5_ex01/tof_gate.json`, `tof_gate_ascan.png`. ✓  
**Deferred:** immersion Ex03; pipe-matched SPECFEM mesh; Lab B on SPECFEM RF; k-Wave / SPECFEM3D.

### Block 6 - Array / acquisition realism - DONE (aperture ablation)

**Purpose:** Multi-element RF, steering, registration (open dense-array subset).  
**Implemented (6a):** coherent delay-and-sum RX (same look delays as TX) in Python + Rust Lab A; `AcquisitionPack` aperture facts (`elem_pitch`, `aperture_span`); CLI `sonolab pipe-aperture-ablation` comparing \(N\in\{2,8,20\}\) through unchanged Lab B `estimate_from_pack`.  
**Accept:** denser aperture improves loc median (or recovers detections when sparse fails) and \(A/R\) RMSE vs \(N=2\); FAR / recall not worse. Evidence (2026-07-29, step 1°, 16 poses, Rust Lab A): \(N=20\) vs \(N=2\) — recall 0.9375 vs 0.0625, loc median 4.3° vs 5.6°, \(A/R\) RMSE 0.047 vs 0.095, FAR 0 both. Files: `datasets/runs/_pipe2d_joint/aperture_ablation.json`, smoke `aperture_ablation_smoke.json`. ✓  
**Deferred (6b+):** dual-obs pitch-catch (2b); CW/CCW steered multi-mode; FMC/TFM; per-element RF matrices.

### Block 7 - Uncertainty & failure modes - DONE (7a)

**Purpose:** When unknowns are too severe, say so.  
**Implemented (7a):** NLLS local-linearization ~95% CIs (gauge-fixed scale-invariant params; SE inflation for nonlinear undercoverage) in [`uncertainty.py`](src/sonolab/pipe2d/uncertainty.py); instrument-only `degenerate`/`refuse` on structural flags (fit fail, high residual, ill-conditioned, bounds, opp-look inconsistent) plus **`dentUncertain` (CI lo≤0)** as hard refuse; CI-τ overlap is a warning. Soft refuse-threshold tuning (2026-07-29) was **rejected** as a scorecard cheat. Honest follow-up: repair bound-saturated null `circle_ref` via τ-gauge + `c_init` (not mill \(R_\mathrm{nominal}\)), reduce `circle_ref` A inflation, recompute final `t_rmse` after post-hoc A. CLI `sonolab pipe-uncertainty-gate`.  
**Accept:** coverage gate on non-refuse poses (A/R, e/R ≥ 70%; θ0 ≥ 50%); sparse stress refuses at least as often in aggregate; FAR stays low. Evidence (2026-07-29/30, step 1°, 16 poses, strict refuse + gauge repair): primary N=20 recall **0.9375**, FAR=0, refuse_on=0, coverage θ0/A/R/e/R = 1.0 on 16 non-refuse; stress N=2 refuse_off=0.875. Pose 4 remains a razor-edge localization miss (~10.4° vs 10° tol) under pick/model bias even with repaired ref. Files: `uncertainty_gate.json`, `uncertainty_gate.png`, smoke `uncertainty_gate_smoke.json`. ✓  
**Deferred (7b):** dual-obs / closure inconsistency refuse once Block 2b exists.

### Block 8 - Portfolio demo narrative (cloud-first) - DONE (2026-07-29)

**Purpose:** Hiring artifact: storyboard from echoes → **dense interactive 3D wall** under unknowns, with truth overlay. Dents are *visible on the surface*, not a separate detector dashboard.  
**Build on:** `sonolab pipe-survey-3d` / `pipe3d` + joint Lab B + `c_cal`; dual acquire/display density.  
**Implemented:** Parallel Lab A looks; acquire default **1° × 16 z**; display mesh **0.25° × 64 z** (resampled fitted wall + analytic truth - not extra FDTD); Three.js mesh viewer with external `survey3d_view_data.js` / `.json`, recon/truth/both toggles, honest HUD (acquire vs display + `physics: 3D_scalar_acoustic`). CLI `sonolab block8-demo` → `datasets/runs/_block8_demo/`; smoke `sonolab block8-demo --fast` (15° × 6).  
**Accept:** Reproducible dense interactive cloud a non-expert can orbit; HUD states acquire vs display density; no commercial ILI mm / sensor-count claim (this scalar FDTD portfolio does not match field imaging density). Shape floors still reported via existing pipe3d score. Evidence: `datasets/runs/_block8_demo/` (full 1°×16), smoke `datasets/runs/_block8_demo_smoke/` (`--fast`). ✓  

---

## 7. Recommended order (quality-first)

```text
0 Vision (done; cloud-first focus locked 2026-07-29)
 → 1 Blind harness (done)
 → 2 Observation model that beats MC failure (done, 2a joint)
 → 3 Absolute-scale path via labeled c_cal (done)
 → 4 True 3D scalar PE survey + wall assembly (done)
 → 5 SPECFEM EXAMPLE 01 ToF fidelity gate (done)
 → 6 Aperture ablation / coherent DAS (done)
 → 7a Uncertainty / refuse (done)
 → 8 Demo package (cloud-first) — done
 → 7b Dual-obs refuse — when 2b exists (deferred)
```

Physics engines (pipe2d, wave3d, SPECFEM) stay in Lab A continuously; they are not “the product,” they are the **referee and data factory**.

---

## 8. Decisions

1. **Name:** Keep working title **Echo-to-Geometry**, or rename (e.g. SonoLab Echolocate)? *(open)*  
2. **Block 2 bet:** **(2a) parametric joint fit** - done.  
3. **Absolute scale:** labeled fluid \(c_{\mathrm{cal}}\) (Block 3) - done. Dual-obs (2b) deferred. Silent \(R_\mathrm{nominal}\) still **forbidden**.

**Resolved:** North star = in-pipe ultrasonic echo→**3D wall geometry** under unknowns; open research / portfolio; no pipe-prior cheat; no commercial vendor naming in public materials.


---

## 9. How we will work

- You ask questions / pick the next block.  
- We implement **only that block**, with tests and evidence.  
- No compromise: if a shortcut breaks the blind contract or silent-fails on the MC failure mode, we reject it.  
- Long horizon is fine; quality of the estimator under unknowns is the goal, not hosting SPECFEM as SaaS.

---

## 10. Pointers

- Joint Lab B estimator: `src/sonolab/pipe2d/joint_geometry.py`  
- Blind harness: `src/sonolab/pipe2d/blind.py`  
- Joint MC evidence: `datasets/runs/_pipe2d_joint/`  
- Block 6 aperture ablation: `src/sonolab/pipe2d/aperture_ablation.py`, evidence `aperture_ablation.json`  
- Block 7a uncertainty: `src/sonolab/pipe2d/uncertainty.py`, evidence `uncertainty_gate.json`  
- Legacy residual baseline / negative result: `wavelab-3d/practice/_monte_carlo_dent_results.json`  
- Broader sim platform plan: `specfem2d-UT/docs/SONOLAB_MASTER_PLAN.md`  
- Orientation: `datasets/runs/_orientation/ORIENTATION.md`
