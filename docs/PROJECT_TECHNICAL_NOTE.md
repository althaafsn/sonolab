# SonoLab — complete project and technical note

**Snapshot date:** 2026-10-08  
**Purpose:** Explain the project, the current code, the physics, the data flow, the changes, the evidence, and the limits.  
**Current presentation:** One 2D video. Two rows compare centered and off-center arrays. Each row shows the rotating array, a channel × time heatmap, and a wall heatmap. A fitted wall then appears over the wall heatmap.

This is open research and portfolio work. It is not a validated inspection instrument or a copy of a closed imaging system.

## Read this first

SonoLab separates two tasks:

1. **Forward simulation:** Choose a pipe, tool position, and pulse. Compute the waves and record echoes.
2. **Inverse estimation:** Give an estimator the recorded echoes and known instrument geometry. Estimate the pipe center and wall shape without giving it the true pipe geometry.

The current video connects these tasks. It does not draw a wall by tracing the brightest part of an image. The wall comes from echo arrival times and a circle-plus-dent model.

The blue cross is the actual pipe center from the simulation. It is a reference for display and error checks. The green ring is the estimated pipe center. The green contour is the estimated wall. Simulation truth does not enter this fit.

The main limits are important:

- The current video uses a 2D scalar wave model, not a fluid–steel elastic model.
- Sound speed is an explicit assumption: `0.45 cells/step`.
- The estimator assumes a roughly circular pipe with one cosine-shaped dent of a specified angular width.
- The wall heatmap uses an approximate path model. It is not full-matrix Total Focusing Method imaging.
- Two successful cases do not establish performance on arbitrary pipes or real data.

### Publication scope

This note describes the local code and saved results inspected on the snapshot date. The public repository was checked separately. At that time, its default branch did not contain `docs/demo/beamforming/` or the current showcase builder. Uploading this note does not upload the new code, video, or local run data.

Code paths below refer to that local snapshot. Local commands need those files. Do not assume that a clone of an older public revision has them. Local run data are excluded by `.gitignore`.

## Contents

1. [Objective and scope](#1-objective-and-scope)
2. [How the project reached this version](#2-how-the-project-reached-this-version)
3. [The ATLUS reference](#3-the-atlus-reference)
4. [Code structure](#4-code-structure)
5. [The two-lab boundary](#5-the-two-lab-boundary)
6. [Coordinates and parameters](#6-coordinates-and-parameters)
7. [Wave physics and the numerical method](#7-wave-physics-and-the-numerical-method)
8. [Array geometry and beamforming](#8-array-geometry-and-beamforming)
9. [Recorded channels and receive processing](#9-recorded-channels-and-receive-processing)
10. [Echo picks and distance](#10-echo-picks-and-distance)
11. [How the center and wall are estimated](#11-how-the-center-and-wall-are-estimated)
12. [Sound speed and scale](#12-sound-speed-and-scale)
13. [How the wall heatmap is made](#13-how-the-wall-heatmap-is-made)
14. [The current video](#14-the-current-video)
15. [Results and their meaning](#15-results-and-their-meaning)
16. [Repairs and cleanup](#16-repairs-and-cleanup)
17. [Other work already in the package](#17-other-work-already-in-the-package)
18. [Verification](#18-verification)
19. [Files and reproduction](#19-files-and-reproduction)
20. [What is not established](#20-what-is-not-established)
21. [Next technical work](#21-next-technical-work)
22. [How to explain the project](#22-how-to-explain-the-project)
23. [Terms and references](#23-terms-and-references)

## 1. Objective and scope

The long-term objective is to recover a 3D inner-wall surface from ultrasonic echoes in a pipe. Tool pose, fluid sound speed, and wall shape can be unknown. Dents and material loss should be inspected from the recovered geometry, not from a separate yes/no score alone.

The locked project direction is in [ECHO_TO_GEOMETRY_VISION.md](ECHO_TO_GEOMETRY_VISION.md). The local `docs/SONOLAB_MASTER_PLAN.md` gives the wider development plan.

The current presentation has a smaller purpose: show the 2D echo-to-wall process and the effect of an off-center tool. This keeps a short demonstration readable. It does not replace the wider 3D objective.

The original ambition included a complete simulation of an industrial ultrasonic inspection problem. The current work is an open model of part of that problem. It does not reproduce proprietary hardware, processing, acquisition schedules, or validated field performance.

The initial request also used the term “Kraken.” The active showcase does not call an engine with that name. It calls SonoLab's NumPy scalar FDTD class. The wider package has optional Rust and SPECFEM paths. These are separate facts, not interchangeable engine names.

## 2. How the project reached this version

The presentation changed in response to a clear product decision: make a showcase, not a full application.

| Stage | Decision or change |
|---|---|
| Initial scope | Research ultrasonic arrays, transmit timing, echoes, and pipe geometry. Inspect the existing code and the supplied ATLUS note. |
| Simpler presentation | Remove application-like controls and long explanations. Keep a large video player and short labels. |
| One main flow | Remove extra recording choices. Keep the 2D process as the current story. |
| Off-center case | Compare the same pipe with a displaced tool. |
| Both cases | Keep centered and off-center cases together, with common scales. |
| Wall inference | Fit a wall from echoes rather than trace a heatmap ridge. |
| Final layout | Keep three panels in each of two rows. Put the fitted wall over the wall heatmap. |
| Center reference | Show the known simulation center separately from the fitted center. |
| Middle-panel correction | Replace the temporary single energy curve with the requested 20-channel × time heatmap. |

There was a reported mismatch in which an older line graph was still visible. The current local MP4 and the MP4 served by the local preview had identical SHA-256 hashes. Their inspected frames contained the channel heatmap. An older loaded video was a possible cause, but it was not proven on the user's browser. A fresh media URL was supplied. The public-site version must also be distinguished from the local version.

This record describes the changes. It does not assign every earlier fault to an agent or person. The package is untracked in a larger local workspace, so that workspace's Git history does not establish authorship of these files.

## 3. The ATLUS reference

ATLUS means **Automotive Tandem Linear Ultrasonic Series-Array**. The supplied note interprets UBC ECE poster `UBC_ECE_071.pdf`. The project authors listed in that note are Brad MacNeil, Charles Clayton, Jeevan Samra, Leo Yuan, and Richard Shi.

The original poster was not independently inspected for this handover. The following are reported by the supplied interpretation, not measured SonoLab results.

The original purpose was to map objects in a 2D plane where visible-light cameras can fail, including darkness, fog, or smoke. The note describes MATLAB waveguide optimization as performed and the custom pulser as built. It describes the processor/FPGA acquisition architecture, but supplies no firmware listing or throughput benchmark. A separate proprietary 3D processing capability is mentioned; its algorithms are not given. Do not turn that statement into a claim that the student array demonstrated validated 3D imaging.

### Reported design

- Eight ultrasonic transmitter/receiver elements in a linear array.
- Airborne 2D object detection and mapping.
- Approximately 59 kHz transmit frequency.
- A MATLAB-optimized, 3D-printed acoustic waveguide.
- A taper described as `2.4λ → 1λ`.
- A custom eight-channel high-voltage pulser, with drive up to `±70 V`.
- An AFE5809 receive front end.
- A Xilinx MicroZed processor/FPGA control path.
- A reported 10 MHz acquisition rate and storage on an SD card.

The AFE5809 is an eight-channel ultrasound receive front end. Its official documentation describes amplification, filtering, ADC conversion, and digital I/Q demodulation. This supports the component's general role, not the performance of the student system. [Official AFE5809 information](https://www.ti.com/product/AFE5809).

The component's available modes do not establish the exact mode, decimation, signal format, or board configuration used by ATLUS. In particular, do not assume an analog demodulation-before-ADC chain solely from a simplified system diagram.

### Timing idea

```text
Target coordinates → compute relative element delays → transmit pulse burst
→ wait t_min → listen for t_en → wait t_interline → next target
```

The supplied note also lists `n_pulses`. It does not give numerical settings for these timing intervals.

### What was an objective, not a proven result

- Detection of a 17 mm object at 2 m.
- Combining multiple units into a 2D array.
- Initial use as a laboratory tool.

The supplied poster interpretation does not include a detection experiment, a complete throughput result, or a quantitative reconstruction benchmark that proves these goals were reached. Its “four samples per cycle” statement is not clearly reconciled with 10 MHz sampling and 59 kHz transmit frequency.

### Why the waveguide matters

Wavelength is `λ = c/f`. A high frequency reduces wavelength. Large physical elements can then be too far apart relative to wavelength. Unwanted interference beams, called grating lobes, can result. A pitch near or below half a wavelength is a common conservative design rule for wide steering; the full limit depends on the steering range and element pattern.

The waveguide was intended to change the effective acoustic arrangement despite the physical element size. A pitch of one wavelength is not a universal guarantee that grating lobes disappear. The note also reports internal losses and ringing. Those require a delay before receive acquisition. No quantitative insertion loss or settling time is supplied.

### What transfers to SonoLab

The transferable ideas are timed sources, multiple receive channels, transmit/receive separation, and echo-based mapping. The numerical values do not transfer directly. An airborne 59 kHz system is not the same as fluid-coupled pipe inspection. SonoLab does not simulate the pulser, AFE, FPGA, SD card, or waveguide in the current clip.

## 4. Code structure

| Local path | Responsibility |
|---|---|
| `scripts/build_showcase_videos.py` | Acquire both current cases, run the fit, score it, and render the comparison video. |
| `scripts/build_circular_comparison.py` | Define `CircularPhasedPipe2D`, the curved array used by the video. |
| `src/sonolab/pipe2d/phased_array.py` | Define the 2D grid, source, pipe masks, FDTD step, and receive samples. Also contains older teaching modes. |
| `src/sonolab/pipe2d/acquire.py` | Supply the blind echo picker and wider Lab A acquisition utilities. |
| `src/sonolab/pipe2d/blind.py` | Define instrument-only acquisition packs, separate truth packs, and scoring. |
| `src/sonolab/pipe2d/joint_geometry.py` | Estimate a circle-plus-dent wall from echo times. |
| `src/sonolab/pipe2d/uncertainty.py` | Add approximate uncertainty and refusal flags. |
| `scripts/envelope_sampling.py` | Interpolate an envelope at fractional sample times. Return zero outside the record. |
| `src/sonolab/pipe3d/` | Separate 3D scalar acquisition, station fitting, wall assembly, and display. |
| `crates/sonolab_pipe2d_fdtd/` | Optional Rust acceleration for the wider 2D Lab A path. |
| `src/sonolab/specfem_gate.py` | Compare an optional SPECFEM example echo with an analytic travel time. |
| `docs/demo/beamforming/` | Current local single-player showcase and its media. |
| `scripts/serve_showcase.py` | Local static preview with HTTP byte-range support. |
| `scripts/check_console.cjs` | Browser checks for the showcase. |
| `tests/` | Package and regression checks. |

The nested SPECFEM solver sources were not rewritten for this showcase. SPECFEM is not the engine that produced the current MP4.

## 5. The two-lab boundary

```text
Lab A: known pipe + known pose + known wave model
       │
       ├── per-channel RF and coherent receive signal
       │       │
       │       ├── illustrative wall heatmap
       │       │
       │       └── blind echo picks → AcquisitionPack → Lab B fit
       │                                           │
       │                                           └── fitted center + wall
       │
       └── separate truth → display reference and score only
```

Lab A must know the true scene to simulate it. That is not a fault. The important boundary is what Lab B receives.

For the current fit, the acquisition pack contains look angles, echo times, pulse timing, element count, effective array offset, and aperture facts. The fit receives an explicit `c_cal` as a separate argument.

The fit does not receive the true pipe center, nominal radius, tool offset in the simulated world, or true dent map. It also does not receive the heatmap.

“Blind” does not mean “no assumptions.” The model family, dent half-width, instrument geometry, parameter bounds, and supplied sound-speed assumption remain known or chosen inputs. The forward and inverse models use the same dent family. This makes the example easier than an unknown real surface.

Older interactive functions in `phased_array.py` include known-radius calibration and nominal-radius gates. Those are not the current showcase inference path. Their existence must not be used to claim that every function in the package has the same blind contract.

## 6. Coordinates and parameters

### Angle convention

The look unit vector is:

$$
\mathbf{u}(\theta)=(\sin\theta,\cos\theta).
$$

Thus, `0°` points along `+Y`, `90°` along `+X`, `180°` along `−Y`, and `270°` along `−X`.

The video uses tool coordinates. The tool is at `(0,0)`. Do not confuse these with simulation-grid coordinates.

### Actual centers

The simulation pipe center is `(95.5,95.5)` in the 192 × 192 grid. The off-center tool is shifted eight cells along `+X`.

| Case | Tool relative to actual pipe center | Actual pipe center in tool coordinates |
|---|---|---|
| Centered | `(0,0)` | `(0,0)` |
| Off-center | `(+8,0)` | `(−8,0)` |

The sign is important. Moving the tool right makes the pipe center appear left in the tool frame. “Pipe center” here means the reference circle center or pipe axis. It is not the centroid of the dented outline.

### Current fixed example

| Parameter | Value | Meaning |
|---|---:|---|
| Domain | 192 × 192 cells | Numerical grid, not physical dimensions. |
| True reference inner radius | 68 cells | Lab A geometry only. |
| True dent depth | 8 cells | Lab A geometry only. |
| True dent direction | 0° | Dent points inward near `+Y`. |
| Dent half-width | 22° | Also an explicit fixed shape assumption in Lab B. |
| Array elements | 20 | Point-source/sample model, not 20 modeled piezoelectric devices. |
| Array radius | 22 cells | Distance of elements from the tool origin. |
| Active array arc | 70° | Elements span `−35°` to `+35°` about the commanded look. |
| Adjacent element chord pitch | About 1.4144 cells | Curved-array instrument geometry. |
| End-to-end chord span | About 25.2374 cells | Not the 38-cell span of the separate linear-array study. |
| Fluid propagation speed | 0.45 cells/step | Lab A scalar model speed. |
| Supplied `c_cal` | 0.45 cells/step | Explicit simulation assumption, not a measured fluid calibration. |
| Source angular frequency | 0.28 rad/step | Not 0.28 Hz or 0.28 MHz. |
| Source envelope center | Step 22 | The code calls this `PULSE_WIDTH`; it is used as the time origin. |
| Source envelope standard deviation | 7.7 steps | `22 × 0.35`. |
| Source amplitude | 0.40 | Uncalibrated numerical amplitude. |
| Receive record per look | 360 steps | Same window for both cases. |
| Acquire angle spacing | 1° | 360 independent looks per case. |
| Display angle spacing | 10° | 36 looks shown in the video. |
| Wave snapshots per displayed look | 24 | One saved field every 15 simulation steps. |
| Effective ranging offset | About 22.4886 cells | Approximate central TX/RX midpoint offset. |

No cell size in metres or time step in seconds is specified for this clip. Therefore, the numbers do not establish a physical frequency, millimetre resolution, drive voltage, or ADC sample rate.

## 7. Wave physics and the numerical method

### Scalar wave model

Inside the uniform fluid region, the code approximates:

$$
\frac{\partial^2 u}{\partial t^2}=c^2\nabla^2u.
$$

Here `u` is an oscillating scalar wave field. Treat it as a pressure-like simulation variable, not a calibrated pressure in pascals. The code does not solve particle velocity, a density-dependent interface system, solid stress, or shear displacement.

### FDTD update

FDTD means finite-difference time-domain. With unit grid spacing and time step, the interior update has the form:

$$
u^{n+1}_{i,j}=2u^n_{i,j}-u^{n-1}_{i,j}
+\nu_{i,j}^2\left(u^n_{i+1,j}+u^n_{i-1,j}+u^n_{i,j+1}+u^n_{i,j-1}-4u^n_{i,j}\right).
$$

The code uses three arrays for the previous, current, and next fields. Its fluid Courant number is `ν = 0.45`. This is below the usual `1/√2` stability bound for this uniform 2D five-point scheme. Stability alone does not prove accuracy.

The source is added at grid points before each update. A pulse source has the form:

$$
s(t)=0.40\exp\left[-\frac12\left(\frac{t-22}{7.7}\right)^2\right]\sin(0.28t).
$$

The curved-array showcase sets all element delays to zero. Each look starts from a new zero-field simulation. The code does not carry the previous look's ringing into the next look or simulate a physical motor during acquisition.

### Wavelength

For this grid model:

$$
f=\frac{0.28}{2\pi}\approx0.0446\ \text{cycles/step},\qquad
\lambda=\frac{0.45}{f}\approx10.1\ \text{cells}.
$$

The grid thus has about ten cells per wavelength. Source placement and receive sampling are rounded to grid points. Numerical dispersion and staircase boundaries can affect timing. A mesh-convergence study is still needed for an accuracy claim.

The current curved-array chord pitch is below half this grid wavelength. This does not solve real transducer packaging: the elements are point sources in a numerical grid, not physically sized devices.

### The wall is not a steel interface

The current solver forces the next field to zero in the wall annulus and exterior cells:

```text
un[wall] = 0
un[exterior] = 0
```

The metadata calls this `2D_scalar_acoustic_hard_wall`. That is a code label for an enforced boundary. It must not be read as a physically rigid steel wall. For a pressure-like variable, a zero-value Dirichlet boundary is pressure-release-like; it is not the rigid-wall zero-normal-velocity condition.

The code defines a different wall speed, but clamping the wall field to zero prevents it from being a transmitting steel layer. The current clip therefore does not establish steel impedance reflection, refraction, mode conversion, outer-wall echoes, or thickness measurement.

The field can show propagation, interference, returning waves, and repeated reflections within this bounded scalar scene. Their real amplitudes and phases are not calibrated. No measured attenuation or transducer response is included.

The drawn tool ring is a geometry reference, not a modeled solid body. The metadata states `tool_body_boundary=false`. Waves can pass through the tool's drawn interior. Tool-body reflection, acoustic shadow, and coupling layers are not part of this clip.

## 8. Array geometry and beamforming

Multiple source waves add. Relative timing can make them add more strongly in one direction or at one point. This is the basic beamforming idea. The official k-Wave steering example shows delayed tone bursts across a linear array. It is a reference for the idea, not the engine used by this clip. [Steering a linear array](https://www.k-wave.org/documentation/example_tvsp_steering_linear_array.php).

`CircularPhasedPipe2D` places 20 elements on a curved arc around the tool. It supports a delay-corrected mode. For a forward reference plane, an edge element has farther to travel than the apex. It must fire earlier. The repaired delay law is:

$$
\tau_i=\frac{r_\mathrm{tool}\left(\cos\alpha_i-\cos\alpha_\mathrm{half}\right)}{c}.
$$

Edges have zero delay; the apex is delayed. Tests check equal predicted arrival times at the forward plane.

**The current video does not enable this mode.** It sets `delay_corrected=False`, so all 20 elements fire together. The array orientation changes between looks. Do not describe this clip as a demonstration of an optimized transmit focus or electronic steering on a fixed array.

The receive class supports delay-aligned averaging. With the current zero transmit delays, that average reduces to a simple mean across channels. A larger aperture can improve direction selection, but it can also introduce unwanted lobes and path differences. The current model is not a finite-element transducer model.

## 9. Recorded channels and receive processing

At each step, the class samples each element about one grid cell forward along the look direction. It stores those values in a per-element receive history.

There is one implementation fallback: if every instantaneous element sample is zero, the class uses a sample near the tool and duplicates it across channels. The note must not present every stored value as a physical hardware measurement. This is a simulation utility, not a receiver electronics model.

For displayed looks, the saved channel tensor has shape:

```text
[look, channel, time] = [36, 20, 360]
```

The middle video panel shows one `20 × 360` slice for the current look. Its vertical axis is channel number, not angle. Its horizontal axis is time step. Color is signed RF amplitude, not measured energy or probability. A moving time cursor reveals the record.

For the fit, the receive samples are averaged into one coherent signal per acquired look:

$$
S_\theta(t)=\frac{1}{20}\sum_{j=1}^{20}RF_{\theta,j}(t).
$$

This tensor has shape `[360,360]`: look × time. It is not the middle channel heatmap. A check confirmed that the means of the saved per-channel records reproduce the matching coherent records to an absolute tolerance of `1e-12`.

The instrument exclusion interval is `TX_EXCLUDE = int(22 × 1.8) = 39`. The display sets samples `0…38` to zero. The picker starts after that interval. The stored raw records retain the original samples. This is software blanking, not a simulated high-voltage transmit/receive switch.

The official k-Wave B-mode example also separates receive processing, input-signal removal, and envelope detection. SonoLab uses a much simpler processing chain. [Receive processing reference](https://www.k-wave.org/documentation/example_us_bmode_linear_transducer.php).

## 10. Echo picks and distance

### Current picker

The showcase calls `blind_pick_echo(trace, prev_t=None)` independently for each look. It does not use a nominal pipe radius or track the preceding angle in this path.

After the transmit exclusion:

1. Take the absolute value of the signal.
2. Estimate a small early-window noise level.
3. Set a threshold to the larger of `4 × noise + 1e-6` and `0.15 × record peak`.
4. Select the earliest local peak above the threshold.
5. If none exists, use the first above-threshold sample or the window maximum.

The metadata calls this a blind first-break pick. More precisely, its main rule is the earliest qualifying local peak. It can select the wrong event in multipath or low signal-to-noise conditions. The broader acquisition module has smoothing and repair functions, but the showcase does not call those functions.

### Time to range

For a central pulse-echo path:

$$
r(\theta)\approx a+\frac{c_\mathrm{cal}}{2}\left[t_\mathrm{echo}(\theta)-t_0\right].
$$

`r` is distance from the tool origin, `a` is the effective instrument offset, and `t0 = 22` is the declared pulse origin. Division by two accounts for the outward and return paths.

The effective offset is the array's largest forward projection plus half a cell. The half cell accounts approximately for the receive location one cell ahead of TX. It does not use the true pipe radius or pose. It is a reduced model of a finite curved aperture, not an exact path for every transmit/receive pair.

Example, using rounded `a = 22.5`: an echo at step 222 gives `r ≈ 22.5 + 0.45 × 200/2 = 67.5 cells`. This is a ranging example, not a separate test result.

The pulse is extended in time, and the picker uses an early peak rather than an exact physical interface impulse. Pulse-origin, aperture, and discretization bias can remain. These biases help explain why a successful fit is not exact.

## 11. How the center and wall are estimated

### Why the tool origin is not the pipe center

For a circular pipe with center `C=(cx,cy)` in tool coordinates, a ray `r u` meets the circle at:

$$
r=C\cdot\mathbf{u}+\sqrt{R^2-\lVert C\rVert^2+(C\cdot\mathbf{u})^2}.
$$

An off-center tool has shorter ranges toward the near wall and longer ranges toward the far wall. For a circle along the displacement axis, the opposing range difference is twice the displacement. A dent and finite-aperture echoes disturb that simple relation, so the code fits the full angular pattern rather than use one pair as the final answer.

### Wall family

The pipe-centered wall model is:

$$
R_\mathrm{wall}(\phi)=R-b(\phi),\qquad
b(\phi)=\frac{A}{2}\left[1+\cos\left(\pi\frac{\Delta\phi}{h}\right)\right]
$$

inside `|Δφ| < h`; outside that interval, `b=0`. `A` is inward dent depth, `theta0` is its direction, and `h=22°` is fixed in this example. The code iterates the ray intersection because the hit angle about the pipe center differs from the look angle about an off-center tool.

The main geometry parameters are `R, cx, cy, A, theta0`. Fluid speed is a sixth parameter in the free-speed mode. The current showcase supplies `c_cal`, so speed is constrained near that value during fitting and reported as the supplied value.

### Fit process

The existing joint estimator uses these stages:

1. Use opposite-look times to initialize a radius/time scale.
2. Convert approximate ranges to points and initialize a free-center circle.
3. Fit a circle with dent amplitude held near zero.
4. Use early residuals to seed a dent direction.
5. Try multiple dent directions while the circle is held nearly fixed.
6. Polish the full joint model, unless a separate data-derived reference circle was supplied.
7. Apply the existing residual/template processing and uncertainty checks.

The core objective compares predicted and picked echo times, with parameter bounds and a small dent-amplitude regularizer. SciPy `least_squares` supplies the numerical optimizer.

**Dense acquisition is not dense optimization at every stage.** The current pack has 360 looks, but `fit_stride = max(1, n//72)`. With 360 looks, the least-squares residual uses every fifth look, or 72 looks. Full-resolution data still enter initialization, dent-direction processing, final wall prediction, and final time RMSE. Saying “the fit uses 360 looks” is acceptable only with this distinction.

### The output is not only a least-squares solution

The estimator also removes a second angular harmonic from residuals, compares cosine templates, and blends matched-filter and peak estimates for dent magnitude. The normal path includes an empirical `1.35` magnitude factor. Some dent-call and eccentricity diagnostics also use thresholds or blends.

These are chosen processing rules, not pure physical laws. They affect the final reported wall. The final time RMSE is recomputed after the amplitude adjustment. The output field `e = sqrt(cx²+cy²)` is the fitted center displacement; the blended `e_over_R` diagnostic is not always exactly `e/R`. Do not use that blended diagnostic to place the center marker.

### Drawing the fitted wall

`fitted_wall` calls `estimate_from_pack`, then evaluates `predict_standoff` using only the estimated parameters. It draws:

$$
(x,y)=r_\mathrm{pred}(\theta)(\sin\theta,\cos\theta).
$$

There is no shift using the true pose and no heatmap-ridge tracing. The green center ring uses the fitted `(cx,cy)` directly.

## 12. Sound speed and scale

Echo time is not a distance until a propagation speed or another scale anchor is supplied. SonoLab permits an explicit `c_cal`. In this video, it is an assumed simulation speed, not a measured calibration.

If all lengths and speed are free, the ideal pulse-echo problem has a scale ambiguity: scaling distance and speed together leaves travel times unchanged. A fixed known instrument offset modifies that exact symmetry and can weakly constrain scale, but it does not justify treating a poorly conditioned free-speed fit as a trusted physical measurement.

The project therefore reports ratios and time scale in free-speed mode, and requires a labeled anchor for absolute geometry. The current output is absolute in **grid cells**, not millimetres.

With a fixed instrument offset, a speed error changes the simple converted range as:

$$
r'=a+\frac{c'}{c}(r-a).
$$

It is not generally correct to say that every fitted length scales by exactly the same factor when the known offset stays fixed.

In real data, the fitted center remains an estimate relative to the tool. An exact actual center needs an independent reference measurement for validation. Neither this fit nor its blue simulation cross supplies a real-world survey position.

## 13. How the wall heatmap is made

The right heatmap is a separate display calculation. For each of 36 displayed looks:

1. Compute the Hilbert envelope of each channel.
2. Approximate TX by the midpoint of the two central elements.
3. For each candidate pixel, add the distance from that TX point and the distance back to each receive point.
4. Convert total path length to a sample time and include the pulse origin.
5. Sample each envelope with fractional interpolation.
6. Sum the sampled envelopes across channels.
7. Apply a Gaussian angular display weight of 12° and exclude the region within 28 cells of the tool.
8. Add that look to the accumulated map once.

The lookup time for channel `j` is:

$$
t_j(x,y)=t_0+\frac{\lVert(x,y)-TX\rVert+\lVert(x,y)-RX_j\rVert}{c_\mathrm{cal}}.
$$

The image grid is 160 × 160 over `−84…84` cells in each tool-coordinate direction. Lookup times outside a channel record contribute zero; they are not clamped to the final sample.

This display uses envelope magnitudes, an approximate central transmit path, and chosen angular weights. Brightness is not a calibrated acoustic energy, confidence value, or exact surface probability.

It is not FMC, which requires distinct transmit events and a transmit × receive × time data matrix. It is not full coherent TFM over those transmit/receive pairs. Some older filenames contain `tfm`; filenames alone do not establish that method.

The wall fit is independent of this image. Agreement between the line and image is useful visually, but the real geometry check compares the fitted ranges with separate simulation truth.

## 14. The current video

```text
                 Left                 Middle                  Right
Centered         Rotating array       20 channels × time      Wall heatmap + fitted wall
Off-center       Rotating array       20 channels × time      Wall heatmap + fitted wall
```

Both rows stay visible. The pipe, pulse, aperture, time window, sound-speed assumption, and estimator are the same. Only tool position changes. Both rows also use common field, RF, and wall-map color scales, and common spatial bounds.

The wavefield snapshots are computed fields, not hand-drawn propagation. Brief transitions interpolate element positions between displayed looks; they do not interpolate physical wave propagation. Each new look starts a separate simulation.

During the scan, the channel record fills with time and the right map gains one completed look. After the scan, the fitted contour is drawn over that map. The final drawing does not imply that a partial contour was fitted in real time; acquisition and fitting occur before video rendering.

The left panel includes the known simulation wall. The right panel includes the known center marker, but not a copied truth contour. The fitted contour remains independent.

The page has one native player, an expandable model note, and a small footer. There are no scene selectors, extra videos, application sidebars, or a 3D link on this showcase page. The wider 3D work remains separate.

The local clip is H.264 MP4, 1280 × 720, 30 fps, and 38.3 seconds long. It has no narration. A video is a saved result, not a solver running inside the browser.

## 15. Results and their meaning

These values are from the current saved run, not from real inspection data:

| Quantity | Centered | Off-center |
|---|---:|---:|
| Actual center in tool frame | `(0,0)` | `(−8,0)` |
| Estimated center | `(0,−0.410931)` | `(−8.343895,−0.027115)` |
| Center error | 0.410931 cells | 0.344962 cells |
| True reference radius | 68 cells | 68 cells |
| Estimated reference radius | 68.332786 cells | 67.812185 cells |
| True dent depth | 8 cells | 8 cells |
| Estimated dent depth | 9.082017 cells | 10.219856 cells |
| True dent direction | 0° | 0° |
| Estimated dent direction | 0° | 352.5062° |
| Wall-range RMSE | 0.511627 cells | 1.323465 cells |
| Final echo-time RMSE | 4.555600 steps | 6.585682 steps |
| Refused fit | No | No |

The off-center direction estimate is about `−7.49°` from truth, not a 352.51° error. Angular differences must wrap around 360°.

Wall-range RMSE is the square root of the mean squared difference between predicted ranges and analytic reference ranges from `true_standoff_from_tool` at matched look angles. It is not an independent boundary pick from the rasterized wave field, an arbitrary nearest-surface distance, or a measured field accuracy. The reference uses the known ideal geometry; the numerical wall is grid-discretized. Center error is the Euclidean distance between fitted and true centers in the same tool frame.

The builder rejects this bounded demo if either fit fails, is refused, or has wall-range RMSE above two cells. This is a demonstration check, not a general acceptance specification.

The center result is good in these two cases, but dent depth is overestimated, especially off-center. Do not summarize the result as “everything is exact.” Agreement of the model family with the simulated dent also limits how much can be inferred about unseen shapes.

## 16. Repairs and cleanup

The repair work addressed both numerical behavior and presentation:

| Area | Change | Why it matters |
|---|---|---|
| Curved-array timing | Corrected the delay sign in curved-array builders. | Edge elements must fire earlier for a common forward plane. |
| Speed in delay calculations | Used propagation speed rather than the unscaled base constant. | Delay units must match the FDTD time scale. |
| Wavelength explanation | Corrected the value to about 10.1 cells. | Frequency is angular frequency in rad/step. |
| Image time origin | Included the pulse-origin offset. | Ignoring it shifts mapped reflector positions. |
| Image sampling | Added fractional interpolation and zero outside the record. | Clipping late arrivals to the last sample can create false image structure. |
| Accumulation | Removed duplicate contributions in an older unified renderer. | Each look must contribute once. |
| Claims and labels | Separated illustrative envelope maps from FMC/TFM claims; corrected radius versus diameter labels. | Display names must match the data and method. |
| Wall inference | Used the existing instrument-only joint fit. | The final wall is not a copied boundary or image ridge. |
| Coordinates | Kept inferred geometry in the tool frame. | True pose must not secretly register the estimate. |
| Middle panel | Restored actual channel × time RF after a temporary energy-curve version. | A combined trace is not a channel heatmap. |
| Video layout | Used two rows and three persistent columns; moved the wall onto the map. | The same process can be compared in both cases. |
| Label spacing | Added a bounding-box regression test. | Header, chart labels, and legend must not collide. |
| Preview seeking | Added byte-range responses. | Native video seeking needs partial-file delivery. |
| Test collection | Scoped pytest to `tests/`. | Nested external solver utilities are not SonoLab unit tests. |

Unused page controls and duplicate labels were removed. Old unlinked media were moved to recoverable local archives, not permanently deleted. Archive directories include `showcase-archive-zP1FOPIm` and `showcase-layout-archive-lUfCj6s0` under `datasets/runs/`.

Some older builder scripts remain because they contain reused classes or regression targets. Removing them blindly would break useful code. The old free-center-circle plus residual-dent path is a failed baseline, not the preferred reconstruction method.

## 17. Other work already in the package

The current clip is not the whole codebase. The following paths and saved evidence already exist. They were inspected here; the expensive full experiments were not rerun for this documentation task.

| Work | What exists | Limit of the evidence |
|---|---|---|
| Blind packs and joint fit | Instrument-only `AcquisitionPack`, separate `TruthPack`, geometry fit, and tests. | Model assumptions still apply. |
| Rust Lab A | Optional FDTD extension and saved benchmarks. | Extension is absent in the current environment; its tests skip. The video uses NumPy. |
| Absolute scale | Explicit `c_cal`, absolute geometry outputs, and synthetic tests. | A supplied value is not automatically a measured calibration. |
| SPECFEM example gate | An example side-drilled-hole pulse-echo result compared with analytic time. | A canned example is not a validated pipe model. |
| Aperture study | Separate linear-array comparisons for 2, 8, and 20 elements. | Do not transfer their metrics directly to the current curved array. |
| Uncertainty/refusal | Local-linearization intervals and instrument-only refusal rules. | Not a Bayesian posterior or a guarantee of coverage on all data. |
| 3D scalar wall | Multi-station acquisition, joint fits, point cloud, radius map, and interactive viewer. | Scalar acoustics, not elastic field NDT. |

### Saved historical results

- A dense 2D joint study used 16 poses, 1° looks, and tool offsets up to eight cells. Its saved metrics report 15/16 localized detections, no false alarms in that study, median direction error 4.275°, and `A/R` RMSE 0.047415. These are synthetic study results with a specified threshold and protocol, not universal detection performance.
- The separate aperture study reports recall 0.0625 for two elements and 0.9375 for 20 elements under that study's definition. This supports an aperture effect in that model. It does not establish a commercial sensor count or resolution.
- The saved SPECFEM example has a picked time of 7.98 µs and an analytic time of 7.73109 µs. Relative error is about 3.22%, below the chosen 10% gate.
- The saved full Block 8 run acquires `1° × 16` axial stations and displays `0.25° × 64`. It reports wall-range RMSE 1.027454 cells, or 0.046702 of its 22-cell reference radius. Its scene and parameters differ from the current 2D video.

Display resampling adds mesh points, not new measured information. The 3D viewer can show fitted geometry and analytic truth separately. That separation must remain clear.

The approximate confidence intervals use a local Jacobian, residual variance, a scale-aware parameterization, and chosen interval inflation. Refusal checks include fit failure, high residual, inconsistent opposite looks, bound saturation, poor conditioning, and some uncertain dent cases. `refuse=false` means those checks passed; it does not prove that the physical model is correct.

The project vision labels its eight foundation blocks as completed. Read that as completion of the documented research blocks and their chosen gates, not completion of an industrial inspection system. Dual-observation inference and elastic 3D pipe work remain future work.

## 18. Verification

Checks rerun for this note on 2026-10-08:

| Check | Observed result |
|---|---|
| `.venv/bin/python -m pytest -q -rs` | 35 passed, 2 skipped. |
| Rust skip reason | `sonolab_pipe2d_fdtd` is not installed. |
| `.venv/bin/python -m compileall -q src scripts` | Passed. |
| Browser showcase check | Passed: one player, two rows in metadata, channel/time display, wall overlay, centers, fit records, play/pause/seek, responsive widths, reduced motion, notes, and error state. |
| Full MP4 decode | Passed. |
| Media metadata | 1280 × 720, 30 fps, 38.3 seconds. |
| Saved channel tensor check | Both are `36 × 20 × 360`; means match coherent RF at corresponding looks to `1e-12`. |

Scan, final overlay, desktop, and mobile frames were visually inspected during the showcase work. The tests cover delayed-array plane arrivals, fractional sampling, out-of-record behavior, pulse origin, preview byte ranges, blind fit behavior, refusal, center-coordinate signs, channel preservation, and label spacing.

The pulse-origin image regression targets the older `build_tfm_reconstruction_demo.compute_tfm_image` path. The current showcase's image-time formula is inline in its builder. It was inspected, but it has no direct isolated pulse-origin regression test. The helper tests do not by themselves validate every current image pixel.

The blind regression changes unrelated truth-like metadata in an acquisition pack and checks that the fitted wall stays unchanged. This is useful evidence, but not a proof that all future code paths cannot leak truth.

Some tests use synthetic signals from the same reduced model as the estimator. They are unit checks. They do not replace independent full-wave benchmarks or real measurements. Full Rust studies, the full 3D survey, and the optional SPECFEM simulation were not rerun for this note.

## 19. Files and reproduction

### Current presentation files

```text
docs/demo/beamforming/
  index.html
  console.css
  console.js
  showcase_flow.mp4
  showcase_flow.png
  showcase_meta.json
```

The current saved local run is `datasets/runs/showcase-offcenter-uls6lhey/`. It has `centered/` and `off-center/` subdirectories. Each contains:

```text
rf.npz             360 coherent receive traces, each 360 samples
channel_rf.npz     36 displayed looks × 20 channels × 360 samples
acquisition.json   instrument facts and picked echo times
truth.json         simulation ranges and center, for scoring/display only
estimate.json      fitted parameters, predicted wall, and diagnostics
```

Wave snapshots and image contributions are held in memory while building. They are not all saved in that run directory. A full video rebuild reruns acquisition.

`showcase_meta.json` is the small public-facing record of settings, layout, scores, and the local run path. A random run directory name can change on each rebuild.

### Rebuild the current full comparison

From a local checkout that contains the current builder, with the project environment and FFmpeg already available:

```bash
.venv/bin/python scripts/build_showcase_videos.py --c-cal 0.45
```

This acquires 360 looks per case and replaces the showcase media and metadata. Archive media you need before rebuilding. The builder writes a new run directory.

A cheaper smoke run is:

```bash
.venv/bin/python scripts/build_showcase_videos.py --c-cal 0.45 --step-deg 10
```

It uses 36 acquired looks, not the full 360-look evidence. Its output overwrites the same media targets. Do not publish smoke output as the full study.

### Preview

```bash
python3 scripts/serve_showcase.py
```

Open `http://127.0.0.1:8095/demo/beamforming/` on the same computer. This is not a public recruiter link. The server supports byte ranges for seeking. An already running server can make that port unavailable; it also accepts `--port`.

If an old video remains loaded, open the MP4 with a new query value or reload the page. Check the actual served file before concluding that caching is the cause. The local preview and a published website can contain different revisions.

### Checks

```bash
.venv/bin/python -m pytest -q -rs
.venv/bin/python -m compileall -q src scripts
node --check scripts/check_console.cjs
node --check docs/demo/beamforming/console.js
ffmpeg -v error -i docs/demo/beamforming/showcase_flow.mp4 -f null -
ffprobe -v error -show_entries stream=width,height,r_frame_rate \
  -show_entries format=duration,size -of json docs/demo/beamforming/showcase_flow.mp4
```

The browser check uses an existing Playwright installation:

```bash
SONOLAB_PLAYWRIGHT=/absolute/path/to/existing/node_modules/playwright \
  node scripts/check_console.cjs
```

The path is a placeholder, not a supplied installation. No dependency was installed for this documentation task. The local environment used Python, NumPy, SciPy, Matplotlib, pytest, Node, FFmpeg, and an existing Playwright/Chromium installation.

## 20. What is not established

The current work does not establish:

- Millimetre resolution, physical bandwidth, maximum inspection speed, or a field detection probability.
- Pressure or energy calibration, voltage-to-pressure conversion, or a modeled piezoelectric response.
- AFE gain, noise, filtering, quantization, overload recovery, or FPGA timing fidelity.
- Real fluid sound-speed calibration.
- A realistic fluid–steel boundary, shear waves, mode conversion, crack scattering, or wall thickness.
- Arbitrary wall shapes, multiple independent dents, severe tool offsets, tilt, or changing pose.
- A robust wall picker under arbitrary multipath, low signal-to-noise ratio, missing channels, or coatings.
- FMC/TFM in the current video.
- An exact real pipe center or a global survey coordinate.
- General accuracy from two successful deterministic cases.
- A current public website containing the same media as the local preview.

The forward and inverse shape families match, the dent width is fixed, and speed matches the simulation by explicit assumption. These choices are valid for a controlled demonstration, but they make it easier than a blind field problem. The note must keep those conditions visible.

## 21. Next technical work

These are proposed steps, not completed work:

1. **Broaden validation.** Use held-out shapes, different dent widths, several dents, null pipes, offset directions, noise, missing channels, and speed errors. Keep truth separate and report failures.
2. **Check numerical accuracy.** Run grid and time-step convergence, validate reflector timing against independent references, and quantify source/picker timing bias.
3. **Improve the observation model.** Replace the central-ray approximation with paths tied to actual array events. Separate improvement of the wall fit from improvement of the display map.
4. **Choose acquisition deliberately.** Select either distinct element transmissions for FMC or declared phased transmit events. Do not mix the data models or rename the current envelope display TFM.
5. **Improve the wall product.** Test less restricted shape models, coverage, uncertainty, and refusal before making broader geometry claims.
6. **Add elastic truth carefully.** Use the existing example gate before adopting a fluid–solid pipe model. Include interface physics, outer-wall echoes, and convergence evidence.
7. **Return to 3D.** Extend validated observations along the pipe and improve the recovered wall cloud. Keep acquired and display sampling separate.
8. **Publish a matched package.** Release the relevant code, current media, small metadata, commands, and this note together only when that publication is requested and checked.

These steps need separate decisions for meaningful model or acquisition changes. They do not require copying a closed industrial implementation.

## 22. How to explain the project

### Short explanation

> SonoLab is an open ultrasonic geometry project. I simulate waves inside a pipe, record array echoes, and estimate the wall and pipe center from echo times. This video compares centered and off-center tools. Simulation truth stays separate from the estimator so I can check its error.

### Technical explanation

> The current forward model is 2D scalar FDTD. Twenty elements fire together on a curved array. I acquire 360 angular looks. The middle panel shows per-channel RF for the displayed looks. The fit uses picked times from coherent receive signals and a reduced circle-plus-dent observation model. Sound speed is an explicit input. The right image is an illustrative envelope map; the green contour is fitted independently.

### Useful questions and answers

**How do you know the actual center?**  
The simulation defines it. The blue cross is that reference in tool coordinates. The green ring is inferred from echoes. In real data, the actual center needs independent measurement for validation.

**Why does off-center matter?**  
Near-wall and far-wall travel times differ. If the tool origin is treated as the pipe center, pose can be mistaken for wall shape.

**Is the middle one channel?**  
No. It contains 20 rows, one per receive channel, against time for the current look. The fit also has a separate combined trace. Do not confuse the two.

**Is the wall just a bright image ridge?**  
No. The wall comes from the echo-time estimator. The image is a separate display.

**Is this a focused phased-array scan?**  
The class supports delay correction, but the current clip uses simultaneous transmit and a rotating array orientation. Describe that mode accurately.

**What worked?**  
Both simulated cases produced non-refused fits. Their center errors are about 0.41 and 0.34 cells. The wall error is larger off-center, and dent depth is biased. Tests and playback checks pass under the stated conditions.

**What is the largest limitation?**  
The wave boundary and wall model are simplified. The next evidence should come from independent shapes, numerical convergence, improved array paths, and elastic or measured data.

**What is the engineering contribution?**  
The integration of simulation, recorded channels, a truth-separated estimator, repeatable scoring, regression checks, and a clear comparison. The project is not the invention of FDTD or a claim to proprietary imaging algorithms. State personal implementation and validation work accurately; do not claim sole authorship of reused methods or external solvers.

## 23. Terms and references

| Term | Meaning here |
|---|---|
| NDT | Non-destructive testing. The wider problem class, not a certification of this demo. |
| TX / RX | Transmit / receive. |
| RF | Signed oscillating receive data. Here it is a simulated scalar signal. |
| A-scan | One receive trace against time. |
| Channel × time heatmap | A row per receive channel, with amplitude shown by color. |
| ToF | Time of flight. Travel time used in ranging. |
| FDTD | Finite-difference time-domain wave calculation. |
| DAS | Delay-and-sum receive processing; a simple mean when delays are zero. |
| FMC | Full matrix capture: distinct TX events with RX channel records. |
| TFM | Total Focusing Method: focusing across declared TX/RX path pairs. |
| Tool frame | Coordinates with the tool origin at zero. |
| Eccentricity | Tool displacement from the reference pipe center. |
| Ground truth | Known simulation geometry used to score an estimate. |
| RMSE | Root mean square error under a stated comparison. |
| Refusal | A decision not to publish a confident fitted result under current quality rules. |

Primary evidence for this note is the inspected local source, saved metadata, saved acquisition/estimate/truth files, and the checks listed above. The July study figures are saved historical evidence, not rerun measurements.

External references support related concepts, not equivalence of SonoLab to those implementations:

- [k-Wave linear-array steering example](https://www.k-wave.org/documentation/example_tvsp_steering_linear_array.php).
- [k-Wave receive beamforming, input-signal removal, and envelope processing](https://www.k-wave.org/documentation/example_us_bmode_linear_transducer.php).
- [Official AFE5809 component information](https://www.ti.com/product/AFE5809).
- ATLUS: the user-supplied Markdown interpretation of `UBC_ECE_071.pdf`. No verified original-poster URL is supplied here.

**Bottom line:** The current showcase demonstrates a repeatable 2D scalar echo-to-wall process for two controlled tool positions. It has real simulated channel data, an independent parametric fit, known references, visible errors, and tested presentation. It remains a research demonstration with explicit assumptions and unfinished field-physics work.
