"""Block 8: dense interactive pipe cloud demo (acquire + display dual density)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sonolab.pipe3d import constants as C
from sonolab.pipe3d.acquire import acquire_pipe_survey_3d
from sonolab.pipe3d.display import (
    DISPLAY_N_Z,
    DISPLAY_STEP_DEG,
    build_display_wall,
)
from sonolab.pipe3d.plot import open_survey_html, write_survey_plots
from sonolab.pipe3d.survey import assemble_wall, estimate_survey, score_survey

# Acquire density (real FDTD measurements).
BLOCK8_ACQUIRE_STEP_DEG = 1.0
BLOCK8_ACQUIRE_N_Z = 16

# Smoke / CI acquire (minutes, not hours).
BLOCK8_FAST_STEP_DEG = 15.0
BLOCK8_FAST_N_Z = 6


def run_block8_demo(
    *,
    c_cal: float,
    out_dir: Path | str = Path("datasets/runs/_block8_demo"),
    fast: bool = False,
    step_deg: float | None = None,
    n_z: int | None = None,
    display_step_deg: float = DISPLAY_STEP_DEG,
    n_z_display: int = DISPLAY_N_Z,
    ecc_mag: float = 2.0,
    ecc_phi_deg: float = 90.0,
    dent: bool = True,
    nxy: int = C.NXY,
    nz: int = C.NZ,
    workers: int | None = None,
    progress: bool = True,
    open_view: bool = False,
) -> dict[str, Any]:
    """Acquire → Lab B → display mesh → evidence under out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if fast:
        step = float(BLOCK8_FAST_STEP_DEG if step_deg is None else step_deg)
        nz_acq = int(BLOCK8_FAST_N_Z if n_z is None else n_z)
    else:
        step = float(BLOCK8_ACQUIRE_STEP_DEG if step_deg is None else step_deg)
        nz_acq = int(BLOCK8_ACQUIRE_N_Z if n_z is None else n_z)

    survey, truth = acquire_pipe_survey_3d(
        n_z=nz_acq,
        step_deg=step,
        dent=dent,
        ecc_mag=float(ecc_mag),
        ecc_phi_deg=float(ecc_phi_deg),
        nxy=int(nxy),
        nz=int(nz),
        progress=progress,
        workers=workers,
    )
    estimates = estimate_survey(survey, c_cal=float(c_cal))
    product = assemble_wall(estimates)
    score = score_survey(product, truth, estimates=estimates)
    display = build_display_wall(
        product,
        truth,
        display_step_deg=float(display_step_deg),
        n_z_display=int(n_z_display),
        acquire_step_deg=step,
        acquire_n_z=nz_acq,
    )

    out = {
        "block": 8,
        "acquireStepDeg": step,
        "acquireNz": nz_acq,
        "displayStepDeg": float(display_step_deg),
        "displayNz": int(n_z_display),
        "survey": survey.to_dict(),
        "truth": truth.to_dict(),
        "product": product,
        "display": {
            "z": display["z"],
            "theta_deg": display["theta_deg"],
            "nZ": display["nZ"],
            "nTheta": display["nTheta"],
            "meta": display["meta"],
            # Keep full mesh arrays in sidecar JSON; demo JSON stores meta + light grids.
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
            "Block 8 dual density: FDTD acquire at step×n_z; display mesh at "
            f"{display_step_deg}°×{n_z_display} z from Lab B + analytic truth. "
            "No commercial ILI mm-density claim. physics=3D_scalar_acoustic."
        ),
    }

    demo_json = out_dir / "block8_demo.json"
    demo_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Block 8 - dense interactive pipe cloud",
                "",
                f"- **Acquire:** {step:g}° × {nz_acq} z (Lab A 3D scalar FDTD, parallel)",
                f"- **Display:** {display_step_deg:g}° × {n_z_display} z mesh "
                "(resampled fitted wall + truth)",
                f"- **Physics:** `{C.PHYSICS_LABEL}`",
                "- **Honesty:** display sampling is not field mm-resolution; "
                "do not claim commercial ILI imaging density here.",
                "",
                "Open `survey3d_view.html` (needs sibling `survey3d_view_data.js`).",
                "",
                "```bash",
                "sonolab block8-demo --c-cal 0.45          # full 1°×16",
                "sonolab block8-demo --c-cal 0.45 --fast   # smoke",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )

    paths = write_survey_plots(
        out_dir,
        product=product,
        truth=truth,
        score=score,
        display=display,
        display_step_deg=float(display_step_deg),
        n_z_display=int(n_z_display),
        acquire_step_deg=step,
    )
    if open_view and "html" in paths:
        open_survey_html(paths["html"])

    return {
        "out_dir": str(out_dir),
        "demo_json": str(demo_json),
        "score": score,
        "paths": {k: str(v) for k, v in paths.items()},
        "acquireStepDeg": step,
        "acquireNz": nz_acq,
        "displayStepDeg": float(display_step_deg),
        "displayNz": int(n_z_display),
        "elapsedSec": survey.meta.get("elapsedSec"),
        "workers": survey.meta.get("workers"),
    }
