"""Collect SPECFEM SU seismograms into standardized SonoLab artifacts."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml


def _load_params(job_dir: Path) -> dict[str, Any]:
    path = job_dir / "DATA" / "00_parameters.yaml"
    with path.open() as f:
        return yaml.safe_load(f)


def _read_par_float(par_text: str, key: str) -> float | None:
    m = re.search(rf"^{re.escape(key)}\s*=\s*([^\s#!]+)", par_text, re.MULTILINE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _read_par_int(par_text: str, key: str) -> int | None:
    v = _read_par_float(par_text, key)
    return int(v) if v is not None else None


def seismogram_dt_s(job_dir: Path) -> float:
    """Sample interval of SU seismograms: Par_file DT × output subsample factor.

    SU headers often truncate dt; CFL-only estimates ignore NTSTEP_BETWEEN_OUTPUT_SAMPLE.
    """
    par_path = job_dir / "DATA" / "Par_file"
    if par_path.exists():
        text = par_path.read_text(encoding="utf-8", errors="replace")
        dt = _read_par_float(text, "DT")
        subsample = _read_par_int(text, "NTSTEP_BETWEEN_OUTPUT_SAMPLE") or 1
        if dt is not None and dt > 0:
            return float(dt) * float(max(subsample, 1))
    # Fallback: EXAMPLE 01 CFL rounding (legacy packs / missing Par_file)
    params = _load_params(job_dir)
    return _cfl_time_step_s(params)


def _cfl_time_step_s(params: dict[str, Any]) -> float:
    """Match EXAMPLE 01 plot_A_scan.py CFL rounding (avoid SU header truncation)."""
    lambda_min = params["material"]["vs"] / params["transducer"]["f0"]
    min_element_size = (lambda_min * 0.8) / 3.0
    dt_cfl = 0.3 * ((min_element_size / 5.0) / params["material"]["vp"])
    exponent = np.floor(np.log10(dt_cfl))
    mantissa = np.floor(dt_cfl / (10**exponent) * 10) / 10.0
    return float(mantissa * (10**exponent))


def read_uz_su(job_dir: Path) -> tuple[np.ndarray, float]:
    """Return (amp[n_rx, n_t], dt_s) from Uz_file_single_v.su."""
    import obspy

    su_path = job_dir / "OUTPUT_FILES" / "Uz_file_single_v.su"
    if not su_path.exists():
        raise FileNotFoundError(f"Missing SU output: {su_path}")
    dt = seismogram_dt_s(job_dir)
    st = obspy.read(str(su_path), format="SU", byteorder="<", unpack_trace_headers=True)
    amp = np.stack([tr.data.astype(np.float64) for tr in st], axis=0)
    return amp, dt


def plot_ascan_png(
    amp: np.ndarray,
    dt_s: float,
    params: dict[str, Any],
    out_png: Path,
) -> None:
    trace = amp[0]
    norm = float(np.percentile(np.abs(trace), 99.5)) or float(np.max(np.abs(trace))) or 1.0
    t_us = np.arange(trace.size) * dt_s * 1e6
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_us, trace / norm, color="#0077B6", linewidth=1.2)
    f0_mhz = params["transducer"]["f0"] / 1e6
    ax.set_title(f"Pulse-Echo A-Scan ({f0_mhz:.2f} MHz) — SonoLab", fontweight="bold")
    ax.set_xlabel("Time of Flight (µs)")
    ax.set_ylabel("Amplitude (norm.)")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def pack_artifacts(
    job_dir: Path,
    *,
    job_spec: dict[str, Any] | None = None,
    platform_version: str = "0.1.0",
    elapsed_s: float | None = None,
) -> Path:
    """
    Write artifacts/ascan.npz, meta.json, a_scan.png under job_dir.

    Returns the artifacts directory.
    """
    amp, dt = read_uz_su(job_dir)
    params = _load_params(job_dir)
    art = job_dir / "artifacts"
    art.mkdir(parents=True, exist_ok=True)

    t_s = np.arange(amp.shape[1], dtype=np.float64) * dt
    np.savez_compressed(
        art / "ascan.npz",
        t_s=t_s,
        amp=amp,
        dt_s=np.array(dt, dtype=np.float64),
    )

    plot_ascan_png(amp, dt, params, art / "a_scan.png")

    # Also keep upstream plot if present
    upstream_png = job_dir / "A_Scan_Result.png"
    if upstream_png.exists():
        shutil.copy2(upstream_png, art / "A_Scan_Result_upstream.png")

    meta = {
        "schema_version": "0.1",
        "platform": "sonolab",
        "platform_version": platform_version,
        "packed_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_dir": str(job_dir.resolve()),
        "backend": "example01_conventional_pulse_echo",
        "n_rx": int(amp.shape[0]),
        "n_t": int(amp.shape[1]),
        "dt_s": dt,
        "duration_s": float(t_s[-1]) if t_s.size else 0.0,
        "specfem_params": params,
        "job_spec": job_spec,
        "elapsed_s": elapsed_s,
        "su_file": "OUTPUT_FILES/Uz_file_single_v.su",
    }
    with (art / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    return art
