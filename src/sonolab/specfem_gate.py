"""Block 5: SPECFEM2D-UT EXAMPLE 01 ToF fidelity gate vs analytic SDH."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from sonolab.config import runs_root
from sonolab.pack import pack_artifacts, read_uz_su, seismogram_dt_s

# EXAMPLE 01 steel_sdh.geo + 00_parameters.yaml (SI meters).
EX01_SURFACE_Z_M = 0.050
EX01_SDH_CENTER_Z_M = 0.025
EX01_SDH_RADIUS_M = 0.002
EX01_VP_DEFAULT = 5950.0

# Primary acceptance: relative ToF error vs analytic round-trip.
REL_TOF_TOL = 0.10

# Gate out main-bang / near-field before searching for SDH echo.
# 3 cycles at 2.25 MHz ≈ 1.3 µs; use 3 µs conservative TX exclusion.
TX_EXCLUDE_S = 3.0e-6


@dataclass
class Ex01Geometry:
    surface_z_m: float = EX01_SURFACE_Z_M
    sdh_center_z_m: float = EX01_SDH_CENTER_Z_M
    sdh_radius_m: float = EX01_SDH_RADIUS_M
    vp_m_s: float = EX01_VP_DEFAULT

    @property
    def one_way_m(self) -> float:
        """Vertical path from free surface TX to near rim of SDH."""
        rim_z = float(self.sdh_center_z_m) + float(self.sdh_radius_m)
        return float(self.surface_z_m) - rim_z

    def analytic_tof_s(self) -> float:
        """Round-trip ToF (s): t = 2 L / vp."""
        L = self.one_way_m
        if L <= 0:
            raise ValueError(f"invalid one-way path L={L}")
        return float(2.0 * L / max(float(self.vp_m_s), 1e-12))


def analytic_sdh_tof_s(
    *,
    vp_m_s: float = EX01_VP_DEFAULT,
    surface_z_m: float = EX01_SURFACE_Z_M,
    sdh_center_z_m: float = EX01_SDH_CENTER_Z_M,
    sdh_radius_m: float = EX01_SDH_RADIUS_M,
) -> float:
    return Ex01Geometry(
        surface_z_m=surface_z_m,
        sdh_center_z_m=sdh_center_z_m,
        sdh_radius_m=sdh_radius_m,
        vp_m_s=vp_m_s,
    ).analytic_tof_s()


def geometry_from_job(job_dir: Path) -> Ex01Geometry:
    """Build geometry from job YAML when present; else EXAMPLE 01 defaults."""
    params_path = job_dir / "DATA" / "00_parameters.yaml"
    vp = EX01_VP_DEFAULT
    surface_z = EX01_SURFACE_Z_M
    if params_path.exists():
        with params_path.open() as f:
            params = yaml.safe_load(f)
        vp = float(params.get("material", {}).get("vp", vp))
        surface_z = float(params.get("transducer", {}).get("zs", surface_z))
    return Ex01Geometry(surface_z_m=surface_z, vp_m_s=vp)


def pick_echo_tof_s(
    amp: np.ndarray,
    dt_s: float,
    *,
    tx_exclude_s: float = TX_EXCLUDE_S,
) -> float:
    """First-break pick past TX exclusion (instrument-style, no geometry prior)."""
    trace = np.asarray(amp, dtype=np.float64).ravel()
    n = int(trace.size)
    if n < 4:
        raise ValueError("trace too short to pick")
    dt = max(float(dt_s), 1e-15)
    lo = int(max(1, round(float(tx_exclude_s) / dt)))
    lo = min(lo, n - 3)
    env = np.abs(trace)
    window = env[lo:]
    noise_n = min(50, max(4, window.size // 20))
    noise = float(np.median(window[:noise_n]))
    peak = float(np.max(window))
    thresh = max(4.0 * noise + 1e-12, 0.15 * peak)
    # First sample above threshold (first-break), not peak of the burst.
    above = np.where(window >= thresh)[0]
    if above.size:
        return float((lo + int(above[0])) * dt)
    return float((lo + int(np.argmax(window))) * dt)


def compare_tof(
    pick_s: float, analytic_s: float, *, rel_tol: float = REL_TOF_TOL
) -> dict[str, Any]:
    pick_s = float(pick_s)
    analytic_s = float(analytic_s)
    abs_err = abs(pick_s - analytic_s)
    rel_err = abs_err / max(analytic_s, 1e-15)
    return {
        "pick_s": pick_s,
        "analytic_s": analytic_s,
        "abs_err_s": abs_err,
        "rel_err": rel_err,
        "rel_tol": float(rel_tol),
        "pass": bool(rel_err <= float(rel_tol)),
    }


def load_job_trace(job_dir: Path) -> tuple[np.ndarray, float]:
    """Load amplitude + correct seismogram dt (repack-aware)."""
    job_dir = Path(job_dir)
    su = job_dir / "OUTPUT_FILES" / "Uz_file_single_v.su"
    if su.exists():
        amp, dt = read_uz_su(job_dir)
        return amp[0], float(dt)
    art = job_dir / "artifacts" / "ascan.npz"
    if not art.exists():
        raise FileNotFoundError(f"no SU or ascan.npz under {job_dir}")
    data = np.load(art)
    amp = np.asarray(data["amp"][0], dtype=np.float64)
    # Prefer Par_file dt even if packed dt was wrong (legacy packs).
    try:
        dt = seismogram_dt_s(job_dir)
    except Exception:  # noqa: BLE001
        dt = float(np.asarray(data["dt_s"]).reshape(-1)[0])
    return amp, dt


def annotate_ascan_png(
    out_png: Path,
    *,
    amp: np.ndarray,
    dt_s: float,
    pick_s: float,
    analytic_s: float,
    title: str = "Block 5 EXAMPLE 01 ToF gate",
) -> Path:
    trace = np.asarray(amp, dtype=np.float64).ravel()
    norm = float(np.percentile(np.abs(trace), 99.5)) or float(np.max(np.abs(trace))) or 1.0
    t_us = np.arange(trace.size) * float(dt_s) * 1e6
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_us, trace / norm, color="#0077B6", lw=1.2, label="Uz")
    ax.axvline(pick_s * 1e6, color="#c45c26", ls="--", lw=1.4, label=f"pick {pick_s*1e6:.2f} µs")
    ax.axvline(
        analytic_s * 1e6,
        color="#2a9d4a",
        ls=":",
        lw=1.6,
        label=f"analytic {analytic_s*1e6:.2f} µs",
    )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Time of Flight (µs)")
    ax.set_ylabel("Amplitude (norm.)")
    ax.grid(True, ls="--", alpha=0.7)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def resolve_job_dir(job_id: str) -> Path:
    return runs_root() / job_id


def ensure_job_artifacts(job_dir: Path, *, repack: bool = True) -> Path:
    """Ensure artifacts/ exist; repack from SU with corrected dt when possible."""
    job_dir = Path(job_dir)
    su = job_dir / "OUTPUT_FILES" / "Uz_file_single_v.su"
    art = job_dir / "artifacts"
    if su.exists() and (repack or not (art / "ascan.npz").exists()):
        return pack_artifacts(job_dir)
    if (art / "ascan.npz").exists():
        return art
    raise FileNotFoundError(
        f"No SPECFEM outputs under {job_dir}. Run: sonolab run-example01 --job-id {job_dir.name}"
    )


def run_tof_gate(
    job_dir: Path,
    *,
    out_dir: Path | None = None,
    rel_tol: float = REL_TOF_TOL,
    repack: bool = True,
) -> dict[str, Any]:
    """Pick SDH echo ToF and compare to analytic; write evidence under out_dir."""
    job_dir = Path(job_dir)
    out_dir = Path(out_dir or (runs_root() / "_block5_ex01"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_job_artifacts(job_dir, repack=repack)
    amp, dt = load_job_trace(job_dir)
    geo = geometry_from_job(job_dir)
    analytic = geo.analytic_tof_s()
    pick = pick_echo_tof_s(amp, dt)
    cmp_ = compare_tof(pick, analytic, rel_tol=rel_tol)

    png = annotate_ascan_png(
        out_dir / "tof_gate_ascan.png",
        amp=amp,
        dt_s=dt,
        pick_s=pick,
        analytic_s=analytic,
    )
    evidence = {
        "schema_version": "0.1",
        "block": 5,
        "case": "example01_conventional_pulse_echo_sdh",
        "job_dir": str(job_dir.resolve()),
        "geometry": {
            "surface_z_m": geo.surface_z_m,
            "sdh_center_z_m": geo.sdh_center_z_m,
            "sdh_radius_m": geo.sdh_radius_m,
            "one_way_m": geo.one_way_m,
            "vp_m_s": geo.vp_m_s,
            "formula": "t = 2 * (surface_z - (sdh_center_z + sdh_radius)) / vp",
        },
        "pick": {
            "method": "first-break after TX_EXCLUDE",
            "tx_exclude_s": TX_EXCLUDE_S,
            "dt_s": dt,
            "n_t": int(amp.size),
        },
        "comparison": cmp_,
        "artifacts": {
            "annotated_ascan": str(png.resolve()),
        },
        "pass": bool(cmp_["pass"]),
    }
    out_json = out_dir / "tof_gate.json"
    out_json.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    evidence["artifacts"]["tof_gate_json"] = str(out_json.resolve())
    return evidence
