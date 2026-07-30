"""Lab A sector-scan acquisition: FDTD + blind picks → AcquisitionPack + TruthPack.

Default backend is the Rust engine ``sonolab_pipe2d_fdtd`` when installed.
Force the NumPy reference with ``SONOLAB_LABA_BACKEND=python``.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from sonolab.pipe2d import phased_array as pa
from sonolab.pipe2d.blind import AcquisitionPack, TruthPack

_RUST = None
_RUST_ERR: str | None = None


def _load_rust():
    global _RUST, _RUST_ERR
    if _RUST is not None or _RUST_ERR is not None:
        return _RUST
    try:
        import sonolab_pipe2d_fdtd as eng  # type: ignore

        _RUST = eng
        return _RUST
    except Exception as exc:  # noqa: BLE001
        _RUST_ERR = str(exc)
        return None


def laba_backend() -> str:
    """Return active Lab A backend name: ``rust`` or ``python``."""
    forced = os.environ.get("SONOLAB_LABA_BACKEND", "").strip().lower()
    if forced in ("python", "numpy", "ref", "reference"):
        return "python"
    if forced in ("rust", "fast"):
        if _load_rust() is None:
            raise RuntimeError(
                f"SONOLAB_LABA_BACKEND=rust but extension unavailable: {_RUST_ERR}"
            )
        return "rust"
    return "rust" if _load_rust() is not None else "python"


def ascan_peak_snr(series: np.ndarray, t_echo: float) -> float:
    """Instrument pick SNR: peak near t_echo over early-window noise floor."""
    trace = np.asarray(series, dtype=np.float64).ravel()
    n = int(trace.size)
    if n < 8:
        return 0.0
    env = np.abs(trace)
    lo_noise = int(pa.TX_EXCLUDE)
    hi_noise = min(n, lo_noise + max(8, n // 20))
    noise = float(np.median(env[lo_noise:hi_noise])) + 1e-12
    i0 = int(max(0, min(n - 1, round(float(t_echo)))))
    w0 = max(0, i0 - 8)
    w1 = min(n, i0 + 9)
    peak = float(np.max(env[w0:w1])) if w1 > w0 else float(env[i0])
    return float(peak / noise)


def blind_pick_echo(
    series: np.ndarray,
    *,
    prev_t: float | None = None,
) -> float:
    """Pick wall echo using instrument facts only (no R_known / pipe prior)."""
    n = int(series.size)
    if n < 4:
        return float(pa.TX_EXCLUDE)

    if prev_t is None:
        lo = int(pa.TX_EXCLUDE)
        hi = n
        window = np.abs(series[lo:hi])
        if window.size < 3:
            return float(lo)
        noise_n = min(24, max(4, window.size // 20))
        noise = float(np.median(window[:noise_n]))
        peak = float(np.max(window))
        thresh = max(4.0 * noise + 1e-6, 0.15 * peak)
        candidates: list[int] = []
        for i in range(1, window.size - 1):
            if (
                window[i] >= thresh
                and window[i] >= window[i - 1]
                and window[i] >= window[i + 1]
            ):
                candidates.append(lo + i)
        if not candidates:
            above = np.where(window >= thresh)[0]
            if above.size:
                return float(lo + int(above[0]))
            return float(lo + int(np.argmax(window)))
        return float(min(candidates))

    half = max(22.0, 0.07 * float(pa.ASCAN_LEN))
    d = pa.pick_echo_result(
        series,
        gate_center=float(prev_t),
        gate_half=half,
        c_assumed=1.0,
        R_known=1.0,
        prefer_nearest=True,
    )
    return float(d["t_echo"])


def _smooth_echo_times(t_echo: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """Circular median smooth to kill isolated bad picks before the joint fit."""
    t = np.asarray(t_echo, dtype=np.float64).copy()
    n = t.size
    if n < 5:
        return t
    out = t.copy()
    for i in range(n):
        idxs = [(i - 1) % n, i, (i + 1) % n]
        out[i] = float(np.median(t[idxs]))
    for i in range(n):
        nbr = 0.5 * (out[(i - 1) % n] + out[(i + 1) % n])
        if abs(out[i] - nbr) > 40.0:
            out[i] = nbr
    return out


def _circular_local_median(t_echo: np.ndarray, half_w: int) -> np.ndarray:
    t = np.asarray(t_echo, dtype=np.float64)
    n = t.size
    half_w = max(1, int(half_w))
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        idxs = [(i + k) % n for k in range(-half_w, half_w + 1)]
        out[i] = float(np.median(t[idxs]))
    return out


def _repair_early_multipath(
    t_raw: np.ndarray, ascan_store: list[np.ndarray]
) -> np.ndarray:
    """Fix far-wall multipath; preserve true dents / near-wall eccentricity."""
    t = np.asarray(t_raw, dtype=np.float64).copy()
    n = t.size
    if n < 5 or len(ascan_store) != n:
        return t
    med = float(np.median(t))
    hard_floor = med - 100.0
    local = _circular_local_median(t, max(7, n // 48))

    def _retrack(i: int, prior: float) -> float:
        half = max(22.0, 0.07 * float(pa.ASCAN_LEN))
        d = pa.pick_echo_result(
            ascan_store[i],
            gate_center=float(max(prior, hard_floor)),
            gate_half=max(half, 40.0),
            c_assumed=1.0,
            R_known=1.0,
            prefer_nearest=False,
        )
        return float(d["t_echo"])

    def _retrack_near(i: int, prior: float) -> float:
        half = max(22.0, 0.07 * float(pa.ASCAN_LEN))
        d = pa.pick_echo_result(
            ascan_store[i],
            gate_center=float(max(prior, hard_floor)),
            gate_half=max(half, 40.0),
            c_assumed=1.0,
            R_known=1.0,
            prefer_nearest=True,
        )
        return float(d["t_echo"])

    for i in range(n):
        if t[i] < hard_floor:
            prior = med if float(local[i]) < med - 30.0 else float(local[i])
            t[i] = _retrack(i, prior)

    if n >= 8 and n % 2 == 0:
        half = n // 2
        for _ in range(3):
            sums = t[:half] + t[half:]
            sum_med = float(np.median(sums))
            changed = False
            for i in range(n):
                j = (i + half) % n
                pred = sum_med - float(t[j])
                if t[i] < pred - 28.0 and t[i] > med - 40.0:
                    new_t = _retrack(i, pred)
                    if abs(new_t - t[i]) > 1.0:
                        t[i] = new_t
                        changed = True
            if not changed:
                break
    return t


def acquire_sector_scan_python(
    *,
    dent: bool = True,
    n_elem: int = pa.N_ELEM_DEFAULT,
    step_deg: float = 15.0,
    ecc_mag: float = 0.0,
    ecc_phi_deg: float = 90.0,
    progress: bool = False,
    store_ascans: bool = False,
) -> tuple[AcquisitionPack, TruthPack]:
    """NumPy FDTD reference path (slow; golden tests / fallback)."""
    ecc_mag = float(np.clip(ecc_mag, 0.0, pa.ECC_MAX))
    ecc_phi_deg = pa.wrap_deg(ecc_phi_deg)
    ex, ey = pa.tool_offset_xy(ecc_mag, ecc_phi_deg)
    thetas = np.arange(0.0, 360.0, float(step_deg))
    t_raw = np.zeros(thetas.size, dtype=np.float64)
    r_true = np.zeros(thetas.size, dtype=np.float64)
    ascan_store: list[np.ndarray] = []
    t0 = time.perf_counter()
    for i, th in enumerate(thetas):
        series = pa.record_pulse_window(
            theta_deg=float(th),
            dent=dent,
            n_elem=n_elem,
            ecc_mag=ecc_mag,
            ecc_phi_deg=ecc_phi_deg,
        )
        ascan_store.append(series.copy())
        te = blind_pick_echo(series, prev_t=None)
        t_raw[i] = te
        r_true[i] = pa.true_standoff_from_tool(
            float(th), dent=dent, ecc_x=ex, ecc_y=ey
        )
        if progress and (i % 8 == 0 or i == thetas.size - 1):
            elapsed = time.perf_counter() - t0
            print(
                f"  acquire {i + 1}/{thetas.size}  θ={th:.0f}°  "
                f"t={te:.1f}  r_true={r_true[i]:.2f}  ({elapsed:.1f}s)",
                flush=True,
            )

    if thetas.size >= 180:
        t_raw = _repair_early_multipath(t_raw, ascan_store)
    t_smooth = _smooth_echo_times(t_raw, thetas)
    t_echo = np.array(
        [
            blind_pick_echo(series, prev_t=float(t_smooth[i]))
            for i, series in enumerate(ascan_store)
        ],
        dtype=np.float64,
    )
    if thetas.size >= 180:
        t_echo = _repair_early_multipath(t_echo, ascan_store)
    t_echo = _smooth_echo_times(t_echo, thetas)

    snrs = [
        ascan_peak_snr(series, float(t_echo[i]))
        for i, series in enumerate(ascan_store)
    ]
    acq = AcquisitionPack(
        theta_deg=[float(v) for v in thetas],
        t_echo=[float(v) for v in t_echo],
        array_look_offset=float(pa.ARRAY_LOOK_OFFSET),
        pulse_width=float(pa.PULSE_WIDTH),
        n_elem=int(n_elem),
        step_deg=float(step_deg),
        elem_pitch=float(pa.ELEM_PITCH),
        aperture_span=float(pa.aperture_span_cells(n_elem)),
        meta={
            "ascanLen": int(pa.ASCAN_LEN),
            "pickMode": "blind first-break + early-repair + smooth-retrack",
            "noPipeRPrior": True,
            "labaBackend": "python",
            "rxMode": "coherent_das",
            "pickSnrMedian": round(float(np.median(snrs)), 4) if snrs else None,
            "pickSnrMean": round(float(np.mean(snrs)), 4) if snrs else None,
        },
    )
    truth = TruthPack(
        ecc_mag=float(ecc_mag),
        ecc_phi_deg=float(ecc_phi_deg),
        c_prop=float(pa.C_PROP),
        pipe_r=float(pa.PIPE_R),
        dent_on=bool(dent),
        dent_theta0_deg=float(pa.DENT_THETA0_DEG),
        dent_depth=float(pa.DENT_DEPTH),
        dent_half_deg=float(pa.DENT_HALF_DEG),
        r_true=[float(v) for v in r_true],
        theta_deg=[float(v) for v in thetas],
        meta={
            "trueToolOffsetX": round(ex, 6),
            "trueToolOffsetY": round(ey, 6),
        },
    )
    if store_ascans:
        acq.meta["nAscans"] = len(ascan_store)
        truth.meta["hasAscans"] = True
    return acq, truth


def acquire_sector_scan_rust(
    *,
    dent: bool = True,
    n_elem: int = pa.N_ELEM_DEFAULT,
    step_deg: float = 1.0,
    ecc_mag: float = 0.0,
    ecc_phi_deg: float = 90.0,
    progress: bool = False,
    store_ascans: bool = False,
    parallel: bool = True,
) -> tuple[AcquisitionPack, TruthPack]:
    """Rust FDTD Lab A path (fast). Passes live ``pa.DENT_*`` for MC pose patches."""
    eng = _load_rust()
    if eng is None:
        raise RuntimeError(f"Rust Lab A unavailable: {_RUST_ERR}")
    if progress:
        print(
            f"  acquire rust step={step_deg}° dent={dent} ecc={ecc_mag:.2f} …",
            flush=True,
        )
    d = eng.sector_scan(
        dent=bool(dent),
        n_elem=int(n_elem),
        step_deg=float(step_deg),
        ecc_mag=float(ecc_mag),
        ecc_phi_deg=float(ecc_phi_deg),
        dent_theta0_deg=float(pa.DENT_THETA0_DEG),
        dent_depth=float(pa.DENT_DEPTH),
        dent_half_deg=float(pa.DENT_HALF_DEG),
        parallel=bool(parallel),
    )
    thetas = [float(v) for v in np.asarray(d["theta_deg"]).ravel()]
    t_echo = [float(v) for v in np.asarray(d["t_echo"]).ravel()]
    r_true = [float(v) for v in np.asarray(d["r_true"]).ravel()]
    n_e = int(d["n_elem"])
    acq = AcquisitionPack(
        theta_deg=thetas,
        t_echo=t_echo,
        array_look_offset=float(d["array_look_offset"]),
        pulse_width=float(d["pulse_width"]),
        n_elem=n_e,
        step_deg=float(d["step_deg"]),
        elem_pitch=float(pa.ELEM_PITCH),
        aperture_span=float(pa.aperture_span_cells(n_e)),
        meta={
            "ascanLen": int(d["ascan_len"]),
            "pickMode": "blind first-break + early-repair + smooth-retrack",
            "noPipeRPrior": True,
            "labaBackend": "rust",
            "rxMode": "coherent_das",
            "elapsedMs": float(d["elapsed_ms"]),
        },
    )
    truth = TruthPack(
        ecc_mag=float(d["ecc_mag"]),
        ecc_phi_deg=float(d["ecc_phi_deg"]),
        c_prop=float(d["c_prop"]),
        pipe_r=float(d["pipe_r"]),
        dent_on=bool(d["dent_on"]),
        dent_theta0_deg=float(d["dent_theta0_deg"]),
        dent_depth=float(d["dent_depth"]),
        dent_half_deg=float(d["dent_half_deg"]),
        r_true=r_true,
        theta_deg=thetas,
        meta={
            "trueToolOffsetX": round(float(d["true_tool_offset_x"]), 6),
            "trueToolOffsetY": round(float(d["true_tool_offset_y"]), 6),
        },
    )
    if store_ascans:
        acq.meta["nAscans"] = len(thetas)
        truth.meta["hasAscans"] = False
    if progress:
        print(
            f"  acquire rust done  n={len(thetas)}  "
            f"elapsed={float(d['elapsed_ms']):.1f} ms",
            flush=True,
        )
    return acq, truth


def acquire_sector_scan(
    *,
    dent: bool = True,
    n_elem: int = pa.N_ELEM_DEFAULT,
    step_deg: float = 1.0,
    ecc_mag: float = 0.0,
    ecc_phi_deg: float = 90.0,
    progress: bool = False,
    store_ascans: bool = False,
) -> tuple[AcquisitionPack, TruthPack]:
    """Run Lab A sector sweep; return blind pack + truth sidecar."""
    if laba_backend() == "rust":
        return acquire_sector_scan_rust(
            dent=dent,
            n_elem=n_elem,
            step_deg=step_deg,
            ecc_mag=ecc_mag,
            ecc_phi_deg=ecc_phi_deg,
            progress=progress,
            store_ascans=store_ascans,
        )
    return acquire_sector_scan_python(
        dent=dent,
        n_elem=n_elem,
        step_deg=step_deg,
        ecc_mag=ecc_mag,
        ecc_phi_deg=ecc_phi_deg,
        progress=progress,
        store_ascans=store_ascans,
    )


def acquire_to_dicts(
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    acq, truth = acquire_sector_scan(**kwargs)
    return acq.to_dict(), truth.to_dict()
