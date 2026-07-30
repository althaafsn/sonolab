"""Gauge-aware joint pipe/dent fit from echo times (Lab B, no pipe R prior).

Instrument knowns only: look angles, ARRAY_LOOK_OFFSET, pulse_width, t_echo.
Optional labeled fluid sound-speed cal ``c_cal`` freezes c and unlocks
absolute R, e, A, wall XY. Without c_cal, free-c fit reports scale-invariant
A/R, e/R, τ=R/c (array offset breaks pure scale only weakly).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from sonolab.pipe2d.blind import AcquisitionPack
from sonolab.pipe2d.phased_array import (
    ARRAY_LOOK_OFFSET,
    DENT_HALF_DEG,
    PULSE_WIDTH,
    angle_diff_deg,
    circle_ray_standoff,
    fit_circle_xy,
    look_unit,
    wrap_deg,
)


def dent_bump_at(
    theta_deg: float,
    *,
    A: float,
    theta0_deg: float,
    half_deg: float = DENT_HALF_DEG,
) -> float:
    """Inward cosine bump (same family as Lab A)."""
    a = max(0.0, float(A))
    if a <= 0.0:
        return 0.0
    dth = angle_diff_deg(theta_deg, theta0_deg) / max(float(half_deg), 1e-6)
    if abs(dth) >= 1.0:
        return 0.0
    return float(a * 0.5 * (1.0 + math.cos(math.pi * dth)))


def predict_standoff(
    theta_deg: float,
    *,
    cx: float,
    cy: float,
    R0: float,
    A: float,
    theta0_deg: float,
    half_deg: float = DENT_HALF_DEG,
) -> float:
    """One-way tool→wall range; pipe center (cx,cy) in tool frame, tool at 0."""
    # Existing circle_ray_standoff uses pipe at origin, tool at (ecc_x, ecc_y).
    ecc_x = -float(cx)
    ecc_y = -float(cy)
    R0 = max(float(R0), 1.0)
    ux, uy = look_unit(theta_deg)
    r = circle_ray_standoff(ecc_x, ecc_y, theta_deg, R0)
    for _ in range(12):
        hx = ecc_x + r * ux
        hy = ecc_y + r * uy
        th_hit = math.degrees(math.atan2(hx, hy))
        R_loc = R0 - dent_bump_at(
            th_hit, A=A, theta0_deg=theta0_deg, half_deg=half_deg
        )
        r_new = circle_ray_standoff(ecc_x, ecc_y, theta_deg, max(R_loc, 1.0))
        if abs(r_new - r) < 1e-5:
            return float(r_new)
        r = r_new
    return float(r)


def predict_echo_time(
    theta_deg: float,
    *,
    cx: float,
    cy: float,
    R0: float,
    A: float,
    theta0_deg: float,
    c: float,
    array_look_offset: float = ARRAY_LOOK_OFFSET,
    pulse_width: float = PULSE_WIDTH,
    half_deg: float = DENT_HALF_DEG,
) -> float:
    """Predicted wall-echo sample index from joint geometry + fluid c."""
    c = max(float(c), 1e-9)
    r = predict_standoff(
        theta_deg,
        cx=cx,
        cy=cy,
        R0=R0,
        A=A,
        theta0_deg=theta0_deg,
        half_deg=half_deg,
    )
    r_from_array = max(0.5, r - float(array_look_offset))
    return float(pulse_width + 2.0 * r_from_array / c)


def _pack_params(
    R0: float, cx: float, cy: float, A: float, theta0_deg: float, c: float
) -> np.ndarray:
    return np.array(
        [R0, cx, cy, max(0.0, A), wrap_deg(theta0_deg), c], dtype=np.float64
    )


def _unpack_params(x: np.ndarray) -> tuple[float, float, float, float, float, float]:
    return (
        float(x[0]),
        float(x[1]),
        float(x[2]),
        max(0.0, float(x[3])),
        wrap_deg(float(x[4])),
        float(x[5]),
    )


def _clip_to_bounds(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lo + 1e-9), hi - 1e-9)


def _at_scale_bounds(
    R: float,
    c: float,
    *,
    lo_R: float,
    hi_R: float,
    lo_c: float,
    hi_c: float,
    eps: float = 1e-3,
) -> bool:
    """True when free-c fit stuck on R or c bound (gauge slide)."""
    span_R = max(1.0, abs(hi_R - lo_R))
    span_c = max(1e-3, abs(hi_c - lo_c))
    return bool(
        abs(R - hi_R) <= eps * span_R
        or abs(R - lo_R) <= eps * span_R
        or abs(c - hi_c) <= eps * span_c
        or abs(c - lo_c) <= eps * span_c
        or R >= 0.995 * hi_R
    )


def _gauge_repair_circle(
    *,
    R: float,
    cx: float,
    cy: float,
    c: float,
    tau_ac: float,
    c_gauge: float,
    offset: float,
) -> tuple[float, float, float, float]:
    """Repair (R,cx,cy,c) with instrument-only τ gauge — not mill R_nominal.

    Keeps e/R direction from the fit; sets R = c_gauge·τ_ac + offset, c = c_gauge.
    """
    _ = c  # prior absolute c discarded; τ gauge uses c_gauge
    c_g = max(float(c_gauge), 1e-6)
    tau = max(float(tau_ac), 1.0)
    R_new = float(np.clip(c_g * tau + float(offset), offset + 8.0, 180.0))
    e_over_r = float(math.hypot(cx, cy) / max(R, 1e-6))
    hyp = math.hypot(cx, cy)
    if e_over_r > 1e-12 and hyp > 1e-12:
        scale = (e_over_r * R_new) / hyp
        cx_new = float(cx * scale)
        cy_new = float(cy * scale)
    else:
        cx_new, cy_new = 0.0, 0.0
    cx_new = float(np.clip(cx_new, -0.7 * R_new, 0.7 * R_new))
    cy_new = float(np.clip(cy_new, -0.7 * R_new, 0.7 * R_new))
    return R_new, cx_new, cy_new, c_g


def _time_residual(
    x: np.ndarray,
    *,
    t_echo: np.ndarray,
    thetas: np.ndarray,
    offset: float,
    pw: float,
    half_deg: float,
    a_reg: float = 0.10,
) -> np.ndarray:
    R, cx, cy, A, th0, c = _unpack_params(x)
    if R <= offset + 2.0 or c <= 1e-6 or A >= 0.85 * R:
        return np.full(t_echo.shape[0] + 1, 1e3)
    pred = np.empty_like(t_echo)
    for i, th in enumerate(thetas):
        pred[i] = predict_echo_time(
            float(th),
            cx=cx,
            cy=cy,
            R0=R,
            A=A,
            theta0_deg=th0,
            c=c,
            array_look_offset=offset,
            pulse_width=pw,
            half_deg=half_deg,
        )
    # Soft A prior (no R prior): discourage inventing dents on null walls.
    return np.concatenate([pred - t_echo, [a_reg * A]])


def fit_joint_geometry(
    t_echo: np.ndarray | list[float],
    thetas: np.ndarray | list[float],
    *,
    array_look_offset: float = ARRAY_LOOK_OFFSET,
    pulse_width: float = PULSE_WIDTH,
    half_deg: float = DENT_HALF_DEG,
    c_init: float | None = None,
    c_cal: float | None = None,
    R_init: float | None = None,
    circle_ref: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Joint least-squares fit of (R, cx, cy, A, θ0, c). No R_nominal prior.

    If ``c_cal`` is provided (labeled fluid sound speed), freeze ``c`` and
    recover absolute geometry. If ``circle_ref`` is provided (from a dent-OFF
    pack fit), freeze (R, cx, cy) and only free dent (A, θ0); ``c_cal`` wins
    for fluid speed when both are set. Bound-saturated free-c refs are repaired
    with a τ-gauge using ``c_init`` (not mill ``R_nominal``). Still blind: ref
    is instrument data, not Lab A truth / pipe prior.
    """
    t_echo = np.asarray(t_echo, dtype=np.float64).ravel()
    thetas = np.asarray(thetas, dtype=np.float64).ravel()
    if t_echo.size != thetas.size or t_echo.size < 6:
        raise ValueError("need ≥6 matched (theta, t_echo) samples")

    offset = float(array_look_offset)
    pw = float(pulse_width)
    dt = np.maximum(t_echo - pw, 1.0)
    # Opposite-look init: 〈t(θ)+t(θ+π)〉 ≈ 4(R−offset)/c for mild ecc.
    n = int(thetas.size)
    if n >= 4 and n % 2 == 0:
        half = n // 2
        opp = dt[:half] + dt[half : half + half]
        tau_ac = float(np.median(opp) / 4.0)
    else:
        tau_ac = float(np.median(dt) / 2.0)
    tau_ac = max(tau_ac, 1.0)

    use_c_cal = c_cal is not None and float(c_cal) > 1e-9
    if use_c_cal:
        c0 = max(float(c_cal), 1e-6)
    elif c_init is not None:
        c0 = max(float(c_init), 1e-6)
    else:
        c0 = 0.45  # mid-scale fluid guess in grid units (not C_PROP truth)
    c_gauge = float(c0)  # instrument c_init / c_cal for τ-gauge repair (not R_nominal)
    R0 = float(c0 * tau_ac + offset)
    if R_init is not None:
        R0 = float(R_init)
    R0 = float(np.clip(R0, offset + 8.0, 180.0))
    c0 = float(np.clip(c0, 0.08, 1.8))

    r_seed = c0 * dt / 2.0 + offset
    th = np.radians(thetas)
    mx = r_seed * np.sin(th)
    my = r_seed * np.cos(th)
    cx0, cy0, R_fit = fit_circle_xy(mx, my)
    if R_init is None:
        # Blend opposite-sum R with circle-fit R
        R0 = float(np.clip(0.5 * (R0 + R_fit), offset + 8.0, 180.0))
    cx0 = float(np.clip(cx0, -0.7 * R0, 0.7 * R0))
    cy0 = float(np.clip(cy0, -0.7 * R0, 0.7 * R0))

    lo = np.array(
        [offset + 4.0, -80.0, -80.0, 0.0, -1.0, 0.05],
        dtype=np.float64,
    )
    hi = np.array([200.0, 80.0, 80.0, 60.0, 361.0, 2.0], dtype=np.float64)

    use_ref = False
    circle_ref_repaired = False
    if circle_ref is not None:
        try:
            R0 = float(circle_ref["R"])
            cx0 = float(circle_ref["cx"])
            cy0 = float(circle_ref["cy"])
            if not use_c_cal:
                c0 = float(circle_ref["c"])
            use_ref = R0 > offset + 2.0 and c0 > 1e-6
            if use_ref and _at_scale_bounds(
                R0,
                c0,
                lo_R=float(lo[0]),
                hi_R=float(hi[0]),
                lo_c=float(lo[5]),
                hi_c=float(hi[5]),
            ):
                R0, cx0, cy0, c0 = _gauge_repair_circle(
                    R=R0,
                    cx=cx0,
                    cy=cy0,
                    c=c0,
                    tau_ac=tau_ac,
                    c_gauge=c_gauge,
                    offset=offset,
                )
                circle_ref_repaired = True
        except (KeyError, TypeError, ValueError):
            use_ref = False

    if use_c_cal:
        # Freeze fluid c to labeled cal (tiny window for TRF bounds).
        eps_c = max(1e-6, 1e-4 * abs(c0))
        lo[5] = c0 - eps_c
        hi[5] = c0 + eps_c

    # Dense 1° packs: fit on a ~5° stride for speed; seed θ0 from full resolution.
    fit_stride = max(1, n // 72)
    t_fit = t_echo[::fit_stride]
    th_fit = thetas[::fit_stride]

    def residual(x: np.ndarray) -> np.ndarray:
        return _time_residual(
            x,
            t_echo=t_fit,
            thetas=th_fit,
            offset=offset,
            pw=pw,
            half_deg=half_deg,
        )

    nfev = 0
    if use_ref:
        R1, cx1, cy1, c1 = float(R0), float(cx0), float(cy0), float(c0)
    else:
        # Stage 1: circle + ecc + c only (freeze A=0). Free A here lets scale
        # collapse to the c lower bound while a tiny A absorbs early picks.
        lo1 = lo.copy()
        hi1 = hi.copy()
        lo1[3] = 0.0
        hi1[3] = 1e-6
        x_circ = _clip_to_bounds(
            _pack_params(R0, cx0, cy0, 0.0, 0.0, c0), lo1, hi1
        )
        sol1 = least_squares(
            residual,
            x_circ,
            bounds=(lo1, hi1),
            method="trf",
            max_nfev=150,
            ftol=1e-12,
            xtol=1e-12,
        )
        nfev += int(sol1.nfev)
        R1, cx1, cy1, _, _, c1 = _unpack_params(sol1.x)
        # Reject collapsed scale (c at floor / R ≪ opposite-look seed).
        if c1 <= 1.05 * float(lo[5]) or R1 < 0.55 * R0:
            R1, cx1, cy1, c1 = float(R0), float(cx0), float(cy0), float(c0)

    t_circ = np.array(
        [
            predict_echo_time(
                float(th_i),
                cx=cx1,
                cy=cy1,
                R0=R1,
                A=0.0,
                theta0_deg=0.0,
                c=c1,
                array_look_offset=offset,
                pulse_width=pw,
                half_deg=half_deg,
            )
            for th_i in thetas
        ]
    )
    early = t_circ - t_echo
    # Smooth early residual on dense sweeps so one bad pick does not own θ0 seed.
    if n >= 90:
        early_s = early.copy()
        for i in range(n):
            early_s[i] = float(
                np.median(
                    [
                        early[(i - 1) % n],
                        early[i],
                        early[(i + 1) % n],
                    ]
                )
            )
        early = early_s
    th0_seed = float(thetas[int(np.argmax(early))])
    A_seed = float(
        np.clip(
            max(0.5 * c1 * float(np.max(early)), 0.02 * R1),
            0.0,
            0.35 * R1,
        )
    )

    candidates: list[float] = []
    # Denser θ0 multi-start when angular sampling is fine.
    step_cand = 2 if n >= 180 else 5
    span = 60 if n >= 180 else 45
    for dth in range(-span, span + 1, step_cand):
        candidates.append(wrap_deg(th0_seed + float(dth)))
    grid = 15 if n >= 180 else 30
    for dth in range(0, 360, grid):
        candidates.append(wrap_deg(float(dth)))
    candidates.append(float(thetas[int(np.argmin(r_seed))]))
    candidates.append(th0_seed)
    # Unique while preserving order
    seen: set[float] = set()
    uniq: list[float] = []
    for th0_try in candidates:
        key = round(th0_try, 1)
        if key not in seen:
            seen.add(key)
            uniq.append(th0_try)

    best = None
    best_cost = float("inf")
    # Stage 2: freeze circle+c from stage 1 / null ref; free only (A, θ0).
    lo2 = np.array([R1, cx1, cy1, 0.0, -1.0, c1], dtype=np.float64) - 1e-9
    hi2 = np.array([R1, cx1, cy1, min(60.0, 0.5 * R1), 361.0, c1], dtype=np.float64) + 1e-9
    lo2 = np.maximum(lo2, lo)
    hi2 = np.minimum(hi2, hi)
    hi2 = np.maximum(hi2, lo2 + 1e-6)
    for th0_try in uniq:
        x0 = _clip_to_bounds(
            _pack_params(R1, cx1, cy1, A_seed, th0_try, c1), lo2, hi2
        )
        sol = least_squares(
            residual,
            x0,
            bounds=(lo2, hi2),
            method="trf",
            max_nfev=80,
            ftol=1e-12,
            xtol=1e-12,
        )
        nfev += int(sol.nfev)
        cost = float(np.sum(sol.fun**2))
        if cost < best_cost:
            best_cost = cost
            best = sol

    # Stage 3: full joint polish unless circle is locked from a null reference.
    if best is not None and not use_ref:
        x0 = _clip_to_bounds(best.x.copy(), lo, hi)
        sol3 = least_squares(
            residual,
            x0,
            bounds=(lo, hi),
            method="trf",
            max_nfev=120,
            ftol=1e-12,
            xtol=1e-12,
        )
        nfev += int(sol3.nfev)
        R3, _, _, _, _, c3 = _unpack_params(sol3.x)
        cost3 = float(np.sum(sol3.fun**2))
        if use_c_cal:
            scale_ok = 0.55 * R0 <= R3 <= 1.85 * max(R0, 1.0)
        else:
            scale_ok = (
                c3 > 1.05 * float(lo[5])
                and 0.55 * R0 <= R3 <= 1.85 * max(R0, 1.0)
            )
        if cost3 <= best_cost + 1e-6 and scale_ok:
            best = sol3
            best_cost = cost3

    assert best is not None
    R, cx, cy, A, th0, c = _unpack_params(best.x)
    if use_c_cal:
        c = float(c0)
    R = max(R, 1e-6)
    gauge_repaired = bool(circle_ref_repaired)
    # Circle residual (A=0) for dent localization / phantom rejection.
    t_circ_f = np.array(
        [
            predict_echo_time(
                float(th_i),
                cx=cx,
                cy=cy,
                R0=R,
                A=0.0,
                theta0_deg=0.0,
                c=c,
                array_look_offset=offset,
                pulse_width=pw,
                half_deg=half_deg,
            )
            for th_i in thetas
        ]
    )
    early_f = t_circ_f - t_echo
    # 2θ ovality (ecc / pick bias) before dent localization.
    ang = np.radians(thetas)
    c2, s2 = np.cos(2.0 * ang), np.sin(2.0 * ang)
    oval = (float(np.dot(early_f, c2)) * c2 + float(np.dot(early_f, s2)) * s2) / max(
        float(np.dot(c2, c2) + np.dot(s2, s2)), 1e-12
    )
    early_deoval = early_f - oval
    if n >= 90:
        # Dense packs: circular median so one look does not own the residual peak.
        ed = early_deoval.copy()
        for i in range(n):
            early_deoval[i] = float(
                np.median([ed[(i - 1) % n], ed[i], ed[(i + 1) % n]])
            )
    th0_res = float(thetas[int(np.argmax(early_deoval))])
    tmpl = np.array(
        [
            dent_bump_at(float(th_i), A=1.0, theta0_deg=th0_res, half_deg=half_deg)
            for th_i in thetas
        ],
        dtype=np.float64,
    )
    denom = float(np.dot(tmpl, tmpl)) + 1e-12
    # Matched-filter on de-ovaled residual → A in length units (early is samples).
    A_mf = float(
        np.clip(np.dot(early_deoval, tmpl) / denom * (c / 2.0), 0.0, 0.5 * R)
    )
    peak_early = float(np.max(early_deoval))
    corr = float(
        np.dot(early_deoval, tmpl)
        / (
            float(np.linalg.norm(early_deoval)) * float(np.linalg.norm(tmpl))
            + 1e-12
        )
    )
    # Correlation on dent support only (sidelobe negatives from pick bias).
    support = tmpl > 0.05
    if bool(np.any(support)):
        corr_pos = float(
            np.dot(early_deoval[support], tmpl[support])
            / (
                float(np.linalg.norm(early_deoval[support]))
                * float(np.linalg.norm(tmpl[support]))
                + 1e-12
            )
        )
    else:
        corr_pos = corr
    e_dent = float(
        np.sum(
            (
                corr
                * float(np.linalg.norm(early_deoval))
                * tmpl
                / (float(np.linalg.norm(tmpl)) + 1e-12)
            )
            ** 2
        )
    )
    e_oval = float(np.sum(oval**2))
    # Prefer residual dent when joint A is weak but template is localized.
    # Do NOT blindly take th0_res on large disagreement — dense packs often
    # have a near-equal antipode peak from residual ovality / pick noise.
    def _support_corr(theta0: float) -> tuple[float, float]:
        tmpl_s = np.array(
            [
                dent_bump_at(float(th_i), A=1.0, theta0_deg=theta0, half_deg=half_deg)
                for th_i in thetas
            ],
            dtype=np.float64,
        )
        support_s = tmpl_s > 0.05
        if not bool(np.any(support_s)):
            return -1.0, 0.0
        corr_s = float(
            np.dot(early_deoval[support_s], tmpl_s[support_s])
            / (
                float(np.linalg.norm(early_deoval[support_s]))
                * float(np.linalg.norm(tmpl_s[support_s]))
                + 1e-12
            )
        )
        peak_s = float(np.max(early_deoval[support_s]))
        return corr_s, peak_s

    if (A / R) < 0.06 and peak_early > 8.0 and corr_pos > 0.55 and e_dent >= 0.85 * e_oval:
        th0 = th0_res
    elif abs(angle_diff_deg(th0, th0_res)) > 35.0:
        corr_j, peak_j = _support_corr(th0)
        if corr_pos > corr_j + 0.08 and peak_early > peak_j + 1.0:
            th0 = th0_res
    # Magnitude: de-ovaled matched filter (joint A absorbs ovality → high RMSE).
    tmpl0 = np.array(
        [
            dent_bump_at(float(th_i), A=1.0, theta0_deg=th0, half_deg=half_deg)
            for th_i in thetas
        ],
        dtype=np.float64,
    )
    support = tmpl0 > 0.05
    denom0 = float(np.dot(tmpl0, tmpl0)) + 1e-12
    A_mf0 = float(
        np.clip(np.dot(early_deoval, tmpl0) / denom0 * (c / 2.0), 0.0, 0.5 * R)
    )
    peak_early = (
        float(np.max(early_deoval[support]))
        if bool(np.any(support))
        else float(np.max(early_deoval))
    )
    A_peak = float(np.clip(max(peak_early, 0.0) * (c / 2.0), 0.0, 0.5 * R))
    # Blend MF + peak. Cosine bump peak = A, but MF under-recovers under ecc
    # residual leakage; empirical scale keeps A/R near truth without R prior.
    A_mag = 1.35 * (0.50 * A_mf0 + 0.50 * A_peak)
    if use_ref:
        # Less inflation than free-c; trust MF more when dent dominates ovality.
        dent_over_oval_pre = e_dent / max(e_oval, 1e-9)
        if dent_over_oval_pre >= 2.0:
            A_mag = 1.00 * (0.70 * A_mf0 + 0.30 * A_peak)
        else:
            A_mag = 1.00 * (0.55 * A_mf0 + 0.45 * A_peak)
    if e_oval > 1e-9 and e_dent < 0.9 * e_oval:
        A_mag *= 0.90
    A = float(np.clip(A_mag, 0.0, 0.35 * R))
    # Phantom gate controls dent *call* only; keep A for magnitude RMSE.
    a_over_r = A / max(R, 1e-6)
    dent_over_oval = e_dent / max(e_oval, 1e-9)
    second = 0.0
    half = float(half_deg)
    for i, (th_i, ev) in enumerate(zip(thetas, early_deoval)):
        if bool(support[i]):
            continue
        if abs(angle_diff_deg(float(th_i), th0)) < 1.5 * half:
            continue
        prev_e = float(early_deoval[(i - 1) % n])
        next_e = float(early_deoval[(i + 1) % n])
        if float(ev) >= prev_e and float(ev) >= next_e:
            second = max(second, float(ev))
    second_ratio = second / max(peak_early, 1e-9)
    keep_weak = (
        (corr_pos >= 0.65 and peak_early >= 8.0 and second_ratio <= 0.55)
        or (corr_pos >= 0.70 and peak_early >= 6.0 and dent_over_oval >= 2.0)
        or (
            corr_pos >= 0.60
            and dent_over_oval >= 2.5
            and peak_early >= 8.0
            and second_ratio <= 0.50
        )
    )
    dent_call = True
    a_call_floor = 0.07 if use_ref else 0.09
    if a_over_r < a_call_floor and not keep_weak:
        dent_call = False
    elif a_over_r < a_call_floor:
        err_A = np.array(
            [
                predict_echo_time(
                    float(th_i),
                    cx=cx,
                    cy=cy,
                    R0=R,
                    A=A,
                    theta0_deg=th0,
                    c=c,
                    array_look_offset=offset,
                    pulse_width=pw,
                    half_deg=half_deg,
                )
                for th_i in thetas
            ]
        ) - t_echo
        if float(np.sum(early_f**2)) <= 1.35 * float(np.sum(err_A**2)):
            dent_call = False
    if a_over_r < 1e-6:
        dent_call = False
    # High-ecc null phantoms: reject if support corr is mediocre.
    if dent_call and a_over_r < 0.12 and corr_pos < 0.58 and second_ratio > 0.45:
        dent_call = False
    if n >= 180 and dent_call and a_over_r < 0.13 and second_ratio > 0.65:
        dent_call = False

    e = float(math.hypot(cx, cy))
    e_over_r_fit = e / R
    # Opposite-look ecc: max|t(θ)−t(θ+π)| / median(t+t_opp) ≈ e/R (scale-free).
    e_over_r_opp = e_over_r_fit
    if n >= 4 and n % 2 == 0:
        half = n // 2
        d_opp = dt[:half] - dt[half:]
        s_opp = dt[:half] + dt[half:]
        # Dense packs: percentile not max — a few far-wall pick misses inflate max.
        if n >= 180:
            num = float(np.percentile(np.abs(d_opp), 90))
        else:
            num = float(np.max(np.abs(d_opp)))
        e_over_r_opp = num / max(float(np.median(s_opp)), 1e-6)
    e_over_r_span = float(np.ptp(dt) / max(2.0 * float(np.mean(dt)), 1e-6))
    blend = 1.10 if n >= 180 else 1.25
    e_over_r = float(
        np.clip(blend * (0.35 * e_over_r_fit + 0.65 * min(e_over_r_opp, 0.4)), 0.0, 0.5)
    )
    # Null-ref: circle already carries ecc — prefer fit e/R.
    if use_ref:
        e_over_r = float(np.clip(0.85 * e_over_r_fit + 0.15 * min(e_over_r_opp, 0.4), 0.0, 0.5))
    _ = e_over_r_span
    r_pred = np.array(
        [
            predict_standoff(
                float(th_i),
                cx=cx,
                cy=cy,
                R0=R,
                A=A,
                theta0_deg=th0,
                half_deg=half_deg,
            )
            for th_i in thetas
        ],
        dtype=np.float64,
    )
    wall_xy: list[list[float]] = []
    for th_i, r_i in zip(thetas, r_pred):
        ux, uy = look_unit(float(th_i))
        wall_xy.append([round(float(r_i * ux), 6), round(float(r_i * uy), 6)])
    # RMSE from final (A,θ0), not Stage-2 only — post-hoc A is the reported model.
    t_final = np.array(
        [
            predict_echo_time(
                float(th_i),
                cx=cx,
                cy=cy,
                R0=R,
                A=A,
                theta0_deg=th0,
                c=c,
                array_look_offset=offset,
                pulse_width=pw,
                half_deg=half_deg,
            )
            for th_i in thetas
        ],
        dtype=np.float64,
    )
    rmse_t = float(np.sqrt(np.mean((t_final - t_echo) ** 2)))
    best.x = np.array([R, cx, cy, A, th0, c], dtype=np.float64)
    scale_mode = "c_cal" if use_c_cal else "free_c"
    method = (
        "joint parametric wall (circle+cosine dent) + fluid c; "
        "2-stage + θ0 multi-start; instrument array_look_offset only; "
        "no R_nominal prior"
    )
    if use_c_cal:
        method += "; c_cal frozen (absolute scale)"
    out: dict[str, Any] = {
        "R": round(R, 6),
        "cx": round(cx, 6),
        "cy": round(cy, 6),
        "A": round(A, 6),
        "theta0_deg": round(th0, 4),
        "c": round(c, 8),
        "A_over_R": round(A / R, 6),
        "dentCall": bool(dent_call),
        "e_over_R": round(e_over_r, 6),
        "e_over_R_fit": round(e_over_r_fit, 6),
        "e_over_R_opp": round(e_over_r_opp, 6),
        "e_over_R_span": round(e_over_r_span, 6),
        "e": round(e, 6),
        "dentCorr": round(corr, 4),
        "dentCorrPos": round(corr_pos, 4),
        "ovalEnergy": round(e_oval, 4),
        "dentEnergy": round(e_dent, 4),
        "secondRatio": round(second_ratio, 4),
        "tau_R_over_c": round(R / c, 6),
        "r_pred": [round(float(v), 6) for v in r_pred],
        "wall_xy": wall_xy,
        "theta_deg": [float(v) for v in thetas],
        "t_echo": [float(v) for v in t_echo],
        "t_rmse": round(rmse_t, 6),
        "success": bool(best.success),
        "nfev": nfev,
        "method": method,
        "scaleMode": scale_mode,
        "gaugeRepaired": bool(gauge_repaired),
        "array_look_offset": offset,
        "pulse_width": pw,
        "dent_half_deg": float(half_deg),
    }
    if use_c_cal:
        out["c_cal"] = round(float(c0), 8)

    # Block 7a: approximate CIs + instrument-only refuse (no dual-obs yet).
    from sonolab.pipe2d.uncertainty import annotate_uncertainty, free_param_indices

    fit_lo = lo2 if use_ref else lo
    fit_hi = hi2 if use_ref else hi
    annotate_uncertainty(
        out,
        x=np.asarray(best.x, dtype=np.float64),
        t_echo=t_fit,
        thetas=th_fit,
        offset=offset,
        pw=pw,
        half_deg=half_deg,
        lo=fit_lo,
        hi=fit_hi,
        free_idx=free_param_indices(use_ref=use_ref, use_c_cal=use_c_cal),
    )
    return out


def estimate_from_pack(
    pack: AcquisitionPack,
    *,
    c_init: float | None = None,
    c_cal: float | None = None,
    R_init: float | None = None,
    half_deg: float = DENT_HALF_DEG,
    circle_ref: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Lab B entrypoint: AcquisitionPack only (no truth fields).

    Optional ``c_cal`` freezes fluid sound speed for absolute geometry.
    Optional ``circle_ref`` from a dent-OFF fit freezes (R,cx,cy); ``c_cal``
    wins for c when both are set.
    """
    return fit_joint_geometry(
        pack.t_echo,
        pack.theta_deg,
        array_look_offset=pack.array_look_offset,
        pulse_width=pack.pulse_width,
        half_deg=half_deg,
        c_init=c_init,
        c_cal=c_cal,
        R_init=R_init,
        circle_ref=circle_ref,
    )
