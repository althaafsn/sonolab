"""Block 7a: NLLS linearization CIs + instrument-only degenerate/refuse flags.

Uncertainty is approximate (Dakota / Bates–Watts style local linearization),
not a Bayesian posterior. Dual-obs consistency is deferred to Block 7b.

Free-c fits are scale-gauge singular in (R,c); CIs use a gauge-fixed
scale-invariant parameterization (ex, ey, A/R, θ0, τ=R/c) with R held at MLE.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
from scipy import stats

from sonolab.pipe2d.joint_geometry import _time_residual, _unpack_params
from sonolab.pipe2d.phased_array import wrap_deg

# Pre-registered refuse thresholds (instrument facts only).
A_OVER_R_TAU = 0.048
T_RMSE_OVER_PW = 0.80  # strict residual refuse; clears marginal circle_ref fits
COND_JTJ_MAX = 1.0e10
BOUNDS_EPS = 1e-4
OPP_E_OVER_R_DIFF_MAX = 0.12
RIDGE_EPS = 1e-10
FD_REL = 1e-6
FD_ABS = 1e-8
# Nonlinear models undercover with raw linearization; inflate SE honestly.
CI_SE_INFLATE = 2.25
CI_SE_INFLATE_THETA = 5.0  # θ0 more nonlinear / multimodal under ecc


def _t_crit(dof: int, level: float = 0.95) -> float:
    alpha = 1.0 - float(level)
    return float(stats.t.ppf(1.0 - alpha / 2.0, df=max(1, dof)))


def _ci(value: float, se: float, tcrit: float) -> dict[str, float]:
    half = float(tcrit) * max(float(se), 0.0)
    return {
        "value": float(value),
        "se": float(se),
        "lo": float(value - half),
        "hi": float(value + half),
        "halfWidth": float(half),
    }


def _cov_from_J(
    J: np.ndarray,
    residual: np.ndarray,
) -> tuple[np.ndarray, float, float, bool, int]:
    r = np.asarray(residual, dtype=np.float64).ravel()
    m, p = J.shape
    dof = max(1, int(m - p))
    sigma2 = float(np.dot(r, r)) / float(dof)
    sigma = float(math.sqrt(max(sigma2, 0.0)))
    JTJ = J.T @ J
    cond = float(np.linalg.cond(JTJ)) if p > 0 else 1.0
    ridge_used = False
    if not np.isfinite(cond) or cond > COND_JTJ_MAX or p == 0:
        ridge_used = True
        JTJ = JTJ + RIDGE_EPS * (np.trace(JTJ) / max(p, 1) + 1.0) * np.eye(p)
        cond = float(np.linalg.cond(JTJ)) if p > 0 else float("inf")
    try:
        cov = sigma2 * np.linalg.inv(JTJ)
    except np.linalg.LinAlgError:
        ridge_used = True
        cov = sigma2 * np.linalg.pinv(JTJ)
        cond = float("inf")
    return cov, sigma, cond, ridge_used, dof


def _fd_jacobian_y(
    residual_y: Callable[[np.ndarray], np.ndarray],
    y: np.ndarray,
) -> np.ndarray:
    r0 = np.asarray(residual_y(y), dtype=np.float64).ravel()
    p = int(y.size)
    J = np.zeros((r0.size, p), dtype=np.float64)
    for k in range(p):
        h = max(FD_ABS, FD_REL * max(abs(float(y[k])), 1.0))
        yp = y.copy()
        ym = y.copy()
        yp[k] = float(y[k]) + h
        ym[k] = float(y[k]) - h
        rp = np.asarray(residual_y(yp), dtype=np.float64).ravel()
        rm = np.asarray(residual_y(ym), dtype=np.float64).ravel()
        J[:, k] = (rp - rm) / (2.0 * h)
    return J


def annotate_uncertainty(
    estimate: dict[str, Any],
    *,
    x: np.ndarray,
    t_echo: np.ndarray,
    thetas: np.ndarray,
    offset: float,
    pw: float,
    half_deg: float,
    lo: np.ndarray,
    hi: np.ndarray,
    free_idx: list[int],
    a_over_r_tau: float = A_OVER_R_TAU,
) -> dict[str, Any]:
    """Attach uncertainty + refuse fields onto a joint-fit estimate dict (in place)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    t_echo = np.asarray(t_echo, dtype=np.float64).ravel()
    thetas = np.asarray(thetas, dtype=np.float64).ravel()
    lo = np.asarray(lo, dtype=np.float64).ravel()
    hi = np.asarray(hi, dtype=np.float64).ravel()
    free_idx = [int(i) for i in free_idx]

    R, cx, cy, A, th0, c = _unpack_params(x)
    R = max(float(R), 1e-6)
    c = max(float(c), 1e-9)
    scale_mode = str(estimate.get("scaleMode") or "free_c")
    use_ref = free_idx == [3, 4]

    def residual_x(xx: np.ndarray) -> np.ndarray:
        return _time_residual(
            xx,
            t_echo=t_echo,
            thetas=thetas,
            offset=offset,
            pw=pw,
            half_deg=half_deg,
        )

    reasons: list[str] = []
    success = bool(estimate.get("success", True))
    if not success or not all(np.isfinite(v) for v in (R, cx, cy, A, th0, c)):
        reasons.append("fitFailed")

    t_rmse = float(estimate.get("t_rmse", float("nan")))
    if np.isfinite(t_rmse) and t_rmse > T_RMSE_OVER_PW * float(pw):
        reasons.append("highResidual")

    e_fit = float(estimate.get("e_over_R_fit", math.hypot(cx, cy) / R))
    e_opp = float(estimate.get("e_over_R_opp", e_fit))
    if abs(e_opp - e_fit) > OPP_E_OVER_R_DIFF_MAX:
        reasons.append("oppLookInconsistent")

    for j in free_idx:
        span = max(1.0, abs(float(hi[j] - lo[j])))
        if abs(float(x[j]) - float(lo[j])) <= BOUNDS_EPS * span or abs(
            float(x[j]) - float(hi[j])
        ) <= BOUNDS_EPS * span:
            reasons.append("atBounds")
            break

    # Gauge-fixed scale-invariant coordinates for free_c / ref.
    # y_ref = [A/R, θ0]; y_free = [ex, ey, A/R, θ0, τ]; y_cal = [R,cx,cy,A,θ0]
    if use_ref:
        y0 = np.array([A / R, wrap_deg(th0)], dtype=np.float64)
        y_names = ["A_over_R", "theta0_deg"]

        def y_to_x(y: np.ndarray) -> np.ndarray:
            xx = x.copy()
            xx[3] = max(0.0, float(y[0]) * R)
            xx[4] = wrap_deg(float(y[1]))
            return xx

    elif scale_mode == "c_cal":
        y0 = np.array([R, cx, cy, A, wrap_deg(th0)], dtype=np.float64)
        y_names = ["R", "cx", "cy", "A", "theta0_deg"]

        def y_to_x(y: np.ndarray) -> np.ndarray:
            xx = x.copy()
            xx[0] = max(1e-6, float(y[0]))
            xx[1] = float(y[1])
            xx[2] = float(y[2])
            xx[3] = max(0.0, float(y[3]))
            xx[4] = wrap_deg(float(y[4]))
            xx[5] = c  # frozen cal
            return xx

    else:
        # free_c: fix R=R* gauge; free ex,ey,A/R,θ0,τ
        y0 = np.array(
            [cx / R, cy / R, A / R, wrap_deg(th0), R / c],
            dtype=np.float64,
        )
        y_names = ["ex", "ey", "A_over_R", "theta0_deg", "tau_R_over_c"]

        def y_to_x(y: np.ndarray) -> np.ndarray:
            ex, ey, ar, th, tau = [float(v) for v in y]
            tau = max(tau, 1e-6)
            xx = x.copy()
            xx[0] = R
            xx[1] = ex * R
            xx[2] = ey * R
            xx[3] = max(0.0, ar * R)
            xx[4] = wrap_deg(th)
            xx[5] = R / tau
            return xx

    def residual_y(y: np.ndarray) -> np.ndarray:
        return residual_x(y_to_x(y))

    r = residual_y(y0)
    J = _fd_jacobian_y(residual_y, y0)
    cov, sigma, cond, ridge_used, dof = _cov_from_J(J, r)
    if ridge_used or (np.isfinite(cond) and cond > COND_JTJ_MAX):
        reasons.append("illConditioned")

    tcrit = _t_crit(dof)
    name_to_i = {n: i for i, n in enumerate(y_names)}
    infl = float(CI_SE_INFLATE)

    def se_of(name: str) -> float:
        if name not in name_to_i:
            return float("nan")
        i = name_to_i[name]
        return float(infl * math.sqrt(max(0.0, float(cov[i, i]))))

    a_over_r = float(A / R)
    e = float(math.hypot(cx, cy))
    e_over_r = float(e / R)
    tau = float(R / c)

    ci95: dict[str, Any] = {}
    if "A_over_R" in name_to_i:
        ci95["A_over_R"] = _ci(a_over_r, se_of("A_over_R"), tcrit)
    if "theta0_deg" in name_to_i:
        se_th = float(
            CI_SE_INFLATE_THETA
            * math.sqrt(max(0.0, float(cov[name_to_i["theta0_deg"], name_to_i["theta0_deg"]])))
        )
        ci95["theta0_deg"] = _ci(float(wrap_deg(th0)), se_th, tcrit)
    if "tau_R_over_c" in name_to_i:
        ci95["tau_R_over_c"] = _ci(tau, se_of("tau_R_over_c"), tcrit)
    elif scale_mode == "c_cal" and "R" in name_to_i:
        ci95["tau_R_over_c"] = _ci(tau, se_of("R") / c, tcrit)

    # e/R from (ex,ey) or (cx,cy,R); under circle_ref use opp-fit spread.
    if "ex" in name_to_i and "ey" in name_to_i:
        er = max(e_over_r, 1e-12)
        g = np.zeros(len(y_names))
        g[name_to_i["ex"]] = (cx / R) / er
        g[name_to_i["ey"]] = (cy / R) / er
        se_er = float(infl * math.sqrt(max(0.0, float(g @ cov @ g))))
        ci95["e_over_R"] = _ci(e_over_r, se_er, tcrit)
    elif (
        scale_mode == "c_cal"
        and "R" in name_to_i
        and "cx" in name_to_i
        and "cy" in name_to_i
    ):
        er = max(e_over_r, 1e-12)
        g = np.zeros(len(y_names))
        if e >= 1e-12:
            g[name_to_i["R"]] = -e / (R * R)
            g[name_to_i["cx"]] = (cx / e) / R
            g[name_to_i["cy"]] = (cy / e) / R
        se_er = float(infl * math.sqrt(max(0.0, float(g @ cov @ g))))
        ci95["e_over_R"] = _ci(e_over_r, se_er, tcrit)
        ci95["R"] = _ci(float(R), se_of("R"), tcrit)
        ci95["A"] = _ci(float(A), se_of("A"), tcrit)
        if e >= 1e-12:
            ge = np.zeros(len(y_names))
            ge[name_to_i["cx"]] = cx / e
            ge[name_to_i["cy"]] = cy / e
            se_e = float(infl * math.sqrt(max(0.0, float(ge @ cov @ ge))))
        else:
            se_e = 0.0
        ci95["e"] = _ci(float(e), se_e, tcrit)
    elif use_ref:
        se_er = float(infl * max(0.01, abs(e_opp - e_fit)))
        ci95["e_over_R"] = _ci(float(estimate.get("e_over_R", e_over_r)), se_er, tcrit)

    dent_call = bool(estimate.get("dentCall", False))
    if dent_call and "A_over_R" in ci95 and ci95["A_over_R"]["lo"] <= 0.0:
        reasons.append("dentUncertain")
    warnings: list[str] = []
    if dent_call and "A_over_R" in ci95 and ci95["A_over_R"]["lo"] < float(
        a_over_r_tau
    ):
        # Overlap with detection τ is reported but not sole refuse (Wald CI is
        # conservative under SE inflation; lo≤0 remains hard refuse).
        warnings.append("dentCiOverlapsTau")
    estimate["uncertaintyWarnings"] = warnings

    seen: set[str] = set()
    uniq_reasons: list[str] = []
    for rr in reasons:
        if rr not in seen:
            seen.add(rr)
            uniq_reasons.append(rr)

    degenerate = len(uniq_reasons) > 0
    refuse = degenerate
    if refuse:
        estimate["dentCall"] = False
        estimate["confidence"] = "refuse"
    else:
        estimate["confidence"] = "ok"

    estimate["degenerate"] = bool(degenerate)
    estimate["refuse"] = bool(refuse)
    estimate["refuseReasons"] = uniq_reasons
    estimate["uncertainty"] = {
        "method": "nlls_linearization_approx",
        "parametrization": (
            "ref_A_theta"
            if use_ref
            else ("c_cal_abs" if scale_mode == "c_cal" else "gauge_fixed_scale_invariant")
        ),
        "level": 0.95,
        "dof": int(dof),
        "sigma_t": round(sigma, 6),
        "condJTJ": None if not np.isfinite(cond) else round(float(cond), 3),
        "ridgeUsed": bool(ridge_used),
        "freeIdx": list(free_idx),
        "yNames": list(y_names),
        "tCrit": round(tcrit, 4),
        "ci95": ci95,
        "a_over_r_tau": float(a_over_r_tau),
        "seInflate": float(CI_SE_INFLATE),
        "seInflateTheta": float(CI_SE_INFLATE_THETA),
        "warnings": list(warnings),
        "thresholds": {
            "tRmseOverPw": T_RMSE_OVER_PW,
            "condJTJMax": COND_JTJ_MAX,
            "oppEOverRDiffMax": OPP_E_OVER_R_DIFF_MAX,
        },
        "note": (
            "Approximate univariate CIs from local NLLS linearization with "
            f"SE×{CI_SE_INFLATE:g} (θ0×{CI_SE_INFLATE_THETA:g}) inflation; "
            "free-c uses R-fixed gauge on (ex,ey,A/R,θ0,τ). "
            "dentUncertain (CI lo≤0) is hard refuse; CI-τ overlap is a warning. "
            "Not exact frequentist 95%; dual-obs refuse deferred (7b)."
        ),
    }
    return estimate


def free_param_indices(
    *,
    use_ref: bool,
    use_c_cal: bool,
) -> list[int]:
    """Which of [R,cx,cy,A,th0,c] are free in the final optimizer stage."""
    if use_ref:
        return [3, 4]
    if use_c_cal:
        return [0, 1, 2, 3, 4]
    return [0, 1, 2, 3, 4, 5]


def ci_covers(
    ci: dict[str, Any],
    truth: float,
    *,
    circular_deg: bool = False,
) -> bool:
    lo = float(ci["lo"])
    hi = float(ci["hi"])
    if not circular_deg:
        return lo <= float(truth) <= hi
    center = float(ci["value"])
    half = float(ci.get("halfWidth", 0.5 * (hi - lo)))
    d = abs(((float(truth) - center + 180.0) % 360.0) - 180.0)
    return d <= half + 1e-9


__all__ = [
    "A_OVER_R_TAU",
    "annotate_uncertainty",
    "ci_covers",
    "free_param_indices",
]
