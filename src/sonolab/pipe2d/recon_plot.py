"""Truth-vs-recon wall overlay plots (scorer / demo only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sonolab.pipe2d.blind import TruthPack
from sonolab.pipe2d.phased_array import look_unit


def write_recon_overlay(
    path: Path | str,
    *,
    estimate: dict[str, Any],
    truth: TruthPack,
    title: str | None = None,
) -> Path:
    """PNG: tool origin, true wall, estimated wall polyline."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    wall = estimate.get("wall_xy") or []
    if wall:
        ex = np.array([p[0] for p in wall], dtype=np.float64)
        ey = np.array([p[1] for p in wall], dtype=np.float64)
    else:
        r_pred = np.asarray(estimate.get("r_pred") or [], dtype=np.float64)
        th = np.asarray(estimate.get("theta_deg") or truth.theta_deg, dtype=np.float64)
        ex, ey = [], []
        for th_i, r_i in zip(th, r_pred):
            ux, uy = look_unit(float(th_i))
            ex.append(r_i * ux)
            ey.append(r_i * uy)
        ex = np.asarray(ex, dtype=np.float64)
        ey = np.asarray(ey, dtype=np.float64)

    r_true = np.asarray(truth.r_true, dtype=np.float64)
    th_t = np.asarray(truth.theta_deg, dtype=np.float64)
    tx, ty = [], []
    for th_i, r_i in zip(th_t, r_true):
        ux, uy = look_unit(float(th_i))
        tx.append(r_i * ux)
        ty.append(r_i * uy)
    tx = np.asarray(tx, dtype=np.float64)
    ty = np.asarray(ty, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=140)
    ax.set_aspect("equal")
    ax.plot(0.0, 0.0, "k+", markersize=10, label="tool")
    if tx.size:
        ax.plot(
            np.r_[tx, tx[0]],
            np.r_[ty, ty[0]],
            color="#1f4e79",
            lw=2.0,
            label="truth wall",
        )
    if ex.size:
        ax.plot(
            np.r_[ex, ex[0]],
            np.r_[ey, ey[0]],
            color="#c45c26",
            lw=1.8,
            ls="--",
            label="Lab B recon",
        )
    cx = float(estimate.get("cx", 0.0))
    cy = float(estimate.get("cy", 0.0))
    ax.plot(cx, cy, "o", color="#c45c26", ms=5, label="est center")
    th0 = float(estimate.get("theta0_deg", 0.0))
    R = float(estimate.get("R", 0.0))
    if estimate.get("dentCall") and R > 0:
        ux, uy = look_unit(th0)
        ax.annotate(
            "",
            xy=(R * ux, R * uy),
            xytext=(0.55 * R * ux, 0.55 * R * uy),
            arrowprops=dict(arrowstyle="->", color="#c45c26", lw=1.4),
        )
    mode = str(estimate.get("scaleMode") or "free_c")
    ttl = title or f"pipe recon ({mode})"
    if mode == "c_cal":
        ttl += (
            f"  R={estimate.get('R'):.2f}  e={estimate.get('e'):.2f}  "
            f"A={estimate.get('A'):.2f}"
        )
    ax.set_title(ttl)
    ax.set_xlabel("x (grid)")
    ax.set_ylabel("y (grid)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
