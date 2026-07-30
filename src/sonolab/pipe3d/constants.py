"""Pipe3d constants - short-segment 3D scalar acoustic Lab A (smoke grid)."""

from __future__ import annotations

import math

# Transverse / axial grid (smoke envelope - true 3D, not full joint).
NXY = 64
NZ = 80  # long enough that wall ToF beats axial-face echoes
CFL = 0.45
C0 = 1.0
C_WALL = 0.55
C_PROP = CFL * C0

CX = (NXY - 1) * 0.5
CY = (NXY - 1) * 0.5
CZ = (NZ - 1) * 0.5

PIPE_R = 22.0
WALL_THICK = 4.0
DENT_DEPTH = 3.0
DENT_HALF_DEG = 25.0
DENT_HALF_Z = 6.0  # axial cosine half-width (cells)
DENT_THETA0_DEG = 0.0
DENT_Z0 = CZ  # dent centered mid-slab

ARRAY_LOOK_OFFSET = 2.0
PULSE_WIDTH = 12.0
AMP = 0.55
FREQ = 0.35
TX_SIGMA = 1.6

ECC_MAX = 6.0
ASCAN_MARGIN = 40.0
_ASCAN_FAR = float(PIPE_R + ECC_MAX - ARRAY_LOOK_OFFSET)
ASCAN_LEN = int(
    math.ceil(PULSE_WIDTH + 2.0 * _ASCAN_FAR / C_PROP + ASCAN_MARGIN)
)
TX_EXCLUDE = int(PULSE_WIDTH * 1.8)

PHYSICS_LABEL = "3D_scalar_acoustic"
