"""Resolve SPECFEM2D-UT and SonoLab paths."""

from __future__ import annotations

import os
from pathlib import Path


def sonolab_root() -> Path:
    """Package / repo root containing ``src/sonolab``."""
    return Path(__file__).resolve().parents[2]


def default_specfem_root() -> Path:
    """Prefer nested ``sonolab/specfem2d-UT``, else sibling ``../specfem2d-UT``."""
    nested = sonolab_root() / "specfem2d-UT"
    if nested.is_dir():
        return nested
    return sonolab_root().parent / "specfem2d-UT"


def specfem_root() -> Path:
    """
    SPECFEM2D-UT root.

    Override with env SONOLAB_SPECFEM_ROOT or SPECFEM2D_DIR.
    Else nested checkout, else sibling.
    """
    for key in ("SONOLAB_SPECFEM_ROOT", "SPECFEM2D_DIR"):
        raw = os.environ.get(key)
        if raw:
            return Path(raw).expanduser().resolve()
    return default_specfem_root().resolve()


def example01_template() -> Path:
    return specfem_root() / "EXAMPLES" / "01_Conventional_Pulse_Echo"


def runs_root() -> Path:
    return sonolab_root() / "datasets" / "runs"


def require_specfem_binaries(root: Path | None = None) -> Path:
    root = (root or specfem_root()).resolve()
    xmesh = root / "bin" / "xmeshfem2D"
    xspec = root / "bin" / "xspecfem2D"
    gmsh_util = (
        root
        / "utils"
        / "Gmsh"
        / "LibGmsh2Specfem_convert_Gmsh_to_Specfem2D_official.py"
    )
    missing = [p for p in (xmesh, xspec, gmsh_util) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "SPECFEM2D-UT binaries/utils missing. Expected under "
            f"{root}. Build with: ./configure FC=gfortran && make all\n"
            f"Missing: {', '.join(str(p) for p in missing)}"
        )
    if not os.access(xmesh, os.X_OK) or not os.access(xspec, os.X_OK):
        raise PermissionError(f"SPECFEM binaries not executable under {root / 'bin'}")
    return root
