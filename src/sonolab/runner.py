"""Run SPECFEM2D-UT EXAMPLE 01 via a materialized SonoLab job directory."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from sonolab import __version__
from sonolab.config import example01_template, require_specfem_binaries, runs_root
from sonolab.pack import pack_artifacts

# Files/dirs to copy from the stock example (exclude heavy outputs / venvs).
_COPY_NAMES = (
    "DATA",
    "dir_setup.sh",
    "plot_A_scan.py",
    "create_gif.py",
    "requirements.txt",
    "environment.yml",
    "README.md",
)


def _copy_template(template: Path, job_dir: Path) -> None:
    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True)
    for name in _COPY_NAMES:
        src = template / name
        if not src.exists():
            continue
        dst = job_dir / name
        if src.is_dir():
            shutil.copytree(
                src,
                dst,
                ignore=shutil.ignore_patterns(
                    "__pycache__",
                    "*.pyc",
                    "MESH",
                    "OUTPUT_FILES",
                ),
            )
        else:
            shutil.copy2(src, dst)
    # Clean generated mesh/output dirs if copied
    for sub in ("DATA/MESH", "OUTPUT_FILES"):
        p = job_dir / sub
        if p.exists():
            shutil.rmtree(p)
    (job_dir / "OUTPUT_FILES").mkdir(parents=True, exist_ok=True)
    # Drop any leftover MESH under DATA from a previous messy tree
    mesh = job_dir / "DATA" / "MESH"
    if mesh.exists():
        shutil.rmtree(mesh)


def _apply_job_spec_overlays(job_dir: Path, job_spec: dict[str, Any]) -> None:
    """Optionally override a few EXAMPLE 01 YAML fields from job_spec."""
    params_path = job_dir / "DATA" / "00_parameters.yaml"
    with params_path.open() as f:
        params = yaml.safe_load(f)

    mat = job_spec.get("materials", {}).get("wall") or job_spec.get("material")
    if isinstance(mat, dict):
        for k in ("vp", "vs", "rho"):
            if k in mat:
                params["material"][k] = float(mat[k])

    tr = job_spec.get("array") or job_spec.get("transducer")
    if isinstance(tr, dict):
        if "f0_hz" in tr:
            params["transducer"]["f0"] = float(tr["f0_hz"])
        if "f0" in tr:
            params["transducer"]["f0"] = float(tr["f0"])
        if "aperture_m" in tr:
            params["transducer"]["aperture"] = float(tr["aperture_m"])
        if "aperture" in tr:
            params["transducer"]["aperture"] = float(tr["aperture"])

    acq = job_spec.get("acquisition") or job_spec.get("simulation")
    if isinstance(acq, dict):
        if "total_time_s" in acq:
            params["simulation"]["total_time"] = float(acq["total_time_s"])
        if "total_time" in acq:
            params["simulation"]["total_time"] = float(acq["total_time"])

    with params_path.open("w") as f:
        yaml.safe_dump(params, f, default_flow_style=False, sort_keys=False)


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("+", " ".join(cmd), f"(cwd={cwd})")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def run_example01(
    job_id: str,
    *,
    job_spec: dict[str, Any] | None = None,
    job_spec_path: Path | None = None,
    skip_gif: bool = True,
    python_bin: str | None = None,
) -> Path:
    """
    Materialize EXAMPLE 01 under datasets/runs/<job_id>, run SPECFEM, pack artifacts.

    Returns the job directory path.
    """
    if job_spec is None and job_spec_path is not None:
        with Path(job_spec_path).open() as f:
            job_spec = yaml.safe_load(f)
    job_spec = job_spec or {}

    if "job_id" in job_spec and not job_id:
        job_id = str(job_spec["job_id"])
    if not job_id:
        raise ValueError("job_id is required")

    specfem = require_specfem_binaries()
    template = example01_template()
    if not template.is_dir():
        raise FileNotFoundError(f"EXAMPLE 01 template missing: {template}")

    job_dir = runs_root() / job_id
    runs_root().mkdir(parents=True, exist_ok=True)
    _copy_template(template, job_dir)

    # Persist job_spec for provenance
    with (job_dir / "job_spec.yaml").open("w") as f:
        yaml.safe_dump(
            {**job_spec, "job_id": job_id, "backend": "example01"},
            f,
            default_flow_style=False,
            sort_keys=False,
        )
    _apply_job_spec_overlays(job_dir, job_spec)

    py = python_bin or os.environ.get("PYTHON_BIN") or shutil.which("python3") or "python3"
    env = os.environ.copy()
    env["SPECFEM2D_DIR"] = str(specfem)
    env["SONOLAB_SPECFEM_ROOT"] = str(specfem)
    env["PATH"] = f"{specfem / 'bin'}:{env.get('PATH', '')}"

    t0 = time.perf_counter()
    _run([py, str(job_dir / "DATA" / "01_create_mesh.py")], cwd=job_dir, env=env)
    _run([py, str(job_dir / "DATA" / "03_update_par_file.py")], cwd=job_dir, env=env)
    _run([py, str(job_dir / "DATA" / "02_create_source_station.py")], cwd=job_dir, env=env)
    _run([str(specfem / "bin" / "xmeshfem2D")], cwd=job_dir, env=env)
    _run([str(specfem / "bin" / "xspecfem2D")], cwd=job_dir, env=env)
    _run([py, str(job_dir / "plot_A_scan.py")], cwd=job_dir, env=env)
    if not skip_gif:
        _run([py, str(job_dir / "create_gif.py")], cwd=job_dir, env=env)
    elapsed = time.perf_counter() - t0

    art = pack_artifacts(
        job_dir,
        job_spec=job_spec,
        platform_version=__version__,
        elapsed_s=elapsed,
    )
    print(f"SonoLab artifacts → {art}")
    print(f"elapsed_s={elapsed:.1f}")
    return job_dir
