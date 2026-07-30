#!/usr/bin/env bash
# Open SPECFEM3D smoke outputs in ParaView (mesh + sources/receivers + surface movie).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
READY="$ROOT/datasets/runs/_orientation/paraview_ready"

if ! command -v paraview >/dev/null 2>&1; then
  echo "paraview not found. Install with: sudo apt install paraview"
  exit 1
fi

if [[ ! -f "$READY/sr.vtk" ]]; then
  echo "Missing GUI files under $READY"
  echo "See datasets/runs/_orientation/PARAVIEW_GUI.md"
  exit 1
fi

# Ubuntu GNOME Wayland often has /tmp/.X11-unix/X0 and X1. hwloc's GL
# component probes every X socket and can hang forever on X1, so ParaView
# never shows a window. Disable that probe.
export HWLOC_COMPONENTS=-gl

# Prefer a clean system PATH so micromamba MPI/Qt does not shadow apt ParaView.
export PATH="/usr/bin:/bin:${PATH}"

echo "Opening ParaView with mesh Vp + sources/receivers."
echo "Then: File → Open → select all AVS_movie_*.inp in:"
echo "  $READY"
echo "→ Apply → press Play for the surface-velocity movie."

exec /usr/bin/paraview \
  --data="$READY/mesh_vp_proc0.vtk" \
  --data="$READY/sr.vtk"
