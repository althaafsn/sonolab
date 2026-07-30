# Block 8 demo (GitHub bundle)

Interactive **3D pipe wall** from ultrasonic pulse-echo looks (Lab A scalar FDTD) + blind Lab B joint fit with labeled fluid \(c_{\mathrm{cal}}\).

## Open the viewer

```bash
# from repo root — open in a browser (needs sibling survey3d_view_data.js)
xdg-open docs/demo/block8/survey3d_view.html   # Linux
# open docs/demo/block8/survey3d_view.html     # macOS
```

Toggles: **recon** / **truth** / **both**. Color = radius from tool center (near → red, far → blue).

## What’s in this folder

| File | Role |
|------|------|
| `survey3d_view.html` + `survey3d_view_data.js` | Orbit / zoom / pan mesh viewer |
| `survey3d_cloud.png` / `survey3d_Rmap.png` | Still figures (full-run evidence) |
| `orbit.gif` / `orbit.mp4` | Short spin for README / LinkedIn (prefer `.mp4` on LinkedIn) |
| `scorecard.json` | Shape metrics + honesty notes |

## Density note

This GitHub mesh is **1° × 32 z** (~11k vertices) so the clone stays small. The full portfolio evidence used **display 0.25° × 64 z** on an acquire of **1° × 16 z**. Re-run locally:

```bash
pip install -e .
sonolab block8-demo --c-cal 0.45 --fast          # smoke, minutes
sonolab block8-demo --c-cal 0.45                 # full 1°×16 (tens of minutes)
```

Physics label: `3D_scalar_acoustic`. Not a field ILI product; do not claim commercial mm imaging density.
