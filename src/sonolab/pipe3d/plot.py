"""Plots for pipe3d survey: unwrapped R(θ,z), PNG overlay, interactive HTML."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sonolab.pipe3d.display import (
    DISPLAY_N_Z,
    DISPLAY_STEP_DEG,
    build_display_wall,
    viewer_payload,
)
from sonolab.pipe3d.survey import SurveyTruth


def write_survey_plots(
    plot_dir: Path | str,
    *,
    product: dict[str, Any],
    truth: SurveyTruth,
    score: dict[str, Any] | None = None,
    display: dict[str, Any] | None = None,
    display_step_deg: float = DISPLAY_STEP_DEG,
    n_z_display: int = DISPLAY_N_Z,
    acquire_step_deg: float | None = None,
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    R_est = np.asarray(product["R_map"], dtype=np.float64)
    R_true = np.asarray(truth.r_true, dtype=np.float64)
    th = np.asarray(product["theta_deg"], dtype=np.float64)
    z = np.asarray(product["z"], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), dpi=130)
    extent = [float(th[0]), float(th[-1]), float(z[0]), float(z[-1])]
    for ax, data, title in zip(
        axes,
        [R_true, R_est, R_est - R_true],
        ["truth R(θ,z)", "Lab B R(θ,z)", "residual"],
    ):
        im = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis" if title != "residual" else "coolwarm",
        )
        ax.set_xlabel("θ (deg)")
        ax.set_ylabel("z (grid)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    ttl = "pipe3d survey (3D scalar)"
    if score is not None:
        ttl += f"  rAbs/R={score.get('rAbsRmseOverR')}"
    fig.suptitle(ttl)
    fig.tight_layout()
    p_map = plot_dir / "survey3d_Rmap.png"
    fig.savefig(p_map)
    plt.close(fig)
    out["rmap"] = p_map

    # 3D cloud PNG (acquire-resolution points)
    fig = plt.figure(figsize=(7.0, 6.0), dpi=130)
    ax = fig.add_subplot(111, projection="3d")
    for zi, row in zip(truth.z, truth.r_true):
        xs, ys = [], []
        for th_i, r in zip(truth.theta_deg, row):
            rad = np.radians(th_i)
            xs.append(r * np.sin(rad))
            ys.append(r * np.cos(rad))
        xs.append(xs[0])
        ys.append(ys[0])
        ax.plot(xs, ys, [zi] * len(xs), color="#1f4e79", lw=1.0, alpha=0.7)
    xyz = np.asarray(product.get("xyz") or [], dtype=np.float64)
    if xyz.size:
        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=8,
            c="#c45c26",
            alpha=0.85,
            label="Lab B",
        )
    ax.plot([0], [0], [float(np.mean(z))], "k+", markersize=10, label="tool")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("3D wall: truth rings + Lab B cloud")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    p_cloud = plot_dir / "survey3d_cloud.png"
    fig.savefig(p_cloud)
    plt.close(fig)
    out["cloud"] = p_cloud

    if display is None:
        display = build_display_wall(
            product,
            truth,
            display_step_deg=float(display_step_deg),
            n_z_display=int(n_z_display),
            acquire_step_deg=acquire_step_deg,
            acquire_n_z=int(len(product["z"])),
        )
    paths = write_survey_view(
        plot_dir,
        display=display,
        score=score,
    )
    out.update(paths)
    return out


def write_survey_view(
    plot_dir: Path | str,
    *,
    display: dict[str, Any],
    score: dict[str, Any] | None = None,
    html_name: str = "survey3d_view.html",
    data_json_name: str = "survey3d_view_data.json",
    data_js_name: str = "survey3d_view_data.js",
) -> dict[str, Path]:
    """Write external viewer data + HTML (mesh, toggles, honest HUD)."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    payload = viewer_payload(display, score=score)

    p_json = plot_dir / data_json_name
    p_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Classic script for file:// open (fetch of JSON is blocked by browsers).
    p_js = plot_dir / data_js_name
    p_js.write_text(
        "window.SURVEY3D_DATA = " + json.dumps(payload) + ";\n",
        encoding="utf-8",
    )

    p_html = write_survey_html(plot_dir / html_name, data_js_name=data_js_name)
    return {"html": p_html, "view_data": p_json, "view_js": p_js}


def write_survey_html(
    path: Path | str,
    *,
    data_js_name: str = "survey3d_view_data.js",
    # Back-compat kwargs ignored when external data is used
    product: dict[str, Any] | None = None,
    truth: SurveyTruth | None = None,
    score: dict[str, Any] | None = None,
) -> Path:
    """Interactive Three.js mesh viewer; loads external SURVEY3D_DATA sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Legacy callers that still pass product/truth: build display + sidecar first.
    if product is not None and truth is not None:
        display = build_display_wall(
            product,
            truth,
            display_step_deg=DISPLAY_STEP_DEG,
            n_z_display=DISPLAY_N_Z,
        )
        write_survey_view(
            path.parent,
            display=display,
            score=score,
            html_name=path.name,
        )
        return path

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>SonoLab pipe3d view</title>
  <style>
    html, body {{ margin: 0; height: 100%; background: #0e1116; color: #e8eaed;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif; overflow: hidden; }}
    #left {{ position: absolute; left: 16px; top: 14px; z-index: 2; max-width: 420px;
      display: flex; flex-direction: column; gap: 10px; }}
    #hud {{ pointer-events: none; text-shadow: 0 1px 2px #000; }}
    #hud h1 {{ margin: 0 0 6px; font-size: 18px; font-weight: 600; letter-spacing: 0.02em; }}
    #hud a.home {{
      pointer-events: auto; display: inline-block; margin: 0 0 8px; font-size: 12px;
      color: #9aa3af; text-decoration: none; border-bottom: 1px solid transparent;
    }}
    #hud a.home:hover {{ color: #e8eaed; border-bottom-color: #6a8fad; }}
    #hud p {{ margin: 0; font-size: 12.5px; opacity: 0.88; line-height: 1.45; white-space: pre-line; }}
    #toggles {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    #toggles button {{
      pointer-events: auto; cursor: pointer; border: 1px solid #3a4250;
      background: #1a2030; color: #e8eaed; font-size: 12px; padding: 6px 10px;
      border-radius: 4px; font-family: inherit;
    }}
    #toggles button.on {{ background: #2a4060; border-color: #6a9cc8; }}
    #cbar {{ position: absolute; right: 16px; top: 14px; z-index: 2;
      width: 18px; height: 160px; border-radius: 3px; border: 1px solid #3a4250;
      background: linear-gradient(to top, #2b6cb0, #38b2ac, #ecc94b, #e53e3e);
      pointer-events: none; box-shadow: 0 1px 4px rgba(0,0,0,0.45); }}
    #cbar-labels {{ position: absolute; right: 40px; top: 14px; z-index: 2;
      height: 160px; display: flex; flex-direction: column; justify-content: space-between;
      font-size: 11px; opacity: 0.9; text-align: right; pointer-events: none;
      text-shadow: 0 1px 2px #000; }}
    #hint {{ position: absolute; right: 16px; bottom: 14px; z-index: 2; font-size: 12px;
      opacity: 0.7; pointer-events: none; }}
    canvas {{ display: block; }}
  </style>
</head>
<body>
  <div id="left">
    <div id="hud">
      <a class="home" href="../../index.html">← SonoLab</a>
      <h1>pipe3d wall</h1>
      <p id="meta">Loading…</p>
    </div>
    <div id="toggles">
      <button type="button" id="btn-recon" class="on">recon</button>
      <button type="button" id="btn-truth" class="on">truth</button>
      <button type="button" id="btn-both" class="on">both</button>
    </div>
  </div>
  <div id="cbar" title="radius from tool center"></div>
  <div id="cbar-labels">
    <span id="cbar-max">far</span>
    <span>radius</span>
    <span id="cbar-min">near</span>
  </div>
  <div id="hint">drag orbit · scroll zoom · right-drag pan</div>
  <script src="{data_js_name}"></script>
  <script type="importmap">
  {{
    "imports": {{
      "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
      "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
    }}
  }}
  </script>
  <script type="module">
    import * as THREE from "three";
    import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";

    const DATA = window.SURVEY3D_DATA;
    if (!DATA) {{
      document.getElementById("meta").textContent =
        "Missing survey3d_view_data.js sidecar next to this HTML.";
      throw new Error("SURVEY3D_DATA missing");
    }}

    const meta = document.getElementById("meta");
    const sc = DATA.score || {{}};
    const m = DATA.meta || {{}};
    const acq = (m.acquireStepDeg ?? "?") + "° × " + (m.acquireNz ?? "?") + " z";
    const disp = (m.displayStepDeg ?? "?") + "° × " + (m.displayNz ?? "?") + " z";
    meta.textContent =
      (DATA.title || "SonoLab pipe3d") + "\\n" +
      "acquire " + acq + "  ·  display " + disp + "\\n" +
      (m.honesty || "") + "\\n" +
      "physics=" + (sc.physics || m.physics || "?") + "\\n" +
      "rAbs/R=" + (sc.rAbsRmseOverR ?? "—") +
      "  θerr=" + (sc.dentThetaErrDeg ?? "—") + "°" +
      "  zerr=" + (sc.dentZErr ?? "—") + "\\n" +
      "color = radius from tool center (near→red, far→blue)\\n" +
      "not field mm-resolution; scalar FDTD portfolio demo";

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0e1116);
    const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.1, 2000);
    camera.position.set(55, 40, 70);

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setSize(innerWidth, innerHeight);
    document.body.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    const zMid = (DATA.dentHighlight && DATA.dentHighlight.z != null)
      ? DATA.dentHighlight.z : 40;
    controls.target.set(0, 0, zMid);

    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const key = new THREE.DirectionalLight(0xffffff, 0.75);
    key.position.set(40, 60, 30);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xa0c4ff, 0.25);
    fill.position.set(-30, -20, 50);
    scene.add(fill);

    {{
      const g = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, zMid - 20), new THREE.Vector3(0, 0, zMid + 20)
      ]);
      scene.add(new THREE.Line(g, new THREE.LineBasicMaterial({{ color: 0x888888 }})));
      const mark = new THREE.Mesh(
        new THREE.SphereGeometry(0.55, 12, 12),
        new THREE.MeshBasicMaterial({{ color: 0xffffff }})
      );
      mark.position.set(0, 0, zMid);
      scene.add(mark);
    }}

    // Radius colormap: near (dent / small R) → red, mid → amber/teal, far → blue
    const RAMP = [
      new THREE.Color(0xe53e3e),
      new THREE.Color(0xecc94b),
      new THREE.Color(0x38b2ac),
      new THREE.Color(0x2b6cb0),
    ];
    function colorAtT(t, out) {{
      const x = Math.min(1, Math.max(0, t)) * (RAMP.length - 1);
      const i0 = Math.floor(x);
      const i1 = Math.min(RAMP.length - 1, i0 + 1);
      out.copy(RAMP[i0]).lerp(RAMP[i1], x - i0);
      return out;
    }}
    function radiusRange(pos) {{
      let rMin = Infinity, rMax = -Infinity;
      for (let i = 0; i < pos.length; i += 3) {{
        const r = Math.hypot(pos[i], pos[i + 1]);
        if (r < rMin) rMin = r;
        if (r > rMax) rMax = r;
      }}
      if (!(rMax > rMin)) {{ rMin -= 0.5; rMax += 0.5; }}
      return [rMin, rMax];
    }}
    function radiusColors(pos, rMin, rMax) {{
      const n = pos.length / 3;
      const cols = new Float32Array(n * 3);
      const tmp = new THREE.Color();
      const span = Math.max(rMax - rMin, 1e-9);
      for (let i = 0; i < n; i++) {{
        const r = Math.hypot(pos[i * 3], pos[i * 3 + 1]);
        colorAtT((r - rMin) / span, tmp);
        cols[i * 3] = tmp.r;
        cols[i * 3 + 1] = tmp.g;
        cols[i * 3 + 2] = tmp.b;
      }}
      return cols;
    }}

    function meshFrom(block, colors, opacity) {{
      const pos = new Float32Array(block.positions);
      const idx = new Uint32Array(block.indices);
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      g.setIndex(new THREE.BufferAttribute(idx, 1));
      g.computeVertexNormals();
      const mat = new THREE.MeshStandardMaterial({{
        vertexColors: true,
        metalness: 0.08,
        roughness: 0.68,
        transparent: opacity < 0.999,
        opacity: opacity,
        side: THREE.DoubleSide,
      }});
      return new THREE.Mesh(g, mat);
    }}

    const reconPos = new Float32Array(DATA.recon.positions);
    const truthPos = new Float32Array(DATA.truth.positions);
    const [rMin0, rMax0] = radiusRange(reconPos);
    const [rMin1, rMax1] = radiusRange(truthPos);
    const rMin = Math.min(rMin0, rMin1);
    const rMax = Math.max(rMax0, rMax1);
    document.getElementById("cbar-min").textContent = rMin.toFixed(1) + " near";
    document.getElementById("cbar-max").textContent = rMax.toFixed(1) + " far";

    const reconMesh = meshFrom(DATA.recon, radiusColors(reconPos, rMin, rMax), 1.0);
    const truthMesh = meshFrom(DATA.truth, radiusColors(truthPos, rMin, rMax), 0.32);
    scene.add(reconMesh);
    scene.add(truthMesh);

    // Dent highlight ray
    if (DATA.dentHighlight && DATA.dentHighlight.thetaDeg != null && DATA.dentHighlight.z != null) {{
      const th = DATA.dentHighlight.thetaDeg * Math.PI / 180;
      // look_unit: x=sin, y=cos in this codebase
      const ux = Math.sin(th), uy = Math.cos(th);
      const r = 28;
      const g = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, DATA.dentHighlight.z),
        new THREE.Vector3(r * ux, r * uy, DATA.dentHighlight.z)
      ]);
      scene.add(new THREE.Line(g, new THREE.LineBasicMaterial({{ color: 0xffdd66 }})));
    }}

    function setMode(mode) {{
      const showR = mode === "recon" || mode === "both";
      const showT = mode === "truth" || mode === "both";
      reconMesh.visible = showR;
      truthMesh.visible = showT;
      document.getElementById("btn-recon").classList.toggle("on", mode === "recon");
      document.getElementById("btn-truth").classList.toggle("on", mode === "truth");
      document.getElementById("btn-both").classList.toggle("on", mode === "both");
    }}
    document.getElementById("btn-recon").onclick = () => setMode("recon");
    document.getElementById("btn-truth").onclick = () => setMode("truth");
    document.getElementById("btn-both").onclick = () => setMode("both");
    setMode("both");

    addEventListener("resize", () => {{
      camera.aspect = innerWidth / innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(innerWidth, innerHeight);
    }});

    (function tick() {{
      requestAnimationFrame(tick);
      controls.update();
      renderer.render(scene, camera);
    }})();
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
    return path


def open_survey_html(path: Path | str) -> None:
    """Open interactive HTML in the default browser."""
    import webbrowser

    path = Path(path).resolve()
    webbrowser.open(path.as_uri())


def rebuild_view_from_demo_json(
    json_path: Path | str, html_path: Path | str | None = None
) -> Path:
    """Rebuild interactive HTML + sidecar from a prior survey demo JSON."""
    from sonolab.pipe3d.geometry import true_standoff

    json_path = Path(json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    product = data["product"]
    score = data.get("score") or {}

    if "truth" in data:
        truth = SurveyTruth.from_dict(data["truth"])
    else:
        z_list = [float(v) for v in product["z"]]
        thetas = [float(v) for v in product["theta_deg"]]
        dent_z0 = float(score.get("dentZTrue", float(np.mean(z_list))))
        dent_th0 = float(score.get("dentThetaTrue", 0.0))
        pipe_r = float(score.get("pipeR", 22.0))
        r_true = [
            [
                true_standoff(
                    th,
                    z,
                    dent=True,
                    ecc_mag=2.0,
                    dent_theta0=dent_th0,
                    dent_z0=dent_z0,
                    pipe_r=pipe_r,
                )
                for th in thetas
            ]
            for z in z_list
        ]
        truth = SurveyTruth(
            z=z_list,
            theta_deg=thetas,
            r_true=r_true,
            pipe_r=pipe_r,
            c_prop=0.45,
            dent_on=True,
            dent_theta0_deg=dent_th0,
            dent_z0=dent_z0,
            dent_depth=3.0,
            ecc_mag=2.0,
            ecc_phi_deg=90.0,
        )

    display_meta = (data.get("display") or {}).get("meta") or {}
    display = data.get("display")
    if display is None or "reconPositions" not in display:
        disp_step = float(
            display_meta.get("displayStepDeg")
            or data.get("displayStepDeg")
            or DISPLAY_STEP_DEG
        )
        n_z_d = int(
            display_meta.get("displayNz") or data.get("displayNz") or DISPLAY_N_Z
        )
        acq_step = float(
            display_meta.get("acquireStepDeg")
            or data.get("acquireStepDeg")
            or (
                float(product["theta_deg"][1] - product["theta_deg"][0])
                if len(product["theta_deg"]) > 1
                else 15.0
            )
        )
        display = build_display_wall(
            product,
            truth,
            display_step_deg=disp_step,
            n_z_display=n_z_d,
            acquire_step_deg=acq_step,
            acquire_n_z=len(product["z"]),
        )

    html_path = Path(html_path or json_path.parent / "survey3d_view.html")
    write_survey_view(
        html_path.parent,
        display=display,
        score=score,
        html_name=html_path.name,
    )
    return html_path
