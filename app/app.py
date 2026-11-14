"""Gradio demo: text prompt -> 2D image + floor plan + interactive 3D viewer.

Main outputs: the generated image, an architect-style floor plan (walls, doors / windows, labelled
furniture footprints, approximate dimensions; src/floorplan.py) and the 3D viewer. The depth map
(grayscale), the Canny edges and the segmentation are in a collapsed "Analysis" section.

Runs the same code as scripts/run_pipeline.py (src/pipeline.py):
    prompt -> SD v1.5 + LoRA -> noise-gated preprocessing -> Canny + SegFormer -> metric depth
           -> depth refinement -> point cloud -> textured grid mesh with filled occlusions (.glb)
           -> floor plan (.png / .svg)

Two modes:
  Quick (single view)  the pipeline above; 2D image + floor plan + 3D relief of the view.
  360° Room            src/pipeline_360.py: SD 2.1 panorama model -> Real-ESRGAN -> DAP (wrap fix)
                       -> SegFormer on 14 views -> full-room mesh + dollhouse + 360° floor plan.

3D viewers (tabs):
  "3D viewer"      <model-viewer> web component (script from a CDN, needs internet).
                   Quick: opens exactly at the camera of the generated image (60 degree FOV) and
                   limits the orbit to about +-30 degrees.
                   360°: inside view, camera at the panorama centre; drag to look around, the
                   orbit radius (zoom) is limited to 5 cm - 1 m around the centre.
  "Dollhouse"      360° only: the room without its ceiling, orbited from outside and above.
  "Free orbit"     gr.Model3D, no limits (works offline).

GPU memory (8 GB card): segmenter and depth model stay on the GPU; only the diffusion models of
the active mode are on the GPU, the other mode's are moved to the CPU when the mode changes
(a few seconds). Measured peaks: README section 8. One request runs at a time.

Launch (from the project root):   .venv\\Scripts\\python.exe app\\app.py
then open http://127.0.0.1:7860
"""

import sys
import time
from pathlib import Path
from urllib.parse import quote

import gradio as gr
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import analysis as an  # noqa: E402
from src import depth as dp  # noqa: E402
from src import floorplan as fp  # noqa: E402
from src import generate_2d as g2  # noqa: E402
from src import generate_pano as gp  # noqa: E402
from src import pano_depth as pdp  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import pipeline_360 as p3  # noqa: E402
from src import reconstruct_3d as r3  # noqa: E402

OUT = ROOT / "outputs" / "app"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = {}
ORBIT_LIMIT_DEG = 30          # left / right; up / down is limited to 20 above and 12 below the camera
MODEL_VIEWER_JS = "https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"
HEAD = f'<script type="module" src="{MODEL_VIEWER_JS}"></script>'
QUICK, ROOM360 = "Quick (single view)", "360° Room"
VIEWER_EMPTY = ('<div style="height:540px;display:flex;align-items:center;justify-content:center;'
                'color:#888;border:1px solid #ddd;border-radius:8px">The 3D model appears here.</div>')


def viewer_html(glb_path: str, target_z: float, fov_deg: float = r3.DEFAULT_FOV_DEG) -> str:
    """<model-viewer> element that opens at the original camera and limits the orbit.

    The .glb has the camera of the generated image at the origin looking down -Z. The viewer
    orbits around the point on the optical axis at the median scene depth (target_z), at radius
    target_z, so orbit angle 0 puts the viewer's camera exactly at the origin. The element is
    square like the image, so its vertical field of view equals the pipeline's FOV.
    """
    z = float(target_z)
    lim = ORBIT_LIMIT_DEG
    src = "/gradio_api/file=" + quote(str(Path(glb_path).resolve()).replace("\\", "/"), safe="/:")
    return (
        f'<model-viewer src="{src}" camera-controls disable-pan interaction-prompt="none" '
        f'camera-target="0m 0m {-z:.3f}m" camera-orbit="0deg 90deg {z:.3f}m" '
        f'min-camera-orbit="{-lim}deg 70deg {0.6 * z:.3f}m" max-camera-orbit="{lim}deg 102deg {z:.3f}m" '
        f'field-of-view="{fov_deg:.0f}deg" min-field-of-view="20deg" max-field-of-view="{fov_deg:.0f}deg" '
        f'interpolation-decay="100" shadow-intensity="0" exposure="1" tone-mapping="neutral" '
        f'style="width:100%;max-width:540px;aspect-ratio:1/1;height:auto;margin:0 auto;display:block;'
        f'background:#ffffff;border:1px solid #ddd;border-radius:8px"></model-viewer>'
        f'<p style="text-align:center;color:#777;font-size:12px;margin:4px 0 0">Drag to look around '
        f'(limited to ±{lim}°), scroll to zoom. Needs internet to load the viewer script.</p>')


def _src(glb_path: str) -> str:
    return "/gradio_api/file=" + quote(str(Path(glb_path).resolve()).replace("\\", "/"), safe="/:")


def _mv(src: str, attrs: str, note: str) -> str:
    return (f'<model-viewer src="{src}" camera-controls disable-pan interaction-prompt="none" {attrs} '
            f'interpolation-decay="100" shadow-intensity="0" exposure="1" tone-mapping="neutral" '
            f'style="width:100%;max-width:760px;aspect-ratio:4/3;height:auto;margin:0 auto;display:block;'
            f'background:#ffffff;border:1px solid #ddd;border-radius:8px"></model-viewer>'
            f'<p style="text-align:center;color:#777;font-size:12px;margin:4px 0 0">{note} '
            f'Needs internet to load the viewer script.</p>')


def inside_html(glb_path: str) -> str:
    """360° room from the inside: the orbit target is the panorama centre (the origin of the .glb),
    so dragging turns the view around the room; the radius (zoom) moves the camera 5 cm - 1 m."""
    return _mv(_src(glb_path), 'camera-target="0m 0m 0m" camera-orbit="0deg 90deg 0.3m" '
               'min-camera-orbit="auto 10deg 0.05m" max-camera-orbit="auto 170deg 1m" '
               'field-of-view="75deg" min-field-of-view="30deg" max-field-of-view="90deg"',
               "Drag to look around the room, scroll to move up to 1 m from the centre.")


def dollhouse_html(glb_path: str, camera_height: float) -> str:
    """Room without ceiling, orbited from outside and above (polar angle 0-80 deg)."""
    y = -float(camera_height) + 0.9                                   # target 0.9 m above the floor
    return _mv(_src(glb_path), f'camera-target="0m {y:.2f}m 0m" camera-orbit="30deg 45deg auto" '
               'min-camera-orbit="auto 0deg auto" max-camera-orbit="auto 80deg auto" field-of-view="45deg"',
               "Drag to orbit around the room, scroll to zoom.")


DOLL_EMPTY = ('<div style="height:300px;display:flex;align-items:center;justify-content:center;'
              'color:#888;border:1px solid #ddd;border-radius:8px">The dollhouse view needs the 360° Room mode.</div>')


def free_orbit_camera(mesh: dict) -> tuple[float, float, float]:
    """(alpha, beta, radius) for gr.Model3D that puts its camera at the original camera position.

    gr.Model3D orbits around the centre of the model's bounding box. In the exported file (Y up,
    scene along -Z) the original camera sits at the origin, i.e. at  -centre  relative to the
    orbit centre: alpha = angle in the X-Z plane, beta = angle from the +Y axis.
    """
    v = mesh["vertices"] * np.array([1, -1, -1], np.float32)
    c = (v.min(0) + v.max(0)) / 2
    radius = float(np.linalg.norm(c))
    alpha = float(np.degrees(np.arctan2(-c[2], -c[0])))
    beta = float(np.degrees(np.arccos(np.clip(-c[1] / max(radius, 1e-6), -1, 1))))
    return alpha, beta, radius


def load_models(mode: str = QUICK) -> dict:
    """Make the models of `mode` ready on the GPU (loading them on first use).

    Segmenter + depth model always stay on the GPU. Quick mode: SD 1.5 + LoRA. 360° mode: the
    panorama SD 2.1 model, Real-ESRGAN and DAP. The diffusion model(s) of the other mode are moved
    to the CPU first, so both modes never sit on the 8 GB card at the same time.
    """
    t0 = time.perf_counter()
    if "seg" not in MODELS:
        MODELS["seg"] = an.load_segmenter(DEVICE)
        MODELS["depth"] = dp.load_depth_model(pl.DEPTH_MODEL_3D, DEVICE)
    if mode == QUICK:
        for k in ("alt", "dap"):
            if k in MODELS:
                MODELS[k] = MODELS[k].to("cpu")
        torch.cuda.empty_cache()
        if "sd" not in MODELS:
            MODELS["sd"] = g2.load_pipeline(g2.DEFAULT_LORA if g2.DEFAULT_LORA.exists() else None, DEVICE)
        else:
            MODELS["sd"] = MODELS["sd"].to(DEVICE)
    else:
        if "sd" in MODELS:
            MODELS["sd"] = MODELS["sd"].to("cpu")
        torch.cuda.empty_cache()
        if "alt" not in MODELS:
            MODELS["alt"] = gp.load_alt_pipeline(DEVICE)
            MODELS["dap"] = pdp.load_dap(DEVICE)[0]
            MODELS["esrgan"] = gp.load_esrgan(DEVICE)
        else:
            MODELS["alt"], MODELS["dap"] = MODELS["alt"].to(DEVICE), MODELS["dap"].to(DEVICE)
    used = torch.cuda.memory_allocated() / 2 ** 30 if DEVICE.type == "cuda" else 0.0
    print(f"{mode}: models ready in {time.perf_counter() - t0:.1f} s, {used:.2f} GB VRAM in use")
    return MODELS


def run(mode: str, prompt: str, seed: float, steps: float, cfg: float):
    """One request in the chosen mode."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Please enter a prompt, e.g. 'a coastal style bedroom'.")
    return run_360(prompt, seed, steps, cfg) if mode == ROOM360 else run_quick(prompt, seed, steps, cfg)


def run_360(prompt: str, seed: float, steps: float, cfg: float):
    """360° Room: panorama -> distance -> full-room mesh, dollhouse and 360° floor plan."""
    models = load_models(ROOM360)
    seed, steps = int(seed), int(steps)
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    watch = pl.Stopwatch()
    with watch.stage("panorama"):
        pano = p3.generate(models["alt"], prompt, seed, steps=steps, guidance=cfg)
    with watch.stage("super_res"):
        tex = p3.super_resolve(models["esrgan"], pano)
    with watch.stage("depth"):
        dist = p3.distance(models["dap"], pano, DEVICE)
    with watch.stage("segmentation"):
        cm = p3.segment(models["seg"], pano, DEVICE)
    with watch.stage("mesh_plan"):
        res = p3.build(dist, tex, cm, models["seg"][0].config.id2label)
    out_dir = OUT / f"room360_{pl.slug(prompt, seed)}_st{steps}_cfg{cfg:g}"
    with watch.stage("export"):
        files = p3.save(res, out_dir, pano=pano, plan_title=prompt)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if DEVICE.type == "cuda" else 0.0
    s, m = watch.seconds, res["mesh"]
    masks = an.room_masks(cm, models["seg"][0].config.id2label)
    edges = an.canny_edges(pano)
    info = (f"**360° room done in {sum(s.values()):.1f} s** (panorama {s['panorama']:.1f} s, super-resolution "
            f"{s['super_res']:.1f} s, depth {s['depth']:.1f} s, segmentation {s['segmentation']:.1f} s, mesh + plan "
            f"{s['mesh_plan']:.1f} s, export {s['export']:.1f} s). Peak VRAM {peak:.1f} GB.\n\n"
            f"Distance {np.percentile(dist, 1):.1f} to {np.percentile(dist, 99):.1f} m; camera "
            f"{m['camera_height']:.2f} m above the floor, room box {m['box']['ceiling_height']:.2f} m high "
            f"({m['box']['source']}). Mesh: {m['stats']['triangles']:,} textured triangles.\n\n"
            f"Floor plan: {plan_text(res['plan'])}")
    return (tex, res["plan_image"], inside_html(files["room_glb"]),
            dollhouse_html(files["dollhouse_glb"], m["camera_height"]), gr.Model3D(value=files["room_glb"]),
            pl.depth_gray(dist), an.overlay_edges(pano, edges), an.overlay_masks(pano, masks),
            files["pano"], files["plan_svg"], files["room_glb"], files["dollhouse_glb"], info)


def run_quick(prompt: str, seed: float, steps: float, cfg: float):
    """Quick mode: returns the pictures, the viewers, the download files and a summary."""
    models = load_models(QUICK)
    seed, steps = int(seed), int(steps)
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    watch = pl.Stopwatch()
    with watch.stage("generate"):
        img = g2.generate(models["sd"], prompt, seed, steps=steps, guidance=cfg)
    out_dir = OUT / f"{pl.slug(prompt, seed)}_st{steps}_cfg{cfg:g}"
    res = pl.image_to_3d(img, models["seg"], models["depth"], DEVICE, watch=watch, out_dir=out_dir)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if DEVICE.type == "cuda" else 0.0
    s, st, d, mesh = watch.seconds, res["stats"], res["depth"], res["mesh"]
    total = sum(s.values())
    planes = ", ".join(p["kind"] for p in mesh["planes"] if p["accepted"]) or "none"
    info = (f"**Done in {total:.1f} s** (image {s['generate']:.1f} s, analysis + depth "
            f"{s['preprocess'] + s['analysis'] + s['depth'] + s['refine']:.2f} s, point cloud {s['pointcloud']:.1f} s, "
            f"mesh {s['mesh']:.1f} s, export {s['export']:.1f} s). Peak VRAM {peak:.1f} GB.\n\n"
            f"Depth {d.min():.1f} to {d.max():.1f} m. Mesh: {st['triangles']:,} textured triangles; "
            f"room planes used: {planes}. Point cloud: {st['points']:,} points. "
            f"Estimated image noise σ = {res['info']['noise_est']:.1f} "
            f"({'denoised' if res['info']['denoised'] else 'not denoised'}). "
            f"Assumed camera: {r3.DEFAULT_FOV_DEG:.0f}° horizontal field of view.\n\n"
            f"Floor plan ({s['floorplan']:.2f} s): {plan_text(res['plan'])}")
    f = res["files"]
    return (img, res["plan_image"], viewer_html(f["mesh_glb"], mesh["target_z"]), DOLL_EMPTY,
            gr.Model3D(value=f["mesh_glb"], camera_position=free_orbit_camera(mesh)),
            pl.depth_gray(d), an.overlay_edges(res["pre"], res["edges"]), an.overlay_masks(res["pre"], res["masks"]),
            f["image"], f["plan_svg"], f["mesh_glb"], f["pointcloud"], info)


def plan_text(plan: dict) -> str:
    """One-line summary of the floor plan for the info box."""
    s = fp.plan_summary(plan)
    furn = sorted({f["label"] for f in plan["furniture"]})
    area = (f"floor area ≈ {s['area_m2']:.1f} m²" if s["closed"]
            else f"visible floor ≈ {s['visible_floor_m2']:.1f} m² (room not closed in one view)")
    return (f"{s['walls']} wall(s), {s['doors']} door(s), {s['windows']} window(s), {s['furniture']} furniture item(s)"
            f"{' (' + ', '.join(furn) + ')' if furn else ''}; {area}; gravity from the {s['gravity']}. "
            "Dimensions are approximate (monocular depth, about 20% scale error on NYU).")


def build() -> gr.Blocks:
    with gr.Blocks(title="RoomGen3D") as demo:
        gr.Markdown("# RoomGen3D\nText prompt → interior design image → depth → 3D room model. "
                    "Prompts like *a coastal style bedroom* work best. **Quick** builds a 3D relief from one "
                    "image (looks right from the original viewpoint). **360° Room** generates a full panorama "
                    "and builds the whole room: look around inside, view it as a dollhouse, and get a closed "
                    "floor plan (about 20 s; the first 360° request loads its models, about a minute).")
        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio([QUICK, ROOM360], value=QUICK, label="Mode")
                prompt = gr.Textbox(label="Prompt", value=pl.DEMO_PROMPTS[1][0], lines=2)
                seed = gr.Number(label="Seed", value=pl.DEMO_PROMPTS[1][1], precision=0)
                steps = gr.Slider(10, 50, value=g2.GEN_STEPS, step=1, label="Steps")
                cfg = gr.Slider(1.0, 15.0, value=g2.GEN_GUIDANCE, step=0.5, label="CFG (guidance scale)")
                btn = gr.Button("Generate", variant="primary")
                info = gr.Markdown()
                with gr.Row():
                    dl_image = gr.DownloadButton("Download image (.png)")
                    dl_plan = gr.DownloadButton("Download floor plan (.svg)")
                with gr.Row():
                    dl_mesh = gr.DownloadButton("Download mesh (.glb)")
                    dl_cloud = gr.DownloadButton("Download point cloud (.ply) / dollhouse (.glb)")
                gr.Examples([[p, s] for p, s in pl.DEMO_PROMPTS], inputs=[prompt, seed], label="Example prompts")
            with gr.Column(scale=2):
                with gr.Row():
                    image = gr.Image(label="2D image / panorama", type="numpy", height=420)
                    plan = gr.Image(label="Floor plan (approximate dimensions)", type="numpy", height=420)
                with gr.Tabs():
                    with gr.Tab("3D viewer (original view / inside the 360° room)"):
                        viewer = gr.HTML(VIEWER_EMPTY)
                    with gr.Tab("Dollhouse (360°)"):
                        doll = gr.HTML(DOLL_EMPTY)
                    with gr.Tab("3D: free orbit (offline)"):
                        model3d = gr.Model3D(label="Textured mesh (drag to rotate, scroll to zoom)", height=540,
                                             clear_color=[1.0, 1.0, 1.0, 1.0])
                with gr.Accordion("Analysis: depth, edges, segmentation", open=False):
                    with gr.Row():
                        depth = gr.Image(label="Depth map (white = near, black = far)", type="numpy", height=260)
                        edges = gr.Image(label="Canny edges", type="numpy", height=260)
                        seg = gr.Image(label="Segmentation", type="numpy", height=260)
                    gr.Markdown("Segmentation colours: blue = wall, orange = floor, green = ceiling, pink = furniture.")
        btn.click(run, [mode, prompt, seed, steps, cfg],
                  [image, plan, viewer, doll, model3d, depth, edges, seg, dl_image, dl_plan, dl_mesh, dl_cloud, info],
                  concurrency_limit=1)
    return demo


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    load_models(QUICK)
    build().queue(max_size=8).launch(server_name="127.0.0.1", server_port=7860, allowed_paths=[str(OUT)], head=HEAD)
