"""Gradio demo: text prompt -> 2D image + floor plan + interactive 3D viewer.

PLAN_V4 (default 360° pipeline "v4", src/pipeline_v4.py): a gallery of readable design shots (hero + wall views)
next to the floor plan with view cones, a 4096 x 2048 sharpened panorama, a sharp inside view that opens facing
the hero, optional TripoSR furniture objects (slider) with relief for uncertain pieces, open sides on the plan;
tabs Inside / Dollhouse / Top-down / 360° panorama / free orbit, then Analysis. v3 and v2 stay selectable.

Main outputs: the generated image, an architect-style floor plan (walls, doors / windows, labelled
furniture footprints, approximate dimensions; src/floorplan.py) and the 3D viewer. The depth map
(grayscale), the Canny edges and the segmentation are in a collapsed "Analysis" section.

Runs the same code as scripts/run_pipeline.py (src/pipeline.py):
    prompt -> SD v1.5 + LoRA -> noise-gated preprocessing -> Canny + SegFormer -> metric depth
           -> depth refinement -> point cloud -> textured grid mesh with filled occlusions (.glb)
           -> floor plan (.png / .svg)

Two modes:
  Quick (single view)  the pipeline above; 2D image + floor plan + 3D relief of the view.
  360° Room            v4 (default, src/pipeline_v4.py, PLAN_V4): v3 below + readable design shots (hero view
                       + four wall views, perspective, level, 1024 x 768) shown in a gallery next to the floor
                       plan, whose camera and view cones carry the same letters; the panorama itself moves to
                       the "360° panorama" viewer tab and the Analysis section.
                       v3 (src/pipeline_v3.py): 1-4 candidate panoramas (SD 2.1 panorama model,
                       one-room prompt) -> DAP + SegFormer -> the most box-like candidate is kept
                       (src/boxiness.py) -> Real-ESRGAN -> Mask2Former instances -> room box shell +
                       solid furniture -> architectural plan (PNG / SVG / DXF).
                       v2 (comparison, src/pipeline_360.py): SD 2.1 panorama -> Real-ESRGAN -> DAP ->
                       SegFormer -> full-room mesh + dollhouse + 360° floor plan.
  Layout-first         src/layout.py: rule-based room layout (room type from the prompt, the seed
                       picks the layout) -> layout depth with shaped furniture proxies and door / window
                       recesses -> SD 2.1 panorama model + SD 2.1 depth ControlNet (scale 0.75, first 60%
                       of the steps; PLAN_V3 milestone 7) -> Real-ESRGAN -> mesh on the EXACT layout
                       geometry + the true floor plan.

3D viewers (tabs):
  "3D viewer"      <model-viewer> web component (script from a CDN, needs internet).
                   Quick: opens exactly at the camera of the generated image (60 degree FOV) and
                   limits the orbit to about +-30 degrees.
                   360°: inside view, camera at the panorama centre; drag to look around, the
                   orbit radius (zoom) is limited to 5 cm - 1 m around the centre.
  "Dollhouse"      360° / layout-first: the room without its ceiling, orbited from outside and above.
  "360° panorama"  the panorama on an inward-facing sphere (src/viewers.pano_sphere): drag to look around;
                   v4 opens facing the hero view.
  "Free orbit"     gr.Model3D, no limits (works offline).

GPU memory (8 GB card): segmenter and depth model stay on the GPU; only the diffusion models of
the active mode are on the GPU, the other mode's are moved to the CPU when the mode changes
(a few seconds). Measured peaks: README section 8. One request runs at a time.

Launch (from the project root):   .venv\\Scripts\\python.exe app\\app.py
then open http://127.0.0.1:7860
"""

import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import quote

import cv2
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
from src import layout as ly  # noqa: E402
from src import room_mesh as rm  # noqa: E402
from src import textured_mesh as tm  # noqa: E402
from src import pano_depth as pdp  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import instances as ins  # noqa: E402
from src import pipeline_v3 as p3v  # noqa: E402
from src import pipeline_v4 as p4v  # noqa: E402
from src import design_shots as ds  # noqa: E402
from src import equirect as eq  # noqa: E402
from src import viewers as vw  # noqa: E402
from src import pano_refine as pr  # noqa: E402
from src import objects3d as o3  # noqa: E402
from src import pipeline_360 as p3  # noqa: E402
from src import reconstruct_3d as r3  # noqa: E402
from src import pipeline_v5 as p5v  # noqa: E402
from src import memory as mem  # noqa: E402
from src import quick_v5 as q5  # noqa: E402

OUT = ROOT / "outputs" / "app"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = {}
ORBIT_LIMIT_DEG = 30          # left / right; up / down is limited to 20 above and 12 below the camera
MODEL_VIEWER_JS = "https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"
HEAD = f'<script type="module" src="{MODEL_VIEWER_JS}"></script>'
QUICK, ROOM360, LAYOUT = "Quick (single view)", "360° Room", "Layout-first"
BIG = {QUICK: ("sd",), ROOM360: ("alt", "dap", "m2f"), LAYOUT: ("cn",)}     # large models of each mode
V5 = "v5: SDXL panorama, re-rendered design shots, 360° inside view, clean dollhouse (default)"
V4, V3, V2 = "v4: design shots + clean box room (comparison)", "v3: clean box room (comparison)", "v2: previous mesh (comparison)"
V5_CANDIDATES = 2             # v5 default candidates (each SDXL panorama ~55 s; PLAN_V5 milestone 2)
NEUTRAL_BG = "#e9e8e4"
V4_TONE = "neutral"           # v4 viewers' tone mapping (PLAN_V4 milestone 3, scripts/viewer_check_v4.py)
DETAIL_STRENGTH = 0.3
DEFAULT_OBJECTS = 4           # v4 generated furniture objects per room (PLAN_V4 milestone 4)         # v4 detail pass (results/tables/pano_refine_decision.json)        # dollhouse background (v3)
LAYOUT_CN_SCALE, LAYOUT_CN_END = 0.75, 0.6   # layout-first: SD 2.1 panorama model + SD 2.1 depth ControlNet (milestone 7)
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


def _mv(src: str, attrs: str, note: str, bg: str = "#ffffff", tone: str = "neutral") -> str:
    return (f'<model-viewer src="{src}" camera-controls disable-pan interaction-prompt="none" {attrs} '
            f'interpolation-decay="100" shadow-intensity="0" exposure="1" tone-mapping="{tone}" '
            f'style="width:100%;max-width:760px;aspect-ratio:4/3;height:auto;margin:0 auto;display:block;'
            f'background:{bg};border:1px solid #ddd;border-radius:8px"></model-viewer>'
            f'<p style="text-align:center;color:#777;font-size:12px;margin:4px 0 0">{note} '
            f'Needs internet to load the viewer script.</p>')


def inside_html(glb_path: str, bg: str = "#ffffff", yaw_deg: float | None = None) -> str:
    """360° room from the inside: the orbit target is the panorama centre (the origin of the .glb),
    so dragging turns the view around the room; the radius (zoom) moves the camera 5 cm - 1 m.
    yaw_deg (v4): longitude in the mesh frame A to face first (the hero view); the viewer then opens LEVEL
    (polar 90°) at eye height (the panorama centre), 5 cm from the centre, and model-viewer's theta = -yaw."""
    if yaw_deg is None:
        return _mv(_src(glb_path), 'camera-target="0m 0m 0m" camera-orbit="0deg 90deg 0.3m" '
                   'min-camera-orbit="auto 10deg 0.05m" max-camera-orbit="auto 170deg 1m" '
                   'field-of-view="75deg" min-field-of-view="30deg" max-field-of-view="90deg"',
                   "Drag to look around the room, scroll to move up to 1 m from the centre.", bg)
    return _mv(_src(glb_path), f'camera-target="0m 0m 0m" camera-orbit="{-yaw_deg:.1f}deg 90deg 0.05m" '
               'min-camera-orbit="auto 10deg 0.05m" max-camera-orbit="auto 170deg 1m" '
               'field-of-view="64deg" min-field-of-view="30deg" max-field-of-view="90deg"',
               "Opens facing the hero view (H). Drag to look around, scroll to move up to 1 m from the centre.",
               bg, tone=V4_TONE)


def dollhouse_html(glb_path: str, camera_height: float, bg: str = "#ffffff") -> str:
    """Room without ceiling, orbited from outside and above (polar angle 0-80 deg)."""
    y = -float(camera_height) + 0.9                                   # target 0.9 m above the floor
    return _mv(_src(glb_path), f'camera-target="0m {y:.2f}m 0m" camera-orbit="30deg 45deg auto" '
               'min-camera-orbit="auto 0deg auto" max-camera-orbit="auto 80deg auto" field-of-view="45deg"',
               "Drag to orbit around the room, scroll to zoom.", bg)


def pano_html(glb_path: str, yaw_deg: float = 0.0) -> str:
    """Interactive 360° panorama: the camera sits at the centre of the textured sphere (radius 10 m) and only
    turns. yaw_deg = panorama longitude to face first (model-viewer theta = -longitude, see viewers.pano_sphere)."""
    return _mv(_src(glb_path), f'camera-target="0m 0m 0m" camera-orbit="{-yaw_deg:.1f}deg 90deg 0.01m" '
               'min-camera-orbit="auto 5deg 0.01m" max-camera-orbit="auto 175deg 0.01m" '
               'field-of-view="80deg" min-field-of-view="30deg" max-field-of-view="100deg"',
               "Drag to look around the panorama, scroll to zoom.", "#000000")


PANO_EMPTY = ('<div style="height:300px;display:flex;align-items:center;justify-content:center;'
              'color:#888;border:1px solid #ddd;border-radius:8px">The 360° panorama viewer needs the 360° Room or '
              'Layout-first mode.</div>')


DOLL_EMPTY = ('<div style="height:300px;display:flex;align-items:center;justify-content:center;'
              'color:#888;border:1px solid #ddd;border-radius:8px">The dollhouse view needs the 360° Room or Layout-first mode.</div>')


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


def _load_m2f():
    """Mask2Former panoptic (v3 instances); the processor is kept separately (it has no .to())."""
    model, proc = ins.load_panoptic(DEVICE)
    MODELS["m2f_proc"] = proc
    return model


def load_models(mode: str = QUICK) -> dict:
    """Make the models of `mode` ready on the GPU (loading them on first use).

    Segmenter, depth model and Real-ESRGAN always stay on the GPU (small). The large models of the
    other modes (BIG) are moved to the CPU first, so two modes never sit on the 8 GB card together.
    Quick: SD 1.5 + LoRA. 360°: SD 2.1 panorama model + DAP. Layout-first: SD 1.5 + ControlNet depth.
    """
    t0 = time.perf_counter()
    if "seg" not in MODELS:
        MODELS["seg"] = an.load_segmenter(DEVICE)
        MODELS["depth"] = dp.load_depth_model(pl.DEPTH_MODEL_3D, DEVICE)
    for other, keys in BIG.items():
        if other != mode:
            for k in keys:
                if k in MODELS:
                    MODELS[k] = MODELS[k].to("cpu")
    for kk in ("i2i", "triposr"):                        # v4 detail pass / objects: only on the GPU while they run
        if kk in MODELS:
            MODELS[kk] = MODELS[kk].to("cpu")
    torch.cuda.empty_cache()
    loaders = {"sd": lambda: g2.load_pipeline(g2.DEFAULT_LORA if g2.DEFAULT_LORA.exists() else None, DEVICE),
               "alt": lambda: gp.load_alt_pipeline(DEVICE), "dap": lambda: pdp.load_dap(DEVICE)[0],
               "cn": lambda: gp.load_controlnet_pipeline_sd21(DEVICE), "m2f": _load_m2f}
    for k in BIG[mode]:
        MODELS[k] = loaders[k]() if k not in MODELS else MODELS[k].to(DEVICE)
    if mode != QUICK and "esrgan" not in MODELS:
        MODELS["esrgan"] = gp.load_esrgan(DEVICE)
    used = torch.cuda.memory_allocated() / 2 ** 30 if DEVICE.type == "cuda" else 0.0
    print(f"{mode}: models ready in {time.perf_counter() - t0:.1f} s, {used:.2f} GB VRAM in use")
    return MODELS


def run(mode: str, prompt: str, seed: float, steps: float, cfg: float, version: str = V4, candidates: float = 1,
        detail: bool = False, n_objects: float = 0, sharp_shots: bool = True):
    """One request in the chosen mode. The first output becomes the gallery: v4 returns its design shots
    (hero first), the other modes their single image / panorama. Then: the 360° panorama viewer, the panorama
    for the Analysis section and the design-shot zip (v4)."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Please enter a prompt, e.g. 'a coastal style bedroom'.")
    extra = {}
    if mode == ROOM360:
        if version == V2:
            out = [*run_360(prompt, seed, steps, cfg), None, None]
        elif version == V5:
            out = list(run_360_v5(prompt, seed, steps, cfg, int(candidates), extra=extra))
        else:
            out = list(run_360_v3(prompt, seed, steps, cfg, int(candidates), v4=(version != V3), extra=extra,
                                       detail=bool(detail), n_objects=int(n_objects),
                                       sharp_shots=bool(sharp_shots)))
    elif mode == LAYOUT:
        out = [*run_layout(prompt, seed, steps, cfg), None, None]
    else:
        out = [*run_quick(prompt, seed, steps, cfg), None, None]
    first = out[0]
    if isinstance(first, list):
        gallery = first
    else:
        gallery = [(first, "generated image" if mode == QUICK else "panorama (equirectangular, 360° x 180°)")]
    pano_view, pano_img = PANO_EMPTY, None
    if mode != QUICK:
        pano_path = extra.get("pano_path", out[8])
        pano_img = cv2.cvtColor(cv2.imread(pano_path), cv2.COLOR_BGR2RGB)
        sphere = extra.get("sphere")
        if sphere is None:
            sphere = str(Path(pano_path).with_name("pano_sphere.glb"))
            vw.pano_sphere_glb(pano_img, sphere)
        pano_view = pano_html(sphere, extra.get("hero_lon", 0.0))
    return (gallery, *out[1:], pano_view, pano_img, extra.get("shots_zip"), extra.get("mesh_tab", MESH_EMPTY),
            extra.get("raw"))


def run_layout(prompt: str, seed: float, steps: float, cfg: float):
    """Layout-first: layout -> depth condition -> ControlNet panorama -> mesh on the exact geometry."""
    models = load_models(LAYOUT)
    seed, steps = int(seed), int(steps)
    try:
        room = ly.room_type(prompt.lower())
    except ValueError:
        raise gr.Error("Layout-first needs a room type in the prompt: " + ", ".join(ly.ROOMS) + ".")
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    watch = pl.Stopwatch()
    with watch.stage("layout"):
        lay = ly.generate_layout(room, seed)
        dist, lab, _ = ly.render_full(lay, shapes=True)            # shaped proxies + door / window recesses
    with watch.stage("panorama"):                                   # PLAN_V3 milestone 7 selection
        pano = gp.generate_pano_controlled(models["cn"], prompt, seed, ly.control_image(dist), LAYOUT_CN_SCALE,
                                           steps=steps, guidance=cfg, end=LAYOUT_CN_END,
                                           prompt=gp.ALT_TEMPLATE.format(room=prompt), negative=gp.PANO_NEGATIVE)
    with watch.stage("super_res"):
        tex = gp.esrgan_upscale(models["esrgan"], pano)
    id2label = models["seg"][0].config.id2label
    with watch.stage("mesh"):
        mesh = rm.build_room_mesh(dist, tex, lab, id2label)
        doll = rm.dollhouse(mesh)
    out_dir = OUT / f"layout_{pl.slug(prompt, seed)}_st{steps}_cfg{cfg:g}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with watch.stage("export"):
        files = {"room_glb": str(out_dir / "room.glb"), "dollhouse_glb": str(out_dir / "dollhouse.glb"),
                 "plan_svg": str(out_dir / "floorplan.svg"), "pano": str(out_dir / "pano.png")}
        tm.export_glb(mesh, files["room_glb"])
        tm.export_glb(doll, files["dollhouse_glb"])
        plan_img = fp.draw_floorplan(ly.plan_dict(lay), out_dir / "floorplan.png", files["plan_svg"],
                                     title=f"{prompt} (layout {seed})", note="Exact layout dimensions (the image was generated to follow them)")
        ly.save_json(lay, out_dir / "layout.json")
        cv2.imwrite(files["pano"], cv2.cvtColor(pano, cv2.COLOR_RGB2BGR))
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if DEVICE.type == "cuda" else 0.0
    s = watch.seconds
    masks = an.room_masks(lab, id2label)
    info = (f"**Layout-first room done in {sum(s.values()):.1f} s** (layout {s['layout']:.1f} s, panorama "
            f"{s['panorama']:.1f} s, super-resolution {s['super_res']:.1f} s, mesh {s['mesh']:.1f} s, export "
            f"{s['export']:.1f} s). Peak VRAM {peak:.1f} GB.\n\n"
            f"Layout: {room}, {lay['W']:.2f} × {lay['D']:.2f} m ({lay['W'] * lay['D']:.1f} m²), "
            f"ceiling {lay['H']:.2f} m; " + ", ".join(b["label"] for b in lay["boxes"]) +
            f". The 3D geometry is the layout itself; the panorama is projected onto it "
            f"({mesh['stats']['triangles']:,} triangles). Change the seed for another layout.")
    return (tex, plan_img, inside_html(files["room_glb"]), dollhouse_html(files["dollhouse_glb"], ly.CAMERA_HEIGHT),
            gr.Model3D(value=files["room_glb"]), pl.depth_gray(dist), ly.control_image(dist),
            an.overlay_masks(pano, masks), files["pano"], files["plan_svg"], files["room_glb"],
            files["dollhouse_glb"], info)


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


def run_360_v3(prompt: str, seed: float, steps: float, cfg: float, k: int = 1, v4: bool = True,
               extra: dict | None = None, detail: bool = False, n_objects: int = 0, sharp_shots: bool = True):
    """360° Room, v3 (PLAN_V3): k candidate panoramas (one-room prompt), the most box-like one kept ->
    Real-ESRGAN -> panoptic instances -> room box shell + solid furniture -> architectural plan.
    v4 (PLAN_V4): the same room plus design shots (gallery, view cones on the plan, panorama sphere)."""
    extra = {} if extra is None else extra
    P = p4v if v4 else p3v
    models = load_models(ROOM360)
    seed, steps = int(seed), int(steps)
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    id2label = {int(i): v for i, v in models["seg"][0].config.id2label.items()}
    watch = pl.Stopwatch()
    with watch.stage("candidates"):
        best = p3v.generate_candidates(models["alt"], models["dap"], models["seg"], prompt, seed, k, DEVICE,
                                       steps=steps, guidance=cfg)
    pano, dist, cm = best["pano"], best["dist"], best["class_map"]
    with watch.stage("super_res"):
        tex = gp.esrgan_upscale(models["esrgan"], pano)
    with watch.stage("instances"):
        ana = p3v.analyse((models["m2f"], models["m2f_proc"]), pano, cm, id2label, DEVICE)
    hires = None
    if v4:                       # PLAN_V4 milestone 2: 4096 x 2048 panorama from 26 refined perspective views
        with watch.stage("sharpen"):
            i2i = None
            if detail:           # img2img + MMIS LoRA (selected on the dev rooms); park the 360° models first
                for kk in BIG[ROOM360]:
                    MODELS[kk] = MODELS[kk].to("cpu")
                torch.cuda.empty_cache()
                MODELS["i2i"] = (pr.load_img2img(DEVICE, g2.DEFAULT_LORA) if "i2i" not in MODELS
                                 else MODELS["i2i"].to(DEVICE))
                i2i = MODELS["i2i"]
            hires, _ = p4v.sharpen(tex, models["esrgan"], i2i, prompt, DETAIL_STRENGTH, seed)
            if detail:
                MODELS["i2i"] = MODELS["i2i"].to("cpu")
                load_models(ROOM360)
    out_dir = OUT / f"room360{'v4' if v4 else 'v3'}_{pl.slug(prompt, seed)}_k{k}_st{steps}_cfg{cfg:g}"
    with watch.stage("room"):
        if v4 and n_objects > 0:     # PLAN_V4 milestone 4: TripoSR objects; park the 360° models meanwhile
            for kk in BIG[ROOM360]:
                MODELS[kk] = MODELS[kk].to("cpu")
            torch.cuda.empty_cache()
            MODELS["triposr"] = o3.load_triposr(DEVICE) if "triposr" not in MODELS else MODELS["triposr"].to(DEVICE)
            res = P.build(dist, tex, ana, id2label, hires=hires, prompt=prompt,
                          shot_net=MODELS["esrgan"] if sharp_shots else None, triposr=MODELS["triposr"], device=DEVICE,
                          max_objects=n_objects, cache_dir=out_dir / "objects")      # cached per prompt + seed
            MODELS["triposr"] = MODELS["triposr"].to("cpu")
            torch.cuda.empty_cache()
        elif v4:
            res = P.build(dist, tex, ana, id2label, hires=hires, prompt=prompt,
                          shot_net=MODELS["esrgan"] if sharp_shots else None)
        else:
            res = P.build(dist, tex, ana, id2label)
    with watch.stage("export"):
        files = (p4v.save(res, out_dir, title=prompt, pano=tex if hires is None else hires) if v4
                 else p3v.save(res, out_dir, title=prompt))
        files["pano"] = str(out_dir / "pano.png")
        cv2.imwrite(files["pano"], cv2.cvtColor(pano, cv2.COLOR_RGB2BGR))
        if v4:
            zp = out_dir / "design_shots.zip"
            with zipfile.ZipFile(zp, "w") as z:
                for s in res["shots"]["shots"]:
                    z.write(files[f"shot_{s['letter']}"], f"{s['letter']}.png")
                z.write(files["plan_png"], "floorplan_with_views.png")
            hero = res["shots"]["shots"][0]
            dc = np.array([0.0, 0.0, 1.0]) @ ds.view_to_cam(res["box"], hero["yaw"])
            extra.update({"shots_zip": str(zp), "sphere": files["pano_sphere_glb"], "pano_path": files["pano"],
                          "hero_lon": float(np.degrees(np.arctan2(dc[0], dc[2])))})
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if DEVICE.type == "cuda" else 0.0
    s, b = watch.seconds, res["box"]
    cands = ", ".join(f"seed {c['seed']}: {c['score']:.2f}" for c in best["candidates"])
    furn = ", ".join(sorted({x["label"] for x in res["solids"]})) or "none"
    shots_txt = ""
    if v4:
        sh = res["shots"]["shots"]
        shots_txt = (f"\n\nDesign shots ({res['shots']['seconds']:.1f} s): {', '.join(x['letter'] for x in sh)}; "
                     f"hero view facing {sh[0]['yaw']:.0f}° in the room frame with {sh[0]['measures']['inst']} furniture "
                     f"piece(s) in view. Camera at the panorama centre, {b['camera_height']:.2f} m above the floor; "
                     "letters match the view cones on the floor plan.")
    if v4:
        shots_txt += (f" Panorama sharpened to 4096 × 2048 in {s['sharpen']:.1f} s ("
                      + ("img2img detail pass with the MMIS LoRA + " if detail else "") + "Real-ESRGAN per view).")
        if res.get("objects"):
            ob = res["objects"]
            shots_txt += (f"\n\nFurniture in the dollhouse: {len(ob['objects'])} generated 3D object(s) (TripoSR, "
                          f"{ob['seconds']:.1f} s; " + ", ".join(sorted({x['solid']['label'] for x in ob['objects']}) or ["none"])
                          + f"), {len(ob['blocks'])} block(s), {ob['n_relief']} uncertain piece(s) shown as relief.")
    info = (f"**360° room ({'v4' if v4 else 'v3'}) done in {sum(s.values()):.1f} s** ({k} candidate(s) {s['candidates']:.1f} s, "
            f"super-resolution {s['super_res']:.1f} s, instances {s['instances']:.1f} s, room {s['room']:.1f} s, "
            f"export {s['export']:.1f} s). Peak VRAM {peak:.1f} GB.\n\n"
            f"Boxiness (1 = one closed room): {cands}; kept seed {best['seed']}.\n\n"
            f"Room ≈ {b['width']:.2f} × {b['depth']:.2f} m ({b['area']:.1f} m²), ceiling {b['ceiling_height']:.2f} m; "
            f"{b['n_detected']} of 4 walls measured (the others hatched on the plan). Furniture: {len(res['solids'])} "
            f"solid(s) ({furn}); {len(res['openings'])} door(s)/window(s). Dimensions are approximate "
            "(monocular depth)." + shots_txt)
    masks = an.room_masks(ana["class_map"], id2label)
    return (p4v.gallery(res) if v4 else tex, res["plan_image"],
            inside_html(files["room_glb"], res["bg_color"], p4v.hero_yaw_A(res) if v4 else None),
            dollhouse_html(files["dollhouse_glb"], b["camera_height"], NEUTRAL_BG),
            gr.Model3D(value=files["room_glb"]), pl.depth_gray(dist), an.overlay_edges(pano, an.canny_edges(pano)),
            an.overlay_masks(pano, masks), files["pano"], files["plan_svg"], files["room_glb"],
            files["dollhouse_glb"], info, files["plan_dxf"], res["topdown"])


# --------------------------------------------------------------------------- #
# PLAN_V5
# --------------------------------------------------------------------------- #
def inside_v5_html(html_path: str) -> str:
    """The v5 Inside view: the self-contained panorama viewer page (src/pano_viewer.py) in an iframe. Camera fixed at
    the panorama centre, turn + zoom only, opens level facing the hero shot; no CDN needed."""
    return (f'<iframe src="{_src(html_path)}" style="width:100%;max-width:960px;aspect-ratio:4/3;border:1px solid #ddd;'
            f'border-radius:8px;display:block;margin:0 auto" allow="fullscreen"></iframe>'
            '<p style="text-align:center;color:#777;font-size:12px;margin:4px 0 0">Drag to look around, scroll to zoom, '
            'double-click to return to the hero view. The camera stays where the panorama was taken, so nothing tears.</p>')


def mesh_v5_html(glb_path: str, bg: str, yaw_deg: float, limit_m: float) -> str:
    """'Experimental: walkable mesh': the v4 depth mesh, translation limited to the measured hole threshold."""
    lim = max(0.05, float(limit_m))
    return _mv(_src(glb_path), f'camera-target="0m 0m 0m" camera-orbit="{-yaw_deg:.1f}deg 90deg 0.05m" '
               f'min-camera-orbit="auto 10deg 0.05m" max-camera-orbit="auto 170deg {lim:.2f}m" '
               'field-of-view="64deg" min-field-of-view="30deg" max-field-of-view="90deg"',
               f"Experimental depth mesh: scroll moves the camera at most {lim:.2f} m from the panorama centre "
               "(beyond that, surfaces the camera never saw appear as holes / smears).", bg, tone=V4_TONE)


MESH_EMPTY = ('<div style="height:300px;display:flex;align-items:center;justify-content:center;'
              'color:#888;border:1px solid #ddd;border-radius:8px">The walkable mesh is shown for 360° Room v5.</div>')


def free_diffusion() -> None:
    """Drop every large model from memory (RAM is the limit for SDXL on the 16 GB machine)."""
    for k in ("sd", "alt", "cn", "i2i", "triposr", "dap", "m2f", "m2f_proc", "q5_i2i"):
        MODELS.pop(k, None)
    mem.free()


def run_360_v5(prompt: str, seed: float, steps: float, cfg: float, k: int = V5_CANDIDATES, extra: dict | None = None):
    """360° Room, v5 (PLAN_V5): k SDXL panoramas (v5 prompt) -> DAP + SegFormer -> candidate with the best boxiness +
    room size + camera clearance -> Mask2Former -> v4 room -> 4096 panorama -> valid design shots, re-rendered with
    SD 1.5 + MMIS LoRA + depth ControlNet -> clean dollhouse -> panorama Inside viewer."""
    from src import camera_v5 as cam
    from src import dollhouse_v5 as dv
    from src import generate_pano_xl as gx
    from src import room_v4 as r4
    from src import shot_render as sr
    from src import shots_v5 as sv
    extra = {} if extra is None else extra
    seed = int(seed)
    gen = p5v.source()
    watch = pl.Stopwatch()
    peaks = {}
    free_diffusion()
    with mem.PeakMonitor() as pm_all:
        with watch.stage("candidates"):
            mem.wait_for_ram(6.0, label="v5 panorama generator")
            pipe = gp.load_alt_pipeline(DEVICE) if gen == "sd21" else gx.load_xl_pipeline(gen.split("_", 1)[1])
            panos = [p5v.generate(pipe, gen, prompt, seed + i, spacious=True) for i in range(max(1, int(k)))]
            pipe = None
            free_diffusion()
            peaks["sdxl_vram_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30 if DEVICE.type == "cuda" else 0.0
        if "seg" not in MODELS:
            MODELS["seg"] = an.load_segmenter(DEVICE)
            MODELS["depth"] = dp.load_depth_model(pl.DEPTH_MODEL_3D, DEVICE)
        if "esrgan" not in MODELS:
            MODELS["esrgan"] = gp.load_esrgan(DEVICE)
        id2label = {int(i): v for i, v in MODELS["seg"][0].config.id2label.items()}
        with watch.stage("scoring"):
            MODELS["dap"] = pdp.load_dap(DEVICE)[0]
            if gen == "sd21":
                panos = [gp.esrgan_upscale(MODELS["esrgan"], x) for x in panos]
            cands = [p5v.score_candidate(x, MODELS["dap"], MODELS["seg"], id2label, DEVICE) for x in panos]
            best = cam.pick_candidate([c["score"] for c in cands], p5v.selection_weights())
            MODELS.pop("dap")
            mem.free()
        pano, dist, cm = panos[best], cands[best]["dist"], cands[best]["class_map"]
        with watch.stage("instances"):
            MODELS["m2f"] = _load_m2f()
            small = cv2.resize(pano, (1024, 512), interpolation=cv2.INTER_AREA)
            ana = p3v.analyse((MODELS["m2f"], MODELS["m2f_proc"]), small, cm, id2label, DEVICE)
            MODELS.pop("m2f")
            MODELS.pop("m2f_proc", None)
            mem.free()
        with watch.stage("sharpen"):
            hires, _ = p4v.sharpen(pano, MODELS["esrgan"])
        with watch.stage("room_shots"):
            res = p5v.build(dist, pano, hires, ana, id2label, prompt)
        with watch.stage("rerender"):
            dec = p5v.decision("v5_shots_decision.json") or {"selected": "cn_lora_s0.45"}
            c = dec["selected"]
            if c != "esrgan":
                pipe = sr.load_cn_img2img(DEVICE, lora=c.startswith("cn_lora")) if c.startswith("cn") else sr.load_xl_img2img()
                size = (sr.W, sr.H) if c.startswith("cn") else (1024, 768)
                ims = sr.rerender_shots(pipe, res["shots"]["shots"], dist, res["box"], prompt, float(c.rsplit("_s", 1)[1]),
                                        seed, size)
                pipe = None
                mem.free()
            else:
                ims = [s["raw"] for s in res["shots"]["shots"]]
            for s, im in zip(res["shots"]["shots"], ims):
                s["image"] = sr.upscale(MODELS["esrgan"], im)
        with watch.stage("dollhouse"):
            dd = p5v.decision("v5_doll_decision.json") or {"fill": "lama", "clean": "bilateral"}
            res["doll_v5"] = dv.build_dollhouse(dist, hires, res, ana, id2label, DEVICE, dd["fill"], dd["clean"])
            dv.free_lama()
            dv.free_sd()
            mem.free()
            m = r4.build_meshes(dist, hires, res, ana, id2label)
            res["mesh"], res["dollhouse"] = m["inside"], res["doll_v5"]
        out_dir = OUT / f"room360v5_{pl.slug(prompt, seed)}_k{k}"
        with watch.stage("export"):
            files = p5v.save(res, out_dir, prompt)
            files["pano"] = str(out_dir / "pano.png")
            cv2.imwrite(files["pano"], cv2.cvtColor(pano, cv2.COLOR_RGB2BGR))
            vw.pano_sphere_glb(hires, out_dir / "pano_sphere.glb")
            zp = out_dir / "design_shots.zip"
            with zipfile.ZipFile(zp, "w") as z:
                for s in res["shots"]["shots"]:
                    z.write(files[f"shot_{s['letter']}"], f"{s['letter']}.jpg")
                z.write(files["plan_png"], "floorplan_with_views.png")
            hero = res["shots"]["shots"][0]
            dc = np.array([0.0, 0.0, 1.0]) @ ds.view_to_cam(res["box"], hero["yaw"])
            lim = (p5v.decision("v5_viewer_selection.json") or {}).get("max_translation_m", p5v.MESH_MAX_TRANSLATION)
            extra.update({"shots_zip": str(zp), "sphere": str(out_dir / "pano_sphere.glb"), "pano_path": files["pano"],
                          "hero_lon": float(np.degrees(np.arctan2(dc[0], dc[2]))),
                          "raw": [(s["raw"], f"{s['letter']}: crop from the panorama (before re-rendering)")
                                  for s in res["shots"]["shots"]],
                          "mesh_tab": mesh_v5_html(files["room_glb"], res["bg_color"], p4v.hero_yaw_A(res), lim)})
    s, b = watch.seconds, res["box"]
    peaks.update({"total_vram_gb": max(peaks.get("sdxl_vram_gb", 0), torch.cuda.max_memory_allocated() / 2 ** 30
                                       if DEVICE.type == "cuda" else 0.0), "peak_ram_gb": pm_all.peak_ram_gb})
    extra["timing"] = {**s, **peaks, "total_s": sum(s.values())}
    sh = res["shots"]["shots"]
    cs = "; ".join(f"seed {seed + i}: boxiness {c['score']['score']:.2f}, {c['score']['area_m2']:.1f} m², nearest furniture "
                   f"{c['score']['d_near']:.2f} m" for i, c in enumerate(cands))
    info = (f"**360° room (v5) done in {sum(s.values()):.0f} s** ({len(panos)} panorama(s) {s['candidates']:.0f} s, "
            f"scoring {s['scoring']:.0f} s, instances {s['instances']:.0f} s, 4096 panorama {s['sharpen']:.0f} s, room + shots "
            f"{s['room_shots']:.0f} s, re-rendering {s['rerender']:.0f} s, dollhouse {s['dollhouse']:.0f} s, export "
            f"{s['export']:.0f} s). Peak VRAM {peaks['total_vram_gb']:.1f} GB, peak RAM {pm_all.peak_ram_gb:.1f} GB.\n\n"
            f"Candidates: {cs}; kept seed {seed + best}.\n\n"
            f"Room ≈ {b['width']:.2f} × {b['depth']:.2f} m ({b['area']:.1f} m²), ceiling {b['ceiling_height']:.2f} m, camera "
            f"{b['camera_height']:.2f} m above the floor. Hero: {sv.caption(sh[0])}. "
            f"{sum(bool(x.get('valid')) for x in sh)} of {len(sh)} shots pass the distance test. Shots re-rendered "
            f"({dec['selected']}) and upscaled to 1536 × 1152. Dimensions are approximate (monocular depth).")
    masks = an.room_masks(ana["class_map"], id2label)
    small = cv2.resize(pano, (1024, 512), interpolation=cv2.INTER_AREA)
    return ([(x["image"], sv.caption(x)) for x in sh], res["plan_image"], inside_v5_html(files["inside_html"]),
            dollhouse_html(files["dollhouse_glb"], b["camera_height"], NEUTRAL_BG), gr.Model3D(value=files["room_glb"]),
            pl.depth_gray(dist), an.overlay_edges(small, an.canny_edges(small)), an.overlay_masks(small, masks),
            files["pano"], files["plan_svg"], files["room_glb"], files["dollhouse_glb"], info, files["plan_dxf"],
            res["topdown"])


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
                    "floor plan (about 20 s). **Layout-first** designs the room plan first (bedroom, living room, "
                    "kitchen, dining room or bathroom; the seed picks the layout), generates the panorama to follow "
                    "it, and builds exact 3D. The first request of a mode loads its models.")
        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio([QUICK, ROOM360, LAYOUT], value=QUICK, label="Mode")
                prompt = gr.Textbox(label="Prompt", value=pl.DEMO_PROMPTS[1][0], lines=2)
                seed = gr.Number(label="Seed", value=pl.DEMO_PROMPTS[1][1], precision=0)
                steps = gr.Slider(10, 50, value=g2.GEN_STEPS, step=1, label="Steps")
                cfg = gr.Slider(1.0, 15.0, value=g2.GEN_GUIDANCE, step=0.5, label="CFG (guidance scale)")
                with gr.Accordion("360° Room options", open=False):
                    version = gr.Radio([V5, V4, V3, V2], value=V5, label="Pipeline")
                    candidates = gr.Slider(1, 4, value=V5_CANDIDATES, step=1, label="Candidates: panoramas generated; v5 keeps the "
                                           "most box-like, largest room with the camera clear of furniture, v3 / v4 the most "
                                           "box-like (v5: ~55 s per SDXL panorama; v4 used 3)")
                    sharp_shots = gr.Checkbox(value=True, label="v4 sharp design shots: Real-ESRGAN on each shot, "
                                              "1536 x 1152 (~1 s per shot; off = fast 1024 x 768 shots)")
                    n_objects = gr.Slider(0, 12, value=DEFAULT_OBJECTS, step=1,
                                          label="v4 furniture objects: how many pieces (largest first) are generated "
                                                "as real 3D shapes with TripoSR (~3 s each; 0 = v3 blocks)")
                    detail = gr.Checkbox(value=False, label="v4 detail pass: img2img (SD 1.5 + MMIS LoRA, strength 0.3) "
                                         "on the 26 views before upscaling (+~75 s; selected on the dev rooms, "
                                         "see README)")
                btn = gr.Button("Generate", variant="primary")
                info = gr.Markdown()
                with gr.Row():
                    dl_image = gr.DownloadButton("Download image (.png)")
                    dl_plan = gr.DownloadButton("Download floor plan (.svg)")
                with gr.Row():
                    dl_mesh = gr.DownloadButton("Download mesh (.glb)")
                    dl_cloud = gr.DownloadButton("Download point cloud (.ply) / dollhouse (.glb)")
                dl_dxf = gr.DownloadButton("Download floor plan (.dxf, 360° v3 / v4)")
                dl_shots = gr.DownloadButton("Download design shots (.zip, 360° v4)")
                gr.Examples([[p, s] for p, s in pl.DEMO_PROMPTS], inputs=[prompt, seed], label="Example prompts")
            with gr.Column(scale=2):
                with gr.Row():
                    with gr.Column(scale=3):
                        image = gr.Gallery(label="Design shots (H = hero view, A-D = wall views; letters as on the "
                                           "plan) / generated image", columns=3, height=420, object_fit="contain",
                                           preview=True, show_label=True)
                    with gr.Column(scale=2):
                        plan = gr.Image(label="Floor plan with camera and view cones (approximate dimensions)",
                                        type="numpy", height=420)
                with gr.Tabs():
                    with gr.Tab("Inside"):
                        viewer = gr.HTML(VIEWER_EMPTY)
                    with gr.Tab("Dollhouse (360°)"):
                        doll = gr.HTML(DOLL_EMPTY)
                    with gr.Tab("Top-down"):
                        topview = gr.Image(label="Top view (orthographic, same orientation as the floor plan; v3 / v4)",
                                           type="numpy", height=420)
                    with gr.Tab("360° panorama"):
                        pano_view = gr.HTML(PANO_EMPTY)
                    with gr.Tab("Experimental: walkable mesh"):
                        mesh_tab = gr.HTML(MESH_EMPTY)
                    with gr.Tab("3D: free orbit (offline)"):
                        model3d = gr.Model3D(label="Textured mesh (drag to rotate, scroll to zoom)", height=540,
                                             clear_color=[1.0, 1.0, 1.0, 1.0])
                with gr.Accordion("Analysis: depth, edges, segmentation", open=False):
                    pano_img = gr.Image(label="Panorama (equirectangular, 360° x 180°)", type="numpy", height=300)
                    raw_gallery = gr.Gallery(label="v5: design shots as cut from the panorama, before re-rendering",
                                             columns=5, height=220, object_fit="contain")
                    with gr.Row():
                        depth = gr.Image(label="Depth map (white = near, black = far)", type="numpy", height=260)
                        edges = gr.Image(label="Canny edges", type="numpy", height=260)
                        seg = gr.Image(label="Segmentation", type="numpy", height=260)
                    gr.Markdown("Segmentation colours: blue = wall, orange = floor, green = ceiling, pink = furniture.")
        btn.click(run, [mode, prompt, seed, steps, cfg, version, candidates, detail, n_objects, sharp_shots],
                  [image, plan, viewer, doll, model3d, depth, edges, seg, dl_image, dl_plan, dl_mesh, dl_cloud, info,
                   dl_dxf, topview, pano_view, pano_img, dl_shots, mesh_tab, raw_gallery],
                  concurrency_limit=1)
    return demo


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    load_models(QUICK)
    build().queue(max_size=8).launch(server_name="127.0.0.1", server_port=7860, allowed_paths=[str(OUT)], head=HEAD)
