"""Gradio demo: text prompt -> 2D image + depth + edges/segmentation + interactive 3D viewer.

Runs the same code as scripts/run_pipeline.py (src/pipeline.py):
    prompt -> SD v1.5 + LoRA -> noise-gated preprocessing -> Canny + SegFormer -> metric depth
           -> depth refinement -> point cloud -> Poisson mesh (.glb)

GPU memory (8 GB card): the three models stay loaded together in fp16. Measured peak while
generating: 2.9 GB (torch peak, all three loaded), so no
load / unload is needed between requests. One request runs at a time.

Launch (from the project root):   .venv\\Scripts\\python.exe app\\app.py
then open http://127.0.0.1:7860
"""

import sys
import time
from pathlib import Path

import gradio as gr
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import analysis as an  # noqa: E402
from src import depth as dp  # noqa: E402
from src import generate_2d as g2  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import reconstruct_3d as r3  # noqa: E402

OUT = ROOT / "outputs" / "app"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = {}


def load_models() -> dict:
    """Load SD + LoRA, the segmenter and the depth model once (all fp16) and keep them."""
    if not MODELS:
        t0 = time.perf_counter()
        MODELS["sd"] = g2.load_pipeline(g2.DEFAULT_LORA if g2.DEFAULT_LORA.exists() else None, DEVICE)
        MODELS["seg"] = an.load_segmenter(DEVICE)
        MODELS["depth"] = dp.load_depth_model(pl.DEPTH_MODEL_3D, DEVICE)
        used = torch.cuda.memory_allocated() / 2 ** 30 if DEVICE.type == "cuda" else 0.0
        print(f"models loaded in {time.perf_counter() - t0:.1f} s, {used:.2f} GB VRAM in use")
    return MODELS


def run(prompt: str, seed: float, steps: float, cfg: float):
    """One request: returns the pictures, the mesh for the viewer, the download files and a summary."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Please enter a prompt, e.g. 'a coastal style bedroom'.")
    models = load_models()
    seed, steps = int(seed), int(steps)
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    watch = pl.Stopwatch()
    with watch.stage("generate"):
        img = g2.generate(models["sd"], prompt, seed, steps=steps, guidance=cfg)
    out_dir = OUT / f"{pl.slug(prompt, seed)}_st{steps}_cfg{cfg:g}"
    res = pl.image_to_3d(img, models["seg"], models["depth"], DEVICE, watch=watch, out_dir=out_dir)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if DEVICE.type == "cuda" else 0.0
    s, st, d = watch.seconds, res["stats"], res["depth"]
    total = sum(s.values())
    info = (f"**Done in {total:.1f} s** (image {s['generate']:.1f} s, analysis + depth "
            f"{s['preprocess'] + s['analysis'] + s['depth'] + s['refine']:.2f} s, point cloud {s['pointcloud']:.1f} s, "
            f"mesh {s['mesh']:.1f} s, export {s['export']:.1f} s). Peak VRAM {peak:.1f} GB.\n\n"
            f"Depth {d.min():.1f} to {d.max():.1f} m. Point cloud: {st['points']:,} points "
            f"({st['flying_removed']:,} flying pixels and {st['outliers_removed']:,} outliers removed). "
            f"Mesh: {st['triangles']:,} triangles. "
            f"Estimated image noise σ = {res['info']['noise_est']:.1f} "
            f"({'denoised' if res['info']['denoised'] else 'not denoised'}). "
            f"Assumed camera: {r3.DEFAULT_FOV_DEG:.0f}° horizontal field of view.")
    f = res["files"]
    return (img, pl.depth_image(d), pl.combined_overlay(res["pre"], res["edges"], res["masks"]), f["mesh_glb"],
            f["image"], f["mesh_glb"], f["pointcloud"], info)


def build() -> gr.Blocks:
    with gr.Blocks(title="RoomGen3D") as demo:
        gr.Markdown("# RoomGen3D\nText prompt → interior design image → depth → 3D room model. "
                    "The LoRA was trained on captions like *a coastal style bedroom*, so that form works best. "
                    "The 3D model is built from a single view: surfaces hidden in the image are missing.")
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Prompt", value=pl.DEMO_PROMPTS[1][0], lines=2)
                seed = gr.Number(label="Seed", value=pl.DEMO_PROMPTS[1][1], precision=0)
                steps = gr.Slider(10, 50, value=g2.GEN_STEPS, step=1, label="Steps")
                cfg = gr.Slider(1.0, 15.0, value=g2.GEN_GUIDANCE, step=0.5, label="CFG (guidance scale)")
                btn = gr.Button("Generate", variant="primary")
                info = gr.Markdown()
                with gr.Row():
                    dl_image = gr.DownloadButton("Download image (.png)")
                    dl_mesh = gr.DownloadButton("Download mesh (.glb)")
                    dl_cloud = gr.DownloadButton("Download point cloud (.ply)")
                gr.Examples([[p, s] for p, s in pl.DEMO_PROMPTS], inputs=[prompt, seed], label="Example prompts")
            with gr.Column(scale=2):
                with gr.Row():
                    image = gr.Image(label="2D image", type="numpy", height=320)
                    depth = gr.Image(label="Depth map (light = near)", type="numpy", height=320)
                    overlay = gr.Image(label="Canny edges (white) + segmentation", type="numpy", height=320)
                model3d = gr.Model3D(label="3D mesh (drag to rotate, scroll to zoom)", height=520,
                                     clear_color=[1.0, 1.0, 1.0, 1.0])
        gr.Markdown("Segmentation colours: blue = wall, orange = floor, green = ceiling, pink = furniture.")
        btn.click(run, [prompt, seed, steps, cfg], [image, depth, overlay, model3d, dl_image, dl_mesh, dl_cloud, info],
                  concurrency_limit=1)
    return demo


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    load_models()
    build().queue(max_size=8).launch(server_name="127.0.0.1", server_port=7860, allowed_paths=[str(OUT)])
