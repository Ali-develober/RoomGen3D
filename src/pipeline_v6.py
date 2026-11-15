"""v6 room pipeline (PLAN_V6): scene first, image second (Ctrl-Room order).

    prompt + seed
      -> layout (src/layout_v6.py: room size, door, windows, furniture by rules)        CPU, < 0.1 s
      -> proxy scene (src/scene_v6.py: shell with openings + CC0 furniture models)      CPU, < 0.2 s
      -> panorama: SDXL + 360Redmond + depth conditioning on the proxy depth (src/pano_v6.py)   GPU
      -> Real-ESRGAN x2 -> 4096 x 2048 for the Inside viewer                                  GPU
      -> SegFormer class map of the panorama (class-consistency test of the dollhouse)          GPU
      -> cameras in the scene (src/cameras_v6.py) -> shots: reprojection through the proxy, LaMa holes,
         SD 1.5 + MMIS LoRA + depth ControlNet on the exact depth, Real-ESRGAN (src/shots_v6.py)  GPU
      -> dollhouse (.glb) + top-down from the textured proxy (src/dollhouse_v6.py)              CPU
      -> floor plan from the layout with every shot cone (src/plan_v6.py)                      CPU
One GPU model at a time; each is loaded once per request and freed before the next (src/memory.py).
Choices (conditioning, strength) are read from results/tables/v6_*_decision.json when present.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np


from . import cameras_v6 as CV
from . import layout_v6 as LV
from . import memory as mem
from . import plan_v6 as PV
from . import scene_v6 as S

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
DEFAULTS = {"pano": {"cond": "cn_small", "scale": 0.7, "end": 1.0, "steps": 25},
            "shots": {"strength": 0.45, "lora": True, "cn_scale": 0.8}}
SHOT_ORDER = ["H", "H2", "A", "B", "C", "D"]


def decision(name: str, key: str) -> dict:
    p = TAB / name
    d = dict(DEFAULTS[key])
    if p.exists():
        d.update(json.loads(p.read_text()).get("selected_params", {}))
    return d


def required_models() -> dict:
    """Local files every v6 stage needs (offline check)."""
    from . import generate_pano_xl as gx
    from . import pano_v6 as P6
    from . import shot_render as SR
    cond = decision("v6_pano_decision.json", "pano")["cond"]
    return {"SDXL base": gx.SDXL_BASE / "unet", "SDXL VAE fp16-fix": gx.SDXL_VAE, "360Redmond LoRA": gx.REDMOND,
            "SDXL depth conditioning": P6.CONDS[cond], "SD 1.5 depth ControlNet": SR.CONTROLNET_DEPTH,
            "MMIS LoRA": SR.LORA, "LaMa": ROOT / "checkpoints" / "lama" / "lama.pt",
            "Real-ESRGAN": ROOT / "checkpoints" / "realesrgan"}


def missing_models() -> list[str]:
    return [k for k, p in required_models().items() if not Path(p).exists()]


class Stopwatch:
    def __init__(self):
        self.s = {}

    def stage(self, name):
        sw = self

        class _C:
            def __enter__(self_):
                self_.t = time.perf_counter()

            def __exit__(self_, *a):
                sw.s[name] = sw.s.get(name, 0.0) + time.perf_counter() - self_.t
        return _C()


def build_scene(prompt: str, seed: int) -> tuple[dict, S.Renderer]:
    sc = S.build(LV.generate(prompt, seed))
    return sc, S.Renderer(sc)


def panorama(sc: dict, rd: S.Renderer, seed: int, device="cuda", pipe=None) -> tuple[np.ndarray, dict]:
    from . import pano_v6 as P6
    p = decision("v6_pano_decision.json", "pano")
    lay = sc["layout"]
    e = rd.equirect(lay["camera"], 2048, 1024)
    ctrl = S.depth_control(e["t"])
    prompt = P6.pano_prompt(lay)
    own = pipe is None
    if own:
        mem.wait_for_ram(7.0, label="v6 SDXL")
        pipe = P6.load_pipeline(p["cond"])
    img = P6.generate(pipe, ctrl, prompt, seed, scale=p["scale"], end=p["end"], steps=int(p["steps"]))
    if own:
        mem.free(pipe)
        del pipe
        mem.free()
    return img, {"prompt": prompt, "control": ctrl, "tokens": P6.n_tokens(prompt), **p}


def segment(pano: np.ndarray, device="cuda") -> np.ndarray:
    from . import analysis as an
    model, proc = an.load_segmenter(device)
    seg = an.segment_classes(model, proc, pano, device)
    mem.free(model, proc)
    return seg


def run(prompt: str, seed: int, out_dir, device="cuda", with_shots: bool = True, log=print) -> dict:
    """Full v6 room. Returns paths, images, metrics and timing."""
    from . import dollhouse_v6 as DH
    from . import generate_pano as gp
    from . import pano_viewer as PVW
    from . import shot_render as SR
    from . import shots_v6 as SH
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    watch = Stopwatch()
    res = {"prompt": prompt, "seed": int(seed)}
    with mem.PeakMonitor() as pm:
        with watch.stage("scene"):
            sc, rd = build_scene(prompt, seed)
            lay = sc["layout"]
        with watch.stage("panorama"):
            pano, pinfo = panorama(sc, rd, seed, device)
        log(f"panorama {watch.s['panorama']:.0f} s")
        with watch.stage("esrgan_seg"):
            esr = gp.load_esrgan(device)
            hires = gp.esrgan_upscale(esr, pano)                     # 4096 x 2048 for the Inside viewer
            seg = segment(pano, device)
        with watch.stage("cameras"):
            cams = CV.choose_cameras(rd, lay)
        shots = {}
        if with_shots:
            with watch.stage("shots"):
                sp = decision("v6_shots_decision.json", "shots")
                raws = {}
                for name in SHOT_ORDER:
                    cam = cams.get(name)
                    if cam is None:
                        continue
                    raw, vis, r = SH.reproject(rd, pano, lay["camera"], cam)
                    raws[name] = (raw, vis, r)
                from . import dollhouse_v5 as dh5
                inits = {n: SH.fill_holes(raw, vis, device) for n, (raw, vis, r) in raws.items()}
                dh5.free_lama()
                mem.free()
                pipe = SR.load_cn_img2img(device, lora=bool(sp["lora"]))
                prompt_s = SH.shot_prompt(lay, "H")
                for k, (n, (raw, vis, r)) in enumerate(raws.items()):
                    base = SH.rerender(pipe, inits[n], r["z"], prompt_s, float(sp["strength"]), seed + k,
                                       float(sp.get("cn_scale", SR.CN_SCALE)))
                    shots[n] = {"base": base, "raw": raw, "init": inits[n], "hole_share": float(1 - vis.mean())}
                mem.free(pipe)
                del pipe
                mem.free()
                for n in shots:
                    shots[n]["image"] = SR.upscale(esr, shots[n]["base"])
        mem.free(esr)
        del esr
        mem.free()
        with watch.stage("dollhouse"):
            tp = DH.textured_parts(rd, sc, pano, seed, seg)
            DH.export_glb(tp, out_dir / "dollhouse_v6.glb")
            top = DH.topdown(tp, sc)
        with watch.stage("export"):
            plan_shots = [{"name": n, "pos": c["pos"], "yaw": c["yaw"], "hfov": c["hfov"]}
                          for n, c in cams.items() if c is not None]
            plan = PV.draw(lay, out_dir / "floorplan_v6.png", shots=plan_shots, models=sc["models"])
            PV.draw(lay, out_dir / "floorplan_v6.svg", shots=plan_shots, models=sc["models"])
            cv2.imwrite(str(out_dir / "pano_2048.jpg"), cv2.cvtColor(pano, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(out_dir / "pano_4096.jpg"), cv2.cvtColor(hires, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(out_dir / "topdown_v6.png"), cv2.cvtColor(top, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_dir / "control.png"), pinfo["control"])
            (out_dir / "shots").mkdir(exist_ok=True)
            for n, s in shots.items():
                cv2.imwrite(str(out_dir / "shots" / f"{n}.jpg"), cv2.cvtColor(s["image"], cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 93])
                cv2.imwrite(str(out_dir / "shots" / f"{n}_reprojected.jpg"), cv2.cvtColor(s["raw"], cv2.COLOR_RGB2BGR))
            hero = cams["H"]
            PVW.save_viewer(out_dir / "inside_v6.html", hires, None, hero["yaw"], 80.0, title=prompt)
            LV.save(lay, out_dir / "layout.json")
            meta = {"cameras": {n: c for n, c in cams.items() if c is not None}, "models": sc["models"],
                    "pano": {k: v for k, v in pinfo.items() if k != "control"},
                    "seen_share": tp["tex"]["seen"], "pieces": len(lay["items"])}
            (out_dir / "summary_v6.json").write_text(json.dumps(meta, indent=1, default=float))
    res.update({"layout": lay, "scene": sc, "pano": pano, "hires": hires, "seg": seg, "cams": cams, "shots": shots,
                "plan": plan, "topdown": top, "dir": out_dir, "timing": dict(watch.s), "peak_ram_gb": pm.peak_ram_gb,
                "peak_vram_gb": pm.peak_vram_gb, "pano_info": pinfo, "doll_seen": tp["tex"]["seen"]})
    (out_dir / "timing.json").write_text(json.dumps({**watch.s, "total_s": sum(watch.s.values()),
                                                     "peak_ram_gb": pm.peak_ram_gb, "peak_vram_gb": pm.peak_vram_gb},
                                                    indent=1))
    return res
