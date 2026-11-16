"""Post-freeze measurement of the v6 panorama (tag v6-final; no pipeline code, default or app behaviour is changed here).

Why: results/tables/v6_pano_summary.csv was measured before the window phrase moved forward in the prompt and before
the window repair, and the v6 ranking rule (results/tables/v6_pano_decision.json) was written after the dev sweep.
This script (1) re-measures the FINAL v6 panorama against the FINAL v5 panorama on the same 26 rooms and seeds and
(2) evaluates the frozen ranking rule once on the 10 held-out rooms, which the sweep never used.

Panoramas (outputs/v6_final/<config>/<room id>.jpg + .json with seconds / peak VRAM / peak RAM):
    cn_small_s0.9_e1.0   v6 raw: src/pipeline_v6.panorama (final prompt, selected conditioning), before any repair
    v6_final             v6 shipped: the raw panorama through src/pipeline_v6.run(with_shots=False, reuse_pano=True),
                         i.e. the pipeline's own segmentation + window repair + extra-window removal
    v5_final             v5 shipped (app run_360_v5 default): 2 SDXL + 360Redmond candidates (seed, seed + 1, v5 prompt)
                         -> DAP + SegFormer scoring -> camera_v5.pick_candidate with the v5 selection weights
    <7 other sweep cfgs> held-out rooms only, same generation code as the sweep (final prompt)
Metrics: the functions of scripts/eval_v6_pano.py (8 level views 90 deg 768 x 576 per panorama): depth agreement vs the
proxy (Depth Anything, scale / shift aligned AbsRel, delta1), instance presence / IoU (SegFormer vs proxy instance ids),
structure-edge recall, straight-line share, window share, BRISQUE, sharpness, seam ratio, CLIP, KID vs MMIS hold-out.
For v5 the proxy metrics are a no-conditioning baseline: v5 never saw the proxy of that prompt / seed.

Stages (each its own process, one GPU model at a time, resumable):
    download  re-fetch cn_mid + t2i fp16 (deleted after the sweep) -> checkpoints/
    gen_v6    SDXL cn_small, all rooms                         gen_sweep  the 7 other configs, held-out rooms
    post_v6   pipeline_v6.run on the cached raw panoramas      gen_v5     2 v5 candidates per room
    pick_v5   v5 candidate scoring                              metrics    views + depth + seg + edges + quality, clip
    report    tables (incl. KID) + the frozen rule on held-out + figure
Usage: python scripts/eval_v6_final.py --stage gen_v6 [--ids demo_a_mid_century_modern_style_dining_room_s44]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import eval_v6_pano as E  # noqa: E402
import v5_common as C  # noqa: E402
from src import memory as mem  # noqa: E402

OUT = ROOT / "outputs" / "v6_final"
E.OUT = OUT                                     # reuse the milestone-2 metric code on the new panoramas
TAB, FIG = C.TAB, C.FIG
DEVICE = "cuda"
SELECTED = "cn_small_s0.9_e1.0"
SWEEP = ["cn_small_s0.5_e1.0", "cn_small_s0.7_e1.0", "cn_small_s0.7_e0.7", "cn_small_s0.9_e1.0",
         "cn_mid_s0.5_e0.8", "cn_mid_s0.7_e1.0", "t2i_s0.8_e1.0", "t2i_s1.0_e1.0"]
DOWNLOADS = {"cn_mid": ("diffusers/controlnet-depth-sdxl-1.0-mid", "cn_depth_sdxl_mid"),
             "t2i": ("TencentARC/t2i-adapter-depth-midas-sdxl-1.0", "t2i_depth_sdxl")}


def rooms(ids=None, sets=None) -> pd.DataFrame:
    df = C.rooms()
    if sets:
        df = df[df.set.isin(sets)]
    if ids:
        df = df[df.id.isin(ids)]
    return df.reset_index(drop=True)


def write_meta(p: Path, **kw):
    p.write_text(json.dumps(kw, indent=1, default=float))


# --------------------------------------------------------------------------- downloads
def stage_download():
    from huggingface_hub import hf_hub_download
    for cond, (repo, folder) in DOWNLOADS.items():
        d = ROOT / "checkpoints" / folder
        if (d / "diffusion_pytorch_model.safetensors").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        tmp = ROOT / "checkpoints" / f"_dl_{folder}"
        shutil.copy(hf_hub_download(repo, "config.json", local_dir=tmp), d / "config.json")
        w = hf_hub_download(repo, "diffusion_pytorch_model.fp16.safetensors", local_dir=tmp)
        shutil.move(w, d / "diffusion_pytorch_model.safetensors")   # same layout as cn_depth_sdxl_small
        shutil.rmtree(tmp, ignore_errors=True)
        print(cond, "->", d, f"{(d / 'diffusion_pytorch_model.safetensors').stat().st_size / 2**30:.2f} GB")


# --------------------------------------------------------------------------- v6 generation
def stage_gen_v6(df):
    from src import pano_v6 as P6
    from src import pipeline_v6 as P
    from src import scene_v6 as S
    d = OUT / SELECTED
    d.mkdir(parents=True, exist_ok=True)
    todo = [r for r in df.itertuples() if not (d / f"{r.id}.json").exists()]
    if not todo:
        return
    p = P.decision("v6_pano_decision.json", "pano")
    assert f"{p['cond']}_s{p['scale']}_e{p['end']}" == SELECTED, p
    t0 = time.perf_counter()
    pipe = P6.load_pipeline(p["cond"])
    load_s = time.perf_counter() - t0
    for r in todo:
        sc, rd = P.build_scene(r.prompt, r.seed)
        with mem.PeakMonitor() as pm:
            img, info = P.panorama(sc, rd, r.seed, DEVICE, pipe=pipe)     # the pipeline's own panorama step
        C.save_rgb(d / f"{r.id}.jpg", img, 95)
        cv2.imwrite(str(d / f"{r.id}_control.png"), info["control"])
        write_meta(d / f"{r.id}.json", seconds=pm.seconds, peak_vram_gb=pm.peak_vram_gb, peak_ram_gb=pm.peak_ram_gb,
                   load_s=load_s, prompt=info["prompt"], tokens=info["tokens"])
        print("gen_v6", r.id, f"{pm.seconds:.0f} s {pm.peak_vram_gb:.2f} GB", flush=True)
    mem.free(pipe)


def stage_post_v6(df):
    """Shipped post-processing of the panorama: pipeline_v6.run on the cached raw panorama (reuse_pano=True)."""
    from src import pipeline_v6 as P
    src, dst = OUT / SELECTED, OUT / "v6_final"
    dst.mkdir(parents=True, exist_ok=True)
    for r in df.itertuples():
        if (dst / f"{r.id}.json").exists() or not (src / f"{r.id}.json").exists():
            continue
        run_dir = OUT / "_runs" / r.id
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(src / f"{r.id}.jpg", run_dir / "pano_raw_2048.jpg")
        shutil.copy(src / f"{r.id}_control.png", run_dir / "control.png")
        res = P.run(r.prompt, r.seed, run_dir, DEVICE, with_shots=False, log=lambda *a: None, reuse_pano=True)
        assert res["pano_info"].get("cached"), f"{r.id}: panorama was regenerated, not reused"
        shutil.copy(run_dir / "pano_2048.jpg", dst / f"{r.id}.jpg")
        g = json.loads((src / f"{r.id}.json").read_text())
        t = res["timing"]
        repair = t.get("window_fix", 0.0)
        write_meta(dst / f"{r.id}.json",
                   seconds=g["seconds"] + t["segmentation"] + repair,      # generation + segmentation + repair
                   gen_s=g["seconds"], seg_s=t["segmentation"], repair_s=repair,
                   sd15_load_s=t.get("shots", 0.0) if repair else 0.0,    # in the app SD 1.5 is loaded for the shots anyway
                   peak_vram_gb=max(g["peak_vram_gb"], res["peak_vram_gb"]),
                   peak_ram_gb=max(g["peak_ram_gb"], res["peak_ram_gb"]),
                   post_peak_vram_gb=res["peak_vram_gb"], post_peak_ram_gb=res["peak_ram_gb"],
                   window_presence=res["window_presence"], window_presence_after=res.get("window_presence_fixed"),
                   extra_windows=res["extra_windows"], extra_windows_after=res.get("extra_windows_after"),
                   repaired=bool(repair))
        for f in ("pano_4096.jpg", "dollhouse_v6.glb", "topdown_v6.png", "inside_v6.html"):   # not needed, disk
            (run_dir / f).unlink(missing_ok=True)
        print("post_v6", r.id, f"presence {res['window_presence']:.2f} -> {res.get('window_presence_fixed')}",
              f"extra {res['extra_windows']}", f"repair {repair:.0f} s", flush=True)


# --------------------------------------------------------------------------- v5
def stage_gen_v5(df):
    from src import generate_pano_xl as gx
    from src import pipeline_v5 as p5v
    gen = p5v.source()
    d = OUT / "_v5_candidates"
    d.mkdir(parents=True, exist_ok=True)
    k = 2                                                     # app.V5_CANDIDATES
    todo = [r for r in df.itertuples() if not (d / f"{r.id}_c{k - 1}.json").exists()]
    if not todo:
        return
    t0 = time.perf_counter()
    pipe = gx.load_xl_pipeline(gen.split("_", 1)[1])
    load_s = time.perf_counter() - t0
    for r in todo:
        for i in range(k):
            if (d / f"{r.id}_c{i}.json").exists():
                continue
            with mem.PeakMonitor() as pm:
                img = p5v.generate(pipe, gen, r.prompt, r.seed + i, spacious=True)
            C.save_rgb(d / f"{r.id}_c{i}.jpg", img, 95)
            write_meta(d / f"{r.id}_c{i}.json", seconds=pm.seconds, peak_vram_gb=pm.peak_vram_gb,
                       peak_ram_gb=pm.peak_ram_gb, load_s=load_s, seed=r.seed + i)
            print("gen_v5", r.id, i, f"{pm.seconds:.0f} s", flush=True)
    mem.free(pipe)


def stage_pick_v5(df):
    import torch
    from src import analysis as an
    from src import camera_v5 as cam
    from src import pano_depth as pdp
    from src import pipeline_v5 as p5v
    d, dst = OUT / "_v5_candidates", OUT / "v5_final"
    dst.mkdir(parents=True, exist_ok=True)
    todo = [r for r in df.itertuples() if not (dst / f"{r.id}.json").exists() and (d / f"{r.id}_c1.json").exists()]
    if not todo:
        return
    dev = torch.device(DEVICE)
    seg = an.load_segmenter(dev)
    id2label = {int(i): v for i, v in seg[0].config.id2label.items()}
    dap = pdp.load_dap(dev)[0]
    w = p5v.selection_weights()
    for r in todo:
        panos = [C.rgb(d / f"{r.id}_c{i}.jpg") for i in range(2)]
        metas = [json.loads((d / f"{r.id}_c{i}.json").read_text()) for i in range(2)]
        with mem.PeakMonitor() as pm:
            cands = [p5v.score_candidate(x, dap, seg, id2label, dev) for x in panos]
            best = cam.pick_candidate([c["score"] for c in cands], w)
        shutil.copy(d / f"{r.id}_c{best}.jpg", dst / f"{r.id}.jpg")
        write_meta(dst / f"{r.id}.json", seconds=sum(m["seconds"] for m in metas) + pm.seconds,
                   gen_s=sum(m["seconds"] for m in metas), scoring_s=pm.seconds, picked=best,
                   peak_vram_gb=max([m["peak_vram_gb"] for m in metas] + [pm.peak_vram_gb]),
                   peak_ram_gb=max([m["peak_ram_gb"] for m in metas] + [pm.peak_ram_gb]))
        print("pick_v5", r.id, "candidate", best, flush=True)
    mem.free(dap, *seg)


# --------------------------------------------------------------------------- sweep configs on held-out
def stage_gen_sweep(df):
    others = [c for c in SWEEP if c != SELECTED]
    for c in others:                                         # one config dir per entry, as in the sweep
        (OUT / c).mkdir(parents=True, exist_ok=True)
    by_cond = {}
    for c in others:
        by_cond.setdefault(E.parse_config(c)["cond"], []).append(c)
    from src import pano_v6 as PV
    from src import scene_v6 as S
    for cond, cs in by_cond.items():
        todo = [(c, r) for c in cs for r in df.itertuples() if not (OUT / c / f"{r.id}.json").exists()]
        if not todo:
            continue
        t0 = time.perf_counter()
        pipe = PV.load_pipeline(cond)
        load_s = time.perf_counter() - t0
        for c, r in todo:
            cfg = E.parse_config(c)
            sc, rd = E.scene_of(r.prompt, r.seed)
            lay = sc["layout"]
            ctrl = S.depth_control(rd.equirect(lay["camera"], 2048, 1024)["t"])
            prompt = PV.pano_prompt(lay)
            with mem.PeakMonitor() as pm:
                img = PV.generate(pipe, ctrl, prompt, r.seed, scale=cfg["scale"], end=cfg["end"])
            C.save_rgb(OUT / c / f"{r.id}.jpg", img, 95)
            write_meta(OUT / c / f"{r.id}.json", seconds=pm.seconds, peak_vram_gb=pm.peak_vram_gb,
                       peak_ram_gb=pm.peak_ram_gb, load_s=load_s, prompt=prompt)
            print("gen_sweep", c, r.id, f"{pm.seconds:.0f} s", flush=True)
        mem.free(pipe)
        del pipe
        mem.free()


# --------------------------------------------------------------------------- metrics
def stage_metrics(df, cfgs):
    for c in cfgs:
        sub = df[[(OUT / c / f"{r.id}.jpg").exists() for r in df.itertuples()]]
        if len(sub):
            E.save_views(c, sub)
            E.stage_metrics(c, sub)
    E.stage_clip(cfgs, df)


# --------------------------------------------------------------------------- rule
RULE = ("gates: depth delta1 >= 0.9, seam ratio <= 1.5, generated window share <= 1.5 x proxy; then mean rank of "
        "depth AbsRel, instance IoU, structure-edge recall, straight-line share, KID, CLIP, BRISQUE")


def apply_rule(s: pd.DataFrame) -> tuple[pd.Series, list]:
    """The v6 ranking rule exactly as written in v6_pano_decision.json (frozen; reproduces the dev5 ranks)."""
    ok = s[(s["depth_d1"] >= 0.9) & (s["seam_ratio"] <= 1.5) & (s["window_share_gen"] <= 1.5 * s["window_share_proxy"])]
    r = pd.DataFrame({"absrel": ok["depth_absrel"].rank(), "iou": ok["inst_iou"].rank(ascending=False),
                      "edge": ok["struct_recall"].rank(ascending=False),
                      "straight": ok["straight_share"].rank(ascending=False), "kid": ok["kid_x1e3"].rank(),
                      "clip": ok["clip"].rank(ascending=False), "brisque": ok["brisque"].rank()})
    return r.mean(axis=1).sort_values(), sorted(set(s.index) - set(ok.index))


def rule_heldout(per: pd.DataFrame, kid: pd.DataFrame, n_boot: int = 1000) -> dict:
    test = per[(per.set == "test") & per.config.isin(SWEEP)]
    cols = ["depth_absrel", "depth_d1", "inst_iou", "struct_recall", "straight_share", "window_share_gen",
            "window_share_proxy", "brisque", "seam_ratio", "clip"]
    s = test.groupby("config")[cols].mean()
    s["n"] = test.groupby("config").size()
    s = s.join(kid[kid.set == "test"].set_index("config")[["kid_x1e3"]])
    assert (s["n"] == 10).all() and len(s) == len(SWEEP), s["n"]
    s.to_csv(TAB / "v6_pano_rule_heldout_summary.csv", float_format="%.4f")
    rank, failed = apply_rule(s)
    # stability: resample the 10 held-out rooms with replacement (KID kept at its full held-out value: it is a set
    # statistic and would need the Inception features per resample)
    rng = np.random.default_rng(0)
    ids = sorted(test.id.unique())
    wins = {}
    for _ in range(n_boot):
        pick = rng.choice(ids, len(ids), replace=True)
        b = pd.concat([test[test.id == i] for i in pick]).groupby("config")[cols].mean().join(s[["kid_x1e3"]])
        rb, _ = apply_rule(b)
        wins[rb.index[0]] = wins.get(rb.index[0], 0) + 1
    dev = json.loads((TAB / "v6_pano_decision.json").read_text())
    dev_order = [x["config"] for x in dev["ranking"]]
    common = [c for c in dev_order if c in rank.index]
    from scipy.stats import kendalltau
    tau = kendalltau([dev_order.index(c) for c in common], [list(rank.index).index(c) for c in common])[0] if len(common) > 2 else np.nan
    out = {"rule": RULE, "rule_frozen_from": "results/tables/v6_pano_decision.json (written after the dev5 sweep)",
           "evaluated_on": "10 held-out rooms (industrial / bohemian x 5 room types, seed 701), never used in the sweep; "
                           "final prompt, raw panoramas (no window repair), same as the sweep",
           "dev_selected": dev["selected"], "heldout_selected": rank.index[0],
           "choice_holds": bool(rank.index[0] == dev["selected"]),
           "heldout_ranking": [{"config": c, "mean_rank": round(float(v), 3)} for c, v in rank.items()],
           "heldout_failed_gates": failed, "dev_failed_gates": dev["failed_gates"],
           "kendall_tau_dev_vs_heldout": float(tau),
           "bootstrap_winner_share": {k: v / n_boot for k, v in sorted(wins.items(), key=lambda kv: -kv[1])}}
    C.write_json(TAB / "v6_pano_rule_heldout.json", out)
    return out


def stage_report(df):
    cfgs = ["v5_final", SELECTED, "v6_final"] + [c for c in SWEEP if c != SELECTED]
    per, _ = E.report([c for c in cfgs if (OUT / c / "metrics.csv").exists()], df, tag="v6_final_pano")
    # timing breakdown from the .json files
    extra = []
    for r in per.itertuples():
        m = json.loads((OUT / r.config / f"{r.id}.json").read_text())
        extra.append({"id": r.id, "config": r.config, **{k: m.get(k) for k in
                      ("gen_s", "seg_s", "repair_s", "scoring_s", "repaired", "picked", "window_presence",
                       "window_presence_after", "extra_windows", "extra_windows_after")}})
    per = per.merge(pd.DataFrame(extra), on=["id", "config"], how="left")
    per.to_csv(TAB / "v6_final_pano_per_room.csv", index=False, float_format="%.4f")
    kid = E.stage_kid([c for c in cfgs if (OUT / c / "views").exists()], df)
    kid.to_csv(TAB / "v6_final_pano_kid.csv", index=False, float_format="%.3f")
    main = per[per.config.isin(["v5_final", SELECTED, "v6_final"])]
    cols = ["depth_absrel", "depth_d1", "inst_iou", "inst_present", "struct_recall", "straight_share",
            "window_share_gen", "window_share_proxy", "brisque", "sharpness", "seam_ratio", "clip", "seconds",
            "peak_vram_gb", "peak_ram_gb"]
    g = main.groupby(["config", "set"])
    summ = g[cols].mean().join(g.size().rename("n")).reset_index()
    summ = summ.merge(kid, on=["config", "set"], how="left")
    allrow = main.groupby("config")[cols].mean().join(main.groupby("config").size().rename("n")).reset_index()
    allrow["set"] = "all"
    summ = pd.concat([summ, allrow], ignore_index=True)
    summ.to_csv(TAB / "v6_final_pano_summary.csv", index=False, float_format="%.4f")
    rule = rule_heldout(per, kid) if per[(per.set == "test") & per.config.isin(SWEEP)].groupby("config").size().eq(10).sum() == len(SWEEP) else None
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string(index=False))
    print(json.dumps(rule, indent=1, default=float))
    figure(df)


def figure(df):
    """v5 vs v6 raw vs v6 final, one row per demo room."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in df[df.set == "demo"].itertuples()]
    cols = [("v5_final", "v5 (shipped)"), (SELECTED, "v6 raw (no repair)"), ("v6_final", "v6 final (window repair)")]
    fig, ax = plt.subplots(len(rows), 3, figsize=(15, 2.6 * len(rows)))
    for i, r in enumerate(rows):
        for j, (c, t) in enumerate(cols):
            p = OUT / c / f"{r.id}.jpg"
            ax[i, j].imshow(cv2.resize(C.rgb(p), (1024, 512)) if p.exists() else np.zeros((512, 1024, 3), np.uint8))
            ax[i, j].set_axis_off()
            ax[i, j].set_title(f"{t} — {r.prompt}, seed {r.seed}" if j == 0 else t, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "v6_final_pano_v5_vs_v6.png", dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--ids", default="")
    a = ap.parse_args()
    ids = [x for x in a.ids.split(",") if x]
    df = rooms(ids)
    st = a.stage
    if st == "download":
        stage_download()
    elif st == "gen_v6":
        stage_gen_v6(df)
    elif st == "post_v6":
        stage_post_v6(df)
    elif st == "gen_v5":
        stage_gen_v5(df)
    elif st == "pick_v5":
        stage_pick_v5(df)
    elif st == "gen_sweep":
        stage_gen_sweep(rooms(ids, ["test"]))
    elif st == "metrics":
        stage_metrics(df, ["v5_final", SELECTED, "v6_final"])
    elif st == "metrics_sweep":
        stage_metrics(rooms(ids, ["test"]), [c for c in SWEEP if c != SELECTED])
    elif st == "report":
        stage_report(df)
    else:
        raise SystemExit(f"unknown stage {st}")


if __name__ == "__main__":
    main()
