"""PLAN_V5 milestone 2 (+ the backbone of milestones 3-5): candidates, camera / room-size measures, shot validity.

Rooms: v5_common (10 dev, 10 held-out, 6 demo). Generator: the milestone-1 choice (results/tables/v5_pano_decision.json).
Stages (resumable, one model in memory at a time; outputs/v5_rooms/<id>/):
    cand    K candidates with the v5 prompt (spacious, near a wall, no close-ups): seeds s .. s+K-1 -> cand_<seed>.png;
            the "before" panorama (v4 one-room prompt, seed s) is taken from milestone 1 (outputs/v5_pano/<gen>/)
    geo     DAP distance + SegFormer classes per panorama -> <name>_geo.npz; boxiness, area, nearest furniture
    inst    Mask2Former instances per panorama -> <name>_inst.npz / .json
    shots   per panorama: room box, v4 hero / wall shots and v5 valid-shot selection, validity measures (CPU)
    report  prompt effect (before vs after, same seed), candidate-selection weight grid chosen on dev (share of rooms
            whose hero is valid, then mean valid shots), v4 vs v5 validity shares on dev / held-out / demo
Usage: python scripts/run_v5_rooms.py --stage cand|geo|inst|shots|report [--sets dev] [--limit 2] [--k 3]
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v5_common as vc  # noqa: E402
from src import boxiness as bx  # noqa: E402
from src import camera_v5 as cam  # noqa: E402
from src import design_shots as ds  # noqa: E402
from src import instances as ins  # noqa: E402
from src import memory as mem  # noqa: E402
from src import pano_depth as pdp  # noqa: E402
from src import pipeline_v3 as p3v  # noqa: E402
from src import pipeline_v5 as p5  # noqa: E402
from src import room_box as rb  # noqa: E402
from src import shots_v5 as sv  # noqa: E402
from src.utils import set_seed  # noqa: E402

ROOT = vc.ROOT
OUT = ROOT / "outputs" / "v5_rooms"
DEVICE = torch.device("cuda")


def panos(df, k, gen) -> list[tuple]:
    """(room row, name, path) of every panorama: 'before' (milestone-1 file) + K candidates."""
    out = []
    for r in df.itertuples():
        d = OUT / r.id
        out.append((r, "before", ROOT / "outputs/v5_pano" / gen / f"{r.id}.png"))
        for i in range(k):
            out.append((r, f"cand_{r.seed + i}", d / f"cand_{r.seed + i}.png"))
    return out


def stage_cand(df, k, gen):
    todo = [(r, n, p) for r, n, p in panos(df, k, gen) if n != "before" and not p.exists()]
    if not todo:
        return
    mem.free()
    if gen == "sd21":
        from src import generate_pano as gp
        pipe = gp.load_alt_pipeline(DEVICE)
    else:
        from src import generate_pano_xl as gx
        pipe = gx.load_xl_pipeline(gen.split("_", 1)[1])
    for r, n, p in tqdm(todo, desc="candidates", unit="pano"):
        seed = int(n.split("_")[1])
        with mem.PeakMonitor() as pm:
            img = p5.generate(pipe, gen, r.prompt, seed, spacious=True)
        if gen == "sd21":
            p = p.with_name(p.stem + "_native.png")
        vc.save_rgb(p, img)
        vc.write_json(p.with_suffix(".json"), {"seconds": pm.seconds, "peak_vram_gb": pm.peak_vram_gb,
                                               "peak_ram_gb": pm.peak_ram_gb})
    pipe = None
    mem.free()


def stage_geo(df, k, gen):
    id2label = vc.ade_id2label()
    todo = [(r, n, p) for r, n, p in panos(df, k, gen) if p.exists() and not (OUT / r.id / f"{n}_geo.npz").exists()]
    if not todo:
        return
    dap = pdp.load_dap(DEVICE)[0]
    dists = {}
    for r, n, p in tqdm(todo, desc="DAP"):
        small = cv2.resize(vc.rgb(p), (1024, 512), interpolation=cv2.INTER_AREA)
        dists[(r.id, n)] = pdp.dap_depth_wrap(dap, small, DEVICE)
    dap = None
    mem.free()
    from src import analysis as an
    seg = an.load_segmenter(DEVICE)
    for r, n, p in tqdm(todo, desc="SegFormer"):
        small = cv2.resize(vc.rgb(p), (1024, 512), interpolation=cv2.INTER_AREA)
        cm = pdp.segment_pano(*seg, small, DEVICE)
        (OUT / r.id).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / r.id / f"{n}_geo.npz", dist=dists[(r.id, n)].astype(np.float32), cm=cm.astype(np.int16))
    seg = None
    mem.free()


def stage_inst(df, k, gen):
    id2label = vc.ade_id2label()
    todo = [(r, n, p) for r, n, p in panos(df, k, gen) if (OUT / r.id / f"{n}_geo.npz").exists()
            and not (OUT / r.id / f"{n}_inst.json").exists()]
    if not todo:
        return
    model, proc = ins.load_panoptic(DEVICE)
    for r, n, p in tqdm(todo, desc="Mask2Former"):
        g = np.load(OUT / r.id / f"{n}_geo.npz")
        small = cv2.resize(vc.rgb(p), (1024, 512), interpolation=cv2.INTER_AREA)
        ana = p3v.analyse((model, proc), small, g["cm"].astype(np.int32), id2label, DEVICE)
        np.savez_compressed(OUT / r.id / f"{n}_inst.npz", class_map=ana["class_map"].astype(np.int16),
                            inst_map=ana["inst_map"].astype(np.int32))
        (OUT / r.id / f"{n}_inst.json").write_text(json.dumps(ana["instances"], default=float))
    model = None
    mem.free()


def load_ana(rid, n):
    a = np.load(OUT / rid / f"{n}_inst.npz")
    return {"class_map": a["class_map"].astype(np.int32), "inst_map": a["inst_map"].astype(np.int32),
            "instances": json.loads((OUT / rid / f"{n}_inst.json").read_text())}


def v4_hero(box, ana, id2label, dist, room):
    """v4 rule (design_shots.design_shots without rendering): hero yaw / FOV + wall views."""
    fam = {int(i["id"]): i.get("family", i.get("label", "")) for i in ana["instances"]}
    sets = ds.class_sets(id2label)
    main = ds.main_piece(ana["inst_map"], ana["instances"], dist, box, room)
    scan = ds.scan_yaws(box, ana["class_map"], ana["inst_map"], fam, sets, dist=dist, main=main)
    h = ds.pick_heroes(scan, ds.selected_weights(), 1)[0]
    return [{"kind": "hero", "yaw": h["yaw"], "fov": ds.HERO_FOV}] + \
        [{"kind": "wall", "yaw": y, "fov": ds.WALL_FOV} for y in ds.wall_yaws(box).values()]


def stage_shots(df, k, gen):
    id2label = vc.ade_id2label()
    path = OUT / "shots.csv"
    have = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["id", "name"])
    rows = []
    for r, n, p in tqdm(panos(df, k, gen), desc="shots"):
        if not (OUT / r.id / f"{n}_inst.json").exists() or ((have["id"] == r.id) & (have["name"] == n)).any():
            continue
        g = np.load(OUT / r.id / f"{n}_geo.npz")
        dist, cm = g["dist"], g["cm"].astype(np.int32)
        ana = load_ana(r.id, n)
        room = ds.room_type_of(r.prompt)
        try:
            box = rb.fit_box(dist, ana["class_map"], id2label)
        except Exception as e:                       # noqa: BLE001
            print(r.id, n, "box failed", e)
            continue
        sc = bx.score(dist, cm, id2label, box)
        dn = cam.nearest_furniture(dist, cm, id2label, box)
        base = {"id": r.id, "set": r.set, "name": n, "boxiness": sc["score"], "area_m2": sc["area_m2"], "d_near": dn,
                "camera_h": box["camera_height"]}
        for s in v4_hero(box, ana, id2label, dist, room):
            m = sv.validity_of(s, ana["class_map"], ana["inst_map"], ana["instances"], id2label, dist, box, room)
            rows.append({**base, "rule": "v4", "kind": s["kind"], "yaw": s["yaw"], "fov": s["fov"],
                         **{c: m[c] for c in ("v_valid", "v_main_share", "v_median_depth", "v_near_share", "v_badness")}})
        sel = sv.select_shots(np.zeros((8, 16, 3), np.uint8), box, ana["class_map"], ana["inst_map"], ana["instances"],
                              id2label, dist, room, size=(16, 12))
        for s in sel["shots"]:
            m = s["measures"]
            rows.append({**base, "rule": "v5", "kind": s["kind"], "yaw": s["yaw"], "fov": s["fov"],
                         "n_valid_yaws": sel["n_valid_yaws"],
                         **{c: m[c] for c in ("v_valid", "v_main_share", "v_median_depth", "v_near_share", "v_badness")}})
    if rows:
        pd.concat([have, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)


GRID = [{"w_area": a, "w_near": b} for a, b in itertools.product([0.0, 0.25, 0.5, 1.0], [0.0, 0.25, 0.5, 1.0])]


def pick_table(sh: pd.DataFrame, k: int, w: dict) -> pd.DataFrame:
    """Chosen candidate per room under weights w (only the K 'cand_*' panoramas compete)."""
    hero = sh[(sh["rule"] == "v5") & (sh["kind"] == "hero") & sh["name"].str.startswith("cand_")]
    out = []
    for rid, g in hero.groupby("id"):
        g = g.sort_values("name").head(k)
        cands = [{"score": row.boxiness, "area_m2": row.area_m2, "d_near": row.d_near} for row in g.itertuples()]
        out.append(g.iloc[cam.pick_candidate(cands, w)])
    return pd.DataFrame(out)


def report(k):
    sh = pd.read_csv(OUT / "shots.csv")
    sh["v_valid"] = sh["v_valid"].astype(bool)
    rows = []
    # (a) prompt effect: before vs first candidate (same seed), v4 shots and v5 shots
    first = {rid: f"cand_{int(rid.rsplit('_s', 1)[1])}" for rid in sh["id"].unique()}
    sh["is_first"] = [n == first[i] for i, n in zip(sh["id"], sh["name"])]
    for split in ("dev", "test", "demo"):
        s = sh[sh["set"] == split]
        for label, sub in (("before (v4 prompt, seed s)", s[s["name"] == "before"]),
                           ("after (v5 prompt, seed s)", s[s["is_first"]])):
            hero = sub[sub["kind"] == "hero"]
            for rule in ("v4", "v5"):
                hr, al = hero[hero["rule"] == rule], sub[sub["rule"] == rule]
                rows.append({"set": split, "panorama": label, "shot_rule": rule, "rooms": hr["id"].nunique(),
                             "area_m2": hr["area_m2"].mean(), "d_near_m": hr["d_near"].replace(np.inf, np.nan).mean(),
                             "d_near_lt_0.5": float((hr["d_near"] < cam.NEAR_REJECT).mean()),
                             "hero_valid": hr["v_valid"].mean(), "shots_valid": al["v_valid"].mean(),
                             "hero_main_share": hr["v_main_share"].mean(), "hero_median_depth": hr["v_median_depth"].mean(),
                             "hero_near_share": hr["v_near_share"].mean()})
    prompt_tab = pd.DataFrame(rows)
    prompt_tab.to_csv(vc.TAB / "v5_camera_prompt.csv", index=False, float_format="%.4f")
    # (b) candidate-selection weights on dev
    grid = []
    for w in GRID:
        for split in ("dev", "test", "demo"):
            pk = pick_table(sh[sh["set"] == split], k, w)
            sel = sh[(sh["rule"] == "v5") & sh.set_index(["id", "name"]).index.isin(list(zip(pk["id"], pk["name"])))]
            grid.append({**w, "set": split, "hero_valid": pk["v_valid"].mean(), "shots_valid": sel["v_valid"].mean(),
                         "area_m2": pk["area_m2"].mean(), "d_near_m": pk["d_near"].replace(np.inf, np.nan).mean(),
                         "boxiness": pk["boxiness"].mean()})
    grid = pd.DataFrame(grid)
    grid.to_csv(vc.TAB / "v5_camera_grid.csv", index=False, float_format="%.4f")
    dev = grid[grid["set"] == "dev"].sort_values(["hero_valid", "shots_valid", "boxiness"], ascending=False)
    best = {"w_area": float(dev.iloc[0]["w_area"]), "w_near": float(dev.iloc[0]["w_near"])}
    vc.write_json(vc.TAB / "v5_camera_selection.json", {"weights": best, "k": k, "rule": "dev: max share of rooms with "
                  "a valid hero, then mean valid shots, then boxiness", "dev_row": dev.iloc[0].to_dict()})
    # (c) v4 vs v5 end to end: v4 = before panorama + v4 shots; v5 = selected candidate + v5 shots
    fin = []
    for split in ("dev", "test", "demo"):
        s = sh[sh["set"] == split]
        b4 = s[(s["name"] == "before") & (s["rule"] == "v4")]
        pk = pick_table(s, k, best)
        b5 = s[(s["rule"] == "v5") & s.set_index(["id", "name"]).index.isin(list(zip(pk["id"], pk["name"])))]
        for lab, b in (("v4", b4), ("v5", b5)):
            h = b[b["kind"] == "hero"]
            fin.append({"set": split, "version": lab, "hero_valid": h["v_valid"].mean(), "shots_valid": b["v_valid"].mean(),
                        "area_m2": h["area_m2"].mean(), "d_near_m": h["d_near"].replace(np.inf, np.nan).mean(),
                        "hero_median_depth": h["v_median_depth"].mean(), "hero_near_share": h["v_near_share"].mean(),
                        "hero_main_share": h["v_main_share"].mean()})
    fin = pd.DataFrame(fin)
    fin.to_csv(vc.TAB / "v5_camera_summary.csv", index=False, float_format="%.4f")
    sel_names = {}
    for split in ("dev", "test", "demo"):
        pk = pick_table(sh[sh["set"] == split], k, best)
        sel_names.update(dict(zip(pk["id"], pk["name"])))
    vc.write_json(OUT / "selected.json", sel_names)
    print(prompt_tab.round(3).to_string())
    print(grid[grid["set"] == "dev"].round(3).to_string())
    print(fin.round(3).to_string())
    print("weights", best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--sets", nargs="+", default=["dev", "test", "demo"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--gen", default=None)
    a = ap.parse_args()
    set_seed(42)
    gen = a.gen or p5.source()
    df = vc.rooms(a.sets)
    if a.limit:
        df = df.groupby("set").head(a.limit).reset_index(drop=True)
    for st, fn in (("cand", stage_cand), ("geo", stage_geo), ("inst", stage_inst), ("shots", stage_shots)):
        if a.stage in ("all", st):
            fn(df, a.k, gen)
    if a.stage in ("all", "report"):
        report(a.k)


if __name__ == "__main__":
    main()
