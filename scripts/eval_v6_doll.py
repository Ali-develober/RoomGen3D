"""PLAN_V6 milestone 4 metrics on the six demo rooms (outputs/v6_demo):
    furniture count        pieces in the layout vs coloured furniture meshes in the dollhouse (must match)
    dollhouse textures     the same measures as PLAN_V5 milestone 5, on the v6 wall + floor textures: residual furniture
                           share (SegFormer on the texture: furniture / object classes), BRISQUE (768 px), flat-area
                           noise sigma (Immerkaer) -> compared with the v5 values of results/tables/v5_doll_summary.csv
    seen share             share of each plane seen from the panorama (the rest is synthesised)
    plan / shot agreement  every layout piece that covers >= 1 % of a design shot (exact, proxy instance ids) and is
                           drawn in that shot's plan cone: is it in the generated shot? (SegFormer classes of the piece
                           cover >= 50 % of its proxy mask)
    window presence        panorama windows before / after the repair (summary_v6.json)
Outputs: results/tables/v6_doll_metrics.csv, v6_plan_shot_agreement.csv, v6_doll_summary.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import v5_common as vc  # noqa: E402
from eval_v6_pano import CLASS_ADE, classes_for  # noqa: E402,F401
from src import analysis as an  # noqa: E402
from src import cameras_v6 as CV  # noqa: E402
from src import dollhouse_v6 as DH  # noqa: E402
from src import memory as mem  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import pipeline_v6 as P6  # noqa: E402
from src import room_v4 as r4  # noqa: E402
from src import scene_v6 as S  # noqa: E402

DEVICE = "cuda"


def main():
    from src import dollhouse_v5 as dv
    from src.floorplan import FURNITURE, class_ids
    id2label = vc.ade_id2label()
    obj_ids = class_ids(id2label, set(FURNITURE) | dv.OBJECT_CLASSES)
    seg_model = an.load_segmenter(DEVICE)
    rows, agree = [], []
    for prompt, seed in pl.DEMO_PROMPTS:
        d = ROOT / "outputs" / "v6_demo" / pl.slug(prompt, seed)
        sc, rd = P6.build_scene(prompt, seed)
        lay = sc["layout"]
        pano = vc.rgb(d / "pano_2048.jpg")
        seg = an.segment_classes(*seg_model, pano, DEVICE)
        tp = DH.textured_parts(rd, sc, pano, seed, seg)
        n_layout = len(lay["items"])
        n_doll = len({i for _, i in tp["furniture"]})
        meta = json.loads((d / "summary_v6.json").read_text())
        for k in ("floor", "S", "N", "W", "E"):
            t = tp["tex"][k]
            s768 = cv2.resize(t, (768, int(768 * t.shape[0] / t.shape[1])) if t.shape[1] >= t.shape[0]
                              else (int(768 * t.shape[1] / t.shape[0]), 768), interpolation=cv2.INTER_AREA)
            cm = an.segment_classes(*seg_model, s768, DEVICE)
            rows.append({"room": d.name, "plane": k, "seen_share": tp["tex"]["seen"][k],
                         "furniture_share": float(np.isin(cm, obj_ids).mean()), "brisque": vc.brisque([s768]),
                         "noise": r4.noise_sigma(t, np.ones(t.shape[:2], bool)), "pieces_layout": n_layout,
                         "pieces_dollhouse": n_doll, "window_presence": meta.get("window_presence"),
                         "window_presence_after_fix": meta.get("window_presence_after_fix")})
        cams = meta["cameras"]
        for n, cam in cams.items():
            p = d / "shots" / f"{n}.jpg"
            if not p.exists():
                continue
            shot = cv2.resize(vc.rgb(p), (768, 576), interpolation=cv2.INTER_AREA)
            r = rd.perspective(cam["pos"], cam["yaw"], 0.0, cam["hfov"], 768, 576, cam.get("shift", 0.0))
            cm = an.segment_classes(*seg_model, shot, DEVICE)
            for it in lay["items"]:
                m = r["inst"] == S.FURN0 + it["id"]
                if m.mean() < 0.01:
                    continue
                cover = float(np.isin(cm[m], classes_for(it["cls"])).mean())
                agree.append({"room": d.name, "shot": n, "cls": it["cls"], "share": float(m.mean()), "cover": cover,
                              "present": cover >= 0.5})
        print(d.name, n_layout, n_doll, flush=True)
    mem.free(seg_model)
    df = pd.DataFrame(rows)
    df.to_csv(vc.TAB / "v6_doll_metrics.csv", index=False)
    ag = pd.DataFrame(agree)
    ag.to_csv(vc.TAB / "v6_plan_shot_agreement.csv", index=False)
    v5 = pd.read_csv(vc.TAB / "v5_doll_summary.csv") if (vc.TAB / "v5_doll_summary.csv").exists() else None
    summ = {"furniture_share_v6": df.furniture_share.mean(), "noise_v6": df.noise.mean(), "brisque_v6": df.brisque.mean(),
            "seen_share_v6": df.seen_share.mean(),
            "pieces_match_rooms": int((df.groupby("room").apply(lambda g: (g.pieces_layout == g.pieces_dollhouse).all())).sum()),
            "rooms": int(df.room.nunique()), "plan_shot_agreement": float(ag.present.mean()) if len(ag) else np.nan,
            "plan_shot_pairs": int(len(ag))}
    if v5 is not None:
        sel = v5[v5.set == "demo"] if "set" in v5 else v5
        summ["v5_table"] = sel.round(4).to_dict("records")
    vc.write_json(vc.TAB / "v6_doll_summary.json", summ)
    print(json.dumps({k: v for k, v in summ.items() if k != "v5_table"}, indent=1, default=float))
    print(df.groupby("room")[["seen_share", "furniture_share", "brisque", "noise", "pieces_layout", "pieces_dollhouse"]].mean().round(3))
    print(ag.groupby("cls").present.mean().round(2))


if __name__ == "__main__":
    main()
