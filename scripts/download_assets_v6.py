"""Download the v6 CC0 furniture library (PLAN_V6 milestone 1).

Sources (both CC0, licences checked on 2026-11-15, listed in assets/LICENSES.md):
  * Kenney Furniture Kit 2.0 (https://kenney.nl/assets/furniture-kit), 140 low-poly models, CC0.
    The site hands out a fresh signed link per request (https://kenney.nl/download-link/<id>).
  * Poly Haven models (https://polyhaven.com, all assets CC0) through the public API
    (https://api.polyhaven.com/files/<id>): glTF with 1k textures.

Files land in data/assets_raw/ (git-ignored, ~0.3 GB). Run:  python scripts/download_assets_v6.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "assets_raw"
UA = {"User-Agent": "Mozilla/5.0 (asset-fetch)"}

KENNEY_PAGE = "https://kenney.nl/assets/furniture-kit"
KENNEY_ID = "6173736574732f6675726e69747572652d6b6974"

# Poly Haven models used by the library (src/assets_v6.CATALOG)
POLYHAVEN = [
    "dining_chair_02", "painted_wooden_chair_01", "gallinera_chair", "GreenChair_01",
    "ArmChair_01", "mid_century_lounge_chair",
    "Sofa_01", "sofa_02", "sofa_03",
    "dining_table", "painted_wooden_table", "wooden_table_02", "round_wooden_table_02",
    "CoffeeTable_01", "modern_coffee_table_01", "modern_coffee_table_02", "coffee_table_round_01",
    "ClassicNightstand_01", "painted_wooden_nightstand", "side_table_01",
    "ClassicConsole_01", "modern_wooden_cabinet", "GothicCommode_01", "painted_wooden_cabinet",
    "painted_wooden_cabinet_02", "vintage_cabinet_01", "chinese_cabinet",
    "Shelf_01", "wooden_bookshelf_worn", "wooden_display_shelves_01",
    "metal_office_desk", "WoodenTable_03", "GothicBed_01",
]


def _get(url: str, headers=None) -> bytes:
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    for k in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001 (retry any network error)
            if k == 3:
                raise
            print("retry", url, e)
            time.sleep(2 + 3 * k)


def kenney() -> None:
    out = RAW / "kenney"
    if (out / "kit" / "Models" / "GLTF format").exists():
        print("kenney: present")
        return
    out.mkdir(parents=True, exist_ok=True)
    import http.cookiejar
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    op.addheaders = list(UA.items()) + [("Referer", KENNEY_PAGE)]
    op.open(KENNEY_PAGE).read()
    link = json.loads(op.open(f"https://kenney.nl/download-link/{KENNEY_ID}").read())["url"]
    data = op.open(link).read()
    (out / "furniture-kit.zip").write_bytes(data)
    with zipfile.ZipFile(out / "furniture-kit.zip") as z:
        z.extractall(out / "kit")
    print(f"kenney: {len(data) / 1e6:.1f} MB")


def polyhaven(res: str = "1k") -> None:
    out = RAW / "polyhaven"
    total = 0
    for aid in POLYHAVEN:
        d = out / aid
        if (d / "done").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        info = json.loads(_get(f"https://api.polyhaven.com/info/{aid}"))
        files = json.loads(_get(f"https://api.polyhaven.com/files/{aid}"))
        g = files["gltf"][res]["gltf"]
        (d / f"{aid}.gltf").write_bytes(_get(g["url"]))
        size = g["size"]
        for rel, f in g["include"].items():
            if "_nor_" in rel or "_arm_" in rel or "_rough_" in rel:   # only the diffuse texture is used
                continue
            p = d / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(_get(f["url"]))
            size += f["size"]
        (d / "info.json").write_text(json.dumps({"id": aid, "name": info.get("name"), "authors": info.get("authors"),
                                                  "dimensions_mm": info.get("dimensions"), "licence": "CC0",
                                                  "url": f"https://polyhaven.com/a/{aid}"}, indent=1))
        (d / "done").write_text("ok")
        total += size
        print(f"polyhaven {aid}: {size / 1e6:.1f} MB")
    print(f"polyhaven total this run: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    kenney()
    polyhaven(sys.argv[1] if len(sys.argv) > 1 else "1k")
