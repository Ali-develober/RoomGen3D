"""Download a balanced ~8K-image MMIS subset (images only, no audio) from Google Drive.

Steps:
  1. Load train/val labels from data/mmis/meta/*.json (downloaded once, ~190 MB).
  2. Pick an equal number of images per (style, room) cell -> data/mmis/subset.csv.
  3. For each style zip on Drive, read its index remotely and fetch only the chosen images.

Resumable: images already on disk are skipped.
Usage:
  python scripts/download_mmis_subset.py --limit 200   # quick test
  python scripts/download_mmis_subset.py               # full subset
"""

import argparse
import json
import sys
from pathlib import Path

import gdown
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_prep import (MMIS_DRIVE_FOLDER, balanced_subset, drive_session, extract_member,  # noqa: E402
                           fetch_range, load_mmis_labels, plan_chunks, remote_zip_index)
from src.utils import set_seed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MMIS = ROOT / "data" / "mmis"


def drive_listing() -> list[dict]:
    """List the Drive folder once (no download) and cache file ids + paths."""
    cache = MMIS / "meta" / "drive_files.json"
    if cache.exists():
        return json.loads(cache.read_text())
    files = gdown.download_folder(url=MMIS_DRIVE_FOLDER, skip_download=True, quiet=True)
    out = [{"id": f.id, "path": f.path.replace("\\", "/")} for f in files]
    cache.write_text(json.dumps(out, indent=1))
    return out


def ensure_meta(files: list[dict]) -> None:
    """Download the train/val annotation JSONs if missing."""
    for f in files:
        name = Path(f["path"]).name
        if name in {"train.json", "val.json"} and not (MMIS / "meta" / name).exists():
            gdown.download(id=f["id"], output=str(MMIS / "meta" / name), quiet=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=8000, help="target subset size")
    ap.add_argument("--limit", type=int, default=0, help="only download the first N images (testing)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    (MMIS / "meta").mkdir(parents=True, exist_ok=True)

    files = drive_listing()
    ensure_meta(files)

    subset_csv = MMIS / "subset.csv"
    sub = balanced_subset(load_mmis_labels(MMIS / "meta"), total=args.total, seed=args.seed)
    sub["path"] = "images/" + sub["member"]
    sub.drop(columns=["text"]).to_csv(subset_csv, index=False)
    print(f"subset: {len(sub)} images, {sub.groupby(['style', 'room']).ngroups} cells -> {subset_csv}")

    todo = sub if not args.limit else sub.sample(n=args.limit, random_state=args.seed)
    todo = todo[[not (MMIS / p).exists() for p in todo["path"]]]
    wanted = set(todo["member"])
    print(f"to download: {len(wanted)} images")
    if not wanted:
        return

    zips = sorted([f for f in files if f["path"].endswith(".zip") and "vgg" not in f["path"]],
                  key=lambda f: f["path"])
    bar = tqdm(total=len(wanted), unit="img")
    for zf in zips:
        if not wanted:
            break
        style = zf["path"].split("/")[0]
        if not any(m.startswith(style + "/") for m in wanted):
            continue
        session, params, size = drive_session(zf["id"])
        infos = [i for i in remote_zip_index(session, params, size) if i.filename in wanted]
        for chunk in plan_chunks(infos):
            end = min(chunk["end"], size) - 1
            buf = fetch_range(session, params, chunk["start"], end)
            for info in chunk["members"]:
                out = MMIS / "images" / info.filename
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(extract_member(buf, chunk["start"], info))
                wanted.discard(info.filename)
                bar.update(1)
    bar.close()
    if wanted:
        print(f"WARNING: {len(wanted)} images not found in any zip, e.g. {sorted(wanted)[:3]}")


if __name__ == "__main__":
    main()
