"""Download NYU Depth V2 labeled split (nyu_depth_v2_labeled.mat, 2.97 GB) + official splits.mat.

The NYU server gives ~150 KB/s per connection, so the file is fetched as 16 MB
byte-range parts over parallel connections, then merged. Resumable: finished
parts are kept in data/nyu/raw/parts/ and skipped on restart.

Usage: python scripts/download_nyu.py [--workers 16]
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "nyu" / "raw"
MAT_URL = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
SPLITS_URL = "http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat"
PART = 16 << 20


def remote_size(url: str) -> int:
    r = requests.head(url, allow_redirects=True, timeout=60)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def fetch_part(url: str, start: int, end: int, out: Path, retries: int = 8) -> int:
    """Download bytes [start, end] into `out` (written via a .tmp file so partial parts never count)."""
    for _ in range(retries):
        try:
            r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
            if r.status_code == 206 and len(r.content) == end - start + 1:
                tmp = out.with_suffix(".tmp")
                tmp.write_bytes(r.content)
                tmp.replace(out)
                return len(r.content)
        except requests.RequestException:
            pass
    raise IOError(f"part {out.name} failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)

    if not (RAW / "splits.mat").exists():
        (RAW / "splits.mat").write_bytes(requests.get(SPLITS_URL, timeout=60).content)

    final = RAW / "nyu_depth_v2_labeled.mat"
    size = remote_size(MAT_URL)
    if final.exists() and final.stat().st_size == size:
        print("already downloaded:", final)
        return
    parts_dir = RAW / "parts"
    parts_dir.mkdir(exist_ok=True)
    jobs = [(i, s, min(s + PART, size) - 1) for i, s in enumerate(range(0, size, PART))]
    todo = [j for j in jobs if not (parts_dir / f"{j[0]:05d}.part").exists()]
    bar = tqdm(total=size, initial=size - sum(e - s + 1 for _, s, e in todo), unit="B", unit_scale=True)
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch_part, MAT_URL, s, e, parts_dir / f"{i:05d}.part") for i, s, e in todo]
        for f in as_completed(futs):
            bar.update(f.result())
    bar.close()

    tmp = final.with_suffix(".mat.tmp")
    with open(tmp, "wb") as out:
        for i, _, _ in jobs:
            out.write((parts_dir / f"{i:05d}.part").read_bytes())
    assert tmp.stat().st_size == size, "merged size mismatch"
    tmp.replace(final)
    for p in parts_dir.glob("*.part"):
        p.unlink()
    parts_dir.rmdir()
    print("done:", final, size, "bytes")


if __name__ == "__main__":
    main()
