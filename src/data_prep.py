"""Dataset preparation: MMIS labels/captions, balanced subset selection, remote zip
extraction from Google Drive, and NYU Depth V2 test-split extraction.

MMIS (Kassab et al., 2024) is shared on Google Drive as 2 zips per style (11 styles).
About 98% of each zip is speech audio (.mp3); the 256x256 images are ~13 KB each.
To avoid downloading ~34 GB, we read each zip's central directory with HTTP range
requests and fetch only the bytes of the images we selected.
"""

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

MMIS_DRIVE_FOLDER = "https://drive.google.com/drive/folders/1FO_sNVZi757I_QBdwibPX--14O24ff0m"
DRIVE_DOWNLOAD = "https://drive.usercontent.google.com/download"
ROOM_TYPES = ["Bathroom", "Bedroom", "Dining Room", "Kitchen", "Living Room"]


# --------------------------------------------------------------------------- #
# MMIS labels and captions
# --------------------------------------------------------------------------- #
def _split_path(p: str) -> tuple[str, str, str]:
    """'Art Deco\\Living Room\\2154.jpg' -> ('Art Deco', 'Living Room', '2154.jpg')."""
    parts = p.replace("\\\\", "/").replace("\\", "/").split("/")
    return parts[0], parts[1], parts[2]


def clean_caption(text: str) -> str:
    """Tidy a dataset caption: fix missing spaces after periods and drop a trailing
    cut-off sentence (some MMIS captions end mid-sentence, e.g. '...In addition to')."""
    text = re.sub(r"\.(?=[A-Z])", ". ", text.strip())
    # drop filler openings such as "The image depicts" / "In the image, there is"
    text = re.sub(r"^(the image (depicts|shows|features|captures)|in (the|this) image,?( there (is|are))?)\s+",
                  "", text, flags=re.IGNORECASE)
    if not text.endswith("."):
        cut = text.rfind(".")
        if cut > 0:
            text = text[: cut + 1]
    return text


def build_caption(style: str, room: str) -> str:
    """Main training caption, built from the labels only, e.g. 'an art deco style living room'.

    The style/room labels come from the folder structure and are reliable. MMIS text
    captions are machine-generated and sometimes name the wrong room, so they are
    kept separately as optional extra text (see balanced_subset).
    """
    article = "an" if style[0].lower() in "aeiou" else "a"
    return f"{article} {style.lower()} style {room.lower()}"


def load_mmis_labels(meta_dir: str | Path) -> pd.DataFrame:
    """Load MMIS train/val annotations into one table of unique images.

    Columns: style, room, file, member (path inside the zip), split, text.
    test.json is not used: it is malformed JSON in the released files.
    Style and room come from the image path (the label fields have inconsistent case).
    """
    rows = []
    for split in ["train", "val"]:
        with open(Path(meta_dir) / f"{split}.json", encoding="utf-8") as f:
            for r in json.load(f):
                style, room, file = _split_path(r["image_path"])
                rows.append({"style": style, "room": room, "file": file,
                             "member": f"{style}/{room}/{file}", "split": split, "text": r["text"]})
    return pd.DataFrame(rows).drop_duplicates("member").reset_index(drop=True)


def balanced_subset(labels: pd.DataFrame, total: int = 8000, seed: int = 42) -> pd.DataFrame:
    """Sample the same number of images from every (style, room) cell.

    per_cell = ceil(total / n_cells); a cell with fewer images contributes all it has.
    """
    groups = labels.groupby(["style", "room"])
    per_cell = int(np.ceil(total / groups.ngroups))
    parts = [g.sample(n=min(per_cell, len(g)), random_state=seed) for _, g in groups]
    sub = pd.concat(parts).sort_values(["style", "room", "file"]).reset_index(drop=True)
    sub["caption"] = [build_caption(s, r) for s, r in zip(sub["style"], sub["room"])]
    sub["extra_text"] = sub["text"].map(clean_caption)  # optional, machine-generated MMIS caption
    return sub


# --------------------------------------------------------------------------- #
# Reading single files out of a remote zip on Google Drive
# --------------------------------------------------------------------------- #
def drive_session(file_id: str) -> tuple[requests.Session, dict, int]:
    """Open a Drive download session for a large file. Returns (session, params, size).

    Large files show a 'can't scan for viruses' page; its hidden form fields
    are the parameters for the real download URL.
    """
    s = requests.Session()
    r = s.get(DRIVE_DOWNLOAD, params={"id": file_id, "export": "download"}, timeout=60)
    params = dict(re.findall(r'name="(\w+)" value="([^"]*)"', r.text)) or {"id": file_id, "export": "download"}
    head = s.get(DRIVE_DOWNLOAD, params=params, headers={"Range": "bytes=0-0"}, timeout=60)
    head.raise_for_status()
    size = int(head.headers["Content-Range"].split("/")[1])
    return s, params, size


def fetch_range(session: requests.Session, params: dict, start: int, end: int, retries: int = 5) -> bytes:
    """Download bytes [start, end] (inclusive) of a Drive file, retrying on errors."""
    for attempt in range(retries):
        try:
            r = session.get(DRIVE_DOWNLOAD, params=params, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
            if r.status_code == 206 and len(r.content) == end - start + 1:
                return r.content
        except requests.RequestException:
            pass
    raise IOError(f"range {start}-{end} failed after {retries} tries")


class RemoteFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (enough for zipfile to read the index)."""

    def __init__(self, session, params, size):
        self.session, self.params, self.size, self.pos = session, params, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, b):
        end = min(self.pos + len(b), self.size) - 1
        if end < self.pos:
            return 0
        data = fetch_range(self.session, self.params, self.pos, end)
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def remote_zip_index(session, params, size) -> list[zipfile.ZipInfo]:
    """Read only the central directory of a remote zip (a few range requests)."""
    with zipfile.ZipFile(io.BufferedReader(RemoteFile(session, params, size), buffer_size=1 << 20)) as z:
        return z.infolist()


def plan_chunks(infos: list[zipfile.ZipInfo], max_gap: int = 1 << 20, max_chunk: int = 32 << 20):
    """Group wanted members into contiguous byte ranges, so nearby files share one request.

    Each member is read from its local header to the end of its compressed data
    (+ 1 KB slack for the local header's variable-length name/extra fields).
    """
    infos = sorted(infos, key=lambda i: i.header_offset)
    chunks, cur = [], None
    for i in infos:
        start, end = i.header_offset, i.header_offset + 30 + len(i.filename.encode()) + 1024 + i.compress_size
        if cur and start - cur["end"] <= max_gap and end - cur["start"] <= max_chunk:
            cur["end"] = max(cur["end"], end)
            cur["members"].append(i)
        else:
            cur = {"start": start, "end": end, "members": [i]}
            chunks.append(cur)
    return chunks


def extract_member(buf: bytes, buf_start: int, info: zipfile.ZipInfo) -> bytes:
    """Decode one zip member from a downloaded byte buffer that contains it."""
    off = info.header_offset - buf_start
    name_len, extra_len = np.frombuffer(buf[off + 26 : off + 30], dtype="<u2")
    data_start = off + 30 + int(name_len) + int(extra_len)
    raw = buf[data_start : data_start + info.compress_size]
    if info.compress_type == zipfile.ZIP_STORED:
        return raw
    if info.compress_type == zipfile.ZIP_DEFLATED:
        import zlib
        return zlib.decompress(raw, -15)
    raise ValueError(f"unsupported compression {info.compress_type} for {info.filename}")


# --------------------------------------------------------------------------- #
# NYU Depth V2 (labeled split, 1,449 RGB-D pairs)
# --------------------------------------------------------------------------- #
def nyu_test_indices(splits_mat: str | Path) -> np.ndarray:
    """0-based indices of the 654 standard test images (from the official splits.mat)."""
    from scipy.io import loadmat
    return loadmat(splits_mat)["testNdxs"].ravel().astype(int) - 1


def nyu_read_pair(h5, idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (rgb uint8 HxWx3, depth float32 HxW in metres) for one labeled-split index.

    The v7.3 .mat (HDF5) stores arrays transposed: images are (N, 3, W, H), depths (N, W, H).
    """
    rgb = np.asarray(h5["images"][idx]).transpose(2, 1, 0)
    depth = np.asarray(h5["depths"][idx]).T.astype(np.float32)
    return np.ascontiguousarray(rgb), np.ascontiguousarray(depth)
