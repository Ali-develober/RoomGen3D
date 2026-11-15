"""Local snapshot paths of Hugging Face models (PLAN_V6 issue 44).

from_pretrained("org/name") makes Hub requests (or, in offline mode, logs that it could not reach the Hub) even when
the model is cached. local("org/name") returns the cached snapshot folder instead, so loading never touches the
network; when the model is not cached it returns the repo id unchanged (the caller then gets a clear error in offline
mode, see app.check_models).
"""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=None)
def local(repo_id: str) -> str:
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo_id, local_files_only=True)
    except Exception:  # noqa: BLE001 (not cached, or a partial snapshot such as fp16-only SD 1.5)
        pass
    try:                                                   # partial snapshot: the folder refs/main points to
        from pathlib import Path
        from huggingface_hub.constants import HF_HUB_CACHE
        d = Path(HF_HUB_CACHE) / ("models--" + repo_id.replace("/", "--"))
        snap = d / "snapshots" / (d / "refs" / "main").read_text().strip()
        if (snap / "model_index.json").exists() or (snap / "config.json").exists():
            return str(snap)
    except Exception:  # noqa: BLE001
        pass
    return repo_id
