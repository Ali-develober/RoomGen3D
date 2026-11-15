"""RAM / VRAM housekeeping for PLAN_V5 (16 GB RAM, 8 GB VRAM).

    free(*objs)          drop references, gc.collect(), torch.cuda.empty_cache()
    ram_free_gb()        available system RAM (psutil)
    wait_for_ram(gb)     block (with retries) until at least `gb` GB of RAM is available, instead of crashing
    PeakMonitor          context manager that samples this process's RSS (and the CUDA peak) in a thread:
                             with PeakMonitor() as pm: ...;  pm.peak_ram_gb, pm.peak_vram_gb, pm.seconds
"""

from __future__ import annotations

import gc
import threading
import time

import psutil
import torch


def free(*objs) -> None:
    """Delete references held here, collect garbage and return cached CUDA memory.
    The caller must also drop its own names (e.g. `pipe = None`)."""
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ram_free_gb() -> float:
    return psutil.virtual_memory().available / 2 ** 30


PROCESS_BASE_GB = 1.5      # RSS of a process with torch + the CUDA context loaded and no model


def ram_available_gb(reuse: bool = False) -> float:
    """Free system RAM; with reuse=True plus this process's RSS above PROCESS_BASE_GB: on Windows a model freed in
    this process stays in the process heap (RSS does not drop) but is reused by the next load (measured: a second
    SDXL load in the same process adds no RSS), so it counts as available for the next model of this process."""
    f = ram_free_gb()
    if reuse:
        f += max(0.0, psutil.Process().memory_info().rss / 2 ** 30 - PROCESS_BASE_GB)
    return f


def wait_for_ram(gb: float, timeout: float = 600.0, poll: float = 10.0, label: str = "", reuse: bool = False) -> float:
    """Wait until `gb` GB of RAM is available (gc + empty_cache on every poll). Returns the free RAM.
    Raises MemoryError after `timeout` seconds so that a resumable job stops cleanly instead of swapping.
    reuse=True counts memory this process already holds and can reuse (see ram_available_gb)."""
    t0 = time.perf_counter()
    while True:
        free()
        f = ram_available_gb(reuse)
        if f >= gb:
            return f
        if time.perf_counter() - t0 > timeout:
            raise MemoryError(f"{label}: only {f:.1f} GB RAM free after {timeout:.0f} s (need {gb:.1f} GB)")
        print(f"[memory] {label}: {f:.1f} GB RAM free, need {gb:.1f} GB - waiting {poll:.0f} s", flush=True)
        time.sleep(poll)


class PeakMonitor:
    """Peak resident RAM of this process (sampled every `interval` s) and peak CUDA memory of a block."""

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.proc = psutil.Process()
        self.peak_ram_gb = 0.0
        self.peak_vram_gb = 0.0
        self.seconds = 0.0
        self._stop = threading.Event()

    def _run(self):
        while not self._stop.is_set():
            self.peak_ram_gb = max(self.peak_ram_gb, self.proc.memory_info().rss / 2 ** 30)
            time.sleep(self.interval)

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.peak_ram_gb = self.proc.memory_info().rss / 2 ** 30
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.peak_vram_gb = torch.cuda.max_memory_allocated() / 2 ** 30
        self.seconds = time.perf_counter() - self._t0
        self._stop.set()
        self._t.join()
        self.peak_ram_gb = max(self.peak_ram_gb, self.proc.memory_info().rss / 2 ** 30)
        return False
