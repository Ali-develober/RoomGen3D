# CLAUDE.md — RoomGen3D (ARTI 403 Image Processing project)

**Read `README.md` first. It is the source of truth for the idea, pipeline, techniques, datasets, evaluation, repo structure, and work plan. Follow it.**

## Environment
- Windows, VS Code, Python 3.12, NVIDIA RTX 5060 (Blackwell, 8 GB VRAM).
- Use the project virtual env `.venv` only. Never install into global Python.
- PyTorch must be a CUDA 12.8 build (`--index-url https://download.pytorch.org/whl/cu128`). Never install a CPU build or downgrade torch.
- Do NOT install xformers. Use PyTorch SDPA attention.
- Keep large data, model weights, and outputs out of git (`data/`, `checkpoints/`, `outputs/` in `.gitignore`).

## GPU memory rules (8 GB)
- fp16 / mixed precision everywhere.
- LoRA training: batch size 1 + gradient accumulation, gradient checkpointing, 8-bit Adam (bitsandbytes), 512×512.
- Inference: `enable_attention_slicing()` / `enable_vae_slicing()` if OOM; free models (`del` + `torch.cuda.empty_cache()`) before loading the next one.
- Never load SD and Depth Anything and the segmenter at the same time unless verified to fit.

## Code rules
- Core logic lives in `src/*.py` as small, documented functions. The notebook `notebooks/RoomGen3D.ipynb` only imports from `src/` and shows results step by step (course submission: clean, commented, logical order).
- Every image-processing step must be visible and explainable: save before/after figures to `results/figures/`.
- Every metric goes into `results/tables/` as CSV.
- Fixed random seeds for reproducibility.
- Long jobs (training, full evaluation) run as scripts with progress bars and resumable checkpoints, not inside the notebook.
- Test each module on a few images before running at full scale.

## Workflow
- Work phase by phase (README section 11). Finish, test, and summarize one phase before starting the next.
- At the end of each phase, report: what was built, files changed, results/metrics, and anything that failed.
- Ask before downloading anything larger than ~5 GB or deleting files.
