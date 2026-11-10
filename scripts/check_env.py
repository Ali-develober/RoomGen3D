"""Phase 0 environment check: GPU, fp16 matmul, SDPA attention, 8-bit Adam, and library imports."""

import importlib

import torch
import torch.nn.functional as F

print(f"torch            {torch.__version__}")
print(f"CUDA available   {torch.cuda.is_available()}")
print(f"CUDA (build)     {torch.version.cuda}")
props = torch.cuda.get_device_properties(0)
print(f"GPU              {props.name}  (sm_{props.major}{props.minor}, {props.total_memory / 1024**3:.1f} GB)")

# fp16 matmul on the GPU
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = a @ a
torch.cuda.synchronize()
print(f"fp16 matmul      ok  shape={tuple(b.shape)} device={b.device}")

# PyTorch SDPA attention (used instead of xformers)
q = torch.randn(1, 8, 256, 64, device="cuda", dtype=torch.float16)
out = F.scaled_dot_product_attention(q, q, q)
torch.cuda.synchronize()
print(f"SDPA attention   ok  shape={tuple(out.shape)}")

# bitsandbytes 8-bit Adam step
import bitsandbytes as bnb

w = torch.nn.Parameter(torch.randn(4096, 64, device="cuda"))
opt = bnb.optim.AdamW8bit([w], lr=1e-4)
w.sum().backward()
opt.step()
print(f"8-bit AdamW      ok  bitsandbytes {bnb.__version__}")

print("\nLibraries:")
for name in ["diffusers", "transformers", "peft", "accelerate", "safetensors", "cv2",
             "skimage", "open3d", "timm", "torchmetrics", "torch_fidelity", "gradio", "matplotlib"]:
    try:
        mod = importlib.import_module(name)
        print(f"  {name:15s} {getattr(mod, '__version__', '?')}")
    except Exception as e:  # report and continue
        print(f"  {name:15s} FAILED: {e}")

try:
    importlib.import_module("xformers")
    print("\nWARNING: xformers is installed (should not be)")
except ImportError:
    print("\nxformers         not installed (as required)")

print(f"VRAM peak used   {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
