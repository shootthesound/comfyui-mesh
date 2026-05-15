"""Standalone smoke test for the server side — validates model loading
and the back-half forward pass without needing a real client.

Synthesizes activations of the right shape, runs them through
double_blocks[start_block:], reports time + output shape. Catches
load-time bugs (wrong FluxParams, missing weight keys, etc) without
needing the network layer.

Run on the 4090 (or on the 5090 for a loopback smoke test):
    python smoke_test_server.py --weights ./flux-2-klein-9b-fp8.safetensors
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Same COMFYUI_PATH discovery as the real server
import os
_comfy_candidates = [
    os.environ.get("COMFYUI_PATH"),
    "S:/Auto/ComfyUI_SEC/ComfyUI",  # the 5090 install (for loopback testing)
    str(HERE / "ComfyUI"),
    str(HERE.parent / "ComfyUI"),
    "C:/ComfyUI",
    "/opt/ComfyUI",
]
for cand in _comfy_candidates:
    if cand and Path(cand).is_dir() and (Path(cand) / "comfy").is_dir():
        if cand not in sys.path:
            sys.path.insert(0, cand)
        print(f"[smoke] using ComfyUI at: {cand}")
        break


from mesh_server import load_flux2_klein, forward_back_half
import vec_io


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--n-blocks", type=int, default=4,
                   help="How many of the LAST double_blocks to load (slim load).")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--T", type=int, default=4096, help="Number of image tokens")
    p.add_argument("--T-text", type=int, default=256, help="Number of text tokens")
    args = p.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    print(f"[smoke] loading model (slim, n_blocks={args.n_blocks})")
    t0 = time.time()
    patcher = load_flux2_klein(args.weights, device, dtype, n_blocks=args.n_blocks)
    model = patcher.model.diffusion_model
    print(f"[smoke] model load: {time.time()-t0:.2f}s")
    H = model.params.hidden_size

    n_double = len(model.double_blocks)
    n_single = len(model.single_blocks)
    print(f"[smoke] hidden={H}  double_blocks={n_double}  single_blocks={n_single}")

    # Synthesise activations roughly matching what FLUX produces mid-stack
    print(f"[smoke] synthesising activations T={args.T} T_text={args.T_text} H={H}")
    img = torch.randn(1, args.T, H, device=device, dtype=dtype) * 0.3
    txt = torch.randn(1, args.T_text, H, device=device, dtype=dtype) * 0.3

    # vec shape depends on global_modulation; build correct fake structure
    if model.params.global_modulation:
        vec = vec_io.fake_modulation_vec(B=1, H=H, device=device, dtype=dtype)
        print(f"[smoke] using fake modulation_tuple vec (global_modulation=True)")
    else:
        vec = torch.randn(1, H, device=device, dtype=dtype) * 0.3
        print(f"[smoke] using single-tensor vec (global_modulation=False)")

    # vec_orig is the un-modulated tensor that single_stream_modulation
    # consumes server-side when single_blocks are loaded. For the smoke
    # test we just synthesise something of the right shape.
    vec_orig = torch.randn(1, H, device=device, dtype=dtype) * 0.3

    # pe is produced by model.pe_embedder(ids) where ids has shape
    # [B, T_total, len(axes_dim)]. Generate synthetic ids and run them
    # through the actual embedder so the pe shape is exactly right
    # (the rope kernels are picky about freqs shape).
    n_axes = len(model.params.axes_dim)
    T_total = args.T + args.T_text
    ids = torch.zeros(1, T_total, n_axes, device=device, dtype=torch.float32)
    # Vary one axis along the token dim so pe isn't degenerate
    ids[0, :, 0] = torch.arange(T_total, device=device, dtype=torch.float32)
    with torch.no_grad():
        pe = model.pe_embedder(ids)
    print(f"[smoke] pe shape={tuple(pe.shape)} dtype={pe.dtype}")

    # Warm-up
    n_total_loaded = n_double + n_single
    print(f"[smoke] warmup forward (running all {n_double}D + {n_single}S = {n_total_loaded} loaded blocks)")
    with torch.no_grad():
        img_w, txt_w = forward_back_half(
            model, img=img, txt=txt, vec=vec, vec_orig=vec_orig, pe=pe,
            attn_mask=None,
        )
    torch.cuda.synchronize()

    # Timed runs
    n_iters = 4
    print(f"[smoke] timed forward x{n_iters}")
    timings = []
    with torch.no_grad():
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.time()
            img_o, txt_o = forward_back_half(
                model, img=img, txt=txt, vec=vec, vec_orig=vec_orig, pe=pe,
                attn_mask=None,
            )
            torch.cuda.synchronize()
            timings.append(time.time() - t0)

    avg_ms = sum(timings) / n_iters * 1000
    print(f"[smoke] back-half forward: {avg_ms:.1f} ms avg over {n_iters} runs")
    print(f"[smoke] {n_total_loaded} blocks at {avg_ms/max(1, n_total_loaded):.1f} ms/block")
    print(f"[smoke] img out shape={tuple(img_o.shape)} dtype={img_o.dtype}")
    print(f"[smoke] txt out shape={tuple(txt_o.shape)} dtype={txt_o.dtype}")
    print("[smoke] OK")


if __name__ == "__main__":
    main()
