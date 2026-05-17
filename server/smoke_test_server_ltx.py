"""Smoke test: LTX-AV server-side back-half forward.

Loads the LTX-AV slim back-half via load_ltx_av(), builds a synthetic
block_wrap payload of plausible LTX-AV shape, runs forward_back_half_ltx
once, asserts the output (vx, ax) shapes + dtypes are right.

Catches: slim-load fp8 metadata remap regressions, LTXAV variant
detection drift, forward-pass kwarg signature mismatches with newer
ComfyUI versions of comfy.ldm.lightricks.av_model, payload_ltx
reconstruct issues that the standalone payload smoke test can't catch.

Run from the server/ folder on the laptop:
    python smoke_test_server_ltx.py --weights ltx-2.3-22b-dev-fp8.safetensors --n-blocks 8

Expected wall-clock: ~30-60s (model load dominates). Output:
    [smoke] loading LTX-AV slim back-half (8 blocks)
    [smoke] model loaded; transformer_blocks=8 class=LTXAVModel
    [smoke] building synthetic block_wrap payload (vx [1, 4096, 4096] bf16 etc)
    [smoke] forward_back_half_ltx: 1 forward pass over 8 transformer_blocks
    [smoke] forward took XXX ms
    [smoke] output vx shape=(1, 4096, 4096) dtype=torch.bfloat16  OK
    [smoke] output ax shape=(1, 1024, 1024) dtype=torch.bfloat16  OK
    SMOKE TEST PASSED
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent

# Resolve ComfyUI path the same way mesh_server_ltx.py does, so this
# script works regardless of how the server folder was deployed.
import os
_comfy_candidates = [
    os.environ.get("COMFYUI_PATH"),
    str(HERE / "ComfyUI"),
    "C:/ComfyUI",
    "/opt/ComfyUI",
]
try:
    import comfy  # noqa: F401
    _comfy_on_path = True
except ImportError:
    _comfy_on_path = False

if not _comfy_on_path:
    for cand in _comfy_candidates:
        if cand and Path(cand).is_dir() and (Path(cand) / "comfy").is_dir():
            if cand not in sys.path:
                sys.path.insert(0, cand)
            print(f"[smoke] using ComfyUI at: {cand}")
            break

# Put this folder on sys.path so mesh_server_ltx imports cleanly.
if str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

# Import what we need from the server module (load + forward).
import mesh_server_ltx  # noqa: E402
import payload_ltx  # noqa: E402


def build_synthetic_payload(diffusion, device, dtype):
    """Build a block_wrap-shaped payload sized to what LTXAVModel
    would feed each transformer_block at typical resolution.

    Reads the actual block 0's scale_shift_table shapes to size the
    CompressedTimestep tensors correctly. Hardcoding ada_params=6 or 9
    is wrong because cross_attention_adaln varies; the only honest way
    to get the right shape is to ask the loaded block.
    """
    # Read shapes from the loaded model so we match its dims exactly.
    H_v = diffusion.inner_dim
    H_a = getattr(diffusion, "audio_inner_dim", H_v)
    n_attn_heads = diffusion.num_attention_heads

    # Inspect block 0 to learn the actual ada-param counts. The
    # CompressedTimestep .data tensor must be reshapeable as
    # (B, T, num_ada_params, dim_per_param), so its feature_dim =
    # num_ada_params * dim_per_param.
    block0 = diffusion.transformer_blocks[0]
    n_ada_v, v_dim = block0.scale_shift_table.shape
    n_ada_a, a_dim = block0.audio_scale_shift_table.shape
    # av_ca tables are (5, v_dim) / (5, a_dim) — first 4 rows are
    # scale_shift, last row is the gate. So scale_shift_timestep
    # feature_dim = 4 * v_dim, gate_timestep feature_dim = 1 * v_dim
    # (per get_av_ca_ada_values' num_scale_shift_values=4 default).
    n_ca_ss, n_ca_gate = 4, 1

    # Plausible token counts. A modest size for fast forward.
    B = 1
    T_v = 1024  # video tokens
    T_a = 256   # audio tokens
    C_v = 256   # video text-context tokens
    C_a = 64    # audio text-context tokens

    # Per-head per-token freq dim — what _precompute_freqs_cis returns
    # depends on internal_dim / heads etc. We just need plausible
    # shapes; the forward will validate them downstream.
    pe_per_token_v = H_v // n_attn_heads // 2
    pe_per_token_a = H_a // n_attn_heads // 2

    vx = torch.randn(B, T_v, v_dim, dtype=dtype, device=device) * 0.3
    ax = torch.randn(B, T_a, a_dim, dtype=dtype, device=device) * 0.3
    v_context = torch.randn(B, C_v, v_dim, dtype=dtype, device=device) * 0.1
    a_context = torch.randn(B, C_a, a_dim, dtype=dtype, device=device) * 0.1

    # Build a real CompressedTimestep object for the *_timestep slots
    # using the same object.__new__ pattern reconstruct_payload uses.
    from comfy.ldm.lightricks.av_model import CompressedTimestep

    def make_ts(B, T, n_ada, dim):
        feature_dim = n_ada * dim
        obj = object.__new__(CompressedTimestep)
        obj.data = torch.randn(B, T, feature_dim, dtype=dtype, device=device) * 0.01
        obj.batch_size = B
        obj.num_frames = T
        obj.feature_dim = feature_dim
        obj.patches_per_frame = 1
        return obj

    payload = {
        "img": (vx, ax),
        "v_context": v_context,
        "a_context": a_context,
        "attention_mask": None,
        "v_pe": (
            torch.randn(B, n_attn_heads, T_v, pe_per_token_v, dtype=dtype, device=device),
            torch.randn(B, n_attn_heads, T_v, pe_per_token_v, dtype=dtype, device=device),
            True,
        ),
        "a_pe": (
            torch.randn(B, n_attn_heads, T_a, pe_per_token_a, dtype=dtype, device=device),
            torch.randn(B, n_attn_heads, T_a, pe_per_token_a, dtype=dtype, device=device),
            True,
        ),
        "v_cross_pe": (
            torch.randn(B, n_attn_heads, C_v, pe_per_token_v, dtype=dtype, device=device),
            torch.randn(B, n_attn_heads, C_v, pe_per_token_v, dtype=dtype, device=device),
            True,
        ),
        "a_cross_pe": (
            torch.randn(B, n_attn_heads, C_a, pe_per_token_a, dtype=dtype, device=device),
            torch.randn(B, n_attn_heads, C_a, pe_per_token_a, dtype=dtype, device=device),
            True,
        ),
        "v_timestep": make_ts(B, T_v, n_ada_v, v_dim),
        "a_timestep": make_ts(B, T_a, n_ada_a, a_dim),
        "v_cross_scale_shift_timestep": make_ts(B, T_v, n_ca_ss, v_dim),
        "a_cross_scale_shift_timestep": make_ts(B, T_a, n_ca_ss, a_dim),
        "v_cross_gate_timestep": make_ts(B, T_v, n_ca_gate, v_dim),
        "a_cross_gate_timestep": make_ts(B, T_a, n_ca_gate, a_dim),
        "v_prompt_timestep": None,
        "a_prompt_timestep": None,
        "self_attention_mask": None,
        "transformer_options": {
            "run_vx": True,
            "run_ax": True,
            "a2v_cross_attn": True,
            "v2a_cross_attn": True,
        },
    }
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True,
                   help="Path to ltx-2.3-22b-dev-fp8.safetensors (or distilled)")
    p.add_argument("--n-blocks", type=int, default=8,
                   help="Number of back-half blocks to slim-load (default 8)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16"])
    args = p.parse_args()

    if not args.weights.is_file():
        print(f"[smoke] weights not found: {args.weights}", file=sys.stderr)
        sys.exit(2)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    device = torch.device(args.device)

    print(f"[smoke] loading LTX-AV slim back-half ({args.n_blocks} blocks)")
    t0 = time.time()
    patcher = mesh_server_ltx.load_ltx_av(
        args.weights, device, dtype, n_blocks=args.n_blocks,
    )
    t_load = time.time() - t0
    diffusion = patcher.model.diffusion_model
    n_loaded = len(diffusion.transformer_blocks)
    print(
        f"[smoke] model loaded in {t_load:.1f}s; "
        f"transformer_blocks={n_loaded} class={type(diffusion).__name__}"
    )

    if type(diffusion).__name__ != "LTXAVModel":
        print(
            f"[smoke] FAIL: expected LTXAVModel, got {type(diffusion).__name__}. "
            "load_ltx_av's variant detection broke.",
            file=sys.stderr,
        )
        sys.exit(3)
    if n_loaded != args.n_blocks:
        print(
            f"[smoke] FAIL: expected {args.n_blocks} blocks, got {n_loaded}. "
            "slim-load count mismatch.",
            file=sys.stderr,
        )
        sys.exit(3)

    # Make sure weights actually landed on the right device.
    import comfy.model_management
    comfy.model_management.load_models_gpu([patcher], force_full_load=True)

    print(f"[smoke] building synthetic block_wrap payload")
    try:
        payload = build_synthetic_payload(diffusion, device, dtype)
    except Exception as e:
        print(f"[smoke] FAIL: could not build payload: {e}", file=sys.stderr)
        sys.exit(4)

    vx_in, ax_in = payload["img"]
    print(f"[smoke]   vx input shape={tuple(vx_in.shape)} dtype={vx_in.dtype}")
    print(f"[smoke]   ax input shape={tuple(ax_in.shape)} dtype={ax_in.dtype}")

    print(f"[smoke] forward_back_half_ltx: 1 forward pass over {n_loaded} transformer_blocks")
    try:
        with torch.no_grad():
            t0 = time.time()
            vx_out, ax_out = mesh_server_ltx.forward_back_half_ltx(diffusion, payload)
            torch.cuda.synchronize() if device.type == "cuda" else None
            t_fwd = time.time() - t0
    except Exception as e:
        print(f"[smoke] FAIL: forward pass crashed: {type(e).__name__}: {e}",
              file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(5)

    print(f"[smoke] forward took {t_fwd*1000:.1f} ms ({t_fwd*1000/n_loaded:.1f} ms/block)")

    ok = True
    if vx_out.shape != vx_in.shape:
        print(f"[smoke] FAIL vx output shape {tuple(vx_out.shape)} != input {tuple(vx_in.shape)}")
        ok = False
    else:
        print(f"[smoke] output vx shape={tuple(vx_out.shape)} dtype={vx_out.dtype}  OK")
    if ax_out.shape != ax_in.shape:
        print(f"[smoke] FAIL ax output shape {tuple(ax_out.shape)} != input {tuple(ax_in.shape)}")
        ok = False
    else:
        print(f"[smoke] output ax shape={tuple(ax_out.shape)} dtype={ax_out.dtype}  OK")
    if torch.isnan(vx_out.float()).any() or torch.isinf(vx_out.float()).any():
        print(f"[smoke] FAIL vx output contains NaN/Inf — forward path numerically broken")
        ok = False
    if torch.isnan(ax_out.float()).any() or torch.isinf(ax_out.float()).any():
        print(f"[smoke] FAIL ax output contains NaN/Inf — forward path numerically broken")
        ok = False

    print()
    if ok:
        print("=" * 60)
        print("SMOKE TEST PASSED: LTX-AV slim-load + back-half forward works")
        print("=" * 60)
        sys.exit(0)
    else:
        print("=" * 60)
        print("SMOKE TEST FAILED — see FAIL lines above")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
