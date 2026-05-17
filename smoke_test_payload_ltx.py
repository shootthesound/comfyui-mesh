"""Smoke test: LTX-AV payload flatten / reconstruct round-trip.

Builds a synthetic block_wrap args dict matching the LTX-AV signature
(tensors, PE 3-tuples, CompressedTimestep, None entries, transformer_options
flags), runs payload_ltx.flatten_payload → payload_ltx.reconstruct_payload,
asserts every value comes back unchanged.

Catches: key naming drift, none_keys handling, CompressedTimestep
serialization, PE 3-tuple split-and-rejoin, flags-dict preservation.

Run from the comfyui-mesh root:
    python smoke_test_payload_ltx.py

ComfyUI must be on sys.path or PYTHONPATH so payload_ltx can lazily
import CompressedTimestep from comfy.ldm.lightricks.av_model.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Locate ComfyUI for the CompressedTimestep import. Same candidates the
# server uses.
_comfy_candidates = [
    os.environ.get("COMFYUI_PATH"),
    str(HERE.parent.parent),  # this lives under custom_nodes/comfyui-mesh, ../..= ComfyUI
    "S:/Auto/ComfyUI_SEC/ComfyUI",
    "C:/ComfyUI",
    "/opt/ComfyUI",
]
for cand in _comfy_candidates:
    if cand and Path(cand).is_dir() and (Path(cand) / "comfy").is_dir():
        if cand not in sys.path:
            sys.path.insert(0, cand)
        print(f"[smoke] using ComfyUI at: {cand}")
        break

import torch

import payload_ltx


def make_fake_compressed_timestep(batch=1, num_frames=4, feature_dim=16):
    """Construct a real CompressedTimestep mirroring what the LTX model
    would hand the block_wrap. Bypasses the constructor's compression
    logic by building one fresh — same pattern reconstruct_payload uses."""
    from comfy.ldm.lightricks.av_model import CompressedTimestep
    obj = object.__new__(CompressedTimestep)
    obj.data = torch.randn(batch, num_frames, feature_dim, dtype=torch.bfloat16)
    obj.batch_size = batch
    obj.num_frames = num_frames
    obj.feature_dim = feature_dim
    obj.patches_per_frame = 8  # some non-trivial value to verify round-trip
    return obj


def build_synthetic_args():
    """Build a block_wrap args dict shaped like what LTX-AV would pass."""
    B, T_v, T_a, H_v, H_a = 1, 32, 16, 4096, 1024
    C_v, C_a = 64, 32  # text-ctx token counts (small for speed)

    vx = torch.randn(B, T_v, H_v, dtype=torch.bfloat16)
    ax = torch.randn(B, T_a, H_a, dtype=torch.bfloat16)

    return {
        "img": (vx, ax),
        "v_context": torch.randn(B, C_v, H_v, dtype=torch.bfloat16),
        "a_context": torch.randn(B, C_a, H_a, dtype=torch.bfloat16),
        "attention_mask": torch.zeros(B, 1, T_v, C_v, dtype=torch.bfloat16),
        "v_pe": (
            torch.randn(T_v, H_v // 2, dtype=torch.bfloat16),
            torch.randn(T_v, H_v // 2, dtype=torch.bfloat16),
            True,  # split_mode bool
        ),
        "a_pe": (
            torch.randn(T_a, H_a // 2, dtype=torch.bfloat16),
            torch.randn(T_a, H_a // 2, dtype=torch.bfloat16),
            False,
        ),
        "v_cross_pe": (
            torch.randn(C_v, H_v // 2, dtype=torch.bfloat16),
            torch.randn(C_v, H_v // 2, dtype=torch.bfloat16),
            True,
        ),
        "a_cross_pe": (
            torch.randn(C_a, H_a // 2, dtype=torch.bfloat16),
            torch.randn(C_a, H_a // 2, dtype=torch.bfloat16),
            False,
        ),
        "v_timestep": make_fake_compressed_timestep(B, 4, 16),
        "a_timestep": make_fake_compressed_timestep(B, 4, 16),
        "v_cross_scale_shift_timestep": make_fake_compressed_timestep(B, 4, 16),
        "a_cross_scale_shift_timestep": make_fake_compressed_timestep(B, 4, 16),
        "v_cross_gate_timestep": make_fake_compressed_timestep(B, 4, 16),
        "a_cross_gate_timestep": make_fake_compressed_timestep(B, 4, 16),
        "v_prompt_timestep": None,  # optional, often None
        "a_prompt_timestep": None,
        "self_attention_mask": None,  # also optional
        "transformer_options": {
            "run_vx": True,
            "run_ax": True,
            "a2v_cross_attn": True,
            "v2a_cross_attn": True,
            "extra_key_we_dont_care_about": "ignored",  # should be dropped
        },
    }


def tensors_equal(a, b, name) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        print(f"  FAIL {name}: one is None ({a is None=}, {b is None=})")
        return False
    if not torch.equal(a, b):
        print(f"  FAIL {name}: tensors differ")
        return False
    return True


def main():
    print("[smoke] building synthetic LTX-AV block_wrap args dict...")
    args = build_synthetic_args()
    print(f"[smoke] args has {len(args)} keys")

    print("[smoke] flatten_payload...")
    meta, named = payload_ltx.flatten_payload(args)
    print(f"  meta keys: {list(meta.keys())}")
    print(f"  named tensors: {len(named)}")
    print(f"  meta['compressed']: {list(meta['compressed'].keys())}")
    print(f"  meta['pe_split_mode']: {meta['pe_split_mode']}")
    print(f"  meta['flags']: {meta['flags']}")
    print(f"  meta['none_keys']: {meta['none_keys']}")

    # Sanity: expected tensor count
    # 2 (vx, ax) + 2 (v_context, a_context) + 1 (attention_mask)
    # + 8 (4 PE 3-tuples * 2 tensors each, cos+sin)
    # + 6 CompressedTimestep .data tensors
    # = 19 tensors. self_attention_mask + v_prompt_timestep + a_prompt_timestep are None.
    expected = 2 + 2 + 1 + 8 + 6
    assert len(named) == expected, f"expected {expected} tensors, got {len(named)}"
    print(f"  OK: {expected} tensors as expected")

    print("[smoke] reconstruct_payload...")
    tensors_by_name = dict(named)
    recovered = payload_ltx.reconstruct_payload(meta, tensors_by_name)
    print(f"  recovered keys: {sorted(recovered.keys())}")

    print("[smoke] verifying round-trip equality per key...")
    ok = True

    # img tuple
    if not (tensors_equal(args["img"][0], recovered["img"][0], "img[0] (vx)")
            and tensors_equal(args["img"][1], recovered["img"][1], "img[1] (ax)")):
        ok = False
    else:
        print("  OK: img (vx, ax)")

    # Plain tensors
    for k in ("v_context", "a_context", "attention_mask", "self_attention_mask"):
        if not tensors_equal(args.get(k), recovered.get(k), k):
            ok = False
        else:
            print(f"  OK: {k}")

    # PE 3-tuples
    for k in ("v_pe", "a_pe", "v_cross_pe", "a_cross_pe"):
        orig = args[k]
        rec = recovered[k]
        if not isinstance(rec, tuple) or len(rec) != 3:
            print(f"  FAIL {k}: not a 3-tuple, got {type(rec).__name__}")
            ok = False
            continue
        if not (tensors_equal(orig[0], rec[0], f"{k}.cos")
                and tensors_equal(orig[1], rec[1], f"{k}.sin")):
            ok = False
            continue
        if orig[2] != rec[2]:
            print(f"  FAIL {k}.split_mode: {orig[2]} vs {rec[2]}")
            ok = False
            continue
        print(f"  OK: {k} (cos + sin + split_mode={rec[2]})")

    # CompressedTimestep entries
    for k in (
        "v_timestep", "a_timestep",
        "v_cross_scale_shift_timestep", "a_cross_scale_shift_timestep",
        "v_cross_gate_timestep", "a_cross_gate_timestep",
    ):
        orig = args[k]
        rec = recovered[k]
        if not hasattr(rec, "data") or not hasattr(rec, "patches_per_frame"):
            print(f"  FAIL {k}: not a CompressedTimestep, got {type(rec).__name__}")
            ok = False
            continue
        if not tensors_equal(orig.data, rec.data, f"{k}.data"):
            ok = False
            continue
        if orig.patches_per_frame != rec.patches_per_frame:
            print(
                f"  FAIL {k}.patches_per_frame: "
                f"{orig.patches_per_frame} vs {rec.patches_per_frame}"
            )
            ok = False
            continue
        print(f"  OK: {k} (CompressedTimestep ppf={rec.patches_per_frame})")

    # None entries
    for k in ("v_prompt_timestep", "a_prompt_timestep"):
        if recovered[k] is not None:
            print(f"  FAIL {k}: expected None, got {type(recovered[k]).__name__}")
            ok = False
        else:
            print(f"  OK: {k} = None preserved")

    # transformer_options flags — only the four known flags should survive
    rec_opts = recovered["transformer_options"]
    if set(rec_opts.keys()) != {"run_vx", "run_ax", "a2v_cross_attn", "v2a_cross_attn"}:
        print(f"  FAIL transformer_options: unexpected key set {set(rec_opts.keys())}")
        ok = False
    elif rec_opts != {"run_vx": True, "run_ax": True, "a2v_cross_attn": True, "v2a_cross_attn": True}:
        print(f"  FAIL transformer_options: values changed {rec_opts}")
        ok = False
    else:
        print(f"  OK: transformer_options flags preserved (extra keys dropped as expected)")

    print()
    if ok:
        print("=" * 60)
        print("SMOKE TEST PASSED: payload_ltx round-trip is bit-identical")
        print("=" * 60)
        sys.exit(0)
    else:
        print("=" * 60)
        print("SMOKE TEST FAILED — see FAIL lines above")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
