"""Smoke test: Nvenc LTX codec round-trip.

Encodes a heavy-tailed synthetic tensor (mimicking LTX activations
with sparse outliers) via the 'Nvenc LTX' codec mode, decodes, asserts
reconstruction error stays in tolerance. Compares against plain 'nvenc'
+ 'raw' for sanity.

Catches: percentile-clip arithmetic bugs, sparse outlier index packing
off-by-ones, codec dispatch drift, WireTensor schema changes.

Run from the comfyui-mesh root:
    S:/Auto/ComfyUI_SEC/ComfyUI/venv/Scripts/python.exe -u smoke_test_codec_ltx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch

import codec


# Tolerances tuned for the Nvenc LTX mode at qp=1 + tile_dim=8 on
# heavy-tailed bf16 activations. Picked from the empirical smoke-test
# numbers we measured during development (mean ~0.018, max ~0.13 on
# similar input). Set to ~2x those values to leave headroom for the
# random seed picking a slightly worse-luck distribution.
NVENC_LTX_MAX_TOL = 0.30
NVENC_LTX_MEAN_TOL = 0.05


def heavy_tailed_synthetic(device, seed=0):
    """Mimic LTX activations: most values in [-0.5, 0.5], 0.5% outliers
    pushed 20x out — the exact pattern that crushed plain nvenc quant
    and motivated the clipsparse mode."""
    torch.manual_seed(seed)
    x = torch.randn(1, 4096, 1024, dtype=torch.bfloat16, device=device) * 0.3
    outlier_mask = torch.rand_like(x.to(torch.float32)) < 0.005
    x = torch.where(outlier_mask, x * 20, x)
    return x


def measure_round_trip(x, mode, **kw):
    w = codec.encode("img", x, mode=mode, **kw)
    d = codec.decode(w.to_header(), w.bytes_payload, device=x.device)
    xf = x.to(torch.float32)
    df = d.to(torch.float32)
    diff = (xf - df).abs()
    return {
        "encoding": w.encoding,
        "bytes": len(w.bytes_payload),
        "max_err": diff.max().item(),
        "mean_err": diff.mean().item(),
        "shape_ok": d.shape == x.shape,
        "dtype_ok": d.dtype == x.dtype,
    }


def main():
    if not torch.cuda.is_available():
        print("[smoke] CUDA not available — codec smoke test skipped (codec requires CUDA)")
        sys.exit(0)
    device = "cuda"
    print(f"[smoke] GPU: {torch.cuda.get_device_name(0)}")

    print("[smoke] building synthetic heavy-tailed LTX-like tensor...")
    x = heavy_tailed_synthetic(device)
    print(f"[smoke] shape={tuple(x.shape)} dtype={x.dtype}")
    print(
        f"[smoke] value range: "
        f"min={x.to(torch.float32).min().item():.3f}  "
        f"max={x.to(torch.float32).max().item():.3f}"
    )

    print()
    print("[smoke] round-tripping through all three modes...")

    results = {}
    # raw: baseline (mathematically lossless)
    results["raw"] = measure_round_trip(x, mode="raw")
    # nvenc (the lossy mode): expected to show contrast crush
    results["nvenc"] = measure_round_trip(x, mode="nvenc", qp=18, tile_dim=8)
    # Nvenc LTX (the tuned mode): expected to behave near-raw
    results["Nvenc LTX"] = measure_round_trip(x, mode="Nvenc LTX", qp=1, tile_dim=8)

    print()
    print(
        f"{'mode':<14} | {'bytes':>8} | {'max err':>8} | {'mean err':>9} | {'shape':>5} {'dtype':>5}"
    )
    print("-" * 65)
    for name, r in results.items():
        print(
            f"{name:<14} | {r['bytes']/1024:>8.1f} | {r['max_err']:>8.4f} | "
            f"{r['mean_err']:>9.6f} | {'OK' if r['shape_ok'] else 'BAD':>5} "
            f"{'OK' if r['dtype_ok'] else 'BAD':>5}"
        )

    print()
    ok = True

    # raw must be bit-exact
    if results["raw"]["max_err"] != 0 or results["raw"]["mean_err"] != 0:
        print(
            f"  FAIL raw: expected 0 err, got max={results['raw']['max_err']} "
            f"mean={results['raw']['mean_err']}"
        )
        ok = False
    else:
        print("  OK: raw is bit-exact")

    # Nvenc LTX must be within tolerance
    nltx = results["Nvenc LTX"]
    if nltx["max_err"] > NVENC_LTX_MAX_TOL:
        print(
            f"  FAIL Nvenc LTX: max_err {nltx['max_err']:.4f} exceeds tolerance "
            f"{NVENC_LTX_MAX_TOL}"
        )
        ok = False
    elif nltx["mean_err"] > NVENC_LTX_MEAN_TOL:
        print(
            f"  FAIL Nvenc LTX: mean_err {nltx['mean_err']:.6f} exceeds tolerance "
            f"{NVENC_LTX_MEAN_TOL}"
        )
        ok = False
    else:
        print(
            f"  OK: Nvenc LTX within tolerance (max {nltx['max_err']:.4f} <= "
            f"{NVENC_LTX_MAX_TOL}, mean {nltx['mean_err']:.6f} <= {NVENC_LTX_MEAN_TOL})"
        )

    # Sanity: Nvenc LTX should be meaningfully better than plain nvenc on this input
    nv = results["nvenc"]
    if nltx["mean_err"] >= nv["mean_err"]:
        print(
            f"  WARN: Nvenc LTX mean_err {nltx['mean_err']:.6f} >= plain nvenc "
            f"{nv['mean_err']:.6f}. Clipsparse should outperform plain nvenc on "
            f"heavy-tailed inputs."
        )
        # Don't fail — it's a regression signal, not a wire-correctness signal.

    # Shape/dtype must be preserved across all three
    for name, r in results.items():
        if not r["shape_ok"]:
            print(f"  FAIL {name}: shape changed")
            ok = False
        if not r["dtype_ok"]:
            print(f"  FAIL {name}: dtype changed")
            ok = False

    print()
    if ok:
        print("=" * 60)
        print("SMOKE TEST PASSED: codec round-trips correct, Nvenc LTX in tolerance")
        print("=" * 60)
        sys.exit(0)
    else:
        print("=" * 60)
        print("SMOKE TEST FAILED — see FAIL lines above")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
