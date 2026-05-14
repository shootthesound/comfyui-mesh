"""Standalone back-half FLUX server for the comfyui-mesh rig.

Runs on the 4090 machine. Loads the FLUX.2 Klein 9B weights, builds
ALL the double_blocks (we'll use blocks [start_block..end] per request),
and listens on TCP for forward_double_blocks requests from the 5090
client.

Usage on the 4090 machine:
    python mesh_server.py \
        --weights /path/to/flux-2-klein-9b-fp8.safetensors \
        --port 7777 \
        --bind 0.0.0.0

Dependencies on the 4090:
    - torch
    - safetensors
    - ComfyUI installed and importable (only its `comfy/` Python package
      is needed; we don't run the UI or its model registry). Get it via:
          git clone https://github.com/comfyanonymous/ComfyUI ComfyUI
          pip install -r ComfyUI/requirements.txt
      then point COMFYUI_PATH at it (or `pip install -e ComfyUI`).
    - nvenc-pframe (optional, for codec-mode requests; raw mode works
      without it)

We import comfy.ldm.flux.model directly rather than vendoring it,
because that subtree pulls in comfy.ldm.common_dit + comfy.patcher_extension
+ a long tail of FLUX2 implementation details that change frequently.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent

# COMFYUI_PATH may be exported by run_server.bat / .sh to point at a
# ComfyUI checkout. Fall back to the user's likely default install
# locations so the script works out of the box on a typical setup.
import os
_comfy_candidates = [
    os.environ.get("COMFYUI_PATH"),
    str(HERE / "ComfyUI"),
    str(HERE.parent / "ComfyUI"),
    "C:/ComfyUI",
    "/opt/ComfyUI",
]
# Try to locate a ComfyUI source tree. If `comfy` already imports
# (e.g. the parent script already added it to sys.path), don't bother.
try:
    import comfy  # noqa: F401
    _comfy_already_on_path = True
except ImportError:
    _comfy_already_on_path = False

if not _comfy_already_on_path:
    for cand in _comfy_candidates:
        if cand and Path(cand).is_dir() and (Path(cand) / "comfy").is_dir():
            if cand not in sys.path:
                sys.path.insert(0, cand)
            print(f"[server] using ComfyUI at: {cand}")
            break
    else:
        print("[server] WARNING: COMFYUI_PATH not set and no ComfyUI checkout found in default locations.")
        print("[server]          Set COMFYUI_PATH=/path/to/ComfyUI before running this script.")

# Insert HERE LAST so our local protocol.py / codec.py / vec_io.py win
# over any same-named modules in ComfyUI's root (ComfyUI ships its own
# top-level protocol.py with a different API).
if str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))

import codec
import protocol
import vec_io


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def load_flux2_klein(weights_path: Path, device: torch.device, dtype: torch.dtype):
    """Load the FLUX.2 Klein 9B model via ComfyUI's own loader.

    Returns the inner `Flux` diffusion module (not the ModelPatcher
    wrapper), already fp8-aware and on `device` in `dtype`.

    Why ComfyUI's loader rather than rolling our own:
    - The fp8 distilled checkpoint has per-tensor scales (`*.input_scale`,
      `*.weight_scale`) that the bare `Flux.load_state_dict` doesn't
      know how to apply. ComfyUI's loader wraps the ops to consume them.
    - The vec/modulation construction for `global_modulation=True` is
      done inside `Flux.forward`, so as long as we feed the inner Flux
      module the same modulated `vec` tuple the front half computed,
      the per-block forward stays consistent.
    """
    import comfy.sd

    print(f"[server] loading FLUX.2 Klein 9B via comfy.sd.load_diffusion_model")
    model_options = {"dtype": dtype}
    patcher = comfy.sd.load_diffusion_model(str(weights_path), model_options=model_options)
    if patcher is None:
        raise RuntimeError(f"comfy.sd.load_diffusion_model returned None for {weights_path}")

    # Force the model onto the requested device. Normally ComfyUI's
    # ModelPatcher does this lazily via .patch_model(); we want it
    # resident now because the server is long-lived.
    import comfy.model_management
    comfy.model_management.load_models_gpu([patcher], force_full_load=True)

    diffusion = patcher.model.diffusion_model
    n_double = len(diffusion.double_blocks)
    n_single = len(diffusion.single_blocks)
    print(f"[server] model loaded; double_blocks={n_double} single_blocks={n_single} "
          f"hidden_size={diffusion.hidden_size} global_modulation={diffusion.params.global_modulation}")
    return diffusion


# ---------------------------------------------------------------------
# Forward pass — back-half double_blocks
# ---------------------------------------------------------------------

@torch.no_grad()
def forward_back_half_double_blocks(
    model,
    *,
    img: torch.Tensor,
    txt: torch.Tensor,
    vec,
    pe: torch.Tensor,
    attn_mask,
    start_block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run model.diffusion_model.double_blocks[start_block:] on the
    given activations. Mirrors the loop in comfy/ldm/flux/model.py
    (lines 219-249) but only for the back half."""
    blocks = model.double_blocks
    for i in range(start_block, len(blocks)):
        block = blocks[i]
        img, txt = block(
            img=img,
            txt=txt,
            vec=vec,
            pe=pe,
            attn_mask=attn_mask,
            transformer_options={},
        )
    if img.dtype == torch.float16:
        img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)
    return img, txt


# ---------------------------------------------------------------------
# TCP server loop
# ---------------------------------------------------------------------

def _decode_request_tensors(header: dict, blobs: list[bytes], device: torch.device):
    wires = header["tensors"]
    by_name = {}
    for w, b in zip(wires, blobs):
        by_name[w["name"]] = (w, b)

    img = codec.decode(*by_name["img"], device=device)
    txt = codec.decode(*by_name["txt"], device=device)

    # Reconstruct vec via the named-tensor scheme from vec_io.
    vec_kind = header.get("vec_kind", "tensor")
    vec_named: dict[str, torch.Tensor] = {}
    for name, (wire, blob) in by_name.items():
        if name == "vec" or name.startswith("vec_"):
            vec_named[name] = codec.decode(wire, blob, device=device)
    vec = vec_io.reconstruct_vec(vec_kind, vec_named)

    pe = codec.decode(*by_name["pe"], device=device)
    attn_mask = None
    if header.get("has_attn_mask"):
        attn_mask = codec.decode(*by_name["attn_mask"], device=device)
    return img, txt, vec, pe, attn_mask


def _encode_response_tensors(img: torch.Tensor, txt: torch.Tensor, codec_mode: str, codec_qp: int, codec_lossless: bool):
    img_w = codec.encode("img", img, mode=codec_mode, qp=codec_qp, lossless=codec_lossless)
    txt_w = codec.encode_raw("txt", txt)
    return [img_w, txt_w]


def serve(model, host: str, port: int, device: torch.device):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(1)
    print(f"[server] listening on {host}:{port}")

    n_double_blocks = len(model.double_blocks)

    while True:
        conn, addr = s.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[server] client connected: {addr}")
        try:
            while True:
                header, blobs = protocol.recv_message(conn)
                kind = header.get("kind")

                if kind == "hello":
                    protocol.send_message(conn, {
                        "kind": "hello_ack",
                        "tensors": [],
                        "server_info": {
                            "device": str(device),
                            "n_double_blocks": n_double_blocks,
                        },
                    }, [])

                elif kind == "forward_double_blocks":
                    start_block = int(header["start_block"])
                    t0 = time.time()
                    img, txt, vec, pe, attn_mask = _decode_request_tensors(header, blobs, device)
                    t_decode = time.time() - t0

                    t0 = time.time()
                    img_out, txt_out = forward_back_half_double_blocks(
                        model,
                        img=img, txt=txt, vec=vec, pe=pe,
                        attn_mask=attn_mask, start_block=start_block,
                    )
                    t_forward = time.time() - t0

                    # Echo the same codec mode that came in for the img tensor
                    img_in_wire = next(t for t in header["tensors"] if t["name"] == "img")
                    codec_mode = img_in_wire["encoding"]
                    codec_qp = 18
                    codec_lossless = False
                    # Re-derive from extra if available
                    if codec_mode == "nvenc":
                        # Match the request's QP heuristic — server doesn't get told,
                        # so default to qp18 unless caller wants otherwise (a future
                        # protocol extension)
                        pass

                    t0 = time.time()
                    wire_outs = _encode_response_tensors(img_out, txt_out, codec_mode, codec_qp, codec_lossless)
                    t_encode = time.time() - t0

                    resp_header = {
                        "kind": "forward_double_blocks_response",
                        "tensors": [w.to_header() for w in wire_outs],
                        "timings_ms": {
                            "decode": t_decode * 1000,
                            "forward": t_forward * 1000,
                            "encode": t_encode * 1000,
                        },
                    }
                    protocol.send_message(conn, resp_header, [w.bytes_payload for w in wire_outs])
                    print(f"[server] forward [{start_block}..{n_double_blocks}): "
                          f"decode {t_decode*1000:.1f} ms  fwd {t_forward*1000:.1f} ms  enc {t_encode*1000:.1f} ms  "
                          f"in {sum(len(b) for b in blobs)/1024/1024:.2f} MB  "
                          f"out {sum(len(w.bytes_payload) for w in wire_outs)/1024/1024:.2f} MB")
                else:
                    print(f"[server] unknown request kind: {kind}")
                    break
        except (ConnectionError, OSError) as e:
            print(f"[server] client disconnected: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True, help="Path to flux-2-klein-9b-fp8.safetensors")
    p.add_argument("--bind", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=7777)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    args = p.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    model = load_flux2_klein(args.weights, device, dtype)
    serve(model, args.bind, args.port, device)


if __name__ == "__main__":
    main()
