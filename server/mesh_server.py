"""Standalone back-half FLUX server for the comfyui-mesh rig.

Runs on the 4090 machine. Slim-loads only the LAST `--n-blocks`
double_blocks of a FLUX.2 checkpoint (via safetensors.safe_open —
never reads the front half from disk, so models too big for either
GPU still fit on the back-half side). Listens on TCP for
`forward_double_blocks` requests from the 5090 client.

Usage on the 4090 machine:
    python mesh_server.py \
        --weights /path/to/flux-2-klein-9b-fp8.safetensors \
        --n-blocks 4 \
        --port 7777 \
        --bind 0.0.0.0

The `--n-blocks` value MUST match the client's `n_blocks_remote`
setting on the Mesh Split node. Mismatch = wrong output (server still
runs all its loaded blocks but the activations land at the wrong
depth in the diffusion stack).

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

def load_flux2_klein(
    weights_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    n_blocks: int | None = None,
):
    """Slim-load the back-half of a FLUX.2 checkpoint.

    Only the LAST `n_blocks` double_blocks are read from disk. Front-half
    double_blocks, all single_blocks, and the final_layer are never
    touched — for models that don't fit on either device whole, this is
    the load-bearing property.

    Reads via `safetensors.safe_open()` so disk I/O is exactly the bytes
    we keep — no full-model spike in CPU RAM or VRAM at any point.

    Encoders, modulation modules, and pe_embedder are kept (~60-80 MB
    combined) because ComfyUI's model detector needs them to identify
    the architecture (e.g. `double_stream_modulation_img.lin.weight` is
    how it distinguishes FLUX2 from FLUX1). They sit unused on the
    server side but cost almost nothing.

    If `n_blocks is None` or `n_blocks >= total_double_blocks`, loads
    every double_block (still skips single_blocks + final_layer).
    """
    import json
    from safetensors import safe_open
    import comfy.sd
    import comfy.model_management

    print(f"[server] reading checkpoint header from {weights_path}")
    with safe_open(str(weights_path), framework="pt", device="cpu") as f:
        all_keys = list(f.keys())
        # Pull the safetensors metadata too — comfy.sd uses it to identify
        # fp8 quant scheme and wrap ops accordingly. Skipping this leaves
        # *.input_scale / *.weight_scale tensors as "unexpected" keys and
        # the fp8 ops never bind.
        metadata = f.metadata() or {}

        # 1. Detect total double_blocks count from key names (header-only,
        #    no tensor data reads).
        db_indices = set()
        for k in all_keys:
            if k.startswith("double_blocks."):
                db_indices.add(int(k.split(".")[1]))
        total_db = (max(db_indices) + 1) if db_indices else 0
        if total_db == 0:
            raise RuntimeError(
                f"no double_blocks.* tensors found in {weights_path} — "
                f"is this really a FLUX safetensors file?"
            )

        if n_blocks is None or n_blocks >= total_db:
            actual_n = total_db
            drop = 0
        else:
            actual_n = n_blocks
            drop = total_db - n_blocks

        print(f"[server] checkpoint has {total_db} double_blocks; "
              f"loading {actual_n} (skipping first {drop})")

        # 2. Build the slim state dict by reading ONLY the needed tensors.
        #    Each .get_tensor() call reads exactly that tensor's bytes
        #    from disk — no full-file load.
        sd_slim = {}
        for k in all_keys:
            # Server-unused, large: skip entirely (no disk read)
            if k.startswith("single_blocks.") or k.startswith("final_layer."):
                continue
            if k.startswith("double_blocks."):
                idx = int(k.split(".")[1])
                if idx < drop:
                    continue  # front-half block, server doesn't need it
                # Remap: source double_blocks.{drop+i} -> slim double_blocks.{i}
                parts = k.split(".")
                parts[1] = str(idx - drop)
                new_key = ".".join(parts)
                sd_slim[new_key] = f.get_tensor(k)
                continue
            # Encoders / modulation / pe_embedder / txt_norm — small and
            # load-bearing for ComfyUI's architecture detection.
            sd_slim[k] = f.get_tensor(k)

        slim_bytes = sum(t.numel() * t.element_size() for t in sd_slim.values())
        print(f"[server] slim state dict: {len(sd_slim)} tensors, "
              f"{slim_bytes/1024/1024/1024:.2f} GB (full would be ~{total_db / actual_n * slim_bytes / 1024/1024/1024:.1f} GB)")

        # 2b. Remap fp8 quantization metadata. _quantization_metadata is a
        #     JSON-encoded dict of {layer_name: {"format": "float8_e4m3fn", ...}}
        #     keyed by ORIGINAL layer names (e.g. "double_blocks.4.img_attn.proj").
        #     Our slim SD has those weights remapped to indices [0..n_blocks),
        #     so the metadata must remap to match — otherwise comfy.utils.
        #     convert_old_quants writes `comfy_quant` markers at the original
        #     indices, the wrapped fp8 Linear at the remapped indices never
        #     gets its quant config, and forward() finds .weight == None.
        if "_quantization_metadata" in metadata:
            qm = json.loads(metadata["_quantization_metadata"])
            layers = qm.get("layers", {})
            new_layers = {}
            for layer_name, cfg in layers.items():
                if layer_name.startswith("single_blocks.") or layer_name.startswith("final_layer."):
                    continue  # not on the server
                if layer_name.startswith("double_blocks."):
                    parts = layer_name.split(".")
                    idx = int(parts[1])
                    if idx < drop:
                        continue  # front-half block, skipped
                    parts[1] = str(idx - drop)
                    new_layers[".".join(parts)] = cfg
                    continue
                # Encoder / modulation / etc — keep as-is
                new_layers[layer_name] = cfg
            qm["layers"] = new_layers
            # Don't mutate the safetensors-returned metadata in place;
            # copy then update.
            metadata = dict(metadata)
            metadata["_quantization_metadata"] = json.dumps(qm)
            print(f"[server] remapped {len(layers)} fp8 layer entries -> {len(new_layers)} for slim model")

    # 3. Hand the slim SD to ComfyUI's loader. Its detector counts
    #    `double_blocks.*` keys (-> depth=actual_n) and `single_blocks.*`
    #    keys (-> depth_single_blocks=0), so it builds a slim Flux model.
    #    fp8 ops binding happens automatically via comfy's normal path.
    print(f"[server] handing slim state dict to comfy.sd.load_diffusion_model_state_dict")
    model_options = {"dtype": dtype}
    patcher = comfy.sd.load_diffusion_model_state_dict(sd_slim, model_options=model_options, metadata=metadata)
    if patcher is None:
        raise RuntimeError(
            "comfy.sd.load_diffusion_model_state_dict returned None for the slim "
            "state dict. Likely a detection failure — check the checkpoint is a "
            "real FLUX safetensors file."
        )
    # Release CPU-side slim SD before staging to GPU; the patcher already holds refs.
    sd_slim.clear()

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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ALL of the server's loaded double_blocks.

    With the slim loader the server only has the LAST n_blocks of the
    original model, remapped to indices [0..n_blocks). The request's
    `start_block` field (if present) is informational only — server
    always runs from its own block 0 to block N-1.

    Match-up is the user's responsibility: client `n_blocks_remote`
    must equal server `--n-blocks`. Mismatch produces wrong output,
    not a crash.
    """
    for block in model.double_blocks:
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
                    # start_block in the request is informational only — the
                    # slim server always runs its full loaded block list.
                    client_start_block = int(header.get("start_block", 0))
                    t0 = time.time()
                    img, txt, vec, pe, attn_mask = _decode_request_tensors(header, blobs, device)
                    t_decode = time.time() - t0

                    t0 = time.time()
                    img_out, txt_out = forward_back_half_double_blocks(
                        model,
                        img=img, txt=txt, vec=vec, pe=pe,
                        attn_mask=attn_mask,
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
                    print(f"[server] forward {n_double_blocks} blocks (client said start={client_start_block}): "
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
    p.add_argument("--n-blocks", type=int, default=None,
                   help="How many of the LAST double_blocks to load. MUST match "
                        "the client node's `n_blocks_remote` setting — user is "
                        "responsible for keeping the two in sync. If omitted, "
                        "loads every double_block (still skips single_blocks and "
                        "final_layer; useful when the client's split_index=0).")
    args = p.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    model = load_flux2_klein(args.weights, device, dtype, n_blocks=args.n_blocks)
    serve(model, args.bind, args.port, device)


if __name__ == "__main__":
    main()
