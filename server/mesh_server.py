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
import json
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
import lora_io


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

    `n_blocks` counts BOTH double_blocks AND single_blocks. The mapping:
        - n_blocks <= n_double:        last n_blocks double_blocks
                                       (no single_blocks loaded)
        - n_blocks > n_double:         all double_blocks loaded, plus
                                       FIRST (n_blocks - n_double) single_blocks
                                       (so client picks up at single_block index
                                        n_blocks - n_double)

    Reads via `safetensors.safe_open()` so disk I/O is exactly the bytes
    we keep — no full-model spike in CPU RAM or VRAM at any point.

    Encoders, modulation modules, and pe_embedder are kept (~60-80 MB
    combined) because ComfyUI's model detector needs them to identify
    the architecture (e.g. `double_stream_modulation_img.lin.weight` is
    how it distinguishes FLUX2 from FLUX1). They sit unused on the
    server side but cost almost nothing.

    If `n_blocks is None`, loads every block (full back-half).
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

        # 1. Detect total double_blocks AND single_blocks counts from key
        #    names (header-only, no tensor data reads).
        db_indices = set()
        sb_indices = set()
        for k in all_keys:
            if k.startswith("double_blocks."):
                db_indices.add(int(k.split(".")[1]))
            elif k.startswith("single_blocks."):
                sb_indices.add(int(k.split(".")[1]))
        total_db = (max(db_indices) + 1) if db_indices else 0
        total_sb = (max(sb_indices) + 1) if sb_indices else 0
        if total_db == 0:
            raise RuntimeError(
                f"no double_blocks.* tensors found in {weights_path} — "
                f"is this really a FLUX safetensors file?"
            )

        total_blocks = total_db + total_sb
        if n_blocks is None or n_blocks >= total_blocks:
            n_double_remote = total_db
            n_single_remote = total_sb
        elif n_blocks <= total_db:
            n_double_remote = n_blocks
            n_single_remote = 0
        else:
            n_double_remote = total_db
            n_single_remote = n_blocks - total_db

        drop_db = total_db - n_double_remote   # how many front-half doubles to skip
        # singles: we load the FIRST n_single_remote of them, no remap needed
        # since they stay at indices [0..n_single_remote).

        print(f"[server] checkpoint has {total_db} double_blocks + {total_sb} single_blocks "
              f"({total_blocks} total)")
        print(f"[server] loading {n_double_remote} doubles (skipping first {drop_db}) "
              f"+ {n_single_remote} singles (first {n_single_remote})")

        # 2. Build the slim state dict by reading ONLY the needed tensors.
        #    Each .get_tensor() call reads exactly that tensor's bytes
        #    from disk — no full-file load.
        sd_slim = {}
        for k in all_keys:
            if k.startswith("final_layer."):
                continue  # client always runs final_layer
            if k.startswith("double_blocks."):
                idx = int(k.split(".")[1])
                if idx < drop_db:
                    continue  # front-half block, client runs it
                # Remap: source double_blocks.{drop_db+i} -> slim double_blocks.{i}
                parts = k.split(".")
                parts[1] = str(idx - drop_db)
                new_key = ".".join(parts)
                sd_slim[new_key] = f.get_tensor(k)
                continue
            if k.startswith("single_blocks."):
                idx = int(k.split(".")[1])
                if idx >= n_single_remote:
                    continue  # this single runs on the client (we only take the first M)
                # No remap — singles stay at their original [0..n_single_remote) indices
                sd_slim[k] = f.get_tensor(k)
                continue
            # Encoders / modulation / pe_embedder / txt_norm — small and
            # load-bearing for ComfyUI's architecture detection.
            sd_slim[k] = f.get_tensor(k)

        slim_bytes = sum(t.numel() * t.element_size() for t in sd_slim.values())
        n_loaded = n_double_remote + n_single_remote
        full_estimate_gb = (total_blocks / max(1, n_loaded) * slim_bytes / 1024/1024/1024) if n_loaded else 0
        print(f"[server] slim state dict: {len(sd_slim)} tensors, "
              f"{slim_bytes/1024/1024/1024:.2f} GB (full would be ~{full_estimate_gb:.1f} GB)")

        # 2b. Remap fp8 quantization metadata to match the slim SD's key
        #     remapping. Otherwise comfy.utils.convert_old_quants writes
        #     `comfy_quant` markers at the original indices, the wrapped
        #     fp8 Linear at the remapped indices never gets its quant
        #     config, and forward() finds .weight == None.
        if "_quantization_metadata" in metadata:
            qm = json.loads(metadata["_quantization_metadata"])
            layers = qm.get("layers", {})
            new_layers = {}
            for layer_name, cfg in layers.items():
                if layer_name.startswith("final_layer."):
                    continue  # not on the server
                if layer_name.startswith("double_blocks."):
                    parts = layer_name.split(".")
                    idx = int(parts[1])
                    if idx < drop_db:
                        continue  # front-half block, skipped
                    parts[1] = str(idx - drop_db)
                    new_layers[".".join(parts)] = cfg
                    continue
                if layer_name.startswith("single_blocks."):
                    parts = layer_name.split(".")
                    idx = int(parts[1])
                    if idx >= n_single_remote:
                        continue  # this single runs on client
                    new_layers[layer_name] = cfg  # no remap
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
    #    `double_blocks.*` keys (-> depth=n_double_remote) and
    #    `single_blocks.*` keys (-> depth_single_blocks=n_single_remote),
    #    so it builds a slim Flux model with exactly the blocks we have
    #    weights for. fp8 ops binding happens automatically via comfy's
    #    normal path.
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

    # Stash slim-load metadata on the patcher so the LoRA loader can
    # remap key indices correctly without re-deriving them.
    patcher._mesh_slim_meta = {
        "drop_db": drop_db,
        "n_single_remote": n_single_remote,
        "total_db": total_db,
        "total_sb": total_sb,
    }
    return patcher


# ---------------------------------------------------------------------
# LoRA application — server-side
# ---------------------------------------------------------------------

def apply_server_lora(patcher, lora_path: Path, strength: float) -> int:
    """Load a LoRA from disk and apply it to the slim-loaded model.

    LoRAs reference ORIGINAL block indices (e.g. `double_blocks.4` for
    the 5th double_block). Our slim model has those weights remapped to
    indices [0..n_loaded). To make ComfyUI's standard LoRA pipeline work,
    we enhance the model's lora key_map with aliases: every entry that
    references slim_idx K gets a parallel entry referencing original_idx
    (K + drop_db) pointing at the same model parameter.

    LoRA references to layers the server doesn't hold (front-half doubles,
    tail singles, final_layer) silently fail to match and get dropped.

    Returns the number of patches actually applied.
    """
    import re
    import comfy.lora
    import comfy.lora_convert
    import comfy.utils

    slim_meta = getattr(patcher, "_mesh_slim_meta", None)
    if slim_meta is None:
        raise RuntimeError("apply_server_lora: patcher has no slim-load metadata")
    drop_db = slim_meta["drop_db"]

    print(f"[server] loading LoRA: {lora_path} (strength={strength})")
    lora_sd = comfy.utils.load_torch_file(str(lora_path), safe_load=True)
    print(f"[server] LoRA has {len(lora_sd)} tensors")

    # Build the slim model's natural key_map
    key_map = comfy.lora.model_lora_keys_unet(patcher.model, {})

    # Add aliases mapping ORIGINAL block indices to slim model targets.
    # We look at every existing key_map entry, find the slim block index
    # if any, and add a parallel entry for the original index.
    pat_kohya = re.compile(r"(double_blocks_)(\d+)(_)")
    pat_native = re.compile(r"(double_blocks\.)(\d+)(\.)")
    new_aliases = {}
    if drop_db > 0:
        for lora_key, model_key in key_map.items():
            for pat, sep in ((pat_kohya, "_"), (pat_native, ".")):
                m = pat.search(lora_key)
                if m:
                    slim_idx = int(m.group(2))
                    orig_idx = slim_idx + drop_db
                    aliased = pat.sub(f"{m.group(1)}{orig_idx}{sep}", lora_key, count=1)
                    new_aliases[aliased] = model_key
                    break  # one pattern match per key
    if new_aliases:
        print(f"[server] added {len(new_aliases)} key-map aliases (orig idx -> slim idx, drop_db={drop_db})")
        key_map.update(new_aliases)

    # Run the standard ComfyUI conversion + matching pipeline
    lora_sd = comfy.lora_convert.convert_lora(lora_sd)
    loaded = comfy.lora.load_lora(lora_sd, key_map, log_missing=False)
    print(f"[server] LoRA matched {len(loaded)} model parameters")

    n_patches = patcher.add_patches(loaded, strength)
    print(f"[server] add_patches accepted {len(n_patches)} patches")

    # Force the LoRA to actually fold into the loaded weights now (slim-load
    # is long-lived; we don't want the per-call patching ComfyUI normally
    # does). load_models_gpu with force_full_load=True re-stages and applies.
    import comfy.model_management
    comfy.model_management.load_models_gpu([patcher], force_full_load=True)

    return len(n_patches)


# ---------------------------------------------------------------------
# Forward pass — back-half (double_blocks + optional single_blocks)
# ---------------------------------------------------------------------

@torch.no_grad()
def forward_back_half(
    model,
    *,
    img: torch.Tensor,
    txt: torch.Tensor,
    vec,                     # double-block modulation tuple
    vec_orig: torch.Tensor | None,  # un-modulated tensor; needed iff singles loaded
    pe: torch.Tensor,
    attn_mask,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ALL of the server's loaded blocks (doubles + any singles).

    Doubles use the `vec` modulation tuple shipped from the client. If
    we also have single_blocks loaded, we compute their modulation
    locally via `model.single_stream_modulation(vec_orig)` — that
    module is in the slim-load alongside the others. So the wire
    payload only needs to carry vec_orig (one tiny tensor) instead
    of a separately-precomputed single-block modulation tuple.

    For the single_blocks portion, FLUX's architecture concatenates
    txt+img into one stream before the singles loop and slices them
    back at the end. We do that internally so the wire format stays
    `(img, txt)` regardless of how many singles ran on the server.

    Match-up is the user's responsibility: client `n_blocks_remote`
    must equal server `--n-blocks`. Mismatch produces wrong output,
    not a crash.
    """
    # Doubles first — operate on separate (img, txt) streams
    for block in model.double_blocks:
        img, txt = block(
            img=img, txt=txt, vec=vec, pe=pe,
            attn_mask=attn_mask, transformer_options={},
        )

    # Singles, if any — operate on the concatenated stream with their
    # own (different) modulation
    if len(model.single_blocks) > 0:
        if vec_orig is None:
            raise RuntimeError(
                "server has single_blocks loaded but client did not send vec_orig — "
                "client/server are on different versions, or n_blocks_remote on the "
                "client is < n_double (no singles expected) but server has singles loaded"
            )
        # Compute single-block modulation locally. global_modulation=True
        # path: returns (ModulationOut, None); we want the first element.
        vec_single, _ = model.single_stream_modulation(vec_orig)

        original_txt_len = txt.shape[1]
        combined = torch.cat((txt, img), 1)
        for block in model.single_blocks:
            combined = block(
                combined, vec=vec_single, pe=pe,
                attn_mask=attn_mask, transformer_options={},
            )
        # Slice back to (img, txt) so the wire response shape is stable
        txt = combined[:, :original_txt_len, ...]
        img = combined[:, original_txt_len:, ...]

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

    # Reconstruct vec (modulation tuple) via vec_io's named-tensor scheme.
    # Note: vec_orig is shipped as a separate top-level tensor (not part of
    # the modulation tuple), so we exclude it from the vec_io reconstruction.
    # client_lora is also not a codec'd tensor — pass-through bytes.
    vec_kind = header.get("vec_kind", "tensor")
    vec_named: dict[str, torch.Tensor] = {}
    for name, (wire, blob) in by_name.items():
        if name in ("vec_orig", "client_lora"):
            continue
        if name == "vec" or name.startswith("vec_"):
            vec_named[name] = codec.decode(wire, blob, device=device)
    vec = vec_io.reconstruct_vec(vec_kind, vec_named)

    # vec_orig is optional — only present if the client knows the server
    # might run single_blocks (newer client). If absent, server can still
    # run a doubles-only forward.
    vec_orig = None
    if "vec_orig" in by_name:
        vec_orig = codec.decode(*by_name["vec_orig"], device=device)

    pe = codec.decode(*by_name["pe"], device=device)
    attn_mask = None
    if header.get("has_attn_mask"):
        attn_mask = codec.decode(*by_name["attn_mask"], device=device)

    # client_lora is opaque safetensors bytes shipped only when the
    # client's LoRA session changed. None = blob not present this call.
    client_lora_blob = None
    if "client_lora" in by_name:
        _wire, client_lora_blob = by_name["client_lora"]

    return img, txt, vec, vec_orig, pe, attn_mask, client_lora_blob


def _encode_response_tensors(img: torch.Tensor, txt: torch.Tensor, codec_mode: str, codec_qp: int, codec_lossless: bool, codec_tile_dim: int = 4):
    img_w = codec.encode("img", img, mode=codec_mode, qp=codec_qp, lossless=codec_lossless, tile_dim=codec_tile_dim)
    txt_w = codec.encode_raw("txt", txt)
    return [img_w, txt_w]


def _apply_client_lora(patcher, blob: bytes, session_id: str, device: torch.device):
    """Apply a client-shipped LoRA bundle to the patcher. Forces a
    re-stage so patches actually fold into the loaded weights (rather
    than ComfyUI's lazy cast-time application). Returns the count of
    patch entries appended.

    NOTE: we write directly to `patcher.patches` rather than using
    `add_patches`. add_patches wraps `patches[k]` inside the 5-tuple
    `(strength, data, strength_model, offset, function)` itself —
    but our decoded entries are ALREADY full 5-tuples in ComfyUI's
    internal format (strengths, offsets, etc baked in via
    decode_patches_from_safetensors). Going through add_patches
    would double-wrap and break calculate_weight.
    """
    import uuid
    import comfy.model_management
    patches = lora_io.decode_patches_from_safetensors(blob, device=device)
    n_keys = len(patches)
    n_entries = sum(len(v) for v in patches.values())

    model_sd = patcher.model.state_dict()
    n_attached = 0
    n_missing = 0
    for key, entries in patches.items():
        if key not in model_sd:
            n_missing += 1
            continue
        existing = patcher.patches.get(key, [])
        existing.extend(entries)
        patcher.patches[key] = existing
        n_attached += len(entries)
    if n_keys > 0:
        patcher.patches_uuid = uuid.uuid4()  # invalidate ComfyUI's cache

    print(f"[server] client LoRA: attached {n_attached} entries across "
          f"{n_keys - n_missing} keys (skipped {n_missing} missing-from-slim) — "
          f"session={session_id}")
    comfy.model_management.load_models_gpu([patcher], force_full_load=True)
    return n_attached


def _unapply_client_lora(patcher):
    """Roll back any previously-applied client LoRA. Uses ComfyUI's
    standard unpatch_model (clears all patches). The server-config
    LoRA from --lora is then RE-applied so it survives the swap.

    NOTE: this also clears server-config LoRA. Caller must reapply it
    via apply_server_lora if it was set."""
    import comfy.model_management
    if patcher.patches:
        try:
            patcher.unpatch_model()
        except Exception as e:
            print(f"[server] unpatch_model failed (will continue): {e}")
        # Clear the patches dict so add_patches starts fresh
        patcher.patches.clear()
    comfy.model_management.load_models_gpu([patcher], force_full_load=True)


def serve(patcher, host: str, port: int, device: torch.device,
          server_lora_path: Path = None, server_lora_strength: float = 1.0):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(1)

    model = patcher.model.diffusion_model
    n_double_blocks = len(model.double_blocks)
    n_single_blocks = len(model.single_blocks)
    n_total_loaded = n_double_blocks + n_single_blocks
    print(f"[server] READY — listening on {host}:{port} "
          f"(n_blocks={n_total_loaded}: {n_double_blocks}D + {n_single_blocks}S)")

    # Track currently-applied client LoRA session id so we know when to
    # swap. None = no client LoRA applied; "empty" = client says "no LoRA".
    current_client_lora_session: str | None = None

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
                            "n_single_blocks": n_single_blocks,
                            "n_total_loaded": n_total_loaded,
                            # The client uses this to detect when its
                            # n_blocks_remote setting drifts from what
                            # the server is actually running; the JS
                            # surfaces a Confirm-restart button on
                            # mismatch and POSTs /mesh/reconfigure to
                            # trigger the kind below.
                            "n_blocks": n_total_loaded,
                        },
                    }, [])

                elif kind == "reconfigure":
                    # Client wants the server to relaunch with a different
                    # --n-blocks. ACK first so the client knows we got it
                    # (and can expect the socket to drop), then write a
                    # small handoff file so the GUI launcher knows to
                    # restart us with the new value, and exit.
                    #
                    # We deliberately do NOT use os.execv here: on Windows
                    # execv spawns a new process and exits the current
                    # one, which detaches the new process from the GUI's
                    # subprocess handle. The GUI would lose the stdout
                    # tail and the user would see no "ready" message.
                    # The handoff-file dance keeps the GUI in the
                    # parent-process role for the relaunched server.
                    new_n_blocks = int(header.get("n_blocks", n_total_loaded))
                    print(f"[server] RESTARTING: reconfigure request "
                          f"--n-blocks {n_total_loaded} -> {new_n_blocks}")
                    protocol.send_message(conn, {
                        "kind": "reconfigure_ack",
                        "tensors": [],
                        "new_n_blocks": new_n_blocks,
                    }, [])
                    try: conn.close()
                    except Exception: pass
                    try: s.close()
                    except Exception: pass

                    handoff = Path(__file__).parent / "mesh_server_reconfig.tmp"
                    try:
                        handoff.write_text(
                            json.dumps({"n_blocks": new_n_blocks}),
                            encoding="utf-8",
                        )
                        print(f"[server] wrote {handoff.name} for GUI relaunch")
                    except Exception as e:
                        print(f"[server] could not write {handoff.name}: {e}")
                    print(f"[server] exiting; GUI launcher will restart with "
                          f"--n-blocks={new_n_blocks}")
                    sys.stdout.flush()
                    sys.exit(0)

                elif kind == "forward_double_blocks":
                    # start_block in the request is informational only — the
                    # slim server always runs its full loaded block list.
                    client_start_block = int(header.get("start_block", 0))
                    t0 = time.time()
                    (img, txt, vec, vec_orig, pe, attn_mask,
                     client_lora_blob) = _decode_request_tensors(header, blobs, device)
                    t_decode = time.time() - t0

                    # ---- Client-LoRA lifecycle ----
                    # Client transmits a session id every call; the blob
                    # itself only when the session changed.
                    incoming_session = header.get("client_lora_session", "") or ""
                    if incoming_session and incoming_session != current_client_lora_session:
                        # Session changed — unpatch any prior LoRAs
                        # (server-config + previous client-supplied),
                        # then re-apply server-config + new client-supplied.
                        print(f"[server] client LoRA session change: "
                              f"{current_client_lora_session!r} -> {incoming_session!r}")
                        _unapply_client_lora(patcher)
                        if server_lora_path is not None:
                            apply_server_lora(patcher, server_lora_path, server_lora_strength)
                        if incoming_session != "empty" and client_lora_blob:
                            _apply_client_lora(patcher, client_lora_blob, incoming_session, device)
                        current_client_lora_session = incoming_session
                    elif incoming_session == "" and current_client_lora_session is not None:
                        # Client turned forwarding off entirely after having
                        # sent something previously. Unpatch + restore server LoRA.
                        print(f"[server] client LoRA forwarding stopped — unpatching")
                        _unapply_client_lora(patcher)
                        if server_lora_path is not None:
                            apply_server_lora(patcher, server_lora_path, server_lora_strength)
                        current_client_lora_session = None

                    t0 = time.time()
                    img_out, txt_out = forward_back_half(
                        model,
                        img=img, txt=txt, vec=vec, vec_orig=vec_orig, pe=pe,
                        attn_mask=attn_mask,
                    )
                    t_forward = time.time() - t0

                    # Echo the same codec mode + tile_dim that came in for
                    # the img tensor. tile_dim lives in the wire's `extra`
                    # dict; QP isn't transmitted so it defaults to 18.
                    img_in_wire = next(t for t in header["tensors"] if t["name"] == "img")
                    codec_mode = img_in_wire["encoding"]
                    codec_qp = 18
                    codec_lossless = False
                    codec_tile_dim = int(img_in_wire.get("extra", {}).get("tile_dim", 4))

                    t0 = time.time()
                    wire_outs = _encode_response_tensors(
                        img_out, txt_out,
                        codec_mode, codec_qp, codec_lossless, codec_tile_dim,
                    )
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
                    print(f"[server] forward {n_double_blocks}D + {n_single_blocks}S blocks "
                          f"(client said start={client_start_block}): "
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
                   help="How many transformer blocks the server runs. Counts "
                        "doubles first, then singles. e.g. for FLUX.2 Klein 9B "
                        "(8 doubles + 24 singles): N=4 -> last 4 doubles only. "
                        "N=8 -> all doubles. N=9 -> all doubles + first 1 single. "
                        "N=32 -> all doubles + all singles (entire back-half). "
                        "MUST match the client node's `n_blocks_remote` setting. "
                        "If omitted, loads every block (full back-half).")
    p.add_argument("--lora", type=Path, default=None,
                   help="Path to a LoRA safetensors file to apply to the slim "
                        "back-half model. References to layers the server doesn't "
                        "hold (front-half doubles, tail singles, encoders, "
                        "final_layer) are silently dropped.")
    p.add_argument("--lora-strength", type=float, default=1.0,
                   help="LoRA strength multiplier (default 1.0).")
    args = p.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    patcher = load_flux2_klein(args.weights, device, dtype, n_blocks=args.n_blocks)
    if args.lora is not None:
        if not args.lora.is_file():
            raise FileNotFoundError(f"LoRA file not found: {args.lora}")
        apply_server_lora(patcher, args.lora, args.lora_strength)
    serve(
        patcher, args.bind, args.port, device,
        server_lora_path=args.lora,
        server_lora_strength=args.lora_strength,
    )


if __name__ == "__main__":
    main()
