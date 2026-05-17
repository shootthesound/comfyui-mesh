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
    str(HERE / "ComfyUI"),     # what install.bat clones
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
# Model loading — LTXAV (audio+video) variant
#
# LTX checkpoints use the HF-style nested prefix:
#   model.diffusion_model.transformer_blocks.{N}.attn1.to_k.weight
# (vs FLUX's bare double_blocks.{N}.img_attn.qkv.weight). Both go through
# ComfyUI's comfy.sd.load_diffusion_model_state_dict the same way, but
# the slim-load + fp8 metadata remap need to preserve the LTX prefix.
#
# LTXAV vs LTXV detection: every block in LTXAV has audio_attn1 / audio_*
# substructures alongside the video attn1/attn2. LTXV blocks don't. We
# raise if it's LTXV (separate iteration).
# ---------------------------------------------------------------------

_LTX_BLOCK_PREFIX = "model.diffusion_model.transformer_blocks."
_LTX_DIFFUSION_PREFIX = "model.diffusion_model."
_LTX_AV_MARKER = "audio_attn1"  # exists only in LTXAV BasicTransformerBlock


def load_ltx_av(
    weights_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    n_blocks: int | None = None,
):
    """Slim-load the back-half of an LTX-AV (audio+video) checkpoint.

    `n_blocks` counts transformer_blocks (a flat list — 48 for the
    22B Dev model). Loads the LAST n_blocks blocks; client runs the
    first (total - n_blocks).

    Reads via safetensors.safe_open() so disk I/O is exactly the bytes
    we keep — no full-model spike in CPU RAM or VRAM at any point.

    Non-block components under model.diffusion_model.* (audio /
    video embeddings connectors, *_adaln_single, av_ca_* modulations,
    *_patchify_proj, *_proj_out) are kept (~hundreds of MB combined)
    because ComfyUI's model detector needs them to identify the
    architecture as LTXAV (vs LTXV — the AV variant has additional
    audio_* heads + av_ca_* modulations). They sit unused on the
    server side but cost relatively little vs the full model.

    Top-level audio_vae / vocoder / vae / text_embedding_projection
    are DROPPED — they run on the client (or via separate ComfyUI
    nodes: LTXVAudioVAELoader, LTXAVTextEncoderLoader).
    """
    import json
    from safetensors import safe_open
    import comfy.sd
    import comfy.model_management

    print(f"[server] reading checkpoint header from {weights_path}")
    with safe_open(str(weights_path), framework="pt", device="cpu") as f:
        all_keys = list(f.keys())
        metadata = f.metadata() or {}

        # 1. Detect block count + AV variant from key names.
        block_indices = set()
        is_av = False
        for k in all_keys:
            if k.startswith(_LTX_BLOCK_PREFIX):
                rest = k[len(_LTX_BLOCK_PREFIX):]
                try:
                    idx = int(rest.split(".")[0])
                except ValueError:
                    continue
                block_indices.add(idx)
                if _LTX_AV_MARKER in rest:
                    is_av = True

        total_blocks = (max(block_indices) + 1) if block_indices else 0
        if total_blocks == 0:
            raise RuntimeError(
                f"no model.diffusion_model.transformer_blocks.* tensors found "
                f"in {weights_path} — is this really an LTX safetensors file?"
            )
        if not is_av:
            raise RuntimeError(
                "this checkpoint appears to be LTXV (video-only). The current "
                "Daedalus LTX implementation targets LTXAV (audio+video, e.g. "
                "ltx-2.3-22b-dev). LTXV support is a separate iteration."
            )

        if n_blocks is None or n_blocks >= total_blocks:
            n_remote = total_blocks
        elif n_blocks <= 0:
            n_remote = total_blocks
        else:
            n_remote = n_blocks
        drop_first = total_blocks - n_remote

        print(f"[server] checkpoint: LTXAV with {total_blocks} transformer_blocks")
        print(f"[server] loading last {n_remote} blocks (skipping first "
              f"{drop_first}); will be remapped to slim indices 0..{n_remote - 1}")

        # 2. Build the slim state dict.
        sd_slim = {}
        for k in all_keys:
            if k.startswith(_LTX_BLOCK_PREFIX):
                rest = k[len(_LTX_BLOCK_PREFIX):]
                idx = int(rest.split(".")[0])
                if idx < drop_first:
                    continue  # front-half block, client runs it
                tail = ".".join(rest.split(".")[1:])
                new_key = f"{_LTX_BLOCK_PREFIX}{idx - drop_first}.{tail}"
                sd_slim[new_key] = f.get_tensor(k)
                continue
            if k.startswith(_LTX_DIFFUSION_PREFIX):
                # Non-block diffusion_model.* components — keep for arch detection
                sd_slim[k] = f.get_tensor(k)
                continue
            # Else: top-level vae / vocoder / audio_vae / text_embedding_projection
            # — drop. The client owns these via separate ComfyUI nodes.

        slim_bytes = sum(t.numel() * t.element_size() for t in sd_slim.values())
        print(f"[server] slim state dict: {len(sd_slim)} tensors, "
              f"{slim_bytes/1024/1024/1024:.2f} GB "
              f"(full model on disk: ~28 GB)")

        # 3a. Patch the `config` metadata so ComfyUI's LTX detector builds
        #     a slim model with N transformer_blocks instead of the full 48.
        #     Without this, count_blocks correctly returns our slim count
        #     but then `dit_config.update(json.loads(metadata["config"])
        #     .get("transformer", {}))` (see comfy/model_detection.py
        #     around the ltxv/ltxav branch) clobbers num_layers back to 48
        #     — the checkpoint's canonical full-model count — and ComfyUI
        #     proceeds to construct a 48-block model, then warns about
        #     missing weights for blocks {n_remote..47}.
        if "config" in metadata:
            cfg = json.loads(metadata["config"])
            if "transformer" in cfg and cfg["transformer"].get("num_layers") != n_remote:
                old_n = cfg["transformer"].get("num_layers")
                cfg["transformer"]["num_layers"] = n_remote
                metadata = dict(metadata)
                metadata["config"] = json.dumps(cfg)
                print(f"[server] patched config.transformer.num_layers "
                      f"{old_n} -> {n_remote} so ComfyUI builds a slim model")

        # 3b. Remap fp8 quantization metadata the same way.
        if "_quantization_metadata" in metadata:
            qm = json.loads(metadata["_quantization_metadata"])
            layers = qm.get("layers", {})
            new_layers = {}
            for layer_name, cfg in layers.items():
                if layer_name.startswith(_LTX_BLOCK_PREFIX):
                    rest = layer_name[len(_LTX_BLOCK_PREFIX):]
                    idx = int(rest.split(".")[0])
                    if idx < drop_first:
                        continue
                    tail = ".".join(rest.split(".")[1:])
                    new_layers[f"{_LTX_BLOCK_PREFIX}{idx - drop_first}.{tail}"] = cfg
                    continue
                if layer_name.startswith(_LTX_DIFFUSION_PREFIX):
                    new_layers[layer_name] = cfg
                    continue
                # Drop top-level vae/vocoder/etc fp8 entries
            qm["layers"] = new_layers
            metadata = dict(metadata)
            metadata["_quantization_metadata"] = json.dumps(qm)
            print(f"[server] remapped {len(layers)} fp8 layer entries "
                  f"-> {len(new_layers)} for slim model")

    # 4. Hand the slim SD to ComfyUI's loader.
    print(f"[server] handing slim state dict to comfy.sd.load_diffusion_model_state_dict")
    model_options = {"dtype": dtype}
    patcher = comfy.sd.load_diffusion_model_state_dict(
        sd_slim, model_options=model_options, metadata=metadata,
    )
    if patcher is None:
        raise RuntimeError(
            "comfy.sd.load_diffusion_model_state_dict returned None for the slim "
            "state dict. Likely a detection failure — check the checkpoint is an "
            "LTX 22B AV safetensors file and that the local ComfyUI version "
            "supports LTX (comfy/ldm/lightricks/av_model.py)."
        )
    sd_slim.clear()
    comfy.model_management.load_models_gpu([patcher], force_full_load=True)

    diffusion = patcher.model.diffusion_model
    n_loaded = len(diffusion.transformer_blocks)
    print(f"[server] LTXAV model loaded; transformer_blocks={n_loaded} "
          f"class={type(diffusion).__name__}")

    patcher._mesh_slim_meta = {
        "drop_first": drop_first,
        "total_blocks": total_blocks,
        "variant": "ltxav",
    }
    return patcher


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

    # Dispatch on the slim-load variant. FLUX uses double_blocks /
    # single_blocks and stashes drop_db; LTX(AV) uses transformer_blocks
    # and stashes drop_first. The remap logic is the same in shape: find
    # every key_map entry that references a slim block index, add a
    # parallel entry pointing at original_idx = slim_idx + drop.
    variant = slim_meta.get("variant", "flux")
    if variant == "ltxav":
        drop_n = slim_meta["drop_first"]
        # LTX LoRA key shapes seen in the wild:
        #   kohya:     transformer_blocks_{N}_<rest>
        #   native HF: transformer_blocks.{N}.<rest>
        block_patterns = (
            (re.compile(r"(transformer_blocks_)(\d+)(_)"), "_"),
            (re.compile(r"(transformer_blocks\.)(\d+)(\.)"), "."),
        )
    else:
        drop_n = slim_meta["drop_db"]
        block_patterns = (
            (re.compile(r"(double_blocks_)(\d+)(_)"), "_"),
            (re.compile(r"(double_blocks\.)(\d+)(\.)"), "."),
        )

    print(f"[server] loading LoRA: {lora_path} (strength={strength})")
    lora_sd = comfy.utils.load_torch_file(str(lora_path), safe_load=True)
    print(f"[server] LoRA has {len(lora_sd)} tensors")

    # Build the slim model's natural key_map
    key_map = comfy.lora.model_lora_keys_unet(patcher.model, {})

    # Add aliases mapping ORIGINAL block indices to slim model targets.
    new_aliases = {}
    if drop_n > 0:
        for lora_key, model_key in key_map.items():
            for pat, sep in block_patterns:
                m = pat.search(lora_key)
                if m:
                    slim_idx = int(m.group(2))
                    orig_idx = slim_idx + drop_n
                    aliased = pat.sub(f"{m.group(1)}{orig_idx}{sep}", lora_key, count=1)
                    new_aliases[aliased] = model_key
                    break  # one pattern match per key
    if new_aliases:
        print(f"[server] added {len(new_aliases)} key-map aliases "
              f"(orig idx -> slim idx, variant={variant}, drop={drop_n})")
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
# Forward pass + wire path — LTX-AV
# ---------------------------------------------------------------------

@torch.no_grad()
def forward_back_half_ltx(model, payload: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Run all of the server's loaded `transformer_blocks` for the LTX-AV
    back-half. `payload` is the reconstructed block_wrap args dict (see
    payload_ltx.reconstruct_payload). Returns the updated (vx, ax).

    transformer_options inside payload is the minimal flag-only dict
    rebuilt from the wire metadata — enough for the block forward's
    `.get(run_vx, True)` lookups but stripped of ComfyUI hooks.
    """
    vx, ax = payload["img"]
    kwargs = {k: v for k, v in payload.items() if k != "img"}
    for block in model.transformer_blocks:
        vx, ax = block((vx, ax), **kwargs)
    return vx, ax


def _decode_request_tensors_ltx(
    header: dict,
    blobs: list[bytes],
    device: torch.device,
    constants_cache: dict,
):
    """Decode an LTX `forward_ltx_blocks` request, transparently
    merging cached constants when the client said `constants_shipped=
    False` and the session id matches.

    `constants_cache` is a mutable dict on the calling serve_ltx loop
    of shape `{"session_id": str, "by_name": {name: tensor}}`. We
    update it in place when a new full payload arrives (so subsequent
    cache-hit calls can satisfy the missing tensors).

    Returns (payload_dict_ready_for_block_forward, client_lora_blob_or_None).
    """
    import payload_ltx

    wires = header["tensors"]
    by_name: dict[str, torch.Tensor] = {}
    client_lora_blob: bytes | None = None
    for w, b in zip(wires, blobs):
        if w.get("encoding") == "lora_safetensors":
            client_lora_blob = b
            continue
        name = w["name"]
        by_name[name] = codec.decode(w, b, device=device)

    incoming_session = header.get("constants_session_id", "") or ""
    constants_shipped = bool(header.get("constants_shipped", True))

    if constants_shipped:
        # Full payload — refresh the cache with the constants subset
        # of what was shipped. Per-timestep tensors stay in by_name as-is.
        new_consts = {n: t for n, t in by_name.items()
                      if n in payload_ltx.CONSTANT_WIRE_NAMES}
        constants_cache["session_id"] = incoming_session
        constants_cache["by_name"] = new_consts
    else:
        # Cache-hit expected. Validate that the client's session id
        # matches what we have. Mismatch = client thinks we have it
        # cached but we don't (e.g. server restarted) — raise so the
        # client's reconnect path forces a re-ship.
        cached_session = constants_cache.get("session_id", "")
        if incoming_session != cached_session:
            raise RuntimeError(
                f"constants cache miss: client wants session "
                f"{incoming_session!r}, server has {cached_session!r}; "
                f"client must retry with full payload"
            )
        # Merge cached constants in (caller-owned dict; merge by
        # adding to by_name so reconstruct sees the full set).
        for name, t in constants_cache["by_name"].items():
            by_name.setdefault(name, t)

    ltx_meta = header.get("ltx_meta", {}) or {}
    payload = payload_ltx.reconstruct_payload(ltx_meta, by_name)
    return payload, client_lora_blob


def _encode_response_tensors_ltx(
    vx: torch.Tensor,
    ax: torch.Tensor,
    codec_mode: str = "raw",
    codec_qp: int = 18,
    codec_lossless: bool = False,
    codec_tile_dim: int = 4,
):
    """Encode the LTX forward response. vx/ax go through whatever codec
    the client used on the request (echoed from the incoming vx wire
    descriptor by the caller)."""
    return [
        codec.encode("vx", vx, mode=codec_mode, qp=codec_qp,
                     lossless=codec_lossless, tile_dim=codec_tile_dim),
        codec.encode("ax", ax, mode=codec_mode, qp=codec_qp,
                     lossless=codec_lossless, tile_dim=codec_tile_dim),
    ]


def serve_ltx(patcher, host: str, port: int, device: torch.device,
              server_lora_path: Path = None, server_lora_strength: float = 1.0,
              server_lora2_path: Path = None, server_lora2_strength: float = 0.5):
    """LTX-AV TCP loop. Mirrors the FLUX `serve()` shape but dispatches
    on `forward_ltx_blocks` and uses the LTX block forward signature.
    Reuses the FLUX hello / reconfigure messages — they're protocol-
    level and don't care about the variant.

    Two server-side LoRA slots: the primary `--lora` (typical style /
    character LoRA) and the secondary `--lora2` (typically the LTX
    distilled LoRA). Both get re-applied after any client-LoRA unpatch
    so a forwarding client doesn't accidentally drop the server's
    static LoRAs.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(1)

    model = patcher.model.diffusion_model
    n_loaded = len(model.transformer_blocks)
    print(f"[server] READY — listening on {host}:{port} "
          f"(LTX-AV, n_blocks={n_loaded} transformer_blocks)")

    current_client_lora_session: str | None = None
    # Per-connection cache of constant tensors (text contexts, PE pairs,
    # attention masks). Reset on every new accept() so a freshly-
    # reconnecting client always re-ships on its first call (matches
    # the client's `_last_shipped_constants_session` reset on close()).
    constants_cache: dict = {"session_id": "", "by_name": {}}

    while True:
        conn, addr = s.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[server] client connected: {addr}")
        constants_cache = {"session_id": "", "by_name": {}}
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
                            "variant": "ltxav",
                            "n_transformer_blocks": n_loaded,
                            "n_blocks": n_loaded,
                            "n_total_loaded": n_loaded,
                        },
                    }, [])

                elif kind == "reconfigure":
                    new_n_blocks = int(header.get("n_blocks", n_loaded))
                    # No-op guard: if the requested value matches what's
                    # already loaded, ack and stay alive instead of
                    # triggering a full server restart. Catches the case
                    # where the user re-confirms an unchanged value in
                    # the GUI, or where the client's bookkeeping has
                    # drifted but the server is actually correct.
                    if new_n_blocks == n_loaded:
                        print(f"[server] reconfigure no-op: --n-blocks "
                              f"already {n_loaded}, staying alive")
                        protocol.send_message(conn, {
                            "kind": "reconfigure_ack",
                            "tensors": [],
                            "new_n_blocks": new_n_blocks,
                            "no_op": True,
                        }, [])
                        continue
                    print(f"[server] RESTARTING: reconfigure request "
                          f"--n-blocks {n_loaded} -> {new_n_blocks}")
                    if new_n_blocks < n_loaded:
                        print(
                            "[server] *** NOTE *** decreasing n_blocks requires the "
                            "CLIENT to restart ComfyUI too — stripped client weights "
                            "are session-scoped and only reload from disk at startup."
                        )
                    protocol.send_message(conn, {
                        "kind": "reconfigure_ack",
                        "tensors": [],
                        "new_n_blocks": new_n_blocks,
                    }, [])
                    try: conn.close()
                    except Exception: pass
                    try: s.close()
                    except Exception: pass
                    handoff = Path(__file__).parent / "mesh_server_ltx_reconfig.tmp"
                    try:
                        handoff.write_text(
                            json.dumps({"n_blocks": new_n_blocks}),
                            encoding="utf-8",
                        )
                        print(f"[server] wrote {handoff.name} for GUI relaunch")
                    except Exception as e:
                        print(f"[server] could not write {handoff.name}: {e}")
                    sys.stdout.flush()
                    sys.exit(0)

                elif kind == "forward_ltx_blocks":
                    client_start_block = int(header.get("start_block", 0))
                    t0 = time.time()
                    payload, client_lora_blob = _decode_request_tensors_ltx(
                        header, blobs, device, constants_cache,
                    )
                    t_decode = time.time() - t0

                    incoming_session = header.get("client_lora_session", "") or ""
                    if incoming_session and incoming_session != current_client_lora_session:
                        print(f"[server] client LoRA session change: "
                              f"{current_client_lora_session!r} -> {incoming_session!r}")
                        _unapply_client_lora(patcher)
                        if server_lora_path is not None:
                            apply_server_lora(patcher, server_lora_path, server_lora_strength)
                        if server_lora2_path is not None:
                            apply_server_lora(patcher, server_lora2_path, server_lora2_strength)
                        if incoming_session != "empty" and client_lora_blob:
                            _apply_client_lora(patcher, client_lora_blob, incoming_session, device)
                        current_client_lora_session = incoming_session
                    elif incoming_session == "" and current_client_lora_session is not None:
                        print(f"[server] client LoRA forwarding stopped — unpatching")
                        _unapply_client_lora(patcher)
                        if server_lora_path is not None:
                            apply_server_lora(patcher, server_lora_path, server_lora_strength)
                        if server_lora2_path is not None:
                            apply_server_lora(patcher, server_lora2_path, server_lora2_strength)
                        current_client_lora_session = None

                    t0 = time.time()
                    vx_out, ax_out = forward_back_half_ltx(model, payload)
                    t_forward = time.time() - t0

                    # Echo the codec the client used for vx so the response
                    # round-trips through the same codec settings. vx is
                    # always shipped (per-timestep), so it's always present
                    # in header["tensors"] regardless of constants_shipped.
                    vx_in_wire = next(t for t in header["tensors"] if t["name"] == "vx")
                    resp_codec_mode = vx_in_wire["encoding"]
                    resp_codec_qp = 18
                    resp_codec_lossless = bool(vx_in_wire.get("extra", {}).get("lossless", False))
                    resp_codec_tile_dim = int(vx_in_wire.get("extra", {}).get("tile_dim", 4))

                    t0 = time.time()
                    wire_outs = _encode_response_tensors_ltx(
                        vx_out, ax_out,
                        codec_mode=resp_codec_mode,
                        codec_qp=resp_codec_qp,
                        codec_lossless=resp_codec_lossless,
                        codec_tile_dim=resp_codec_tile_dim,
                    )
                    t_encode = time.time() - t0

                    resp_header = {
                        "kind": "forward_ltx_blocks_response",
                        "tensors": [w.to_header() for w in wire_outs],
                        "timings_ms": {
                            "decode": t_decode * 1000,
                            "forward": t_forward * 1000,
                            "encode": t_encode * 1000,
                        },
                    }
                    protocol.send_message(conn, resp_header, [w.bytes_payload for w in wire_outs])
                    cache_state = "shipped" if header.get("constants_shipped", True) else "cached"
                    print(f"[server] forward LTX {n_loaded} blocks "
                          f"(client said start={client_start_block}, constants={cache_state}): "
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


# ---------------------------------------------------------------------
# Forward pass — FLUX back-half (double_blocks + optional single_blocks)
# Kept in this file from the duplicate; unused by the LTX serve loop
# but referenced by `serve()` below if the FLUX path is ever needed.
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
                    if new_n_blocks < n_total_loaded:
                        # Heads-up for the user: server-side reconfigure
                        # works either direction, but the CLIENT's
                        # in-place strip is one-way for the session.
                        # Decreasing n_blocks_remote on the client side
                        # requires re-reading the model weights from
                        # disk, which only happens at ComfyUI startup.
                        print(
                            "[server] *** NOTE *** decreasing n_blocks "
                            "requires the CLIENT to restart ComfyUI too — "
                            "the client-side stripped block weights are "
                            "gone for the session and can only be reloaded "
                            "from disk by a fresh ComfyUI launch."
                        )
                    protocol.send_message(conn, {
                        "kind": "reconfigure_ack",
                        "tensors": [],
                        "new_n_blocks": new_n_blocks,
                    }, [])
                    try: conn.close()
                    except Exception: pass
                    try: s.close()
                    except Exception: pass

                    handoff = Path(__file__).parent / "mesh_server_ltx_reconfig.tmp"
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


def _wait_for_vram(device: torch.device,
                   max_wait_seconds: float = 60.0,
                   poll_interval: float = 2.0) -> None:
    """Wait for VRAM on `device` to be reasonably free before loading.

    Addresses the common Linux-after-crash pattern: a previous server
    process crashed, leaving the CUDA context's allocations stranded
    in VRAM (driver hasn't reclaimed them yet). Without this probe,
    the next load_ltx_av() crashes mid-load with an OOM stack trace,
    the user manually retries, and the second attempt works because
    by then the kernel has cleaned up. With this probe, the server
    surfaces a clear "waiting..." message and proceeds once VRAM is
    ready (or after max_wait_seconds, in case the user really did
    have something else legitimately holding it).
    """
    if device.type != "cuda":
        return
    try:
        _free, total_bytes = torch.cuda.mem_get_info(device)
    except Exception:
        return
    # Heuristic: require at least 30% of the card's VRAM free OR 4 GB,
    # whichever is larger. Catches "nearly all VRAM is allocated" cleanly
    # without needing to know the exact model size up front.
    required_bytes = max(int(4 * 1024**3), int(0.30 * total_bytes))
    total_gb = total_bytes / 1024**3
    required_gb = required_bytes / 1024**3
    deadline = time.time() + max_wait_seconds
    waited = False
    while True:
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
        except Exception:
            return
        free_gb = free_bytes / 1024**3
        if free_bytes >= required_bytes:
            if waited:
                print(f"[server] VRAM ready: {free_gb:.1f} / {total_gb:.1f} GB "
                      f"free on {device}, proceeding with model load")
            return
        if time.time() >= deadline:
            print(f"[server] *** WARNING *** still only {free_gb:.1f} GB free "
                  f"on {device} after {max_wait_seconds:.0f}s wait "
                  f"(need ~{required_gb:.1f} GB); proceeding anyway — the "
                  f"load may OOM if a previous crash left orphaned VRAM")
            return
        if not waited:
            print(f"[server] only {free_gb:.1f} / {total_gb:.1f} GB free on "
                  f"{device} — need ~{required_gb:.1f} GB. Waiting (a previous "
                  f"server crash may have left orphaned VRAM that the driver "
                  f"hasn't reclaimed yet)...")
            waited = True
        time.sleep(poll_interval)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True,
                   help="Path to an LTX safetensors file (e.g. ltx-2.3-22b-dev-fp8.safetensors)")
    p.add_argument("--bind", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=7777)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--n-blocks", type=int, default=None,
                   help="How many transformer_blocks the server runs. LTX is a "
                        "flat transformer (no double/single split): "
                        "LTXAV 22B has 48 blocks total. N=24 -> last 24 blocks. "
                        "N=12 -> last 12 blocks. MUST match the Icarus LTX "
                        "node's n_blocks_remote setting. If omitted, loads "
                        "every block (full back-half).")
    p.add_argument("--lora", type=Path, default=None,
                   help="Path to a LoRA safetensors file to apply to the slim "
                        "back-half model. References to layers the server "
                        "doesn't hold (front-half blocks, encoders, etc.) are "
                        "silently dropped via ComfyUI's standard pipeline.")
    p.add_argument("--lora-strength", type=float, default=1.0,
                   help="LoRA strength multiplier (default 1.0).")
    p.add_argument("--lora2", type=Path, default=None,
                   help="Path to a SECOND LoRA safetensors file (intended for "
                        "the LTX 2.3 distilled LoRA). Applied after --lora; "
                        "both stack additively via ComfyUI's standard patch "
                        "pipeline.")
    p.add_argument("--lora2-strength", type=float, default=0.5,
                   help="Second LoRA strength multiplier (default 0.5, the "
                        "typical strength for the LTX distilled LoRA).")
    args = p.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    _wait_for_vram(device)
    patcher = load_ltx_av(args.weights, device, dtype, n_blocks=args.n_blocks)
    if args.lora is not None:
        if not args.lora.is_file():
            raise FileNotFoundError(f"LoRA file not found: {args.lora}")
        apply_server_lora(patcher, args.lora, args.lora_strength)
    if args.lora2 is not None:
        if not args.lora2.is_file():
            raise FileNotFoundError(f"Distill LoRA file not found: {args.lora2}")
        apply_server_lora(patcher, args.lora2, args.lora2_strength)

    serve_ltx(
        patcher, args.bind, args.port, device,
        server_lora_path=args.lora,
        server_lora_strength=args.lora_strength,
        server_lora2_path=args.lora2,
        server_lora2_strength=args.lora2_strength,
    )


if __name__ == "__main__":
    main()
