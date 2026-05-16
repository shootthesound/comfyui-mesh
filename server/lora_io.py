"""Serialize / deserialize ComfyUI ModelPatcher.patches dicts via safetensors.

ComfyUI patches dict shape:

    patches = {
        "diffusion_model.double_blocks.4.img_attn.qkv.weight": [
            (strength_patch, ("lora", (mat1, mat2, alpha, dora_scale)), strength_model, offset, function),
            ...
        ],
        ...
    }

Each entry is a 5-tuple. The 2nd element is itself a 2-tuple of
`(type_str, args_tuple)`. `args_tuple` is a flat sequence whose
elements are tensors, scalars, or None.

We store all tensors as safetensors values and use a JSON `_layout`
metadata entry to describe how to reassemble. Pickle is intentionally
NOT used — safetensors guarantees no code execution at load time.

Filter + remap helper handles the slim-model index translation
(client patches reference original block indices; server has slim
indices remapped from drop_db).

Used by mesh_node.py (encode + filter) and mesh_server.py (decode +
add_patches lifecycle).
"""

from __future__ import annotations

import hashlib
import io
import json
from typing import Any

import torch


# ---------------------------------------------------------------------
# Filter + remap a patches dict for the slim model on the server side
# ---------------------------------------------------------------------

def filter_and_remap_patches(
    patches: dict,
    *,
    drop_db: int,
    n_single_remote: int,
    n_double_total: int,
) -> dict:
    """Take the client's full-model patches dict, return only the entries
    targeting layers the server holds, with double_block indices
    remapped from original (`drop_db..n_double_total`) to slim (`0..n_double_remote`).

    Keys we drop:
      - `diffusion_model.double_blocks.{N}.*` where N < drop_db   (front-half, client runs them)
      - `diffusion_model.single_blocks.{N}.*` where N >= n_single_remote  (tail singles, client runs them)
      - `diffusion_model.final_layer.*`           (server doesn't run final_layer)
      - encoder layers (img_in, txt_in, time_in, vector_in, guidance_in,
        pe_embedder, *_modulation*) — kept by the server only for ComfyUI's
        architecture detection; never actually executed in the back-half
        forward path. LoRA on these would be wasted.

    Keys we keep:
      - `diffusion_model.double_blocks.{N}.*` where N >= drop_db
      - `diffusion_model.single_blocks.{N}.*` where N < n_single_remote
    """
    out = {}
    for key, entries in patches.items():
        if not key.startswith("diffusion_model."):
            # Other top-level keys (e.g. unet/clip patches) — not applicable
            # to our diffusion-model-only server.
            continue
        sub = key[len("diffusion_model."):]

        if sub.startswith("double_blocks."):
            try:
                idx = int(sub.split(".")[1])
            except (ValueError, IndexError):
                continue
            if idx < drop_db:
                continue  # front-half block, client handles it
            new_idx = idx - drop_db
            new_sub = f"double_blocks.{new_idx}." + ".".join(sub.split(".")[2:])
            out[f"diffusion_model.{new_sub}"] = entries
            continue

        if sub.startswith("single_blocks."):
            try:
                idx = int(sub.split(".")[1])
            except (ValueError, IndexError):
                continue
            if idx >= n_single_remote:
                continue  # tail single, client handles it
            out[key] = entries  # no remap, indices align
            continue

        # final_layer or encoders — server has the modules but doesn't run
        # them, so a LoRA patch there would be wasted. Drop.
        continue

    return out


def filter_and_remap_patches_ltx(
    patches: dict,
    *,
    drop_n: int,
    n_total: int,
) -> dict:
    """LTX-equivalent of filter_and_remap_patches. LTX uses the
    nested HF-style `diffusion_model.transformer_blocks.{N}.*` key
    prefix (vs FLUX's flat `double_blocks` / `single_blocks`).

    Keys we drop:
      - `diffusion_model.transformer_blocks.{N}.*` where N < drop_n
        (front-half block, the client runs it locally)
      - everything outside the transformer_blocks namespace (encoders,
        embeddings, projection layers, vae) — the server holds those
        modules only for ComfyUI's architecture detection; they're
        never actually executed in the back-half forward.

    Keys we keep + remap:
      - `diffusion_model.transformer_blocks.{N}.*` where N >= drop_n
        → remapped to `diffusion_model.transformer_blocks.{N-drop_n}.*`
        so the slim back-half model's keys (0..n_remote-1) match.
    """
    out = {}
    prefix = "diffusion_model.transformer_blocks."
    for key, entries in patches.items():
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix):]
        try:
            idx = int(rest.split(".")[0])
        except (ValueError, IndexError):
            continue
        if idx < drop_n:
            continue  # front-half block, client handles it locally
        new_idx = idx - drop_n
        tail = ".".join(rest.split(".")[1:])
        out[f"{prefix}{new_idx}.{tail}"] = entries
    return out


# ---------------------------------------------------------------------
# Encode / decode patches dict via safetensors
# ---------------------------------------------------------------------

def _encode_args_seq(seq, tensors_out: dict, prefix: str) -> list:
    """Walk a flat sequence of weight tuple elements and encode each
    as a layout descriptor + (if tensor) safetensors entry."""
    out = []
    for a_idx, arg in enumerate(seq):
        if isinstance(arg, torch.Tensor):
            tname = f"{prefix}_t{a_idx}"
            tensors_out[tname] = arg.detach().cpu().contiguous()
            out.append({"kind": "tensor", "name": tname})
        elif arg is None:
            out.append({"kind": "none"})
        elif isinstance(arg, bool):
            out.append({"kind": "scalar", "value": arg})
        elif isinstance(arg, int):
            out.append({"kind": "scalar", "value": arg})
        elif isinstance(arg, float):
            out.append({"kind": "scalar", "value": arg})
        elif isinstance(arg, str):
            out.append({"kind": "string", "value": arg})
        elif isinstance(arg, (list, tuple)):
            # Nested list (e.g. reshape param for LoRAAdapter)
            out.append({"kind": "list", "value": list(arg)})
        else:
            # Unknown type — store repr for debugging
            out.append({"kind": "_opaque", "repr": repr(arg)[:200]})
    return out


def encode_patches_to_safetensors(patches: dict) -> bytes:
    """Serialize a patches dict to a safetensors-format bytes blob.

    Handles BOTH forms ComfyUI uses in `ModelPatcher.patches`:

    1. WeightAdapterBase instances (LoRAAdapter, LoHaAdapter, LoKrAdapter,
       GLoRAAdapter, OFTAdapter, BOFTAdapter). Each has:
         .name    — short string identifying the adapter class
         .weights — flat tuple of (tensor | scalar | None | list | str)
       All adapter constructors share the same signature
       `__init__(loaded_keys, weights)`, so on the server we just
       look up the class by name and rebuild.

    2. Tuple-form patches: `("diff", (tensor,))`, `("set", (tensor,))`,
       `("model_as_lora", ...)`. Used by ComfyUI's load_lora for the
       w_norm/diff/diff_b/set_weight code paths. Same flat-args
       structure inside the tuple.

    `add_patches` stores entries as the 5-tuple
    `(strength_patch, patch_data, strength_model, offset, function)`
    where `patch_data` is either an adapter instance or a 2-tuple.
    `function` is a callable and not serialized — server reconstructs
    it as None (the standard for non-custom patches).

    Returns b"" if patches is empty.
    """
    from safetensors.torch import save

    if not patches:
        return b""

    tensors: dict[str, torch.Tensor] = {}
    layout_patches: list = []

    for p_idx, (key, entries) in enumerate(sorted(patches.items())):
        layout_entries = []
        for e_idx, entry in enumerate(entries):
            # ComfyUI's add_patches stores entries as
            # (strength_patch, patch_data, strength_model, offset, function)
            # but earlier versions used shorter arities. Pad with defaults.
            try:
                strength_patch = float(entry[0])
            except (IndexError, TypeError, ValueError):
                strength_patch = 1.0
            patch_data = entry[1] if len(entry) > 1 else None
            try:
                strength_model = float(entry[2]) if len(entry) > 2 else 1.0
            except (TypeError, ValueError):
                strength_model = 1.0
            offset = entry[3] if len(entry) > 3 else None
            # entry[4] is `function` (callable) — dropped

            slot_prefix = f"p{p_idx}_e{e_idx}"

            # Branch on patch_data form
            if hasattr(patch_data, "name") and hasattr(patch_data, "weights"):
                # WeightAdapterBase instance
                adapter_name = patch_data.name
                weights_seq = list(patch_data.weights)
                args_layout = _encode_args_seq(weights_seq, tensors, slot_prefix)
                patch_layout = {
                    "form": "adapter",
                    "name": adapter_name,
                    "args": args_layout,
                }
            elif isinstance(patch_data, tuple) and len(patch_data) == 2 and isinstance(patch_data[0], str):
                # Legacy ("type_str", args_tuple) form
                ptype, pargs = patch_data
                if not isinstance(pargs, (tuple, list)):
                    layout_entries.append({"_skipped": True, "reason": "bad pargs shape"})
                    continue
                args_layout = _encode_args_seq(pargs, tensors, slot_prefix)
                patch_layout = {
                    "form": "tuple",
                    "type": ptype,
                    "args": args_layout,
                }
            else:
                layout_entries.append({
                    "_skipped": True,
                    "reason": f"unknown patch_data type {type(patch_data).__name__}",
                })
                continue

            offset_repr = None
            if offset is not None:
                if isinstance(offset, int):
                    offset_repr = {"kind": "int", "value": offset}
                elif isinstance(offset, (tuple, list)):
                    offset_repr = {"kind": "list", "value": list(offset)}
                else:
                    offset_repr = {"kind": "_opaque", "repr": repr(offset)[:200]}

            layout_entries.append({
                "strength_patch": strength_patch,
                "patch": patch_layout,
                "strength_model": strength_model,
                "offset": offset_repr,
            })

        if layout_entries:
            layout_patches.append({"key": key, "entries": layout_entries})

    layout = {"version": 2, "patches": layout_patches}
    metadata = {"_layout": json.dumps(layout, separators=(",", ":"))}

    if not tensors:
        # safetensors requires at least one tensor
        tensors["_empty_marker"] = torch.zeros(1, dtype=torch.float32)

    return save(tensors, metadata=metadata)


def _decode_args_seq(layout_args: list, tensors: dict) -> list:
    """Inverse of _encode_args_seq."""
    out = []
    for a in layout_args:
        kind = a["kind"]
        if kind == "tensor":
            out.append(tensors[a["name"]])
        elif kind == "none":
            out.append(None)
        elif kind == "scalar":
            out.append(a["value"])
        elif kind == "string":
            out.append(a["value"])
        elif kind == "list":
            out.append(list(a["value"]))
        else:
            # _opaque or unknown — best we can do is None
            out.append(None)
    return out


def decode_patches_from_safetensors(blob: bytes, device) -> dict:
    """Inverse of encode_patches_to_safetensors. Returns a patches dict
    suitable for `patcher.add_patches(patches, strength_patch=1.0)`.

    Adapter-form patches are reconstructed by looking up the adapter
    class in `comfy.weight_adapter.adapters` by `.name` and instantiating
    with `(loaded_keys=set(), weights=tuple(args))`. Tuple-form patches
    are rebuilt as `(type_str, tuple(args))`.

    Strengths are already baked into each entry's `strength_patch` field;
    callers pass `1.0` to add_patches so the embedded strengths apply.
    """
    if not blob:
        return {}

    from safetensors.torch import load as st_load
    raw_tensors = st_load(blob)

    # safetensors.torch.load doesn't expose metadata. Parse the header
    # JSON ourselves — the spec puts it in the first 8 bytes (u64 LE
    # header size) followed by header_size bytes of UTF-8 JSON, with
    # the optional `__metadata__` key inside.
    import struct as _struct
    header_size = _struct.unpack("<Q", blob[:8])[0]
    header = json.loads(blob[8:8 + header_size].decode("utf-8"))
    metadata = header.get("__metadata__", {}) or {}

    layout_str = metadata.get("_layout")
    if not layout_str:
        raise ValueError("decode_patches_from_safetensors: missing _layout metadata")
    layout = json.loads(layout_str)

    if layout.get("version") != 2:
        raise ValueError(
            f"unsupported lora_io layout version: {layout.get('version')} "
            f"(expected 2). Client and server are out of sync."
        )

    # Move tensors to target device once
    raw_tensors = {k: v.to(device) for k, v in raw_tensors.items() if k != "_empty_marker"}

    # Build adapter-class lookup table (by .name)
    adapter_class_by_name: dict = {}
    try:
        from comfy.weight_adapter import adapters as _adapters
        for cls in _adapters:
            adapter_class_by_name[cls.name] = cls
    except Exception:
        # If comfy.weight_adapter isn't importable for some reason, we
        # can still handle tuple-form patches; adapter-form ones will
        # fall through with a warning.
        pass

    out: dict = {}
    for p in layout["patches"]:
        key = p["key"]
        entries = []
        for e in p["entries"]:
            if e.get("_skipped"):
                continue

            patch_layout = e["patch"]
            args_out = _decode_args_seq(patch_layout["args"], raw_tensors)

            if patch_layout["form"] == "adapter":
                cls = adapter_class_by_name.get(patch_layout["name"])
                if cls is None:
                    import logging
                    logging.warning(
                        f"lora_io: adapter type {patch_layout['name']!r} not "
                        f"available in this ComfyUI; dropping patch on {key}"
                    )
                    continue
                patch_data = cls(loaded_keys=set(), weights=tuple(args_out))
            elif patch_layout["form"] == "tuple":
                patch_data = (patch_layout["type"], tuple(args_out))
            else:
                continue

            offset_repr = e.get("offset")
            if offset_repr is None:
                offset = None
            elif offset_repr["kind"] == "int":
                offset = int(offset_repr["value"])
            elif offset_repr["kind"] == "list":
                offset = tuple(offset_repr["value"])
            else:
                offset = None

            entries.append((
                float(e["strength_patch"]),
                patch_data,
                float(e["strength_model"]),
                offset,
                None,  # function — never serialized
            ))

        if entries:
            out[key] = entries

    return out


# ---------------------------------------------------------------------
# Cheap session-id hash for change detection
# ---------------------------------------------------------------------

def patches_session_id(patches: dict) -> str:
    """A fast, deterministic id for a patches dict. Used for client-side
    change detection: 'have I shipped THIS exact set of patches to the
    server already?'. Hashes layout (keys, names/types, strengths,
    tensor shapes/dtypes) — not tensor data — which catches:
      - different LoRA file (different keys / shapes / strengths)
      - same LoRA at a different strength (different strength_patch)
      - LoRA added or removed (different key set)
    Misses only the pathological case of two distinct LoRAs with
    identical structure and identical strengths but different tensor
    values — which doesn't happen in practice.

    Handles both adapter-instance patches and tuple-form patches.
    Returns a 16-char hex string.
    """
    h = hashlib.sha256()
    if not patches:
        return "empty"
    for key in sorted(patches.keys()):
        h.update(key.encode("utf-8"))
        h.update(b"|")
        for entry in patches[key]:
            try:
                strength_patch = entry[0]
                patch_data = entry[1]
                strength_model = entry[2] if len(entry) > 2 else 1.0
            except Exception:
                continue
            # Identify adapter-vs-tuple form + extract its identifier and args
            if hasattr(patch_data, "name") and hasattr(patch_data, "weights"):
                ptype = f"A:{patch_data.name}"
                pargs = patch_data.weights
            elif isinstance(patch_data, tuple) and len(patch_data) == 2:
                ptype = f"T:{patch_data[0]}"
                pargs = patch_data[1] if isinstance(patch_data[1], (tuple, list)) else ()
            else:
                ptype = "?"
                pargs = ()
            h.update(f"{strength_patch}|{strength_model}|{ptype}|".encode("utf-8"))
            for a in pargs:
                if isinstance(a, torch.Tensor):
                    h.update(f"T{tuple(a.shape)}{a.dtype}".encode("utf-8"))
                elif a is None:
                    h.update(b"N")
                else:
                    h.update(f"S{a!r}".encode("utf-8"))
            h.update(b";")
        h.update(b"|")
    return h.hexdigest()[:16]
