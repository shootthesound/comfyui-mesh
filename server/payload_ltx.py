"""Helpers to flatten and reconstruct the LTX-AV per-block payload for
the wire protocol.

ComfyUI's `transformer_options["patches_replace"]["dit"][("double_block", i)]`
mechanism (yes, the key is still `"double_block"` for LTX even though
LTX-AV uses `transformer_blocks` — that's just ComfyUI's naming) invokes
our callback with this `args` dict per block, per timestep:

    args = {
        "img": (vx, ax),                              # tuple of 2 tensors
        "v_context": <tensor>,                         # text cross-attn ctx (video)
        "a_context": <tensor>,                         # text cross-attn ctx (audio)
        "attention_mask": <tensor | None>,             # padding mask
        "v_timestep": <CompressedTimestep>,            # per-frame video t embedding
        "a_timestep": <CompressedTimestep>,            # per-frame audio t embedding
        "v_pe": <tensor>,                              # rotary freqs (video self)
        "a_pe": <tensor>,                              # rotary freqs (audio self)
        "v_cross_pe": <tensor>,                        # rotary freqs (v cross)
        "a_cross_pe": <tensor>,                        # rotary freqs (a cross)
        "v_cross_scale_shift_timestep": <CompressedTimestep>,
        "a_cross_scale_shift_timestep": <CompressedTimestep>,
        "v_cross_gate_timestep": <CompressedTimestep>,
        "a_cross_gate_timestep": <CompressedTimestep>,
        "transformer_options": <dict>,                 # NOT shipped (see below)
        "self_attention_mask": <tensor | None>,        # optional
        "v_prompt_timestep": <CompressedTimestep | None>,
        "a_prompt_timestep": <CompressedTimestep | None>,
    }

`transformer_options` is mostly ComfyUI bookkeeping (sampler hooks,
attention precision, model_options snapshot) and isn't safe to ship as
JSON. We extract the four flags the block forward actually reads
(`run_vx`, `run_ax`, `a2v_cross_attn`, `v2a_cross_attn`) into the
header `meta` and reconstruct a minimal dict on the server. Default
True if absent (matches the block's `transformer_options.get(...,
True)` lookups).

Wire format for each call:
    header = {
        "kind": "forward_ltx_blocks",
        "tensors": [<wire descriptor for each shipped tensor>],
        "ltx_meta": {
            "compressed": {<name>: patches_per_frame, ...},
            "none_keys": [<name>, ...],
            "flags": {"run_vx": bool, "run_ax": bool,
                      "a2v_cross_attn": bool, "v2a_cross_attn": bool},
        },
        "start_block": int,
    }

This module is byte-identical between client (`comfyui-mesh/`) and
server (`comfyui-mesh/server/`) per the project rule that
wire-contract files must stay in lockstep — DO NOT edit one without
mirroring to the other.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch


# Names of the keys we transit (besides the `img` tuple which is split
# into "vx" and "ax"). Order is canonical so we can iterate
# deterministically in both directions.
_TENSOR_KEYS = (
    "v_context", "a_context",
    "attention_mask",
    "self_attention_mask",
)

# Rotary positional-embedding keys. Each is a 3-tuple
# (cos_freq, sin_freq, split_mode) produced by
# LTXVModel._precompute_freqs_cis: two tensors + a bool flag indicating
# whether the rope is in split or interleaved mode. We ship the two
# tensors named `{key}.cos` / `{key}.sin` and stash the bool in meta.
_PE_KEYS = (
    "v_pe", "a_pe",
    "v_cross_pe", "a_cross_pe",
)

# Keys that hold CompressedTimestep objects.
_COMPRESSED_KEYS = (
    "v_timestep", "a_timestep",
    "v_cross_scale_shift_timestep", "a_cross_scale_shift_timestep",
    "v_cross_gate_timestep", "a_cross_gate_timestep",
    "v_prompt_timestep", "a_prompt_timestep",
)

# transformer_options flags the LTX-AV block.forward() actually reads.
_FLAG_KEYS = ("run_vx", "run_ax", "a2v_cross_attn", "v2a_cross_attn")


# Names of the *wire tensors* (after flattening — PE 3-tuples already
# split into `.cos`/`.sin` pairs) that are CONSTANT within a single
# generation. The client hashes these + relevant meta into a
# constants-session-id; on subsequent calls with the same id, it ships
# only the per-timestep tensors and the server pulls these from cache.
#
# Per-timestep tensors (always shipped): vx, ax, all `*_timestep.data`.
CONSTANT_WIRE_NAMES = frozenset({
    "v_context", "a_context",
    "attention_mask", "self_attention_mask",
    "v_pe.cos", "v_pe.sin",
    "a_pe.cos", "a_pe.sin",
    "v_cross_pe.cos", "v_cross_pe.sin",
    "a_cross_pe.cos", "a_cross_pe.sin",
})


def compute_constants_session_id(
    constants_named: list[tuple[str, torch.Tensor]],
    meta: dict[str, Any],
) -> str:
    """Compute a stable id for a set of constant tensors + their meta.

    Fingerprints each tensor by sampling 8 elements at fixed positions
    (start / quarter / mid / 3-quarter / end + 3 random-ish) — enough
    entropy to distinguish text contexts and PE tensors across
    generations, cheap enough to call every timestep.

    Why not `id(t)` (the original attempt): ComfyUI's outer model
    forward re-runs `_prepare_positional_embeddings` on every timestep,
    so the PE tensors get fresh allocations every step. id() differed
    every call, cache missed every call — observed in production.

    Why not full content hash: would require copying the entire tensor
    to CPU which dominates the wire-save the cache is meant to provide.
    8-element samples per tensor totals ~32 bytes of CPU sync across
    ~12 constants → sub-millisecond, indistinguishable in the trace.

    Hashes alongside: shape + dtype (so a resolution change between
    generations invalidates), the constants-relevant meta values
    (pe_split_mode, flags, the constant-keys subset of none_keys).
    Returns a 16-hex-char digest.
    """
    h = hashlib.sha1()
    for name, t in sorted(constants_named, key=lambda x: x[0]):
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        n = t.numel()
        if n > 0:
            flat = t.detach().flatten()
            # Sample positions: start, 1/4, 1/2, 3/4, end, plus three
            # offset reads to catch tensors that share boundary values
            # but differ inside.
            positions = sorted({
                0,
                n // 4,
                n // 2,
                (3 * n) // 4,
                n - 1,
                min(7, n - 1),
                min(n // 3, n - 1),
                min((2 * n) // 5, n - 1),
            })
            # Cast to float32 before .numpy() — numpy lacks a native
            # bfloat16 dtype (PyTorch's text-context and PE tensors are
            # bf16). Lossless for fingerprinting since float32 has
            # strictly more precision than bf16; we're hashing bits, not
            # values, so any bit-distinct bf16 samples remain
            # bit-distinct after the cast.
            sample = flat[positions].to(torch.float32).contiguous().cpu().numpy().tobytes()
            h.update(sample)
    # Constant-relevant meta: pe_split_mode (per-key bool), flags
    # (transformer_options run_vx etc), and the subset of none_keys that
    # are constant-keys. Per-timestep meta (compressed, non-constant
    # none_keys) is intentionally NOT hashed — it changes per call.
    h.update(json.dumps(meta.get("pe_split_mode", {}), sort_keys=True).encode())
    h.update(json.dumps(meta.get("flags", {}), sort_keys=True).encode())
    constant_root_keys = {"v_context", "a_context", "attention_mask",
                          "self_attention_mask", "v_pe", "a_pe",
                          "v_cross_pe", "a_cross_pe"}
    constant_none = sorted(k for k in (meta.get("none_keys", []) or [])
                           if k in constant_root_keys)
    h.update(json.dumps(constant_none).encode())
    return h.hexdigest()[:16]


def partition_named_by_constant(
    named: list[tuple[str, torch.Tensor]],
) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
    """Split flattened named tensors into (constants, per_timestep)."""
    consts = [(n, t) for n, t in named if n in CONSTANT_WIRE_NAMES]
    perstep = [(n, t) for n, t in named if n not in CONSTANT_WIRE_NAMES]
    return consts, perstep


def flatten_payload(args: dict[str, Any]) -> tuple[dict[str, Any], list[tuple[str, torch.Tensor]]]:
    """Flatten the LTX-AV block_wrap args dict into wire form.

    Returns:
        meta: JSON-serialisable dict describing structure (which keys
              were None, which are CompressedTimestep + their
              patches_per_frame, and the transformer_options flags).
        named_tensors: list of (name, tensor) tuples in canonical order.
              The caller wraps each via codec.encode_raw or codec.encode
              and appends to the wire blobs list.
    """
    meta: dict[str, Any] = {
        "compressed": {},
        "none_keys": [],
        "flags": {},
        "pe_split_mode": {},
    }
    named: list[tuple[str, torch.Tensor]] = []

    # img = (vx, ax)
    vx, ax = args["img"]
    named.append(("vx", vx))
    named.append(("ax", ax))

    # Plain tensors (some optional)
    for k in _TENSOR_KEYS:
        v = args.get(k)
        if v is None:
            meta["none_keys"].append(k)
        elif isinstance(v, torch.Tensor):
            named.append((k, v))
        else:
            raise TypeError(f"flatten_payload: {k!r} expected Tensor or None, got {type(v).__name__}")

    # Rotary PE 3-tuples: (cos, sin, split_mode_bool)
    for k in _PE_KEYS:
        v = args.get(k)
        if v is None:
            meta["none_keys"].append(k)
            continue
        if isinstance(v, tuple) and len(v) == 3:
            cos, sin, split_mode = v
            if not (isinstance(cos, torch.Tensor) and isinstance(sin, torch.Tensor)):
                raise TypeError(
                    f"flatten_payload: {k!r} expected (Tensor, Tensor, bool) tuple"
                )
            named.append((f"{k}.cos", cos))
            named.append((f"{k}.sin", sin))
            meta["pe_split_mode"][k] = bool(split_mode)
        else:
            raise TypeError(
                f"flatten_payload: {k!r} expected 3-tuple or None, got {type(v).__name__}"
            )

    # CompressedTimestep entries
    for k in _COMPRESSED_KEYS:
        v = args.get(k)
        if v is None:
            meta["none_keys"].append(k)
            continue
        # Duck-type rather than importing CompressedTimestep here so this
        # module stays importable without comfy on the path (smoke tests,
        # protocol-only sanity checks).
        if hasattr(v, "data") and hasattr(v, "patches_per_frame"):
            named.append((f"{k}.data", v.data))
            meta["compressed"][k] = int(v.patches_per_frame)
        elif isinstance(v, torch.Tensor):
            # Some prompt-timestep entries arrive as bare tensors when
            # cross_attention_adaln is off — ship as raw.
            named.append((k, v))
        else:
            raise TypeError(
                f"flatten_payload: {k!r} expected CompressedTimestep or Tensor or None, "
                f"got {type(v).__name__}"
            )

    # Pull the four runtime flags out of transformer_options if present.
    topts = args.get("transformer_options") or {}
    for fk in _FLAG_KEYS:
        if fk in topts:
            meta["flags"][fk] = bool(topts[fk])

    return meta, named


def reconstruct_payload(
    meta: dict[str, Any],
    tensors_by_name: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Inverse of flatten_payload. Returns an args dict ready to pass
    as the keyword args to a transformer_block forward() on the server.

    transformer_options is built from `meta["flags"]` — defaults to
    True for any flag not present (matches the block's `.get(k, True)`
    lookup semantics).
    """
    # CompressedTimestep is in av_model; only imported on the server side
    # where comfy is on the path. Lazy import so this module imports
    # cleanly in protocol-only contexts.
    from comfy.ldm.lightricks.av_model import CompressedTimestep

    compressed_meta = meta.get("compressed", {}) or {}
    none_keys = set(meta.get("none_keys", []) or [])
    flags = meta.get("flags", {}) or {}
    pe_split_mode = meta.get("pe_split_mode", {}) or {}

    out: dict[str, Any] = {}

    # img = (vx, ax)
    out["img"] = (tensors_by_name["vx"], tensors_by_name["ax"])

    for k in _TENSOR_KEYS:
        if k in none_keys:
            out[k] = None
        else:
            out[k] = tensors_by_name[k]

    for k in _PE_KEYS:
        if k in none_keys:
            out[k] = None
            continue
        cos = tensors_by_name[f"{k}.cos"]
        sin = tensors_by_name[f"{k}.sin"]
        out[k] = (cos, sin, bool(pe_split_mode.get(k, False)))

    for k in _COMPRESSED_KEYS:
        if k in none_keys:
            out[k] = None
            continue
        if k in compressed_meta:
            data = tensors_by_name[f"{k}.data"]
            ppf = compressed_meta[k]
            # CompressedTimestep.__init__ takes a flat (B, T, D) tensor
            # and patches_per_frame; it stores `.data` as the already-
            # compressed form when T % ppf == 0. Re-passing the same
            # compressed data through that constructor would double-
            # compress. Build the object directly instead.
            obj = object.__new__(CompressedTimestep)
            obj.data = data
            obj.batch_size = int(data.shape[0])
            if ppf == 1:
                # Wasn't actually compressible; data IS the expanded form
                obj.num_frames = int(data.shape[1])
                obj.feature_dim = int(data.shape[2])
                obj.patches_per_frame = 1
            else:
                # Compressed: data shape is (B, num_frames, feature_dim)
                obj.num_frames = int(data.shape[1])
                obj.feature_dim = int(data.shape[2])
                obj.patches_per_frame = int(ppf)
            out[k] = obj
        else:
            # Tensor-shaped fallback
            out[k] = tensors_by_name[k]

    topts: dict[str, Any] = {}
    for fk in _FLAG_KEYS:
        topts[fk] = bool(flags.get(fk, True))
    out["transformer_options"] = topts

    return out
