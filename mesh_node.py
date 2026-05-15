"""ComfyUI custom node: FLUX mesh-split rig.

Splits FLUX transformer blocks across two machines:
    - First (n_double - min(n, n_double)) doubles run locally
    - Last  min(n, n_double) doubles run remotely
    - First (n - n_double) single_blocks run remotely (when n > n_double)
    - Remaining single_blocks + final_layer + VAE run locally

So `n_blocks_remote` is a unified counter starting from the END of the
double_blocks stack and walking forward through the single_blocks.

User is responsible for setting `n_blocks_remote` here to match the
server's `--n-blocks` setting. Mismatch produces wrong output, not a
crash — no handshake validation in v1.

The single_blocks portion needs the un-modulated `vec_orig` tensor so
the server can compute single-block modulation locally. We capture
that via a forward_pre_hook on `double_stream_modulation_img` and ship
it as one extra small tensor on every wire request.

Wiring uses ComfyUI's built-in transformer_options["patches_replace"]
mechanism — we register a per-block override callback that, instead of
running the local copy of the block, packages the activations and
sends them to the remote server.

Two registered nodes:
    - MeshSplitFlux:  pass-through MODEL node, configures split point +
                      remote address + codec mode. Sets up the per-block
                      patches.
    - MeshStatus:     pure-info node that reports last-call wire stats
                      (bytes sent, bytes received, codec ratio).
"""

from __future__ import annotations

import socket
import time

import torch

# Package-relative imports so we always pick up the node's own
# codec / protocol / vec_io files, regardless of sys.path ordering.
# ComfyUI ships its own top-level `protocol.py` (BinaryEventTypes) which
# would shadow ours if we used `import protocol` with the wrong path
# order. ComfyUI's custom-node loader sets up __init__.py-rooted specs
# as proper packages, so `from . import X` resolves cleanly.
from . import codec
from . import protocol
from . import vec_io
from . import lora_io


_LAST_STATS: dict = {
    "wire_call_count": 0,
    "bytes_sent": 0,
    "bytes_received": 0,
    "last_call_seconds": 0.0,
    "codec_mode": "n/a",
    "n_blocks_remote": 0,
    "n_double_remote": 0,
    "n_single_remote": 0,
    "split_index": -1,
    "blocks_offloaded": 0,
}


class MeshClient:
    """Persistent TCP connection to the back-half server. Lazy-opened
    on first use, kept alive across timesteps within a generation, and
    re-opened transparently if the peer drops."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._sock: socket.socket | None = None
        # Track which client-LoRA session id was last shipped to this
        # server, to avoid reshipping ~MB-sized LoRA payloads every
        # timestep when nothing changed.
        self._last_sent_lora_session: str = ""

    def _ensure_open(self) -> socket.socket:
        if self._sock is not None:
            return self._sock
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.connect((self.host, self.port))
        self._sock = s
        # Handshake
        protocol.send_message(s, {"kind": "hello", "tensors": []}, [])
        header, _ = protocol.recv_message(s)
        if header.get("kind") != "hello_ack":
            raise RuntimeError(f"unexpected handshake response: {header!r}")
        print(f"[mesh] connected to {self.host}:{self.port}; server reports {header.get('server_info', {})}")
        return s

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def call_double_blocks(
        self,
        *,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor | tuple,
        vec_orig: torch.Tensor | None,
        pe: torch.Tensor,
        attn_mask,
        start_block: int,
        codec_mode: str,
        codec_qp: int,
        codec_lossless: bool,
        codec_tile_dim: int,
        client_lora_session: str = "",       # "" or "empty" -> no LoRA on client
        client_lora_blob: bytes = b"",       # safetensors blob; empty if unchanged
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Send the back-half-double-blocks request and receive the
        post-double-block (img, txt) state."""
        sock = self._ensure_open()
        device = img.device

        # Encode the big tensor (img) via codec; ship the rest raw
        # because they're either tiny (vec, attn_mask), already-cached
        # (pe), or batch-small (txt).
        wire_tensors = []
        blobs = []

        img_w = codec.encode("img", img, mode=codec_mode, qp=codec_qp, lossless=codec_lossless, tile_dim=codec_tile_dim)
        wire_tensors.append(img_w.to_header())
        blobs.append(img_w.bytes_payload)

        txt_w = codec.encode_raw("txt", txt)
        wire_tensors.append(txt_w.to_header())
        blobs.append(txt_w.bytes_payload)

        # vec is either a single tensor (global_modulation=False) or a
        # nested ((img_mod1, img_mod2), (txt_mod1, txt_mod2)) tuple
        # of ModulationOut dataclasses (global_modulation=True, the
        # FLUX.2 path). Flatten via vec_io to a list of named tensors.
        vec_kind, named_vec = vec_io.flatten_vec(vec)
        for tname, t in named_vec:
            w = codec.encode_raw(tname, t)
            wire_tensors.append(w.to_header())
            blobs.append(w.bytes_payload)

        # vec_orig — the un-modulated tensor that single_stream_modulation
        # consumes. Server uses it to compute the single-block modulation
        # locally (only when it has single_blocks loaded; ignored otherwise).
        # ~16 KB, raw.
        if vec_orig is not None:
            vo_w = codec.encode_raw("vec_orig", vec_orig)
            wire_tensors.append(vo_w.to_header())
            blobs.append(vo_w.bytes_payload)

        pe_w = codec.encode_raw("pe", pe)
        wire_tensors.append(pe_w.to_header())
        blobs.append(pe_w.bytes_payload)

        attn_blob = b""
        if attn_mask is not None:
            am_w = codec.encode_raw("attn_mask", attn_mask)
            wire_tensors.append(am_w.to_header())
            blobs.append(am_w.bytes_payload)

        # Client-LoRA blob — included only when the session id changed
        # since last shipped (caller is responsible for tracking that).
        # Treated as opaque bytes by the wire layer; server detects by name
        # and feeds it to lora_io.decode_patches_from_safetensors.
        if client_lora_blob:
            wire_tensors.append({
                "name": "client_lora",
                "encoding": "lora_safetensors",
                "size": len(client_lora_blob),
                "dtype": "bytes",
                "shape": [],
                "extra": {"session_id": client_lora_session},
            })
            blobs.append(client_lora_blob)

        header = {
            "kind": "forward_double_blocks",
            "tensors": wire_tensors,
            "start_block": int(start_block),
            "vec_kind": vec_kind,
            "has_attn_mask": attn_mask is not None,
            # Top-level so server can lifecycle-manage even when the blob
            # is omitted (unchanged session). Empty string = client has
            # forwarding off OR no LoRAs loaded; "empty" = explicit empty
            # set; otherwise a 16-char hex id from lora_io.patches_session_id.
            "client_lora_session": client_lora_session,
        }

        bytes_sent = 4 + 4 + len(_dumps_len(header)) + sum(len(b) for b in blobs)

        t0 = time.time()
        protocol.send_message(sock, header, blobs)
        resp_header, resp_blobs = protocol.recv_message(sock)
        elapsed = time.time() - t0

        if resp_header.get("kind") != "forward_double_blocks_response":
            raise RuntimeError(f"unexpected response kind {resp_header!r}")

        wires = resp_header["tensors"]
        if len(wires) < 2:
            raise RuntimeError(f"response missing img/txt; got {len(wires)} tensors")
        img_back = codec.decode(wires[0], resp_blobs[0], device=device)
        txt_back = codec.decode(wires[1], resp_blobs[1], device=device)

        bytes_received = sum(len(b) for b in resp_blobs)

        _LAST_STATS["wire_call_count"] += 1
        _LAST_STATS["bytes_sent"] += bytes_sent
        _LAST_STATS["bytes_received"] += bytes_received
        _LAST_STATS["last_call_seconds"] = elapsed

        return img_back, txt_back


def _dumps_len(header: dict) -> bytes:
    import json
    return json.dumps(header, separators=(",", ":")).encode("utf-8")


# Module-level connection registry: one client per (host, port). Kept
# alive across forward passes so we don't pay TCP setup per timestep.
_CLIENTS: dict[tuple[str, int], MeshClient] = {}


def _get_client(host: str, port: int) -> MeshClient:
    key = (host, port)
    if key not in _CLIENTS:
        _CLIENTS[key] = MeshClient(host, port)
    return _CLIENTS[key]


def _make_block_replacement(
    client: MeshClient,
    split_index: int,
    n_double_blocks: int,
    n_single_blocks: int,
    n_double_remote: int,
    n_single_remote: int,
    codec_mode: str,
    codec_qp: int,
    codec_lossless: bool,
    codec_tile_dim: int,
    forward_client_loras: bool,
    patcher_capture: dict,
    vec_orig_capture: dict,
):
    """Return a callable that ComfyUI's patches_replace will invoke at
    block `split_index`. It does the remote forward for blocks
    [split_index..n_double_blocks) (and any configured single_blocks)
    and returns the post-back-half state.

    Optionally inspects the live ModelPatcher.patches dict on each call
    and forwards the relevant client-side LoRA patches to the server,
    but only when the patches changed since last shipped (cheap session-
    id hash detects the change).
    """
    drop_db = n_double_blocks - n_double_remote

    def replace_at_split(args, extras):
        img = args["img"]
        txt = args["txt"]
        vec = args["vec"]
        pe = args["pe"]
        attn_mask = args.get("attn_mask")

        # vec_orig was captured by the forward_pre_hook on
        # double_stream_modulation_img earlier in this forward pass.
        # When the server has single_blocks loaded, it uses vec_orig
        # to compute the single-block modulation locally.
        vec_orig = vec_orig_capture.get("vec_orig")

        # ---- Client-LoRA forwarding (optional) ----
        client_lora_session = ""
        client_lora_blob = b""
        if forward_client_loras:
            patcher = patcher_capture.get("patcher")
            if patcher is not None and getattr(patcher, "patches", None):
                # Filter the live patches dict to only the keys targeting
                # server-held layers, with double_block indices remapped
                # to the slim model's coords.
                slim_patches = lora_io.filter_and_remap_patches(
                    patcher.patches,
                    drop_db=drop_db,
                    n_single_remote=n_single_remote,
                    n_double_total=n_double_blocks,
                )
                client_lora_session = lora_io.patches_session_id(slim_patches)
            else:
                # forward toggle on, but no patches loaded — explicit
                # empty session so the server unpatches anything left over.
                client_lora_session = "empty"

            # Only encode + ship if the session changed since we last
            # sent it to THIS client (host, port).
            if client_lora_session and client_lora_session != client._last_sent_lora_session:
                if client_lora_session == "empty":
                    # Signal "no patches" without a blob — server unpatches.
                    client_lora_blob = b""
                else:
                    client_lora_blob = lora_io.encode_patches_to_safetensors(slim_patches)
                # Mark sent regardless of whether the blob has bytes
                client._last_sent_lora_session = client_lora_session

        new_img, new_txt = client.call_double_blocks(
            img=img,
            txt=txt,
            vec=vec,
            vec_orig=vec_orig,
            pe=pe,
            attn_mask=attn_mask,
            start_block=split_index,
            codec_mode=codec_mode,
            codec_qp=codec_qp,
            codec_lossless=codec_lossless,
            codec_tile_dim=codec_tile_dim,
            client_lora_session=client_lora_session,
            client_lora_blob=client_lora_blob,
        )
        return {"img": new_img, "txt": new_txt}

    return replace_at_split


def _make_double_passthrough():
    """No-op replacement for double_blocks AFTER the split point — server
    already ran them, return inputs unchanged."""
    def passthrough(args, extras):
        return {"img": args["img"], "txt": args["txt"]}
    return passthrough


def _make_single_passthrough():
    """No-op replacement for single_blocks the server has already run.
    Single_blocks operate on the concatenated [txt|img] tensor, so the
    args dict carries just `img` (the concatenated form)."""
    def passthrough(args, extras):
        return {"img": args["img"]}
    return passthrough


def _install_vec_orig_hook(diffusion_model, capture_dict):
    """Install a forward_pre_hook on `double_stream_modulation_img` to
    capture vec_orig (the un-modulated tensor passed into the modulation
    modules at the start of each forward pass).

    Idempotent: if a previous hook from this node is already installed,
    it's removed first so re-running the workflow doesn't accumulate
    hooks. Marker is stashed as `_mesh_vec_orig_hook` on the module.

    Without this, the server can't run any single_blocks (its
    single_stream_modulation needs vec_orig to compute the single-block
    modulation tuple).
    """
    mod = getattr(diffusion_model, "double_stream_modulation_img", None)
    if mod is None:
        return  # FLUX1 path (no global_modulation) doesn't have these

    # Remove any prior hook from this node so we don't accumulate
    prior_handle = getattr(mod, "_mesh_vec_orig_hook", None)
    if prior_handle is not None:
        try:
            prior_handle.remove()
        except Exception:
            pass

    def hook(module, inputs):
        # inputs is a tuple of positional args; vec_orig is inputs[0]
        if len(inputs) > 0:
            capture_dict["vec_orig"] = inputs[0]

    mod._mesh_vec_orig_hook = mod.register_forward_pre_hook(hook)


class MeshSplitFlux:
    """Configure FLUX double-block split between local 5090 and a
    remote 4090 server. Pass-through MODEL node — slot it between the
    model loader and the sampler."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "n_blocks_remote": ("INT", {"default": 4, "min": 0, "max": 256,
                                            "tooltip": (
                                                "How many transformer blocks run on the remote server. "
                                                "Counts double_blocks first, then single_blocks. "
                                                "For FLUX.2 Klein 9B (8 doubles + 24 singles): "
                                                "0=nothing remote, 1-8=last N doubles, "
                                                "9-32=all doubles + first (N-8) singles. "
                                                "MUST match the server's --n-blocks setting."
                                            )}),
                "remote_host": ("STRING", {"default": "127.0.0.1",
                                           "tooltip": "Hostname or IP of the back-half server (the 4090)."}),
                "remote_port": ("INT", {"default": 7777, "min": 1, "max": 65535}),
                "codec_mode": (["raw", "nvenc"], {"default": "nvenc"}),
                "codec_qp": ("INT", {"default": 18, "min": 0, "max": 51,
                                     "tooltip": "Lower = higher quality / less compression. 10=near-lossless, 18=standard, 28=high-compression (FLUX absorbs it fine; only a slight softness vs QP=18)."}),
                "codec_lossless": ("BOOLEAN", {"default": False,
                                               "tooltip": "Use NVENC's lossless tuning (overrides QP, much larger bitstream)."}),
                "codec_tile_dim": ([1, 2, 4, 8], {"default": 4,
                                                   "tooltip": (
                                                       "How many channels to tile per Y/U/V plane in each NVENC frame. "
                                                       "Bigger tiles = fewer larger codec frames per encode = much "
                                                       "faster wall clock. 1=legacy (~600ms/round-trip), "
                                                       "4=default (~130ms), 8=most aggressive (~110ms). Compression "
                                                       "ratio is essentially unchanged across values."
                                                   )}),
                "forward_client_loras": ("BOOLEAN", {"default": True,
                                                       "tooltip": (
                                                           "When ON, any LoRAs loaded BEFORE this node in the workflow "
                                                           "(via ComfyUI's standard LoraLoader) get serialized via "
                                                           "safetensors and shipped to the server so the LoRA applies "
                                                           "to the back-half blocks too. Required for full-model LoRA "
                                                           "effect when offloading any blocks. Auto-detects changes; "
                                                           "ships the blob only when LoRA set / strength changes. "
                                                           "Workflow ordering matters: LoraLoader must come BEFORE "
                                                           "MeshSplit FLUX in the graph for this to see them."
                                                       )}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "configure"
    CATEGORY = "mesh"
    OUTPUT_NODE = False

    def configure(self, model, n_blocks_remote, remote_host, remote_port, codec_mode, codec_qp, codec_lossless, codec_tile_dim, forward_client_loras):
        # Reach into the diffusion model to learn block counts
        diffusion = model.model.diffusion_model
        n_double_blocks = len(diffusion.double_blocks)
        n_single_blocks = len(diffusion.single_blocks)
        n_total_blocks = n_double_blocks + n_single_blocks
        if not (0 <= n_blocks_remote <= n_total_blocks):
            raise ValueError(
                f"n_blocks_remote {n_blocks_remote} out of range; model has "
                f"{n_double_blocks} doubles + {n_single_blocks} singles "
                f"= {n_total_blocks} total"
            )

        # Translate the unified n_blocks_remote into per-stack offload counts.
        # Doubles get offloaded first; once n_blocks_remote exceeds n_double,
        # the surplus eats into singles (front-to-back).
        if n_blocks_remote <= n_double_blocks:
            n_double_remote = n_blocks_remote
            n_single_remote = 0
        else:
            n_double_remote = n_double_blocks
            n_single_remote = n_blocks_remote - n_double_blocks

        # Where the wire hook fires in the doubles loop. If no doubles are
        # offloaded (n_double_remote == 0), we don't fire there at all and
        # the wire hook moves down to single_block[0].
        split_index = n_double_blocks - n_double_remote

        # Open / reuse the client so the user gets a connection error
        # at queue-time rather than mid-sample.
        client = _get_client(remote_host, remote_port)
        client._ensure_open()

        # Capture vec_orig via a forward_pre_hook on the modulation module —
        # the server uses it to compute single-block modulation locally.
        vec_orig_capture: dict = {"vec_orig": None}
        _install_vec_orig_hook(diffusion, vec_orig_capture)

        # ModelPatcher copy + register the per-block overrides via the
        # canonical comfy.model_patcher API. Using set_model_patch_replace
        # rather than mutating model_options directly so we don't fight
        # the ModelPatcher's copy-on-write semantics.
        m = model.clone()

        # Capture the (post-clone) patcher reference so the per-call
        # closure can introspect its .patches dict at sample time. This
        # is the patcher that downstream nodes (KSampler) will see; if
        # LoraLoader runs BEFORE this node, m.patches has the LoRA
        # patches at this point and they propagate via clone().
        # If LoraLoader runs AFTER, this reference won't see those
        # later-added patches — see the tooltip on forward_client_loras.
        patcher_capture: dict = {"patcher": m}

        if n_blocks_remote > 0:
            replace_at_split = _make_block_replacement(
                client, split_index, n_double_blocks, n_single_blocks,
                n_double_remote, n_single_remote,
                codec_mode, codec_qp, codec_lossless, codec_tile_dim,
                forward_client_loras,
                patcher_capture,
                vec_orig_capture,
            )
            double_pass = _make_double_passthrough()
            single_pass = _make_single_passthrough()

            if n_double_remote > 0:
                # Wire hook fires inside the double_blocks loop
                m.set_model_patch_replace(replace_at_split, "dit", "double_block", split_index)
                for i in range(split_index + 1, n_double_blocks):
                    m.set_model_patch_replace(double_pass, "dit", "double_block", i)
            else:
                # No doubles offloaded — wire hook moves to single_block[0]
                # (this branch is only reachable if 0 < n_blocks_remote <= n_single_blocks
                # AND n_double_remote == 0, which by our mapping means ... never.
                # We leave this branch unreachable for the current mapping but
                # the structure supports a future "singles-only" mode.)
                pass

            # If the server is also running some single_blocks, passthrough
            # those on the client so its local copies don't run again.
            for i in range(n_single_remote):
                m.set_model_patch_replace(single_pass, "dit", "single_block", i)
        # n_blocks_remote == 0: no patches; entire model runs locally

        _LAST_STATS["codec_mode"] = codec_mode
        _LAST_STATS["codec_tile_dim"] = codec_tile_dim
        _LAST_STATS["n_blocks_remote"] = n_blocks_remote
        _LAST_STATS["n_double_remote"] = n_double_remote
        _LAST_STATS["n_single_remote"] = n_single_remote
        _LAST_STATS["split_index"] = split_index
        _LAST_STATS["blocks_offloaded"] = n_blocks_remote
        _LAST_STATS["wire_call_count"] = 0
        _LAST_STATS["bytes_sent"] = 0
        _LAST_STATS["bytes_received"] = 0
        _LAST_STATS["last_call_seconds"] = 0.0

        print(f"[mesh] offloading {n_double_remote}/{n_double_blocks} doubles + "
              f"{n_single_remote}/{n_single_blocks} singles "
              f"(double_block intercept at index {split_index}); "
              f"server={remote_host}:{remote_port}; "
              f"codec={codec_mode} qp={codec_qp} lossless={codec_lossless} tile_dim={codec_tile_dim}")

        return (m,)


class MeshStatus:
    """Reports stats from the most recent set of wire calls."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "report"
    CATEGORY = "mesh"
    OUTPUT_NODE = True

    def report(self):
        s = _LAST_STATS
        ratio = (s["bytes_sent"] / max(1, s["bytes_received"]))
        n_double = s.get("n_double_remote", s.get("blocks_offloaded", 0))
        n_single = s.get("n_single_remote", 0)
        msg = (
            f"n_blocks_remote={s.get('n_blocks_remote', s['blocks_offloaded'])}  "
            f"({n_double} doubles + {n_single} singles, "
            f"double_block intercept at index {s['split_index']})\n"
            f"codec_mode={s['codec_mode']}\n"
            f"wire_calls={s['wire_call_count']}  "
            f"bytes_sent={s['bytes_sent']/1024/1024:.2f} MB  "
            f"bytes_received={s['bytes_received']/1024/1024:.2f} MB  "
            f"last_call={s['last_call_seconds']*1000:.1f} ms\n"
            f"send/recv ratio={ratio:.2f}x"
        )
        print(f"[mesh status] {msg}")
        return (msg,)


NODE_CLASS_MAPPINGS = {
    "MeshSplitFlux": MeshSplitFlux,
    "MeshStatus": MeshStatus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshSplitFlux": "Mesh Split FLUX (5090 ↔ 4090)",
    "MeshStatus": "Mesh Status",
}
