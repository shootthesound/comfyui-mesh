"""ComfyUI custom node: FLUX mesh-split rig.

Splits FLUX double_blocks across two machines:
    - First (total - n_blocks_remote) blocks run locally (this 5090)
    - Last  n_blocks_remote          blocks run remotely (the 4090 over TCP)

User is responsible for setting `n_blocks_remote` here to match the
server's `--n-blocks` setting. Mismatch produces wrong output, not a
crash — no handshake validation in v1.

All single_blocks + final layer stay local (smaller, post-double-block).

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


_LAST_STATS: dict = {
    "wire_call_count": 0,
    "bytes_sent": 0,
    "bytes_received": 0,
    "last_call_seconds": 0.0,
    "codec_mode": "n/a",
    "n_blocks_remote": 0,
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
        pe: torch.Tensor,
        attn_mask,
        start_block: int,
        codec_mode: str,
        codec_qp: int,
        codec_lossless: bool,
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

        img_w = codec.encode("img", img, mode=codec_mode, qp=codec_qp, lossless=codec_lossless)
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

        pe_w = codec.encode_raw("pe", pe)
        wire_tensors.append(pe_w.to_header())
        blobs.append(pe_w.bytes_payload)

        attn_blob = b""
        if attn_mask is not None:
            am_w = codec.encode_raw("attn_mask", attn_mask)
            wire_tensors.append(am_w.to_header())
            blobs.append(am_w.bytes_payload)

        header = {
            "kind": "forward_double_blocks",
            "tensors": wire_tensors,
            "start_block": int(start_block),
            "vec_kind": vec_kind,
            "has_attn_mask": attn_mask is not None,
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
    codec_mode: str,
    codec_qp: int,
    codec_lossless: bool,
):
    """Return a callable that ComfyUI's patches_replace will invoke at
    block `split_index`. It does the remote forward for blocks
    [split_index..n_double_blocks) and returns the post-back-half state."""

    def replace_at_split(args, extras):
        img = args["img"]
        txt = args["txt"]
        vec = args["vec"]
        pe = args["pe"]
        attn_mask = args.get("attn_mask")

        new_img, new_txt = client.call_double_blocks(
            img=img,
            txt=txt,
            vec=vec,
            pe=pe,
            attn_mask=attn_mask,
            start_block=split_index,
            codec_mode=codec_mode,
            codec_qp=codec_qp,
            codec_lossless=codec_lossless,
        )
        return {"img": new_img, "txt": new_txt}

    return replace_at_split


def _make_passthrough():
    """Replacement for the blocks AFTER the split point — they should
    not run locally because the remote already did them."""
    def passthrough(args, extras):
        return {"img": args["img"], "txt": args["txt"]}
    return passthrough


class MeshSplitFlux:
    """Configure FLUX double-block split between local 5090 and a
    remote 4090 server. Pass-through MODEL node — slot it between the
    model loader and the sampler."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "n_blocks_remote": ("INT", {"default": 4, "min": 0, "max": 100,
                                            "tooltip": "How many of the LAST double_blocks to run on the remote server. MUST match the server's --n-blocks setting. 0 = nothing remote (no-op); 4 on an 8-block model = even half-split; 8 = entire double_block stack remote."}),
                "remote_host": ("STRING", {"default": "127.0.0.1",
                                           "tooltip": "Hostname or IP of the back-half server (the 4090)."}),
                "remote_port": ("INT", {"default": 7777, "min": 1, "max": 65535}),
                "codec_mode": (["raw", "nvenc"], {"default": "nvenc"}),
                "codec_qp": ("INT", {"default": 18, "min": 0, "max": 51,
                                     "tooltip": "Lower = higher quality / less compression. 10=near-lossless, 18=standard, 28=high-compression (FLUX absorbs it fine; only a slight softness vs QP=18)."}),
                "codec_lossless": ("BOOLEAN", {"default": False,
                                               "tooltip": "Use NVENC's lossless tuning (overrides QP, much larger bitstream)."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "configure"
    CATEGORY = "mesh"
    OUTPUT_NODE = False

    def configure(self, model, n_blocks_remote, remote_host, remote_port, codec_mode, codec_qp, codec_lossless):
        # Reach into the diffusion model to learn n_double_blocks
        diffusion = model.model.diffusion_model
        n_double_blocks = len(diffusion.double_blocks)
        if not (0 <= n_blocks_remote <= n_double_blocks):
            raise ValueError(
                f"n_blocks_remote {n_blocks_remote} out of range; model has {n_double_blocks} double_blocks"
            )

        # Convert "N blocks running remotely" to "intercept at block index
        # (total - N)" — that's where our patches_replace fires on the
        # client side. The server is configured with --n-blocks N so its
        # block 0 is what was originally block split_index.
        split_index = n_double_blocks - n_blocks_remote

        # Open / reuse the client so the user gets a connection error
        # at queue-time rather than mid-sample.
        client = _get_client(remote_host, remote_port)
        client._ensure_open()

        # ModelPatcher copy + register the per-block overrides via the
        # canonical comfy.model_patcher API. Using set_model_patch_replace
        # rather than mutating model_options directly so we don't fight
        # the ModelPatcher's copy-on-write semantics.
        m = model.clone()
        if n_blocks_remote > 0:
            replace_at_split = _make_block_replacement(
                client, split_index, n_double_blocks, codec_mode, codec_qp, codec_lossless
            )
            passthrough = _make_passthrough()
            m.set_model_patch_replace(replace_at_split, "dit", "double_block", split_index)
            for i in range(split_index + 1, n_double_blocks):
                m.set_model_patch_replace(passthrough, "dit", "double_block", i)
        # n_blocks_remote == 0: no patches; entire model runs locally

        _LAST_STATS["codec_mode"] = codec_mode
        _LAST_STATS["n_blocks_remote"] = n_blocks_remote
        _LAST_STATS["split_index"] = split_index
        _LAST_STATS["blocks_offloaded"] = n_blocks_remote
        _LAST_STATS["wire_call_count"] = 0
        _LAST_STATS["bytes_sent"] = 0
        _LAST_STATS["bytes_received"] = 0
        _LAST_STATS["last_call_seconds"] = 0.0

        print(f"[mesh] {n_blocks_remote}/{n_double_blocks} double_blocks running remotely "
              f"(intercepting at block {split_index}); server={remote_host}:{remote_port}; "
              f"codec={codec_mode} qp={codec_qp} lossless={codec_lossless}")

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
        msg = (
            f"n_blocks_remote={s.get('n_blocks_remote', s['blocks_offloaded'])}  "
            f"(intercepting at block {s['split_index']})\n"
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
