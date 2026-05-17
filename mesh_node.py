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

Single registered node:
    - MeshSplitFlux:  pass-through MODEL node, configures split point +
                      remote address + codec mode. Sets up the per-block
                      patches. Live status (connection, wire stats,
                      pending changes) surfaces inline on the node via
                      web/mesh.js — no separate status node needed.
"""

from __future__ import annotations

import gc
import socket
import time

import torch
from torch import nn

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


class MeshReconnect(Exception):
    """Raised by MeshClient when a server-side disconnect is detected
    and the persistent socket has been reset. The per-block replacement
    closure catches this and rebuilds its LoRA-forwarding bookkeeping
    (which assumes the server remembers what we shipped last time)
    before retrying the call. Lets the user restart the server without
    relaunching ComfyUI."""
    pass


class MeshDecreaseNeedsReload(Exception):
    """Raised when the user decreases n_blocks_remote mid-session — the
    stripped weights are gone and we can't un-strip. configure() catches
    this, surfaces a user-friendly message in the node UI via websocket,
    then re-raises so ComfyUI flags the node."""
    pass


class MeshServerNeedsReconfigure(Exception):
    """Raised when the client's n_blocks_remote disagrees with the
    server's currently-running --n-blocks. The JS surfaces a Confirm
    button that POSTs /mesh/reconfigure to apply the change; until
    confirmed, KSampler is blocked here so the user can't produce
    silently wrong output with a mismatched split."""
    pass


# ---------------------------------------------------------------------
# HTTP route: /mesh/reconfigure
#
# The JS Confirm-restart button POSTs here with the node's current
# (host, port, n_blocks). We connect to that mesh server, send a
# reconfigure message, and return the result. The server execvs
# itself with the new --n-blocks; subsequent generations reconnect
# transparently and pick up the new server_n_blocks via the existing
# hello handshake.
# ---------------------------------------------------------------------

try:
    from server import PromptServer as _ComfyPromptServer
    from aiohttp import web as _aiohttp_web
    _HAS_COMFY_SERVER = True
except ImportError:
    _HAS_COMFY_SERVER = False

if _HAS_COMFY_SERVER:
    @_ComfyPromptServer.instance.routes.get("/mesh/status")
    async def _mesh_status_route(request):
        """JS-side connection indicator polls this endpoint every few
        seconds. We DON'T do a fresh TCP probe here — that would either
        log spurious disconnects on the server or briefly block the
        legitimate client (mesh server is single-tenant). Instead, we
        report the state of any cached MeshClient for this (host, port):

          - "connected"    — cached client has a live socket
          - "disconnected" — cached client exists but its socket is dead
                             (typically: server died, MeshReconnect cleared
                             the socket, awaiting next call to reopen)
          - "idle"         — no client cached yet (user hasn't queued
                             anything for this host:port in this session)
        """
        host = (request.query.get("host") or "").strip()
        try:
            port = int(request.query.get("port") or "0")
        except ValueError:
            port = 0
        if not host or not (1 <= port <= 65535):
            return _aiohttp_web.json_response(
                {"state": "idle", "error": "bad host/port"}, status=400,
            )
        client = _CLIENTS.get((host, port))
        if client is None:
            return _aiohttp_web.json_response({
                "state": "idle", "host": host, "port": port,
            })

        # If the cached client's socket is dead, try to reconnect now.
        # This is what lets the indicator auto-flip back to green when
        # the server returns from a restart / reconfigure / crash —
        # without the user having to queue a workflow first to trigger
        # _ensure_open. The in-flight guard keeps repeated 3s polls
        # from stacking up while the server is genuinely down.
        if client._sock is None and not client._probe_in_flight:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                # 6s budget — slightly more than the socket connect
                # timeout in _ensure_open, so we don't cut off a
                # successful handshake.
                await asyncio.wait_for(
                    loop.run_in_executor(None, client.try_reconnect),
                    timeout=6.0,
                )
            except (asyncio.TimeoutError, Exception):
                pass

        return _aiohttp_web.json_response({
            "state": "connected" if client._sock is not None else "disconnected",
            "host": host, "port": port,
            "server_n_blocks": getattr(client, "server_n_blocks", None),
        })

    @_ComfyPromptServer.instance.routes.post("/mesh/reconfigure")
    async def _mesh_reconfigure_route(request):
        try:
            data = await request.json()
        except Exception:
            return _aiohttp_web.json_response({"error": "invalid JSON"}, status=400)
        host = (data or {}).get("host", "").strip()
        port = int((data or {}).get("port", 0) or 0)
        n_blocks = int((data or {}).get("n_blocks", -1))
        if not host or not (1 <= port <= 65535) or n_blocks < 0:
            return _aiohttp_web.json_response(
                {"error": "host, port, n_blocks required"}, status=400,
            )
        try:
            client = _get_client(host, port)
            client.reconfigure(n_blocks)
            return _aiohttp_web.json_response({
                "ok": True,
                "host": host,
                "port": port,
                "n_blocks": n_blocks,
            })
        except Exception as e:
            return _aiohttp_web.json_response(
                {"error": f"reconfigure failed: {e}"}, status=500,
            )


def _send_node_message(node_id, level: str, text: str) -> None:
    """Send a `mesh-message` websocket event to ComfyUI's web UI. The
    bundled `web/mesh.js` extension picks it up and renders a banner
    below the matching node. Stays as a fire-and-forget — if anything
    in the dispatch path is broken, log and continue."""
    if not node_id:
        return
    try:
        import server as _comfy_server
        inst = getattr(_comfy_server.PromptServer, "instance", None)
        if inst is None:
            return
        inst.send_sync("mesh-message", {
            "node_id": str(node_id),
            "level": level,
            "text": text,
        })
    except Exception as e:
        print(f"[mesh] could not send node-message ({e})")


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
        # Server's currently-running --n-blocks, learned from the
        # hello_ack handshake. Used to detect client/server drift
        # so MeshSplitFlux.configure() can block the run with a
        # 'Click Confirm to restart server' message.
        self.server_n_blocks: int | None = None
        # Set while a status-driven reconnect probe is in flight, so
        # we don't stack multiple probe threads when the server is
        # down for many polling intervals.
        self._probe_in_flight: bool = False

    def _ensure_open(self) -> socket.socket:
        if self._sock is not None:
            return self._sock
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # 5s connect timeout so probe-driven reconnects fail quickly
        # against an absent server (otherwise OS default can be ~60s).
        # Reverted to blocking after handshake — workflow recvs are
        # long-lived and shouldn't time out.
        s.settimeout(5.0)
        s.connect((self.host, self.port))
        self._sock = s
        # Handshake
        protocol.send_message(s, {"kind": "hello", "tensors": []}, [])
        header, _ = protocol.recv_message(s)
        s.settimeout(None)
        if header.get("kind") != "hello_ack":
            raise RuntimeError(f"unexpected handshake response: {header!r}")
        server_info = header.get("server_info", {}) or {}
        # Older servers may not include "n_blocks" — fall back to
        # n_total_loaded which has always been there.
        self.server_n_blocks = int(
            server_info.get("n_blocks",
                            server_info.get("n_total_loaded", 0)) or 0
        )
        print(f"[mesh] connected to {self.host}:{self.port}; server reports {server_info}")
        return s

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def try_reconnect(self) -> bool:
        """Best-effort reconnect attempt for the status indicator's
        sake. Quick TCP-level fail if the server is down (5s connect
        timeout in _ensure_open); full handshake if it's up. The
        in-flight guard prevents repeated polls (every 3s) from
        stacking probe threads while the server is genuinely down."""
        if self._sock is not None:
            return True
        if self._probe_in_flight:
            return False
        self._probe_in_flight = True
        try:
            self._ensure_open()
            return True
        except Exception:
            return False
        finally:
            self._probe_in_flight = False

    def reconfigure(self, new_n_blocks: int) -> None:
        """Tell the server to re-exec with a different --n-blocks. Sends
        the reconfigure message, waits for ack, then closes the socket.
        The server will execv itself; subsequent calls reconnect via
        the normal _ensure_open path and pick up the new server_n_blocks
        from the fresh hello_ack."""
        sock = self._ensure_open()
        protocol.send_message(sock, {
            "kind": "reconfigure",
            "tensors": [],
            "n_blocks": int(new_n_blocks),
        }, [])
        # Expect a single ack, then the socket dies as the server execvs.
        try:
            resp_header, _ = protocol.recv_message(sock)
            if resp_header.get("kind") != "reconfigure_ack":
                raise RuntimeError(
                    f"server sent {resp_header.get('kind')!r} instead of "
                    f"reconfigure_ack"
                )
        except (OSError, EOFError):
            # Server closed before we read the ack — treat as success;
            # _ensure_open will reconnect later and pick up the new state.
            pass
        # Drop the dead socket + reset LoRA-session bookkeeping (new
        # server process has no memory of what we shipped).
        self.close()
        self._last_sent_lora_session = ""
        # Optimistically record what we asked for; the next _ensure_open
        # will overwrite this with whatever the new server actually reports.
        self.server_n_blocks = int(new_n_blocks)

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
        try:
            protocol.send_message(sock, header, blobs)
            resp_header, resp_blobs = protocol.recv_message(sock)
        except (OSError, EOFError) as e:
            # Server went away (most likely a restart). Drop the dead
            # socket and reset LoRA bookkeeping — the new server process
            # has no memory of which sessions we shipped — then let the
            # closure rebuild and retry.
            print(f"[mesh] connection lost to {self.host}:{self.port} ({e}); will reconnect")
            self.close()
            self._last_sent_lora_session = ""
            raise MeshReconnect(str(e)) from e
        elapsed = time.time() - t0

        if resp_header.get("kind") != "forward_double_blocks_response":
            raise RuntimeError(f"unexpected response kind {resp_header!r}")

        wires = resp_header["tensors"]
        if len(wires) < 2:
            raise RuntimeError(f"response missing img/txt; got {len(wires)} tensors")
        img_back = codec.decode(wires[0], resp_blobs[0], device=device)
        txt_back = codec.decode(wires[1], resp_blobs[1], device=device)

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

        # Retry once on MeshReconnect. On the second pass through, the
        # client's _last_sent_lora_session is "" so the LoRA blob gets
        # rebuilt and re-shipped to the freshly-started server.
        for attempt in (1, 2):
            # ---- Client-LoRA forwarding (optional) ----
            client_lora_session = ""
            client_lora_blob = b""
            if forward_client_loras:
                patcher = patcher_capture.get("patcher")
                # When the client is slim-loaded, patches targeting back-half
                # blocks live in `_mesh_back_half_patches` (split out so
                # patch_model doesn't try to apply them to the stripped
                # stubs). Merge them with the live patches dict so the
                # filter still sees the full picture, then filter+remap.
                live = getattr(patcher, "patches", None) if patcher is not None else None
                stashed = getattr(patcher, "_mesh_back_half_patches", None) if patcher is not None else None
                if live or stashed:
                    combined = dict(live or {})
                    if stashed:
                        combined.update(stashed)
                    slim_patches = lora_io.filter_and_remap_patches(
                        combined,
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

            try:
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
            except MeshReconnect:
                if attempt == 2:
                    raise
                print("[mesh] retrying after reconnect")
                continue
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


def _capture_block_param_signature(block: nn.Module) -> dict:
    """Snapshot {leaf_name: (shape, dtype)} for every parameter and
    buffer in a block. Used at strip time so the stub can reproduce
    the same state_dict keys ComfyUI's LoRA mapper expects."""
    sig: dict = {}
    for name, param in block.named_parameters():
        sig[name] = (tuple(param.shape), param.dtype)
    for name, buf in block.named_buffers():
        sig[name] = (tuple(buf.shape), buf.dtype)
    return sig


class MeshRemoteStub(nn.Module):
    """Placeholder for a transformer block that runs on the remote
    mesh server. Holds NO parameters (zero VRAM), and its forward is
    never supposed to be invoked — patches_replace short-circuits
    these slots to the wire.

    Crucially, the stub still REPORTS the original block's param
    names in state_dict, with zero-storage tensors. ComfyUI's
    `comfy.lora.model_lora_keys_unet` walks state_dict keys to build
    the LoRA-key → model-key map; if the back-half keys vanished,
    every back-half LoRA key would log "lora key not loaded" and the
    server would never receive the LoRA. The synthetic entries cost
    nothing and keep the LoRA pipeline whole.
    """

    _mesh_stub = True

    def __init__(self, param_sig: dict | None = None):
        super().__init__()
        self._mesh_param_sig = param_sig or {}

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # Emit a zero-storage tensor at every key the original block had.
        # model_lora_keys_unet only iterates `.keys()`, so the value is
        # immaterial — but we keep the dtype right so downstream
        # introspection (size estimation, dtype reporting) doesn't choke.
        for leaf_name, (_shape, dtype) in self._mesh_param_sig.items():
            destination[prefix + leaf_name] = torch.empty(0, dtype=dtype)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # Silently consume any keys destined for this stub so that a
        # strict load_state_dict() round-trip doesn't blow up on the
        # synthetic entries. Nothing to actually load — the block lives
        # on the server.
        for k in [k for k in state_dict.keys() if k.startswith(prefix)]:
            state_dict.pop(k, None)

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "MeshRemoteStub.forward called — patches_replace did not "
            "intercept this block. Check that n_blocks_remote on the "
            "node matches the count of stripped blocks."
        )


def _force_release_block_vram(block: nn.Module) -> int:
    """Aggressively release the CUDA memory backing every parameter and
    buffer in `block`. Returns approximate bytes freed.

    Pattern: replace each parameter/buffer's `.data` with a zero-sized
    empty tensor on the same device + dtype. The original storage
    becomes unreferenced — ComfyUI's patcher / LoRA mapper / state_dict
    snapshots that still hold a reference to the Parameter object stay
    valid (the wrapper is alive), but its underlying tensor is now
    empty. After this, gc.collect() + torch.cuda.empty_cache() actually
    triggers the cudaFree.

    Without this step, simply replacing the block in the ModuleList with
    a parameter-less stub doesn't free VRAM, because ComfyUI's
    ModelPatcher cached a state_dict snapshot at model-load time that
    still holds references to every original Parameter object. The
    storage stays alive until those references die — which they don't,
    because the patcher persists for the session.
    """
    freed_bytes = 0
    for name, param in list(block.named_parameters(recurse=True)):
        try:
            sz = param.data.element_size() * param.data.numel()
            param.data = torch.empty(0, device=param.data.device, dtype=param.data.dtype)
            freed_bytes += sz
        except Exception as e:
            print(f"[mesh] _force_release: failed to release param {name}: {e}")
    for name, buf in list(block.named_buffers(recurse=True)):
        try:
            sz = buf.element_size() * buf.numel()
            buf.data = torch.empty(0, device=buf.device, dtype=buf.dtype)
            freed_bytes += sz
        except Exception as e:
            print(f"[mesh] _force_release: failed to release buffer {name}: {e}")
    return freed_bytes


def _strip_diffusion_back_half(
    diffusion: nn.Module,
    n_double_remote: int,
    n_single_remote: int,
):
    """Replace the back-half double_blocks and front N single_blocks
    with parameter-less stubs, freeing their VRAM. Mutates the shared
    nn.Module in place — see the comment in MeshSplitFlux.configure for
    the cache-trade reasoning.

    Three call cases:

    1. **Fresh model** — no prior strip. Strip the requested range.
    2. **Same config** — strip already matches request. No-op.
    3. **Increase** — request strips MORE blocks than prior. The
       additional blocks still hold their original weights, so we can
       extend the strip in place (incrementally).
    4. **Decrease** — request strips FEWER blocks than prior. The
       already-stripped blocks' weights are gone, so we can't restore
       them. Raise — user must reload the model.

    The increase path is what makes "change n_blocks_remote upward
    without relaunching ComfyUI" work."""
    n_total_doubles = len(diffusion.double_blocks)
    n_total_singles = len(diffusion.single_blocks)
    requested = (n_double_remote, n_single_remote, n_total_doubles, n_total_singles)

    prior = getattr(diffusion, "_mesh_strip_config", None)
    if prior is None:
        # Fresh strip — initial range.
        new_db_range = range(n_total_doubles - n_double_remote, n_total_doubles)
        new_sb_range = range(0, n_single_remote)
    else:
        prior_db, prior_sb, prior_td, prior_ts = prior
        if (prior_td, prior_ts) != (n_total_doubles, n_total_singles):
            # The model itself changed shape — different checkpoint
            # loaded into the same MODEL slot. Strip can't continue.
            raise RuntimeError(
                f"Model shape changed since last mesh-strip "
                f"(was {prior_td}+{prior_ts}, now {n_total_doubles}+{n_total_singles}). "
                "Reload the model before re-running with mesh."
            )
        if prior_db == n_double_remote and prior_sb == n_single_remote:
            return 0  # already stripped for this exact config
        if prior_db > n_double_remote or prior_sb > n_single_remote:
            # Decrease — would need to un-strip, but those weights are gone.
            raise MeshDecreaseNeedsReload(
                f"Decreased n_blocks_remote ({prior_db}→{n_double_remote} doubles, "
                f"{prior_sb}→{n_single_remote} singles). The stripped weights are "
                "gone for this session.\n\n"
                "Please restart ComfyUI to reload the model from disk. "
                "(Increasing n_blocks_remote works without restart.)"
            )
        # Increase on at least one axis. Strip only the NEWLY-back-half blocks.
        # Doubles: new range extends the strip earlier in the stack.
        new_db_range = range(
            n_total_doubles - n_double_remote,
            n_total_doubles - prior_db,
        )
        # Singles: new range extends the strip later in the stack.
        new_sb_range = range(prior_sb, n_single_remote)

    # Strip — capture each block's parameter signature first so the stub
    # can keep presenting those keys in state_dict (LoRA mapping needs them).
    # Force-release each block's parameter storage BEFORE swapping in the
    # stub, so the underlying CUDA memory actually becomes unreferenced —
    # see _force_release_block_vram for the why.
    stripped_count = 0
    for i in new_db_range:
        sig = _capture_block_param_signature(diffusion.double_blocks[i])
        _force_release_block_vram(diffusion.double_blocks[i])
        diffusion.double_blocks[i] = MeshRemoteStub(sig)
        stripped_count += 1
    for i in new_sb_range:
        sig = _capture_block_param_signature(diffusion.single_blocks[i])
        _force_release_block_vram(diffusion.single_blocks[i])
        diffusion.single_blocks[i] = MeshRemoteStub(sig)
        stripped_count += 1

    diffusion._mesh_strip_config = requested

    if stripped_count > 0:
        # Drop the now-orphaned tensors from CUDA's caching allocator.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # ensure pending ops finish before empty_cache
            torch.cuda.empty_cache()
        # Ask ComfyUI's model_management to actually evict any stale
        # GPU copies of the now-stripped weights. See the equivalent
        # comment in mesh_node_ltx.py for the reasoning.
        try:
            import comfy.model_management as mm
            mm.free_memory(1e30, mm.get_torch_device())
            mm.soft_empty_cache(True)
        except Exception as e:
            print(f"[mesh] FLUX strip: comfy free_memory call failed (non-fatal): {e}")

    return stripped_count


def _split_back_half_patches(
    patcher,
    n_double_remote: int,
    n_single_remote: int,
    n_total_doubles: int,
):
    """Pull patches that target now-stripped blocks out of
    patcher.patches and stash them on patcher._mesh_back_half_patches.
    ComfyUI's patch_model would otherwise try to apply them to the
    stubs and crash; the wire forwarder reads them from the stash to
    ship to the server.

    Front-half block patches, encoder patches, and final_layer patches
    stay in patcher.patches and apply locally as normal."""
    drop_db = n_total_doubles - n_double_remote
    back: dict = getattr(patcher, "_mesh_back_half_patches", None) or {}
    moved: list[str] = []
    for key in list(patcher.patches.keys()):
        if not key.startswith("diffusion_model."):
            continue
        sub = key[len("diffusion_model."):]
        if sub.startswith("double_blocks."):
            try:
                idx = int(sub.split(".")[1])
            except (ValueError, IndexError):
                continue
            if idx >= drop_db:
                moved.append(key)
        elif sub.startswith("single_blocks."):
            try:
                idx = int(sub.split(".")[1])
            except (ValueError, IndexError):
                continue
            if idx < n_single_remote:
                moved.append(key)
    for k in moved:
        back[k] = patcher.patches.pop(k)
    patcher._mesh_back_half_patches = back
    return len(moved)


class MeshSplitFlux:
    """Configure FLUX double-block split between the local GPU and a
    remote mesh server. Pass-through MODEL node — slot it between the
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
                                           "tooltip": (
                                               "Hostname or IP of the back-half server. "
                                               "127.0.0.1 = same machine (e.g. two GPUs). "
                                               "Possible examples: 192.168.x.x = LAN. "
                                               "100.x.x.x = VPN etc."
                                           )}),
                "remote_port": ("INT", {"default": 7777, "min": 1, "max": 65535,
                                        "tooltip": "TCP port the back-half server is listening on. Default 7777."}),
                "codec_mode": (["raw", "nvenc"], {"default": "nvenc",
                                                  "tooltip": (
                                                      "How activations get put on the wire. "
                                                      "'nvenc' = NVENC HEVC compresses 3-10× before sending — "
                                                      "the right choice for any slow wire (LAN, Tailscale, "
                                                      "residential broadband). "
                                                      "'raw' = uncompressed bf16 — only better when the wire "
                                                      "is faster than the codec encode/decode latency, i.e. "
                                                      "PCIe between two GPUs in the same machine."
                                                  )}),
                "codec_qp": ("INT", {"default": 18, "min": 0, "max": 51,
                                     "tooltip": "Lower = higher quality / less compression. 10 = near-lossless. 18 = sharp (default). Towards 28 the image gets noticeably softer with visible noise."}),
                "codec_lossless": ("BOOLEAN", {"default": False,
                                               "tooltip": "Use NVENC's lossless tuning (overrides QP, much larger bitstream)."}),
                "codec_tile_dim": ([1, 2, 4, 8], {"default": 8,
                                                   "tooltip": (
                                                       "How many channels to tile per Y/U/V plane in each NVENC frame. "
                                                       "Bigger tiles = fewer larger codec frames per encode = much "
                                                       "faster wall clock. 1=legacy (~600ms/round-trip), "
                                                       "4=balanced (~130ms), 8=default & most aggressive (~110ms). "
                                                       "Compression ratio is essentially unchanged across values."
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
            },
            "hidden": {
                # ComfyUI passes the runtime node id here; we use it to
                # send a `mesh-message` websocket event back to the JS
                # extension so it can render an inline banner under
                # this specific node (e.g. "restart ComfyUI" on a
                # decrease).
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "configure"
    CATEGORY = "mesh"
    OUTPUT_NODE = False

    def configure(self, model, n_blocks_remote, remote_host, remote_port, codec_mode, codec_qp, codec_lossless, codec_tile_dim, forward_client_loras, unique_id=None):
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

        # Block KSampler if the client's n_blocks_remote disagrees with
        # the server's currently-running --n-blocks. The JS surfaces a
        # 'Confirm: restart server with N=X' button that POSTs to
        # /mesh/reconfigure; once the server execvs and re-handshakes,
        # this check passes and the run proceeds. We only check when
        # n_blocks_remote > 0 (n=0 means 'no mesh', client runs
        # everything locally and the server isn't consulted).
        if n_blocks_remote > 0 and client.server_n_blocks is not None:
            if client.server_n_blocks != n_blocks_remote:
                pending_msg = (
                    f"n_blocks_remote ({n_blocks_remote}) differs from server "
                    f"({client.server_n_blocks}). Click the Confirm button to "
                    f"restart the server with n_blocks={n_blocks_remote}, "
                    f"or set n_blocks_remote back to {client.server_n_blocks}."
                )
                _send_node_message(unique_id, "warn", pending_msg)
                raise MeshServerNeedsReconfigure(pending_msg)

        # Capture vec_orig via a forward_pre_hook on the modulation module —
        # the server uses it to compute single-block modulation locally.
        vec_orig_capture: dict = {"vec_orig": None}
        _install_vec_orig_hook(diffusion, vec_orig_capture)

        # Free VRAM held by back-half block weights. This mutates the
        # shared diffusion_model in place — clone() does NOT deep-copy
        # the nn.Module, so the change is visible to any cached MODEL
        # reference too. The trade: if the user removes this node or
        # changes n_blocks_remote, they must reload the model (the
        # original weights are gone). _strip_diffusion_back_half is
        # idempotent for identical configs and raises clearly otherwise.
        if n_blocks_remote > 0:
            try:
                stripped = _strip_diffusion_back_half(diffusion, n_double_remote, n_single_remote)
            except MeshDecreaseNeedsReload as e:
                # Surface the decrease-needs-restart message inline on
                # the node so the user sees it without scrolling the
                # console. Then re-raise so ComfyUI flags this run.
                _send_node_message(unique_id, "warn", str(e))
                raise
            # Clear any prior banner from this node — strip succeeded.
            _send_node_message(unique_id, "clear", "")
            if stripped:
                print(f"[mesh] stripped {n_double_remote} double_blocks + "
                      f"{n_single_remote} single_blocks from client VRAM")

        # ModelPatcher copy + register the per-block overrides via the
        # canonical comfy.model_patcher API. Using set_model_patch_replace
        # rather than mutating model_options directly so we don't fight
        # the ModelPatcher's copy-on-write semantics.
        m = model.clone()

        # Move patches targeting now-stripped blocks out of m.patches so
        # ComfyUI's patch_model doesn't try to apply them to the stubs.
        # The wire forwarder reads them back from _mesh_back_half_patches.
        if n_blocks_remote > 0:
            _split_back_half_patches(m, n_double_remote, n_single_remote, n_double_blocks)

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

        print(f"[mesh] offloading {n_double_remote}/{n_double_blocks} doubles + "
              f"{n_single_remote}/{n_single_blocks} singles "
              f"(double_block intercept at index {split_index}); "
              f"server={remote_host}:{remote_port}; "
              f"codec={codec_mode} qp={codec_qp} lossless={codec_lossless} tile_dim={codec_tile_dim}")

        return (m,)


NODE_CLASS_MAPPINGS = {
    "MeshSplitFlux": MeshSplitFlux,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshSplitFlux": "ComfyUI Mesh : Icarus",
}
