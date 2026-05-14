# comfyui-mesh — back-half server (the 4090 side)

Companion to the `comfyui-mesh` ComfyUI custom node that lives on the
5090 (in `custom_nodes/comfyui-mesh/`). This folder gets copied to the
4090 machine and runs as a long-lived TCP server. Per request, it
takes activations from the front half of the FLUX double-block stack,
runs the back half through its own copy of the weights, and ships the
result back over the wire.

## What this folder contains

```
comfyuiserver/
├── mesh_server.py                       — the server (this is what you run)
├── codec.py                             — torch tensor ↔ NVENC bitstream wire codec
├── protocol.py                          — length-prefixed TCP message format
├── run_server.bat                       — Windows launcher
├── flux-2-klein-9b-fp8.safetensors      — FLUX.2 Klein 9B fp8 weights (9.4 GB)
└── README.md                            — this file
```

## Setup on the 4090 machine

1. **Install Python dependencies.** Inside whatever venv you use:

   ```
   pip install torch safetensors einops
   ```

2. **Get ComfyUI's source on the box** (the server imports its FLUX
   implementation directly to avoid version drift). You don't need to
   run the UI; you just need its `comfy/` package importable. Easiest:

   ```
   git clone https://github.com/comfyanonymous/ComfyUI ../ComfyUI
   pip install -r ../ComfyUI/requirements.txt
   ```

   The launcher (`run_server.bat`) defaults to `..\ComfyUI` relative to
   this folder. Override with `COMFYUI_PATH=C:\path\to\ComfyUI`.

3. **Install nvenc-pframe** (optional — only needed if the 5090 sends
   `codec_mode=nvenc`. Without it, the wire falls back to `raw` mode,
   which uses ~5–15× more bandwidth but works for testing).

   nvenc-pframe is the private codec wrapper; install it the same way
   you installed it for `llmtests_native` if you've already done that.

4. **Confirm the model file is here.** It should be 9.4 GB:

   ```
   dir flux-2-klein-9b-fp8.safetensors
   ```

## Running the server

```
run_server.bat
```

You should see something like:

```
[server] using ComfyUI at: U:\ComfyUI
[server] reading state dict from U:\comfyuiserver\flux-2-klein-9b-fp8.safetensors
[server] state dict has NNNN tensors
[server] detected 8 double_blocks, 32 single_blocks
[server] Flux module instantiated
[server] model loaded on cuda:0 as torch.bfloat16
[server] listening on 0.0.0.0:7777
```

The server will sit waiting. When the 5090's ComfyUI runs a workflow
with the `Mesh Split FLUX` node configured to point at this machine's
IP, you'll see per-call timing prints.

## What the wire actually carries

Per FLUX timestep (4 timesteps in distilled mode), one round-trip:

- **Send** (5090 → 4090): img tensor `[B, T, H]` (typically 33 MB at fp16
  for 1024×1024), txt tensor (~2 MB), vec + pe + attn_mask (small)
- **Receive** (4090 → 5090): updated img + txt after running blocks
  `[split_index..end_of_double_blocks)`

In `nvenc` codec mode the img tensor is compressed via NVENC HEVC
YUV444 before transmission. Compression ratio depends on QP setting:

| Mode | Typical CR | Wire bytes for 33 MB tensor |
|---|---:|---:|
| raw | 1× | 33 MB |
| nvenc lossless | 2–4× | 8–17 MB |
| nvenc qp10 | 6–12× | 3–6 MB |
| nvenc qp18 | 15–40× | ~1 MB |

## Network setup notes

- **LAN:** the 5090 connects directly to this 4090's LAN IP. Use
  `ipconfig` here to get the IP, set the `Mesh Split FLUX` node's
  `remote_host` to that.
- **Tailscale:** works the same way — use the 4090's Tailscale IP
  (typically `100.x.x.x`). The codec was designed to be the slow-wire
  win, so Tailscale's encrypted overlay is a perfectly valid target.
- **Single-machine loopback test:** point `remote_host=127.0.0.1` and
  run this server on the same box as ComfyUI. Useful for debugging
  the protocol; obviously no PCIe-vs-network speedup.

## What it does NOT do (yet)

- No multi-client support (one ComfyUI client at a time)
- No KV/cache prefetch — each timestep ships activations fresh
- No batched-timestep mode (4 timesteps × 4 round-trips, not 1)
- No fault tolerance: if the connection drops mid-generation, you
  restart the workflow

For a single-user creative-tooling workflow this is fine. If you start
sharing the 4090 across users, that's the obvious next layer.
