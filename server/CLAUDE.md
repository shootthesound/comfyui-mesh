# CLAUDE.md — comfyui-mesh server (the back-half side)

> **For a Claude Code instance running on the host that holds the back-half GPU.** This is a setup brief — read it end to end before acting. The folder you're in is the deliverable that needs to run on this machine. The client side (ComfyUI custom node) lives on a separate host.

---

## What this is

A two-host split rig for FLUX.2 inference:

- **Client host (the other machine, e.g. a 5090):** runs ComfyUI normally with a custom node (`Mesh Split FLUX`) that intercepts FLUX's double-block forward at a configurable index and ships the remaining double-block work to *this* machine over TCP.
- **Server host (this machine, e.g. a 4090):** runs `mesh_server.py`. **Slim-loads only the back-half double_blocks** of the FLUX checkpoint from disk — does NOT load the full model. Listens on TCP for `forward_double_blocks` requests.
- **The wire** between the two carries codec-compressed activations (NVENC HEVC YUV444 via `nvenc-pframe`). At `codec_qp=18` the wire payload is ~12-15 MB per round-trip, ~120 MB per generation total (4 timesteps × 2 directions). Viable over LAN, Tailscale, even residential broadband.

The slim load is the load-bearing architectural property — it's what makes this rig work for models that don't fit on either device whole.

---

## What's already in this folder

```
.
├── CLAUDE.md                ← you are here
├── README.md                ← human-facing version of this brief
├── requirements.txt         ← `pip install -r` for the standalone server
├── mesh_server.py           ← the server (slim-loads via safetensors.safe_open)
├── mesh_server_gui.py       ← Tkinter wrapper around the server
├── codec.py                 ← tensor ↔ NVENC bitstream (per-channel quant + HEVC)
├── protocol.py              ← length-prefixed TCP framing
├── vec_io.py                ← FLUX.2 vec/modulation tuple serializer
├── nvenc_pframe/            ← BUNDLED codec source — no separate install needed
├── smoke_test_server.py     ← validates model load + back-half forward
├── install_check.py         ← env pre-flight check
├── run_server.bat           ← headless launcher, no GPU pinning
├── run_server_gpu0.bat      ← same-host two-GPU: server on physical GPU 0
├── run_server_gpu1.bat      ← same-host two-GPU: server on physical GPU 1
├── run_server_cpu.bat       ← CPU / system-RAM mode
├── run_server_gui.bat       ← launch the GUI
└── flux-2-klein-9b-fp8.safetensors   ← model weights (9.4 GB), already in place
```

`codec.py / protocol.py / vec_io.py / nvenc_pframe/` must stay byte-identical to the client-side copies. They're the wire contract. **Do not edit them in isolation** — if they need to change, the change happens on the client first and is mirrored here.

**Your job: get the env right, run the smoke test, then launch the server.**

---

## Setup tasks (in order)

### 1. Verify the environment

```
cd /d <this folder>
python install_check.py
```

Reports OK / MISSING / BROKEN for every dependency. Read it, fix anything MISSING before continuing. Covers: torch, safetensors, einops, CUDA, ComfyUI source, nvenc-pframe, the model file.

### 2. Get ComfyUI's source importable

The server imports `comfy.sd.load_diffusion_model_state_dict` to handle fp8 quantization correctly. The UI isn't needed; only the `comfy/` Python package needs to be importable.

```
git clone https://github.com/comfyanonymous/ComfyUI ../ComfyUI
pip install -r ../ComfyUI/requirements.txt
```

Launchers default to `..\ComfyUI` relative to this folder. Override with `COMFYUI_PATH=C:\path\to\ComfyUI` before launching.

**Critical:** the ComfyUI version here should match (or be reasonably close to) the version on the client. The fp8 detection and FLUX implementation evolve; mismatched versions = silent-correctness bugs.

### 3. Install Python dependencies

```
pip install -r requirements.txt
```

That gives you `torch`, `safetensors`, `einops`, and `cuda-bindings`. The codec wrapper itself is **bundled** at `./nvenc_pframe/` — no separate install. Confirm it loads:

```
python -c "import nvenc_pframe; print(nvenc_pframe.__file__)"
```

The `__file__` should point INTO this folder's `nvenc_pframe/__init__.py`, not somewhere in site-packages. If it points elsewhere, an older copy is shadowing — uninstall it (`pip uninstall nvenc-pframe`).

### 4. Smoke-test the server

```
python smoke_test_server.py --weights flux-2-klein-9b-fp8.safetensors --n-blocks 4
```

Expected output:

```
[server] reading checkpoint header from flux-2-klein-9b-fp8.safetensors
[server] checkpoint has 8 double_blocks; loading 4 (skipping first 4)
[server] slim state dict: 119 tensors, 2.22 GB (full would be ~4.4 GB)
[server] remapped 112 fp8 layer entries -> 32 for slim model
[server] model loaded; double_blocks=4 single_blocks=0 hidden_size=4096 global_modulation=True
[smoke] model load: ~4s
[smoke] back-half forward: ~30 ms avg
[smoke] OK
```

Reference timing from a 5090: 28.6 ms for 4 blocks (7.2 ms/block). On a 4090 expect somewhat slower — still well under 100 ms per timestep.

**If the smoke test fails, do not launch the live server.** Debug it first. Common failures:

- `comfy.sd.load_diffusion_model_state_dict returned None` → wrong weight file or ComfyUI version mismatch
- `Freqs dimension N must be head_dim//2` → pe synthesis shape issue; the smoke test should be using `model.pe_embedder(ids)`, verify
- `not enough values to unpack` on vec → global_modulation detection mismatch
- `AttributeError: 'NoneType' object has no attribute 'device'` in cast_bias_weight → fp8 metadata not being remapped (should be fixed in current mesh_server.py; verify you're on the latest)
- CUDA OOM → slim load too generous; lower `--n-blocks`

### 5. Launch the live server

Five launcher variants. Pick the one that matches the deployment:

```
run_server_gui.bat        REM Tkinter UI — recommended for first run
run_server.bat            REM headless cross-machine, no GPU pinning
run_server_gpu0.bat       REM same-host two-GPU: server on physical GPU 0
run_server_gpu1.bat       REM same-host two-GPU: server on physical GPU 1
run_server_cpu.bat        REM CPU / system-RAM mode (slow; requires client codec_mode=raw)
```

Each headless launcher has an EDIT-ME line at the top:

```
set N_BLOCKS=4
```

This MUST match the client's `n_blocks_remote` setting on the Mesh Split FLUX node. **No protocol validation in v1** — if the numbers disagree, output is wrong, no error.

For the typical case (this host is on the OTHER side of the wire from the ComfyUI machine, and has one GPU), `run_server.bat` is the right launcher.

The server should end with `[server] listening on 0.0.0.0:7777`.

### 6. Network: tell Peter the IP/port

The client's Mesh Split FLUX node needs to know how to reach this machine. Two paths:

- **LAN:** `ipconfig` (Windows) or `ip addr` (Linux). Looks like `192.168.x.x`.
- **Tailscale:** `tailscale ip -4`. Looks like `100.x.x.x`. Works off-LAN.

Report which option is available + the IP + port back to Peter.

---

## What to watch for once it's running

Per-call log line per FLUX timestep:

```
[server] client connected: ('192.168.x.x', NNNNN)
[server] forward 4 blocks (client said start=4): decode XX ms  fwd XX ms  enc XX ms  in 14.5 MB  out 4.2 MB
```

You should see 4 of those per generation (FLUX.2 Klein distilled does 4 sampler steps).

If the lines aren't appearing when the client queues a workflow, the problem is upstream: connection refused, firewall, or the node isn't slotted between the loader and the sampler.

---

## What NOT to do

- **Do not modify `codec.py`, `protocol.py`, or `vec_io.py`** in isolation. They must stay byte-identical with the client. Fix upstream and re-copy.
- **Do not try to reimplement or replace the bundled `nvenc_pframe/`.** It's the codec we ship with. If you think it's broken, escalate — don't fork.
- **Do not load the full model "for safety".** The whole point is slim load — for models too big to fit whole, `--n-blocks N` is the only viable path. Server loads only the LAST N double_blocks plus encoders/modulation needed for ComfyUI's detection (~60-80 MB combined).
- **Do not refuse QP=28 requests.** That warning was for CFD chaotic dynamics, not diffusion. FLUX's residual stream tolerates QP=28 fine — produces slightly softer images than QP=18, doesn't collapse. Domain-dependent envelope; client picks per-workload.
- **Do not redistribute the model weights.** Peter's licensed copy; internal use only.
- **Do not modify mesh_server.py to log activations to disk.** No tensor dumps. The wire transport is the point.

---

## What success looks like

- `install_check.py` reports OK across the board
- `smoke_test_server.py` runs end-to-end and prints `[smoke] OK`
- Launcher prints `[server] listening on 0.0.0.0:7777` and stays running
- When Peter queues a FLUX workflow on the client, you see 4 `[server] forward …` lines per generation, with byte counts in the 10-20 MB range at QP=18

Once you've got the server humming, Peter takes the client side for A/B comparison (mesh-split vs all-local).

---

## Escalate if any of these come up

- Bundled `nvenc_pframe/` failing to import (likely missing `cuda-bindings` PyPI dep — `pip install cuda-bindings`)
- ComfyUI version on this host substantially different from the client's (silent-correctness risk)
- Model file size / tensor count doesn't match expected (9.4 GB / 425 tensors for FLUX.2 Klein 9B fp8)
- CUDA driver / NVENC SDK version incompatible with what nvenc-pframe expects
- Slim load reports "slim state dict: X GB" where X is suspiciously close to the full model size (means filtering isn't working)

Otherwise: be brief, report back when it's listening, let Peter drive from the client side.
