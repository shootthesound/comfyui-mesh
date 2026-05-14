# CLAUDE.md — comfyui-mesh server (the 4090 side)

> **For the Claude Code instance running on Peter's 4090.** This is a setup brief — read it end to end before acting. The folder you're in (`U:/comfyuiserver/`) is the deliverable that needs to run on this machine. The other half of the rig is on Peter's 5090, in `S:/Auto/ComfyUI_SEC/ComfyUI/custom_nodes/comfyui-mesh/`.

---

## What this is

A two-machine split rig for FLUX.2 Klein 9B distilled inference:

- **5090 (other machine):** runs ComfyUI normally. Has a custom node (`Mesh Split FLUX`) that intercepts FLUX's double-block forward at a configurable split index and ships the remaining double-block work to *this* machine over TCP.
- **4090 (this machine):** runs `mesh_server.py`, a standalone TCP server that loads the full FLUX.2 Klein 9B weights and serves `forward_double_blocks` requests — taking activations from the front half and returning them after running the back half locally on this 4090.
- **The wire** between the two carries codec-compressed activations (NVENC HEVC YUV444 via `nvenc-pframe`). At `codec_qp=18` the wire payload is ~3 MB per round-trip vs ~30 MB raw. With FLUX.2 Klein distilled's 4 sampler steps and one split point, that's just 4 round-trips per generation — viable over LAN, Tailscale, even residential broadband.

The win the demo is supposed to show: same-quality FLUX generation on the 5090 while half the model's compute happens on the 4090, talking over a wire that compression makes fast enough to be invisible. Same primitive that powers `llmtests_native` — but for diffusion instead of LLM.

## What's already in this folder

```
U:/comfyuiserver/
├── CLAUDE.md                         ← you are here
├── README.md                         ← human-facing version of this brief
├── mesh_server.py                    ← the server (the thing you'll run)
├── codec.py                          ← tensor ↔ NVENC bitstream wire codec
├── protocol.py                       ← length-prefixed TCP framing
├── vec_io.py                         ← FLUX.2 vec/modulation tuple serializer
├── smoke_test_server.py              ← validates model load + back-half forward
├── install_check.py                  ← pre-flight env check
├── run_server.bat                    ← Windows launcher (cross-machine; no GPU pinning)
├── run_server_gpu0.bat                ← same-host two-GPU: pin server to physical GPU 0
├── run_server_gpu1.bat                ← same-host two-GPU: pin server to physical GPU 1
├── run_server_cpu.bat                 ← CPU / system-RAM mode (slow; architectural-proof only)
└── flux-2-klein-9b-fp8.safetensors   ← model weights, 9.4 GB, already in place
```

All the code is verified to run on the 5090 side. The thing that's untested is whether your 4090 environment has the right pieces. **Your job: get the env right, run the smoke test, then launch the server.**

## Setup tasks (in order)

### 1. Verify the environment

```
cd /d U:\comfyuiserver
python install_check.py
```

This reports OK / MISSING / BROKEN for every dependency. **Read what it says. Fix anything MISSING before continuing.** The check covers: torch, safetensors, einops, CUDA, ComfyUI source, nvenc-pframe, the model file.

### 2. Get ComfyUI's source onto the machine

The server imports `comfy.sd.load_diffusion_model` directly — this handles FLUX.2's fp8 quantization correctly. You don't need to run ComfyUI's UI, only its `comfy/` Python package needs to be importable.

```
git clone https://github.com/comfyanonymous/ComfyUI ../ComfyUI
pip install -r ../ComfyUI/requirements.txt
```

The launcher (`run_server.bat`) defaults to `..\ComfyUI` (i.e. `U:\ComfyUI`) relative to this folder. If you put it somewhere else, set `COMFYUI_PATH=C:\path\to\ComfyUI` before launching.

### 3. Install nvenc-pframe

This is the private NVENC codec wrapper. **It's almost certainly already installed on this 4090** because `U:/llmtests_native` requires it and that's been used on this machine before. Verify with:

```
python -c "import nvenc_pframe; print(nvenc_pframe.__version__)"
```

If it's not there, it's in `W:/Peter/Documents/Development/NVENC Activations/vortex/` (a `pip install -e` target). The 4090 might not have W: mounted — in that case ask Peter where to find it. **Do not try to reimplement it; it's the load-bearing IP.**

If nvenc-pframe is genuinely unavailable, the server still works in `codec_mode=raw` (no compression). That's only useful for protocol-validation testing.

### 4. Run the server-side smoke test

```
python smoke_test_server.py --weights flux-2-klein-9b-fp8.safetensors --start-block 4
```

Expected output on success:

```
[server] loading FLUX.2 Klein 9B via comfy.sd.load_diffusion_model
[server] model loaded; double_blocks=8 single_blocks=24 hidden_size=4096 global_modulation=True
[smoke] model load: ~6s (from local SSD) or up to several minutes (from network drive)
[smoke] pe shape=(1, 1, 4352, 64, 2, 2) dtype=torch.float32
[smoke] back-half forward: XX ms avg over 4 runs
[smoke] OK
```

Reference timing from the 5090: 28.6 ms for 4 blocks (7.2 ms/block). On this 4090 expect somewhat slower — that's fine, FLUX.2 Klein distilled only does 4 sampler steps so total back-half compute is well under a second per generation.

**If this smoke test fails, do not launch the live server.** Debug it first. Common failure modes:

- `comfy.sd.load_diffusion_model returned None` → wrong weight file or ComfyUI version mismatch. Check the file is the 9.4 GB fp8 distilled checkpoint Peter expects.
- `Freqs dimension N must be head_dim//2` → pe synthesis shape issue. Should be fixed in the current `smoke_test_server.py` but worth checking it generates pe via `model.pe_embedder(ids)`.
- `not enough values to unpack` on the vec tuple → `global_modulation` detection issue. Check `model.params.global_modulation` matches the smoke test's branching logic.
- CUDA OOM → model is ~15-20 GB in VRAM as bf16. The 4090 has 24 GB; should fit but might be tight if anything else is using the GPU.

### 5. Launch the live server

Four launcher variants — pick the one that matches the deployment:

```
run_server.bat            REM cross-machine network use (default; no GPU pinning)
run_server_gpu0.bat       REM same-host two-GPU rig, server on physical GPU 0
run_server_gpu1.bat       REM same-host two-GPU rig, server on physical GPU 1
run_server_cpu.bat        REM CPU / system-RAM mode — REQUIRES codec_mode=raw on client
```

For the standard scenario where this 4090 is on the OTHER side of the wire
from a 5090, the plain `run_server.bat` is right — there's only one card on
this host, no ambiguity. The `_gpu0` / `_gpu1` variants exist for the
single-machine two-GPU experiment (where ComfyUI runs on one card and the
server pins itself to the other via `CUDA_VISIBLE_DEVICES`).

The `_cpu` variant hides CUDA entirely so the model loads into system RAM
and the back-half forward runs on CPU cores. Slow (~30-90s per timestep)
but proves the architectural decoupling — with raw codec mode the server's
hardware can be anything PyTorch runs on. If `comfy.sd.load_diffusion_model`
errors on fp8 in CPU mode (it might — fp8 ops are CUDA-tuned), the user
needs a bf16 FLUX.2 Klein checkpoint instead and to point the WEIGHTS line
of `run_server_cpu.bat` at it.

Or directly:

```
python mesh_server.py --weights flux-2-klein-9b-fp8.safetensors --bind 0.0.0.0 --port 7777
```

Should end with `[server] listening on 0.0.0.0:7777` and sit waiting for the 5090 to connect.

### 6. Network: tell Peter the IP/port

The 5090's `Mesh Split FLUX` node needs to know how to reach this machine. Two paths:

- **Plain LAN**: get this machine's local IP from `ipconfig`. Looks like `192.168.x.x`. The 5090 enters this in `remote_host`.
- **Tailscale**: get this machine's Tailscale IP (`tailscale status`). Looks like `100.x.x.x`. Works for cross-network or off-LAN use.

Report which option is available + the IP back to Peter. He'll plug it into the node on the 5090 and queue a workflow.

## What to do once it's running

The server is single-client at v1. When the 5090 connects you'll see:

```
[server] client connected: ('192.168.x.x', NNNNN)
[server] forward [4..8): decode XX ms  fwd XX ms  enc XX ms  in 3.2 MB  out 3.4 MB
... (4 of these per generation, one per sampler timestep)
```

If you don't see those lines when Peter queues a workflow:
- Check the connection: is the port open? Did the firewall block it?
- Check the codec: `codec_mode=nvenc` on the client side requires nvenc-pframe here too.
- Check the split index: `split_index=4` with 8 double_blocks runs blocks [4..8) here. `split_index=0` runs all 8 here.

## What NOT to do

- **Do not modify `codec.py`, `protocol.py`, or `vec_io.py`.** They're mirrors of the client-side files on the 5090; they must stay byte-identical or the protocol breaks. If anything looks wrong, fix it on the 5090 and re-copy.
- **Do not try to reimplement nvenc-pframe.** The codec wrapper is private IP — if it's missing, ask Peter, don't roll your own.
- **Codec QP envelope is domain-dependent.** On CFD-style chaotic dynamics QP=28 falls off the attractor (Karman shedding stops) — that's where the original audit said "don't use it". For *diffusion* activations (this rig's actual workload), QP=28 has been empirically tested by Peter and produces only a slightly softer image vs QP=18 at the same seed — well inside FLUX's residual-stream noise tolerance. So QP=10 / 18 / 28 are all valid knobs here; the choice is bandwidth-vs-sharpness, not safety. Don't refuse QP=28 requests from the client.
- **Do not redistribute the model weights.** Klein 9B fp8 is Peter's licensed copy; this is internal use, not redistribution.
- **Do not modify mesh_server.py to log activations to disk.** No tensor dumps. The wire transport is the point; intermediate captures aren't.

## What success looks like

- `install_check.py` reports OK across the board.
- `smoke_test_server.py` runs end-to-end and prints `[smoke] OK`.
- `run_server.bat` prints `[server] listening on 0.0.0.0:7777` and stays running.
- When Peter queues a FLUX workflow on the 5090, you see 4 `[server] forward [...]` lines per generation, in-byte counts ~3 MB at QP=18.

Once you've got this server humming, Peter takes over on the 5090 side for the actual A/B comparison (mesh-split vs all-local, image quality side-by-side).

## Context Peter would want flagged

If any of these come up, raise them before pushing through:

- nvenc-pframe missing AND not findable on disk (means the codec demo isn't reachable, only `raw` mode works)
- ComfyUI version on the 4090 substantially different from the 5090's (FLUX.2 implementation may have drifted; mismatched `comfy.ldm.flux` between machines is the most likely silent-correctness bug)
- The model file at `U:/comfyuiserver/flux-2-klein-9b-fp8.safetensors` doesn't match the 9.4 GB / 425-tensor shape expected
- CUDA driver / NVENC SDK version on this 4090 incompatible with what nvenc-pframe expects

Otherwise: be brief, report back when it's listening, and let Peter drive from the 5090.
