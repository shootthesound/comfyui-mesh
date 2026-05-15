# comfyui-mesh

**Split a diffusion model across two GPUs over a network. The
activations between them get compressed live by NVIDIA's idle video
codec silicon.**

A 9 GB FLUX.2 model running on a 5090 desktop with its back half
offloaded to a 4090 elsewhere on the LAN. Or two cards in the same box
without NVLink. Or your friend's GPU over Tailscale. The bandwidth that
would normally make this miserable stops being the bottleneck because
NVENC compresses the bytes on the wire while they're already on the
GPU.

```
                ┌─────────────────┐                     ┌─────────────────┐
                │  ComfyUI 5090   │   NVENC HEVC wire   │  Server 4090    │
                │                 │ ─── ~10 MB / step ─►│                 │
   img latent ──┤ front-half      │                     │ back-half       │── img latent
                │ blocks + VAE    │ ◄────────────────── │ slim-loaded     │
                └─────────────────┘    ~10 MB / step    └─────────────────┘
                       LoRAs work transparently across the wire
```

For FLUX.2 Klein 9B distilled: **4 sampler timesteps × ~130 ms wire
round-trip = ~0.5 s of wire overhead per generation**. The rest is
just diffusion.

---

## The thing that's new

Every modern Nvidia GPU has dedicated **NVENC** silicon that compresses
H.265/HEVC video at sub-millisecond per frame. During ML inference it
sits 100% idle — none of the compute uses it.

This rig treats ML activations as video frames and feeds them to NVENC.
A FLUX activation tensor is `[batch, tokens, channels]`; we pack
multiple channels per Y/U/V plane in a grid, quantize per-channel to
uint8, and the codec compresses the result by 3–10× depending on QP.
The output bitstream is what crosses the wire. The receiver runs NVDEC
to reverse it.

**The codec runs on dedicated silicon that wasn't going to do anything
else anyway.** So you get compression "for free" in the sense that you
weren't using those transistors. The wire-byte savings convert directly
to wall-clock savings on any bandwidth-limited transport.

That's it. The rest is plumbing.

---

## What works today

- **FLUX.2 (any variant)** — Klein 9B, dev, schnell, whatever Black
  Forest Labs ships under the FLUX.2 architecture. Tested end-to-end.
- **LoRAs**: any format ComfyUI itself supports — Kohya, Diffusers PEFT,
  BFL Flux, USO, Wan Fun, SimpleTuner, native, etc. Includes the full
  weight-adapter family: **lora, loha, lokr, glora, oft, boft**, plus
  `diff` / `set` style patches.
- **Topologies**:
  - Cross-machine over LAN (gigabit OK, 2.5G/10G better)
  - Cross-machine over Tailscale (residential broadband fine for
    FLUX.2-distilled's 4-step samplers)
  - Same machine with two GPUs (no NVLink required)

**Models that are NOT supported (yet)** — anything that isn't FLUX.2.
The architectural differences are real (block signatures, modulation,
vec structure), but they're not blockers — just code that needs
writing. Top of the queue when community demand says go:

- **Wan** (2.1 / 2.2 / VACE) — video model with very similar DiT shape
- **LTX-Video** — distilled video diffusion, would benefit hugely from
  the split-rig (long sequences, big activations)
- **FLUX.1** — same family as FLUX.2, minor handling differences
- **SD3 / SD3.5** — MMDiT, related architecture
- **HunyuanVideo / Qwen-Image / Chroma** — each has its own quirks

**If you want one of these added, support the project** (link below) and
tell me which — community demand drives the priority list.

---

## Quick start — cross-machine (the headline use case)

Two physical machines, one running ComfyUI, the other running the
mesh server. Connected over your LAN or Tailscale.

### On the ComfyUI host

1. Drop the `comfyui-mesh/` folder into `ComfyUI/custom_nodes/`.
2. Restart ComfyUI. The Manager runs `pip install -r requirements.txt`
   automatically — that's a single line, `cuda-bindings`. The codec
   wrapper (`nvenc_pframe/`) is **bundled in the folder**, no separate
   install.
3. The nodes appear under the `mesh` category. You're done.

### On the back-half host

1. Copy the `server/` folder somewhere convenient.
2. Open a terminal in that folder and run **one command**:
   ```
   install.bat
   ```
   It creates a `.venv`, clones ComfyUI as a sibling folder if missing,
   installs torch + cuda-bindings + dependencies, runs an env check.
   Re-running is idempotent.
3. Drop your FLUX.2 safetensors checkpoint in the folder
   (`flux-2-klein-9b-fp8.safetensors` etc).
4. Launch via the GUI (recommended for first run):
   ```
   run_server_gui.bat
   ```
   Pick the model file, pick `n_blocks` (how many blocks to host —
   spinbox shows the range, default 4), pick port, click **Start
   Server**.

The server prints `[server] listening on 0.0.0.0:7777` when ready.

### Wire it up in a workflow

```
UNETLoader  →  (optional LoraLoader)  →  Mesh Split FLUX  →  KSampler
```

Set on the `Mesh Split FLUX` node:

- **`n_blocks_remote`** = same number as the server's GUI (default 4)
- **`remote_host`** = the server's LAN IP, e.g. `192.168.0.18`, or
  Tailscale IP `100.x.x.x`
- **`remote_port`** = `7777`
- **`codec_mode`** = `nvenc`, **`codec_qp`** = `18`, **`codec_tile_dim`** = `4`
- **`forward_client_loras`** = **ON** (so any LoraLoader-loaded LoRA
  affects the back half too)

Queue a generation. Server log shows one `[server] forward …` line
per timestep, with byte counts. Done.

---

## Quick start — same machine, two GPUs (no NVLink)

ComfyUI and the server live on the same machine but pinned to different
GPUs. You get to use a 4090 + 5090 (or any pair) without buying
NVLink-capable cards.

1. On the same host, install the node (as above) and the server
   (as above).
2. Launch ComfyUI normally — it grabs whatever GPU it sees first
   (usually `cuda:0`).
3. Launch the server **pinned to the OTHER GPU**:
   ```
   run_server_gpu1.bat    # pins server to physical GPU 1
   ```
   (or `run_server_gpu0.bat` if ComfyUI is on GPU 1.) These set
   `CUDA_VISIBLE_DEVICES` so the two processes don't fight over the
   same card.
4. In the workflow, set `remote_host = 127.0.0.1` (loopback).

That's it. The two processes share PCIe but each only sees its own
GPU.

**Specifically for same-host setups:** since PCIe between two GPUs in
the same desktop is ~32 GB/s — much faster than what the codec
encode/decode takes — you'll get better wall-clock with
`codec_mode = raw` instead of `nvenc`. The codec is the right tool for
slow wires (LAN, Tailscale, residential broadband); on PCIe it's
overkill and adds latency. Set `codec_mode = raw` for same-host pairs.

---

## What the two nodes do

### `Mesh Split FLUX (5090 ↔ 4090)`

Pass-through MODEL node. Slot it between the model loader (or
LoraLoader) and the sampler. Its parameters:

| Parameter | Default | What it controls |
|---|---|---|
| `model` | — | The loaded FLUX MODEL |
| `n_blocks_remote` | 4 | How many transformer blocks run remotely (counts double-blocks first, then single-blocks). For Klein 9B: max 32. For FLUX.2 dev: max 56. **Must match the server's `--n-blocks`.** |
| `remote_host` | `127.0.0.1` | Server's hostname / LAN IP / Tailscale IP |
| `remote_port` | `7777` | Server's TCP port |
| `codec_mode` | `nvenc` | `nvenc` for slow wires, `raw` for same-host PCIe |
| `codec_qp` | `18` | NVENC quality. 10=near-lossless, 18=default, 28=highest compression |
| `codec_lossless` | OFF | NVENC lossless tuning (still has uint8 quant floor) |
| `codec_tile_dim` | `4` | Channels-per-frame tile size. Higher = fewer larger NVENC frames = ~5× faster. 4 is a strong default. |
| `forward_client_loras` | ON | Ship client-side LoraLoader patches to server so the LoRA effect covers back-half blocks too |

### `Mesh Status`

Pure-output node. Reports per-generation stats (bytes sent / received,
last-call latency, codec ratio). Drop anywhere in the graph and watch
the ComfyUI console.

---

## Honest performance numbers

Measured on FLUX.2 Klein 9B distilled (1024×1024, 4 sampler steps), RTX
5090 client, RTX 4090 server, gigabit LAN, `tile_dim=4` everywhere:

| Setup | Wire / step | Total per generation | vs all-local |
|---|---:|---:|---:|
| Local only (5090) | — | ~14 s | baseline |
| Mesh, half-and-half (n=4) | ~10 MB | ~16 s | ~14% slower |
| Mesh, all remote (n=32) | ~12 MB | ~20 s | ~40% slower |
| Same-host PCIe `raw` (n=4) | ~32 MB (raw) | ~15 s | ~7% slower |

These reflect the codec overhead — about 130 ms per wire round-trip at
QP=18 thanks to channel tiling + per-channel quant. The wire payload
itself is small enough that gigabit ethernet isn't the bottleneck.

**Quality**: cosine similarity > 0.995 per round-trip at QP=18. Output
is visually indistinguishable from all-local at the same seed for any
QP up to 28 (FLUX's residual stream absorbs codec noise comfortably —
this is the load-bearing architectural assumption the whole rig
depends on).

---

## Honest limits

- **Only FLUX.2 family today.** Other architectures need per-model
  code (block signatures, modulation, vec structure differ). Open to
  contributions or sponsored work.
- **Workflow ordering for client-LoRA forwarding**: LoraLoader must
  come BEFORE `Mesh Split FLUX` in the graph. After-Mesh patches don't
  propagate to the captured patcher reference. Tooltip on the node
  warns about this.
- **No back-half VRAM saving on the client side yet.** ComfyUI on the
  client loads the full model normally; back-half blocks then sit
  unused in VRAM because the patches_replace short-circuits them.
  The **server slim-loads** (only the blocks it actually needs); the
  **client doesn't yet**. Real engineering work to fix on the
  client side, hasn't been a blocker for any real workflow yet.
- **Sequential request/response.** No CUDA-stream overlap of codec
  work with compute. The FLUX sampler is inherently sequential per
  timestep, so this caps the headroom anyway.
- **One client at a time.** Server is single-tenant.
- **User-responsible parameter matching.** Client's `n_blocks_remote`
  and server's `--n-blocks` must agree. Mismatch = wrong output, no
  error. Deliberate keep-it-simple choice.

---

## Support the project

This is independent work by one person. If it saves you the cost of an
extra GPU, or you'd just like more FLUX-family models / architectures
supported, **donations make this go faster**:

### ☕ **[buymeacoffee.com/lorasandlenses](https://buymeacoffee.com/lorasandlenses)**

What more support unlocks:
- **More model architectures.** Highest leverage targets:
  **Wan** (image + video, hugely popular ComfyUI workload),
  **LTX-Video** (distilled video — long sequences + big activations
  = ideal codec target), **FLUX.1** (small lift from FLUX.2),
  **SD3.5**, **HunyuanVideo**, **Qwen-Image**, **Chroma**.
- **Multi-LoRA server-side stacking** (multiple LoRA files + per-lora
  strengths in the GUI / launchers)
- **Multi-client server mode** — rent your back-half GPU out
- **Client-side slim load** — free the VRAM currently wasted on
  unused back-half blocks held by the local model
- **CUDA-stream overlap** — codec hides behind compute for genuine
  wall-clock parity with all-local
- **Activation pre-stage cache** — skip re-shipping unchanged `pe` /
  `vec_orig` etc within a generation

---

## Files

```
comfyui-mesh/
├── README.md                     ← this file
├── requirements.txt              ← ComfyUI auto-installs (cuda-bindings)
├── __init__.py                   ← ComfyUI node registration
├── mesh_node.py                  ← MeshSplitFlux + MeshStatus
├── codec.py                      ← tensor ↔ NVENC bitstream (per-channel uint8 + HEVC)
├── protocol.py                   ← length-prefixed TCP framing
├── vec_io.py                     ← FLUX.2 vec/modulation tuple (de)serializer
├── lora_io.py                    ← safetensors-based LoRA patch shipping
├── smoke_test_codec.py           ← standalone codec roundtrip test
├── nvenc_pframe/                 ← BUNDLED NVENC codec wrapper (no separate install)
└── server/                       ← deploy folder for the back-half host
    ├── README.md
    ├── CLAUDE.md
    ├── install.bat               ← one-shot installer
    ├── requirements.txt
    ├── mesh_server.py            ← slim-load TCP server
    ├── mesh_server_gui.py        ← Tkinter wrapper
    ├── codec.py / protocol.py / vec_io.py / lora_io.py / nvenc_pframe/  ← mirror of client (byte-identical)
    ├── smoke_test_server.py
    ├── install_check.py
    └── run_server*.bat           ← six launcher variants (default/gpu0/gpu1/cpu/gui + install)
```

---

## Sibling repos

This rig is one expression of "use NVENC silicon as a wire codec for
non-video state." Related public-or-internal artifacts:

- **`torch-nvenc-compress`** — the original PoC. PCA + per-channel quant
  + NVENC for FLUX mid-block activations and LLM KV cache. Cross-
  architecture transfer numbers measured.
- **`vortex` / `nvenc-pframe`** — the codec package (CFD validation
  suite shipped, public release pending). The DirectBackend ctypes
  wrapper this repo bundles ships from here.
- **`llmtests_native`** — the LLM-side equivalent. Same primitive
  (codec on the wire between two split-model nodes), but for
  Qwen / Mistral / Llama. Validated over LAN + 4G via Tailscale.

---

## License & contact

Author: Peter Neill — `peter@shootthesound.com`

Internal / pre-release. Apache-2.0 components retain their original
license. Bundled `nvenc_pframe` is Apache-2.0.

Bug reports, feature requests, and architecture additions — open an
issue, email me, or donate to push them up the queue. ☕
