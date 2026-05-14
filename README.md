# comfyui-mesh

**Distributed FLUX inference across two GPUs, with NVENC HEVC compression
on the wire between them.**

Two halves of a FLUX double-block stack run on different machines (or
different cards in the same machine). Activations crossing the wire are
compressed by NVENC's idle codec silicon — making LAN, Tailscale, and even
residential broadband fast enough to be usable.

For FLUX.2 Klein 9B distilled (4 sampler timesteps) that's just **4 wire
round-trips per generation total** — well inside the latency budget for
interactive workflows.

This folder is the **client side** — the ComfyUI custom node. The server
side lives in `server/` and gets deployed to wherever the back-half GPU
is (a second machine, or a second card in this one).

---

## What problem this solves

Modern Nvidia consumer cards (4090, 5090) have no NVLink. Splitting a
diffusion model across two of them via PCIe round-trips of full-precision
activations is slower than just running it on one card.

This rig flips the trade. NVENC silicon sits idle during ML inference —
it's hardware that compresses video frames at sub-millisecond latency,
designed for exactly the kind of spatially-coherent data that ML
activations resemble. Use it as a wire codec:

- **Activation bytes** crossing the wire compress by **3-12×** at near-
  lossless quality
- **NVENC encode/decode** runs in parallel to the SM compute, on dedicated
  silicon that wasn't being used anyway
- **The wire becomes effectively wider** — gigabit ethernet stops being
  the bottleneck, Tailscale-over-residential becomes viable, PCIe stops
  losing to NVLink

The same primitive could drive multi-GPU LLM inference, distributed
training, hybrid local-cloud generation. This repo is the diffusion proof.

---

## How it works (one paragraph)

ComfyUI's FLUX implementation already exposes a per-block override
mechanism (`transformer_options["patches_replace"]["dit"][("double_block", i)]`).
Our node hooks into one block boundary, captures the activations at that
point, ships them to the server over TCP, and substitutes the server's
response back into the forward pass. The server runs the remaining
double-blocks and returns the post-back-half state. The single-blocks,
final layer, and VAE all stay local. Wire payload at QP=18 is ~12-15 MB
per timestep per direction; 4 timesteps × 2 directions = ~100 MB total
per generation. Compared to ~280 MB uncompressed.

---

## Quick start

**On the host running ComfyUI (e.g. the 5090):**

1. This folder is already at `ComfyUI/custom_nodes/comfyui-mesh/`. Restart
   ComfyUI to pick it up.
2. The codec (`nvenc_pframe/`) is bundled in this folder — no separate
   install needed. It does have one runtime dep:
   ```
   pip install cuda-bindings
   ```
   Then verify: `python -c "import nvenc_pframe"` (should succeed).

**On the host running the back-half server (e.g. the 4090):**

1. Copy the `server/` folder somewhere convenient (it's self-contained —
   the bundled `nvenc_pframe/` rides along).
2. Get ComfyUI's `comfy/` package importable. Simplest: clone ComfyUI
   next to the server folder.
3. Install the one external dep:
   ```
   pip install torch safetensors einops cuda-bindings
   ```
4. Drop your FLUX safetensors checkpoint into the server folder.
5. Launch:
   - **GUI:** `run_server_gui.bat` — file picker, spinboxes, Start button.
   - **Headless:** edit `N_BLOCKS=4` at the top of `run_server.bat`, then
     run it.

See `server/README.md` for the full server-side setup.

**Then in a ComfyUI workflow:**

1. Load FLUX.2 Klein 9B normally (UNETLoader + dual CLIP).
2. Insert the **`Mesh Split FLUX`** node between the loader and the
   sampler.
3. Set:
   - `n_blocks_remote` to the same number as the server's `--n-blocks`
   - `remote_host` to the server's LAN/Tailscale IP
   - `codec_mode = nvenc`, `codec_qp = 18`
4. Queue. First call opens the TCP connection; subsequent timesteps reuse it.

---

## Nodes added

### `Mesh Split FLUX (5090 ↔ 4090)`

Pass-through MODEL node. Configures the split and wraps the model with
the per-block override.

| Input | Default | Description |
|---|---|---|
| `model` | — | The loaded FLUX MODEL |
| `n_blocks_remote` | 4 | How many of the LAST double-blocks run on the remote server. **Must match the server's `--n-blocks`.** 0 = nothing remote (no-op); 8 (for Klein 9B) = entire stack remote. |
| `remote_host` | `127.0.0.1` | Server hostname or IP. LAN, Tailscale, loopback — all fine. |
| `remote_port` | 7777 | Server TCP port |
| `codec_mode` | `nvenc` | `nvenc` (compressed) or `raw` (no compression — debug / non-NVENC server) |
| `codec_qp` | 18 | NVENC quality. Lower = better quality. **10** = near-lossless. **18** = standard. **28** = highest compression, slightly softer image but still clean on diffusion. |
| `codec_lossless` | false | NVENC lossless tuning. Overrides QP. Smaller wire savings (~1.8×) but bit-exact at the YUV layer (still has the uint8 quant floor). |

### `Mesh Status`

Pure-output node. Reports stats from the most recent generation: bytes
sent/received, wire-call count, last-call latency, codec ratio. Drop it
anywhere in your workflow and check the console output.

---

## Files

```
comfyui-mesh/
├── README.md                     ← this file
├── __init__.py                   ← ComfyUI node registration
├── mesh_node.py                  ← MeshSplitFlux + MeshStatus
├── codec.py                      ← tensor ↔ NVENC bitstream (per-channel uint8 + HEVC)
├── protocol.py                   ← length-prefixed TCP framing
├── vec_io.py                     ← FLUX.2 vec/modulation tuple (de)serializer
├── smoke_test_codec.py           ← standalone codec roundtrip test
├── nvenc_pframe/                 ← BUNDLED codec source (no separate install)
│   ├── __init__.py
│   └── direct/
│       ├── backend.py            ← DirectBackend — main entry point
│       ├── decoder.py            ← cuvid decode wrapper
│       ├── api.py, structs.py    ← NVENC SDK bindings
│       ├── _native.py            ← lazy builds the C helper
│       └── _encode_loop.c        ← compiled on first import (cached under ~/.cache)
└── server/                       ← deploy folder for the back-half host
    ├── README.md                 ← server-side setup
    ├── CLAUDE.md                 ← brief for an AI agent doing back-half setup
    ├── mesh_server.py            ← slim-load TCP server
    ├── mesh_server_gui.py        ← Tkinter wrapper around the server
    ├── codec.py / protocol.py / vec_io.py    ← mirror of client (byte-identical)
    ├── nvenc_pframe/             ← bundled codec source (mirror of client copy)
    ├── smoke_test_server.py      ← model-load + back-half-forward validator
    ├── install_check.py          ← env pre-flight check
    └── run_server*.bat           ← five launcher variants (default/gpu0/gpu1/cpu/gui)
```

---

## Wire format

Per timestep, one round-trip:

```
client → server                          server → client
─────────────────────                    ─────────────────────
img  [B, T,    H]  (codec-compressed)    img' [B, T,    H]  (codec-compressed)
txt  [B, Ttxt, H]  (raw bf16)            txt' [B, Ttxt, H]  (raw bf16)
vec  (12 small tensors, raw)             — (single tuple back)
pe   (raw fp32)
attn_mask  (raw, if present)
+ 4-byte length prefix per message
```

Only `img` (the big tensor, typically ~32 MB at bf16 for 1024×1024) goes
through the codec. Small tensors ship raw because codec overhead would
exceed the bandwidth saved.

The `vec` tensor is the FLUX.2 modulation tuple — a 2×2 grid of
`ModulationOut(shift, scale, gate)` dataclasses (12 small tensors total
when global_modulation=True). Serialized via `vec_io.py`.

---

## Performance numbers (FLUX.2 Klein 9B, 1024×1024, RTX 5090)

| Mode | img wire bytes | Total wire / direction | cos_sim per round-trip |
|---|---:|---:|---:|
| raw | 32 MB | ~38 MB | 1.000 (bit-exact) |
| nvenc lossless | ~18 MB | ~24 MB | 0.99986 |
| nvenc qp=10 | ~12 MB | ~18 MB | 0.99910 |
| nvenc qp=18 | ~9 MB | ~15 MB | 0.99547 |
| nvenc qp=28 | ~6 MB | ~12 MB | ~0.985 |

Per-generation cost: 4 timesteps × 2 directions = 8 wire crossings. At
QP=18 that's ~120 MB total per generation, comfortably handled by gigabit
ethernet or Tailscale.

Codec compression ratios are modest compared to natural-image NVENC use
(~10-40×) because FLUX activations have outlier channels with wide
dynamic range. Per-channel uint8 quantization is what keeps the cosine
similarity high — global quant would crush typical-magnitude channels
to a handful of bins.

For order-of-magnitude better compression, the path is **PCA basis + per-
channel quant + NVENC** (proven in the `torch-nvenc-compress` sibling
repo). That requires per-block calibration captures and is future work.

---

## Same-host two-GPU vs cross-machine

The rig works the same way in both configurations:

- **Cross-machine** (5090 ↔ 4090 on LAN/Tailscale): the wire is slow, the
  codec is the load-bearing optimisation. Use `nvenc qp=18` or similar.
- **Same-host two-GPU** (e.g. 5090 + 4090 in the same desktop): the wire
  is PCIe (~32 GB/s effective). Raw mode is faster than codec mode here
  because codec encode/decode (~700 ms total round-trip) exceeds the PCIe
  cost (~2 ms for 32 MB). **Set `codec_mode = raw` for same-host.**

For same-host two-GPU there are dedicated server launchers
(`run_server_gpu0.bat` / `run_server_gpu1.bat`) that pin the server to a
specific card via `CUDA_VISIBLE_DEVICES`. ComfyUI runs on the other card
normally.

---

## QP envelope is domain-dependent

You may have seen older notes saying QP=28 is unsafe. That came from the
sibling CFD work where chaotic dynamics (Karman vortex shedding) compound
per-step perturbations exponentially and lose the attractor at QP=28.

**Diffusion is the opposite case.** FLUX's residual stream is explicitly
noise-tolerant — that's the architectural property this whole rig
depends on. QP=10/18/28 all produce clean images on FLUX activations.
QP=28 gives a slightly softer image vs QP=18 at the same seed but
doesn't collapse. Use it as the bandwidth/sharpness knob, not as a
safety boundary.

(Outside diffusion — chaotic CFD, autoregressive LLMs at large depths,
or anything where small early-step perturbations cascade — re-derive the
safe envelope per workload. Don't carry numbers across domains.)

---

## Honest scope of v1

- **One split point per generation.** A future version could split into
  N segments for finer-grained pipelining.
- **No back-half VRAM saving on the local side.** ComfyUI on the client
  loads the full FLUX model normally; back-half blocks then sit unused
  in VRAM because the patches_replace short-circuits them. The
  *server* slim-loads (only the blocks it actually needs); the
  *client* doesn't. Future optimization.
- **Sync request/response, no CUDA stream overlap.** NVENC encode and
  local block compute could run on parallel CUDA streams to hide codec
  latency. Not implemented yet — the cross-machine wire dominates
  anyway, and same-host runs are recommended to use raw mode.
- **User-responsible parameter matching.** Client's `n_blocks_remote`
  and server's `--n-blocks` must match. There's no handshake validation
  in v1 — mismatch produces wrong output rather than a clean error.
  Deliberate keep-it-simple choice; protocol-level handshake is a
  future cleanup.
- **One client at a time.** Server is single-tenant. Multi-client / load
  balancing not in scope.

For a single-user creative-tooling workflow, the rig works as-is. The
limits above are the obvious next layers.

---

## Sibling repos

This rig is one expression of a broader primitive ("idle NVENC silicon
as a wire codec for non-video state"). Related work:

- **`torch-nvenc-compress`** — original public artifact. PCA + per-
  channel quant + NVENC HEVC for FLUX activations and LLM KV cache.
  Audited results on cross-architecture transferability.
- **`vortex` / `nvenc-pframe`** — the codec wrapper itself (private),
  plus the CFD validation suite (cylinder vortex-street, 3D Kolmogorov
  turbulence, 5000-step soak). The DirectBackend ctypes wrapper used
  here ships from this repo.
- **`llmtests_native`** — the LLM-side equivalent of this rig. Same
  primitive (codec on the wire between two split-model nodes), but for
  Qwen / Llama / Mistral instead of FLUX. Validated over WiFi, LAN, and
  4G via Tailscale.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `AttributeError: module 'protocol' has no attribute 'recv_message'` | ComfyUI's own `protocol.py` is shadowing ours via sys.path | Make sure `from . import ...` is used in `__init__.py` / `mesh_node.py`. Already fixed in current code. |
| `TypeError: Got unsupported ScalarType BFloat16` | numpy doesn't have bf16; encode_raw needs the uint16 view trick | Fixed in current `codec.py`. |
| Output looks like noise at any QP including lossless | Codec quantization is too coarse (e.g. global instead of per-channel) | Fixed: per-channel quant in current `codec.py`. |
| Server says it has N blocks but client says N+something | `n_blocks_remote` and server's `--n-blocks` disagree | Set both to the same number. |
| `Connection refused` at workflow queue time | Server not running / wrong host/port | Start the server first; check firewall. |
| Server `[server] forward …` lines missing during generation | Patches_replace not registering — check the node is between loader and sampler | Re-check workflow graph. |

---

## License & contact

Internal / pre-release. Not for redistribution. Contact: Peter Neill —
`peter@shootthesound.com`.
