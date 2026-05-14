# comfyui-mesh

Distributed FLUX inference across two GPUs on different machines, with
NVENC HEVC compression of activations on the wire between them.

This is the **client side** — a ComfyUI custom node that registers
a per-block override on FLUX's double-block stack and sends the
back-half computation to a server running on a second GPU. The
server side lives in `U:/comfyuiserver/` (see that folder's
`README.md`).

## What problem this solves

The 4090 is a great GPU. So is the 5090. They're on the same desk.
There's no NVLink between them (Nvidia removed it), and naively
splitting a model across them via PCIe round-trips of full-precision
activations is slower than just running it on one card.

This node uses your idle NVENC silicon as a wire codec. Activations
crossing the wire (PCIe peer-to-peer if the cards are in the same
machine, or LAN/Tailscale if they're in different machines) get
compressed by ~10–40×. That makes the bandwidth sufficient to run
the back half on the second machine in parallel with whatever the
front half is doing.

For FLUX.2 Klein 9B distilled (4 sampler timesteps), this means just
**4 wire round-trips per generation** total — well inside the
latency budget even on Tailscale across a residential connection.

## Nodes added

- **`Mesh Split FLUX (5090 ↔ 4090)`** — pass-through MODEL node. Slot
  it between the model loader and the sampler. Inputs:
    - `model` — the loaded FLUX MODEL
    - `split_index` — at which double_block index to start running
      remotely. `0` = entire double_block stack remote.
      For Klein 9B's 8 double_blocks, `split_index=4` is the natural
      half-split.
    - `remote_host`, `remote_port` — the 4090's address (LAN IP or
      Tailscale IP)
    - `codec_mode` — `raw` (no compression, debug only) or `nvenc`
    - `codec_qp` — NVENC quality. 10=near-lossless, 18=standard,
      28=highest compression. On FLUX activations all three work cleanly
      — QP=28 gives a slightly softer image vs QP=18 at the same seed
      but doesn't fall apart. (The "don't use QP=28" warning you may
      have seen elsewhere is a CFD-domain constraint: chaotic dynamics
      compound small per-step perturbations and lose the attractor.
      Diffusion's residual stream absorbs that level of noise fine.)
    - `codec_lossless` — use NVENC's lossless tuning (overrides QP,
      yields ~2–4× compression instead of ~10–40×)

- **`Mesh Status`** — pure-output node that reports stats (bytes
  sent/received, last call time, codec ratio) from the most recent
  generation. Drop it anywhere in your workflow and check the
  console output.

## Setup

1. Make sure ComfyUI sees this folder as a custom node — it's already
   in `custom_nodes/comfyui-mesh/`, so it should auto-register on
   ComfyUI restart.

2. Set up the back-half server on the 4090. See
   `U:/comfyuiserver/README.md`.

3. Make sure `nvenc-pframe` is importable from this ComfyUI's Python
   env (`python -c "import nvenc_pframe"` should succeed). It's the
   same package the existing `vortex/` and `torch-nvenc-compress/`
   work uses.

4. In a workflow:
   - Load FLUX.2 Klein 9B normally (UNETLoader, dual CLIP loader, etc.)
   - Insert `Mesh Split FLUX` after the loader, before the sampler
   - Set `remote_host` to your 4090's IP
   - Run a generation. First call opens the TCP connection; subsequent
     timesteps reuse it.

## What gets sent on the wire

Per timestep, per split point:

```
client → server:  img[B,T,H]  (codec-compressed)
                  txt[B,T_text,H]  (raw, small)
                  vec, pe, attn_mask  (raw, small)
                  + 4-byte length prefix per message

server → client:  img'[B,T,H]  (codec-compressed)
                  txt'[B,T_text,H]  (raw)
```

Codec compression applies only to the `img` tensor (the big one);
small tensors ship raw because the codec overhead would exceed the
bandwidth saved.

## Files

- `__init__.py` — registers `MeshSplitFlux` and `MeshStatus`
- `mesh_node.py` — the actual node logic, hook installation, TCP client
- `codec.py` — torch tensor ↔ NVENC bitstream
- `protocol.py` — length-prefixed binary message format
- `README.md` — this file

## Limitations of v1

- Only one split point (one wire round-trip per timestep). A future
  version could split into N segments for finer-grained pipelining.
- All single_blocks + final layer + VAE stay local. No back-half VRAM
  saving on the local 5090 — the back-half blocks are still loaded
  but not run. Real VRAM saving would require deleting them after
  load (a future version).
- Sync request/response, no overlap with local compute. NVENC encode
  and the local block compute *could* run on parallel CUDA streams;
  not done yet.

For a first-cut "let's see if it works" rig, this is fine. The
codec-on-wire claim is the load-bearing part and that does work end
to end.
