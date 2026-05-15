# comfyui-mesh — back-half server

Companion to the `comfyui-mesh` ComfyUI custom node (`../README.md`).
This folder gets deployed to whichever host holds the back-half GPU —
a second machine on the LAN/Tailscale, or a second card in the same
desktop. It runs a long-lived TCP server: per request, it takes
activations from the front half of FLUX's transformer block stack,
runs the remaining doubles + singles through its slim-loaded weights,
and ships the result back over the wire (NVENC-compressed).

Two headline architectural properties:

1. **Slim-load.** Server reads ONLY the blocks it needs from disk. For
   FLUX.2 Klein 9B at `n_blocks=4`: ~2.2 GB instead of 9.4 GB. For
   FLUX.2 dev at `n_blocks=12`: ~6 GB instead of ~22 GB. For models
   too big to fit on either device whole, this is the load-bearing
   property.
2. **LoRA support, both ways.** Pick a LoRA at startup (GUI or CLI) +
   accept any LoRA the client forwards over the wire (ComfyUI's
   standard LoraLoader output, serialized via safetensors). Both
   stack. Covers lora / loha / lokr / glora / oft / boft plus the
   `diff` and `set` patch types.

---

## What this folder contains

```
server/
├── README.md                   ← this file
├── CLAUDE.md                   ← brief for an AI agent doing this side's setup
├── install.bat                 ← ONE-SHOT INSTALLER — venv + ComfyUI + deps
├── requirements.txt            ← what install.bat installs (also for manual use)
├── mesh_server.py              ← the server. Slim-loads via safetensors.safe_open.
├── mesh_server_gui.py          ← Tkinter wrapper — pick file, set n_blocks, click Start.
├── codec.py                    ← tensor ↔ NVENC bitstream (per-channel uint8 + HEVC + tile_dim)
├── protocol.py                 ← length-prefixed TCP framing
├── vec_io.py                   ← FLUX.2 vec/modulation tuple (de)serializer
├── lora_io.py                  ← safetensors-based LoRA patch shipping
├── nvenc_pframe/               ← BUNDLED codec source (no separate install)
│   └── direct/...              ←   compiles its C helper on first import
├── smoke_test_server.py        ← validates model load + back-half forward
├── install_check.py            ← env pre-flight (deps + cuda + comfy + weights)
├── run_server_gui.bat          ← launch the GUI (recommended for first run)
├── run_server.bat              ← headless launcher, no GPU pinning
├── run_server_gpu0.bat         ← same-host: pin server to physical GPU 0
├── run_server_gpu1.bat         ← same-host: pin server to physical GPU 1
└── run_server_cpu.bat          ← CPU / system-RAM mode (slow; raw codec only)
```

Files in `codec.py / protocol.py / vec_io.py / lora_io.py / nvenc_pframe/`
MUST stay byte-identical to the client-side copies. They're the wire
contract — drift = silent corruption.

---

## Setup on the back-half host

### Easy path: one-shot installer

```
install.bat
```

That single command:

1. Finds Python (3.10+) on PATH
2. Creates a local `.venv` in this folder
3. Upgrades pip + wheel
4. Clones ComfyUI to `..\ComfyUI` if it's not already there
5. Installs ComfyUI's requirements (this pulls torch with CUDA — multi-GB,
   takes a minute or two on a fast connection)
6. Installs the server's extras (cuda-bindings)
7. Runs `install_check.py` to confirm everything is wired up

Re-running `install.bat` is safe — every step is idempotent.

After it finishes, drop your FLUX safetensors into this folder and launch
the server (see below).

### Manual path: do it yourself

If you'd rather control the install yourself (e.g. you already have a
venv, or you want a different CUDA-version torch wheel):

1. Create / activate a Python 3.10+ venv however you like.
2. Clone ComfyUI somewhere reachable; set `COMFYUI_PATH=C:\path\to\ComfyUI`
   if it's not a sibling of this folder.
3. Install ComfyUI's own requirements (`pip install -r path/to/ComfyUI/requirements.txt`)
   — this gets torch with the right CUDA build.
4. `pip install -r requirements.txt` in this folder (adds cuda-bindings
   on top).
5. `python install_check.py` to verify.

**Important either way:** the ComfyUI version on the back-half host should
match (or be reasonably close to) the version on the ComfyUI client. The
fp8 detection logic and FLUX implementation evolve; mismatched versions
between the two ends is the most likely silent-correctness bug.

### Drop the model weights here

```
flux-2-klein-9b-fp8.safetensors   ← 9.4 GB
```

Or point the launcher at wherever you keep your checkpoints.

### Smoke-test the model load + forward

```
python smoke_test_server.py --weights flux-2-klein-9b-fp8.safetensors --n-blocks 4
```

Expected output on success:

```
[server] reading checkpoint header from flux-2-klein-9b-fp8.safetensors
[server] checkpoint has 8 double_blocks; loading 4 (skipping first 4)
[server] slim state dict: 119 tensors, 2.22 GB (full would be ~4.4 GB)
[server] remapped 112 fp8 layer entries -> 32 for slim model
[server] handing slim state dict to comfy.sd.load_diffusion_model_state_dict
[server] model loaded; double_blocks=4 single_blocks=0 hidden_size=4096 global_modulation=True
[smoke] model load: ~4s (local SSD) or longer (network drive)
[smoke] pe shape=(1, 1, 4352, 64, 2, 2)
[smoke] back-half forward: ~30 ms avg
[smoke] 4 blocks at ~7-8 ms/block
[smoke] OK
```

If this fails, do not launch the live server — debug the smoke test
first. Common failure modes are listed in the Troubleshooting section
below.

---

## Running the server

### Option A: GUI (recommended for first run)

```
run_server_gui.bat
```

Opens a Tkinter window with:

- **Model:** file picker. Picks the safetensors.
- **n_blocks:** spinbox. Auto-bounds its max to (n_double + n_single)
  for the loaded checkpoint. FLUX.2 Klein 9B → 32 max. FLUX.2 dev → 56
  max. `0` = full model.
- **Port / Bind:** defaults 7777 / 0.0.0.0.
- **Device:** dropdown listing nvidia-smi-detected GPUs + "cpu". Sets
  `CUDA_VISIBLE_DEVICES` on the subprocess.
- **dtype:** bfloat16 / float16 / float32. Leave on bfloat16 unless you
  know why you're changing it.
- **LoRA:** optional file picker + strength spinbox. Applied to the
  slim-loaded model at startup. Stacks with any LoRA the client
  forwards.
- **Start Server / Stop:** subprocess lifecycle. Live stdout streams
  into the text area below.

The GUI doesn't add server logic — it just spawns `mesh_server.py`
with the right args. Same generations, friendlier launch.

The device dropdown populates asynchronously. On Windows, `nvidia-smi`
cold-start takes 1-3s (driver + NVML init); rather than block the window
from painting, the GUI opens immediately with a `cuda:0` placeholder and
fills in the real device list (with card names) once `nvidia-smi` returns.
If you click the dropdown within that first second you'll see the
placeholder; it refreshes shortly after.

GUI settings (model path, n_blocks, port, bind, device, dtype, LoRA
path, LoRA strength) are remembered between launches in
`mesh_server_gui_settings.json` next to the script. Saved on close and
on Start. Delete the file to reset to defaults.

### Slow GUI startup?

Each launch writes a timestamped trace to
`mesh_server_gui_startup.log` (truncated each run). If the GUI takes
more than a few seconds to appear, check the log for where time was
spent. Typical culprits:

- **Windows Defender scanning the venv `python.exe` and `_tkinter.pyd`**
  the first time after boot — easily 5-30s. The standing fix is to
  add the server folder (or at least `.venv\Scripts\` and
  `.venv\Lib\site-packages\`) to Defender's exclusions.
- **Cold-cache safetensors header read** for a multi-GB checkpoint when
  a saved model path is restored at launch. This is now done on a
  background thread so the window paints first; the n_blocks_max
  display fills in once the read returns.

### Option B: Headless launchers

Each bat file has an editable line at the top:

```
REM ============================================================
REM  EDIT THIS: how many of the LAST double_blocks to load.
REM  Must match the node's `n_blocks_remote` setting.
REM  Leave at 0 to load the full back-half model.
REM ============================================================
set N_BLOCKS=4
```

Pick the right launcher for your topology:

| Launcher | Use case |
|---|---|
| `run_server.bat` | Cross-machine — this host has one GPU, no ambiguity |
| `run_server_gpu0.bat` | Same-host two-GPU rig, server on physical GPU 0 |
| `run_server_gpu1.bat` | Same-host two-GPU rig, server on physical GPU 1 |
| `run_server_cpu.bat` | CPU / system-RAM mode (slow; requires `codec_mode=raw` on client) |

The `_gpu0` / `_gpu1` variants use `CUDA_VISIBLE_DEVICES` to pin the
process to one card so it doesn't compete with ComfyUI on the other.

### Option C: Direct invocation

```
python mesh_server.py \
    --weights flux-2-klein-9b-fp8.safetensors \
    --n-blocks 4 \
    --port 7777 \
    --bind 0.0.0.0 \
    --device cuda:0 \
    --dtype bfloat16
```

Omit `--n-blocks` to load every double-block (still skips single-blocks
and final layer, which the server never uses).

---

## Server-side log format

When a client connects:

```
[server] client connected: ('192.168.x.x', NNNNN)
[server] forward 4 blocks (client said start=4): decode XX ms  fwd XX ms  enc XX ms  in 14.5 MB  out 4.2 MB
```

One line per timestep per generation. The decode / fwd / enc timings
break down what the server spent its time on. The in / out byte counts
are the wire payloads after the codec.

If you don't see those lines when the client queues a workflow, the
problem is upstream — connection refused, firewall, or the client node
isn't slotted between the loader and the sampler.

---

## Network setup

- **LAN:** get this host's local IP from `ipconfig` / `ip addr`. Looks
  like `192.168.x.x`. Set the client node's `remote_host` to that.
- **Tailscale:** `tailscale ip -4` gives a `100.x.x.x` address. Works
  for off-LAN deployments.
- **Loopback:** `127.0.0.1` works fine for protocol-validation testing
  on a single machine.
- **Firewall:** the server's port (default 7777) must be reachable from
  the client host. Windows Firewall may prompt the first time.

---

## What the wire actually carries

Per timestep, one round-trip:

**Client → server:**
- `img` tensor `[B, T, H]` (~32 MB raw at bf16 / ~9 MB at qp=18)
- `txt` tensor `[B, Ttxt, H]` (raw, ~2 MB)
- `vec` modulation tuple (12 small tensors, raw, few hundred KB)
- `pe` positional encoding (raw fp32, ~4 MB)
- `attn_mask` (raw, optional)

**Server → client:**
- `img'` updated tensor (codec-compressed)
- `txt'` updated tensor (raw)

Total wire per direction at QP=18: roughly 12-15 MB. The `pe` and `vec`
tensors are recomputable on the client side (they're a function of the
prompt and timestep), so a future optimization could send them once per
session instead of per timestep. Not done yet.

---

## Same-host two-GPU notes

Use `run_server_gpu0.bat` / `run_server_gpu1.bat` depending on which
card ComfyUI is using on the other side.

Two important things to know about same-host setups:

1. **Set `codec_mode=raw` on the client** for same-host. PCIe is fast
   enough that codec encode/decode latency exceeds the bandwidth
   savings. Raw mode is the right choice here.

2. **Both sides slim-load.** The server slim-loads from disk; the
   client strips the back-half block weights from its loaded model
   in place (frees ~half the VRAM on `n_blocks_remote=4` of Klein 9B).
   If you change `n_blocks_remote` after a generation, force-reload
   the model on the client — the stripped weights are gone for the
   session. The node raises if you try without reloading.

---

## Limitations of v1

- **One client at a time.** Server is single-tenant. Multi-client
  support not in scope.
- **No protocol-level validation.** User is responsible for setting
  `n_blocks_remote` on the client equal to `--n-blocks` on the server.
  Mismatch produces wrong output rather than a clean error.
- **No fault tolerance.** If the connection drops mid-generation, the
  workflow fails — restart it.
- **Per-timestep wire crossings.** 4 timesteps × 2 directions = 8 wire
  crossings per generation. A future version could overlap codec
  encode with local compute via parallel CUDA streams.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `comfy.sd.load_diffusion_model_state_dict returned None` | Wrong weight file or ComfyUI version too old to detect FLUX2 | Check checkpoint, update ComfyUI |
| `Freqs dimension N must be head_dim//2` | pe synthesis shape issue | Should be fixed in current smoke_test_server.py; check it uses `model.pe_embedder(ids)` |
| `not enough values to unpack` on vec | global_modulation detection mismatch | Make sure both client and server are on the current code |
| CUDA OOM | Slim load too generous OR full load on small VRAM | Lower `n_blocks`; check no other process is on the GPU |
| `AttributeError: 'NoneType' object has no attribute 'device'` (cast_bias_weight) | fp8 metadata not being remapped to slim indices | Already fixed in current mesh_server.py; make sure you're on a recent build |
| Image output is noise at any QP | codec.py was at global quant instead of per-channel | Already fixed in current codec.py; make sure client and server are on the same version |
| `unet unexpected:` warning at load with fp8 scale keys | safetensors metadata not being passed through to load_diffusion_model_state_dict | Already fixed; current mesh_server.py passes `metadata=metadata` |
| `Connection refused` at workflow queue time | Server not running / wrong host/port / firewall | Verify server is up; check the IP and port match |
| `Missing weight for layer final_layer.linear` warning | Expected — server intentionally skips final_layer | Ignore |

---

## See also

- `../README.md` — client-side overview and the broader architecture
- `CLAUDE.md` — concise brief for an AI agent doing the back-half setup
