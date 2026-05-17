# ComfyUI Mesh : Daedalus *(the back-half server)*

Companion to **Icarus**, the ComfyUI custom node living one folder up
(`../README.md`). This folder gets deployed to whichever host holds
the back-half GPU — a second machine on the LAN/Tailscale, or a
second card in the same desktop. It runs a long-lived TCP server:
per request, it takes activations from the front half of FLUX's
(or LTX's) transformer block stack, runs the remaining blocks through
its slim-loaded weights, and ships the result back over the wire
(NVENC-compressed).

**Two servers in one folder:**

- `mesh_server.py` + `mesh_server_gui.py` — for **FLUX.2** Dev / Klein 9B.
- `mesh_server_ltx.py` + `mesh_server_ltx_gui.py` — for **LTX 2.3** (LTX-AV 22B Dev).

Both share the same install, the same codec, the same wire protocol
plumbing — just paired with the matching client node (`Icarus` for
FLUX, `Icarus LTX` for LTX). See the LTX subsection below for the
small UX differences between the two GUIs.

(Daedalus prepares the wings; Icarus rides them. The names are
mythological flair on the underlying engineering split — the server
slim-loads the back-half model, the client node hands it activations
to chew on.)

Two headline architectural properties:

1. **Slim-load.** Server reads ONLY the blocks it needs from disk. For
   FLUX.2 Klein 9B at `n_blocks=4`: ~2.2 GB instead of 9.4 GB. For
   FLUX.2 Dev at `n_blocks=12`: ~6 GB instead of ~22 GB. For models
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
├── install.bat                 ← ONE-SHOT INSTALLER — venv + ComfyUI + cu128 torch + deps
├── update_comfy.bat            ← git pull on .\ComfyUI + re-install requirements
├── requirements.txt            ← what install.bat installs (also for manual use)
│
│ ─── FLUX.2 server pair ───
├── mesh_server.py              ← FLUX server. Slim-loads via safetensors.safe_open.
│                                  Handles reconfigure messages by writing a handoff
│                                  file and exiting; GUI relaunches with new --n-blocks.
├── mesh_server_gui.py          ← FLUX Tkinter wrapper. Settings persist to JSON; restart
│                                  button on form drift; auto-restart on reconfigure;
│                                  startup trace log captures bat→python wall-clock.
│
│ ─── LTX 2.3 server pair ───
├── mesh_server_ltx.py          ← LTX server. Same slim-load pattern, plus LTX-AV
│                                  variant detection and TWO LoRA slots (--lora +
│                                  --lora2; the second slot defaults to strength 0.5
│                                  for the LTX 2.3 Distilled LoRA).
├── mesh_server_ltx_gui.py      ← LTX Tkinter wrapper. Same shape as the FLUX GUI plus
│                                  a second "Distill LoRA" row (default strength 0.5).
│                                  Settings persist independently of the FLUX GUI's.
│
│ ─── wire-contract files (mirror client; MUST stay byte-identical) ───
├── codec.py                    ← tensor ↔ NVENC bitstream (per-channel uint8 + HEVC, plus "Nvenc LTX (5090 optimized)" mode)
├── protocol.py                 ← length-prefixed TCP framing
├── vec_io.py                   ← FLUX.2 vec/modulation tuple (de)serializer
├── payload_ltx.py              ← LTX-AV per-block payload (de)serializer
├── lora_io.py                  ← safetensors-based LoRA patch shipping
├── nvenc_pframe/               ← BUNDLED codec source (no separate install)
│   └── direct/...              ←   compiles its C helper on first import
│
│ ─── helpers + launchers ───
├── smoke_test_server.py        ← validates model load + back-half forward
├── install_check.py            ← env pre-flight (deps + cuda + comfy + weights)
├── _splash_flux2.cmd           ← cmd-console "starting…" splash launched by the
│                                  FLUX 2 GUI bat in parallel with pythonw, polls a
│                                  sentinel file and self-closes when the GUI paints
├── _splash_ltx.cmd             ← same, for the LTX GUI's launcher
├── run_server_flux2_gui.bat    ← launch the FLUX 2 GUI (recommended for first FLUX run)
├── run_server_ltx_gui.bat      ← launch the LTX GUI (recommended for first LTX run)
├── run_server_flux2.bat        ← headless FLUX 2 launcher, no GPU pinning
├── run_server_flux2_gpu0.bat   ← same-host: pin FLUX 2 server to physical GPU 0
├── run_server_flux2_gpu1.bat   ← same-host: pin FLUX 2 server to physical GPU 1
└── run_server_flux2_cpu.bat    ← FLUX 2 CPU / system-RAM mode (slow; raw codec only)
```

Files in `codec.py / protocol.py / vec_io.py / payload_ltx.py /
lora_io.py / nvenc_pframe/` MUST stay byte-identical to the
client-side copies. They're the wire contract — drift = silent
corruption.

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
4. Clones ComfyUI INTO this server folder (`.\ComfyUI`) if it's not
   already there — keeps the whole deploy in one place so you can
   delete the folder to fully uninstall
5. **Installs CUDA-enabled torch from PyTorch's cu128 wheels.** Covers
   every RTX 30/40/50-series card. (50-series Blackwell *requires* cu128;
   older CUDA wheels silently fall back to CPU on those cards. cu128
   works fine on 30/40 too — one wheel set covers all current consumer
   Nvidia GPUs with NVENC.) This is multi-GB; takes a minute or two on
   a fast connection.
6. Installs ComfyUI's other requirements (transformers, einops, etc.)
7. Installs the server's extras (cuda-bindings)
8. Runs `install_check.py` to confirm everything is wired up

Re-running `install.bat` is safe — every step is idempotent.

**Why we install torch ourselves:** ComfyUI's `requirements.txt`
just lists `torch` with no CUDA specifier. On Windows that pulls
the CPU-only wheel from PyPI, which silently breaks NVENC + CUDA
inference. The cu128 step above runs BEFORE the comfy requirements
so the GPU-enabled wheel wins.

### Updating ComfyUI later

```
update_comfy.bat
```

Pulls the latest ComfyUI (`git pull` in `.\ComfyUI`) and re-installs
its requirements + the server's extras. Run this whenever the client
side flags a ComfyUI-version mismatch (the fp8 detection + FLUX
implementation evolve in upstream; mismatched ComfyUI versions between
client and server is the most common silent-correctness gotcha).
Doesn't touch torch — re-run `install.bat` for that.

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
flux-2-klein-9b-fp8.safetensors   ← 9.4 GB and/or flux2_dev_fp8mixed.safetensors
ltx-2.3-22b-dev-fp8.safetensors   ← if you're running LTX 2.3
```

Where to get the right files:

- **FLUX.2 Dev:** the ComfyUI docs page
  [Flux.2 Dev](https://docs.comfy.org/tutorials/flux/flux-2-dev) has
  direct links (the fp8 variants are what fit comfortably on consumer
  cards).
- **FLUX.2 Klein 9B:** Black Forest Labs' HuggingFace repo at
  [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/tree/main).
- **LTX 2.3 (LTX-AV 22B Dev):** Lightricks' HuggingFace repo at
  [Lightricks/LTX-2.3-fp8](https://huggingface.co/Lightricks/LTX-2.3-fp8/tree/main).
  Repo contains both the **base model** and a **pre-distilled
  variant**. If you grab the distilled variant, you don't need to
  populate the Distill LoRA row in the LTX server GUI — the
  distillation is baked into the weights. The Distill LoRA row is
  there for users running the base model who want to apply the
  distill LoRA at runtime instead.

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

The live server adds one more line right after the slim load when it's
actually accepting connections:

```
[server] READY — listening on 0.0.0.0:7777 (n_blocks=4: 4D + 0S)
```

If this fails, do not launch the live server — debug the smoke test
first. Common failure modes are listed in the Troubleshooting section
below.

---

## Running the server

### Option A: GUI (recommended for first run)

```
run_server_flux2_gui.bat     (FLUX 2)
run_server_ltx_gui.bat       (LTX-AV)
```

Opens a Tkinter window with:

- **Model:** file picker. Picks the safetensors.
- **n_blocks:** spinbox. Auto-bounds its max to (n_double + n_single)
  for the loaded checkpoint. FLUX.2 Klein 9B → 32 max. FLUX.2 Dev → 56
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

While the server is running, **changing any setting** in the form
flips the Start button into a "Restart server to apply new settings"
state (wider, bold). Clicking it stops the subprocess and starts it
fresh with the new values. If you lower `n_blocks` the GUI also
prints a warning to the log: the *client* still has its previous
strip applied and needs a ComfyUI restart to pick up the smaller
value (the client's stripped weights are gone for the session).

The GUI also auto-restarts the server when the **client** asks for a
new `--n-blocks` (via the Confirm button on the Icarus node).
The server writes a small handoff file with the new value before
exiting, the GUI picks it up, updates the spinbox visually, and
relaunches the subprocess. Net log:

```
[server] RESTARTING: reconfigure request --n-blocks 4 -> 6
[server] exiting; GUI launcher will restart with --n-blocks=6
[gui] server exited (rc=0)
[gui] server requested reconfigure to n_blocks=6 — applying + restarting...
[gui] launching: ... --n-blocks 6 ...
...
[server] READY — listening on 0.0.0.0:7777 (n_blocks=6: 6D + 0S)
```

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
| `run_server_flux2.bat` / `run_server_ltx.bat` | Cross-machine — this host has one GPU, no ambiguity |
| `run_server_flux2_gpu0.bat` / `run_server_ltx_gpu0.bat` | Same-host two-GPU rig, server on physical GPU 0 |
| `run_server_flux2_gpu1.bat` / `run_server_ltx_gpu1.bat` | Same-host two-GPU rig, server on physical GPU 1 |
| `run_server_flux2_cpu.bat` / `run_server_ltx_cpu.bat` | CPU / system-RAM mode (slow; requires `codec_mode=raw` on client) |

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

## Running the LTX 2.3 server

For LTX 2.3 (the Lightricks LTX-AV 22B Dev model) the install is the
same as the FLUX install — `install.bat` covers both. The launcher
and the script that runs are different:

```
run_server_ltx_gui.bat
```

The LTX GUI is the same shape as the FLUX one (model picker,
n_blocks spinbox, port/bind/device/dtype rows, start/stop, log
view) with two LTX-specific extras:

- **n_blocks defaults to 8** — matches the Icarus LTX node's
  default. LTX-AV 22B has 48 transformer_blocks total.
- **Two LoRA rows** instead of one:
  - **LoRA / LoRA strength** — primary slot, default strength 1.0.
  - **Distill LoRA / Distill strength** — second slot, default
    strength **0.5**. Intended for the **LTX 2.3 Distilled LoRA**
    which most LTX workflows stack on top of the base model.
    Default 0.5 matches the typical strength.

Both slots get applied to the slim-loaded model at server startup,
in order (primary first, then distill). They also both get re-applied
after any client-LoRA unpatch so a client that's forwarding its own
LoRAs doesn't accidentally drop the server's static slot LoRAs.

GUI settings (model path, n_blocks, port, bind, device, dtype, both
LoRA paths + strengths) persist independently of the FLUX GUI's in
`mesh_server_ltx_gui_settings.json` next to the script.

The LTX server also accepts direct CLI invocation with the same flags
as the FLUX one, plus `--lora2` / `--lora2-strength`:

```
python mesh_server_ltx.py \
    --weights ltx-2.3-22b-dev-fp8.safetensors \
    --n-blocks 8 \
    --port 7777 \
    --bind 0.0.0.0 \
    --device cuda:0 \
    --dtype bfloat16 \
    --lora character.safetensors --lora-strength 1.0 \
    --lora2 ltx-2.3-22b-distilled-lora-384.safetensors --lora2-strength 0.5
```

The LTX server reuses the same `--n-blocks` reconfigure handshake +
handoff-file pattern as the FLUX server, so the Icarus LTX node's
Confirm-restart flow works the same way. The two servers default to
the same port (7777) — don't try to run both simultaneously without
changing one's `--port`.

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

Use `run_server_flux2_gpu0.bat` / `run_server_flux2_gpu1.bat` (or the
matching `_ltx_gpu0` / `_ltx_gpu1` for LTX-AV) depending on which card
ComfyUI is using on the other side.

Two important things to know about same-host setups:

1. **Set `codec_mode=raw` on the client** for same-host. PCIe is fast
   enough that codec encode/decode latency exceeds the bandwidth
   savings. Raw mode is the right choice here.

2. **Both sides slim-load.** The server slim-loads from disk; the
   client strips the back-half block weights from its loaded model
   in place (frees ~half the VRAM on `n_blocks_remote=4` of Klein 9B).
   The server can re-strip up or down on each restart (it just re-reads
   from disk). The client's strip is one-way for the session:
   **increasing** `n_blocks_remote` works seamlessly (the strip extends
   incrementally), **decreasing** would require un-stripping the
   weights, which are gone, so a fresh ComfyUI launch is needed to
   re-load them from disk.

3. **Lowering `n_blocks` in the server GUI also requires restarting
   ComfyUI on the client.** Same reason as above: when the GUI restarts
   the server with a smaller `--n-blocks`, the *client* still has the
   bigger strip applied and can't recover. The Icarus node
   surfaces a "Confirm" button on `n_blocks_remote` mismatch, but
   confirming with a smaller value only works for the in-flight session
   if the client's previous strip wasn't already wider — otherwise the
   inline banner tells you to restart ComfyUI. Increasing in the server
   GUI is fine: the client's incremental-strip handles it.

---

## Limitations

- **One client at a time.** Server is single-tenant. A second client
  connecting kicks the first.
- **Per-timestep wire crossings.** 4 timesteps × 2 directions = 8 wire
  crossings per generation. A future version could overlap codec
  encode with local compute via parallel CUDA streams.
- **Decreasing `n_blocks_remote` requires the client to restart
  ComfyUI** even though the server itself can re-strip up or down on
  every relaunch — see the "same-host" section above for the why.
- **GUI cold start can take 10-30s on Windows** (Python + venv site
  init + tkinter DLL load + Defender scan). The cmd-console splash
  during launch shows you something is happening; see the "Slow GUI
  startup?" subsection above for the standing fix (Defender exclusions
  on the venv).

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
