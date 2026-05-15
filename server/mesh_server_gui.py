"""Tkinter GUI for the comfyui-mesh back-half server.

Wraps `mesh_server.py` as a subprocess. Lets the user pick:
    - the model safetensors file
    - n_blocks (autobounded by what the checkpoint actually contains)
    - port to listen on
    - device (cuda:0 / cuda:1 / ... / cpu)
    - dtype

Then Start/Stop with the server's stdout streaming to a Text widget.

No new server logic — just a friendlier launcher than editing a .bat.
The actual back-half forward is still mesh_server.py.

Run on the 4090:
    python mesh_server_gui.py
or
    run_server_gui.bat
"""

from __future__ import annotations

# Capture the launch timestamp before any heavy imports so the startup
# log can attribute the time spent in tkinter/Tcl initialization too.
import time as _time
_GUI_LAUNCH_T0 = _time.perf_counter()

import json
import os
import queue
import signal
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import (
    Tk, Frame, Label, Entry, Button, StringVar, IntVar,
    filedialog, messagebox, END, DISABLED, NORMAL, BOTH, LEFT, RIGHT, X, Y,
)
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

HERE = Path(__file__).parent
SETTINGS_FILE = HERE / "mesh_server_gui_settings.json"
STARTUP_LOG = HERE / "mesh_server_gui_startup.log"
READY_SENTINEL = HERE / "mesh_server_gui_ready.tmp"


def _signal_ready() -> None:
    """Drop the ready sentinel file the launcher splash polls for.
    Called once the main Tk window is actually painted so the splash
    closes the moment the user can interact."""
    try:
        READY_SENTINEL.write_text("ready", encoding="utf-8")
    except Exception:
        pass


def _log_session_header() -> None:
    """Truncate the startup log and write a fresh session header.
    Called once on every GUI launch — only the most recent startup
    is kept on disk."""
    try:
        import platform
        # The launcher .bat drops a marker file just before invoking
        # pythonw. Comparing its mtime to wall-clock now tells us how
        # long python.exe spawn + venv site init + module imports took
        # — i.e. everything BEFORE _GUI_LAUNCH_T0 was captured. This is
        # where 10-40s startup lag typically lives (Defender scan +
        # tkinter DLL load + cold venv).
        bat_marker = HERE / "mesh_server_gui_bat_t0.tmp"
        bat_to_py_line = ""
        if bat_marker.exists():
            try:
                delta = _time.time() - bat_marker.stat().st_mtime
                bat_to_py_line = (
                    f"bat→python : {delta:6.2f}s  "
                    f"(python.exe spawn + venv site init + module imports; "
                    f"NOT script work)\n"
                )
            except Exception:
                pass
        with STARTUP_LOG.open("w", encoding="utf-8") as f:
            f.write("=== mesh_server_gui startup log ===\n")
            f.write(f"local time : {_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"python     : {sys.executable}\n")
            f.write(f"version    : {sys.version.split()[0]}\n")
            f.write(f"platform   : {platform.platform()}\n")
            if bat_to_py_line:
                f.write(bat_to_py_line)
            f.write("---\n")
    except Exception:
        pass


def _log_event(msg: str) -> None:
    """Append one timestamped line to the startup log. Used to localise
    where the 10-40s startup time is being spent (typically Python
    cold-start + tkinter/Tcl DLL load + Windows Defender scan)."""
    try:
        elapsed = _time.perf_counter() - _GUI_LAUNCH_T0
        with STARTUP_LOG.open("a", encoding="utf-8") as f:
            f.write(f"[+{elapsed:7.3f}s] {msg}\n")
    except Exception:
        pass


_log_session_header()
_log_event("module imports complete")


def _load_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_settings(settings: dict) -> None:
    try:
        SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------
# Lightweight detection helpers (no torch import — runs in the parent
# process, must start fast)
# ---------------------------------------------------------------------

def detect_gpus() -> list[str]:
    """Query nvidia-smi for the list of visible CUDA devices. Returns
    something like ['cuda:0 (NVIDIA GeForce RTX 4090)', ...]. Falls
    back to ['cuda:0'] if nvidia-smi isn't on PATH."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
    except Exception:
        return ["cuda:0"]
    devices = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) == 2:
            devices.append(f"cuda:{parts[0]} ({parts[1]})")
    return devices or ["cuda:0"]


def detect_n_blocks_max(weights_path: Path) -> tuple[int, int] | None:
    """Open the safetensors header (no tensor data) and count
    double_blocks.N + single_blocks.N keys. Returns (n_double, n_single)
    or None on failure. The GUI uses (n_double + n_single) as the
    spinbox max — n_blocks_remote is a unified counter spanning both."""
    try:
        from safetensors import safe_open
    except ImportError:
        return None
    try:
        with safe_open(str(weights_path), framework="pt", device="cpu") as f:
            db_idx = set()
            sb_idx = set()
            for k in f.keys():
                if k.startswith("double_blocks."):
                    try:
                        db_idx.add(int(k.split(".")[1]))
                    except (ValueError, IndexError):
                        continue
                elif k.startswith("single_blocks."):
                    try:
                        sb_idx.add(int(k.split(".")[1]))
                    except (ValueError, IndexError):
                        continue
            if not db_idx:
                return None
            return (max(db_idx) + 1, (max(sb_idx) + 1) if sb_idx else 0)
    except Exception:
        return None


def find_python() -> str:
    """Find a sensible python.exe. Prefer the local .venv, else
    whatever's running this GUI."""
    venv_py = HERE / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def find_comfyui_path() -> str | None:
    """Mirror mesh_server.py's COMFYUI_PATH discovery."""
    candidates = [
        os.environ.get("COMFYUI_PATH"),
        str(HERE / "ComfyUI"),
        str(HERE.parent / "ComfyUI"),
        "C:/ComfyUI",
        "/opt/ComfyUI",
    ]
    for c in candidates:
        if c and Path(c).is_dir() and (Path(c) / "comfy").is_dir():
            return c
    return None


# ---------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------

class MeshServerGUI:
    def __init__(self, root: Tk):
        _log_event("MeshServerGUI.__init__ entered")
        self.root = root
        root.title("comfyui-mesh — back-half server")
        root.geometry("780x560")

        self.proc: subprocess.Popen | None = None
        self.output_q: queue.Queue = queue.Queue()
        self.reader_thread: threading.Thread | None = None
        self.settings = _load_settings()
        # Saved device prefix ("cuda:0") used by _on_devices_ready to
        # restore the user's last selection once nvidia-smi returns.
        self._saved_device_prefix = self.settings.get("device", "")
        # Snapshot of settings captured the moment the running subprocess
        # started. When the live form drifts from this snapshot, the
        # Start button morphs into "Restart server to apply new settings".
        self._running_baseline: dict | None = None

        self._build_ui()
        _log_event("_build_ui complete (window can paint)")
        # Schedule the ready signal AFTER the first event-loop tick, so
        # the splash only closes once the user can actually see + click
        # the window — not just when our build code returned.
        self.root.after(0, self._on_window_ready)
        # The startup safetensors-header read happens off-thread so even
        # a slow first read (cold cache on a multi-GB checkpoint) doesn't
        # delay the window paint.
        if self.weights_var.get().strip():
            threading.Thread(target=self._refresh_after_file_change, daemon=True).start()
        else:
            self._refresh_after_file_change()
        self.root.after(100, self._poll_output)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ----- UI build -----

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        s = self.settings

        # Row: weights file picker
        row = Frame(self.root)
        row.pack(fill=X, **pad)
        Label(row, text="Model:", width=12, anchor="w").pack(side=LEFT)
        self.weights_var = StringVar(value=s.get("weights", ""))
        self.weights_entry = Entry(row, textvariable=self.weights_var)
        self.weights_entry.pack(side=LEFT, fill=X, expand=True, padx=4)
        Button(row, text="Browse…", command=self._on_browse).pack(side=LEFT)

        # Row: n_blocks
        row = Frame(self.root)
        row.pack(fill=X, **pad)
        Label(row, text="n_blocks:", width=12, anchor="w").pack(side=LEFT)
        self.n_blocks_var = IntVar(value=int(s.get("n_blocks", 4)))
        self.n_blocks_spin = ttk.Spinbox(
            row, from_=0, to=999, increment=1, textvariable=self.n_blocks_var, width=8,
        )
        self.n_blocks_spin.pack(side=LEFT)
        self.n_blocks_info = Label(row, text="(0 = full model)", anchor="w", fg="#666")
        self.n_blocks_info.pack(side=LEFT, padx=8)

        # Row: port
        row = Frame(self.root)
        row.pack(fill=X, **pad)
        Label(row, text="Port:", width=12, anchor="w").pack(side=LEFT)
        self.port_var = IntVar(value=int(s.get("port", 7777)))
        ttk.Spinbox(row, from_=1, to=65535, increment=1, textvariable=self.port_var, width=8).pack(side=LEFT)
        self.bind_var = StringVar(value=s.get("bind", "0.0.0.0"))
        Label(row, text="  Bind:", anchor="w").pack(side=LEFT, padx=(16, 4))
        Entry(row, textvariable=self.bind_var, width=18).pack(side=LEFT)

        # Row: device + dtype
        row = Frame(self.root)
        row.pack(fill=X, **pad)
        Label(row, text="Device:", width=12, anchor="w").pack(side=LEFT)
        self.device_var = StringVar()
        # nvidia-smi cold start blocks 1-3s on Windows; populate async so the window paints first.
        self.device_combo = ttk.Combobox(
            row, textvariable=self.device_var,
            values=["cuda:0", "cpu"], state="readonly", width=42,
        )
        self.device_combo.current(0)
        self.device_combo.pack(side=LEFT)
        self._populate_devices_async()

        row = Frame(self.root)
        row.pack(fill=X, **pad)
        Label(row, text="dtype:", width=12, anchor="w").pack(side=LEFT)
        self.dtype_var = StringVar(value=s.get("dtype", "bfloat16"))
        ttk.Combobox(
            row, textvariable=self.dtype_var,
            values=["bfloat16", "float16", "float32"], state="readonly", width=18,
        ).pack(side=LEFT)
        Label(row, text="(leave on bfloat16 unless you know why you're changing it)",
              fg="#666", anchor="w").pack(side=LEFT, padx=8)

        # Row: ComfyUI path (read-only display; sourced from env / probed)
        row = Frame(self.root)
        row.pack(fill=X, **pad)
        Label(row, text="ComfyUI:", width=12, anchor="w").pack(side=LEFT)
        comfy_path = find_comfyui_path() or "(not found — set COMFYUI_PATH before launching)"
        Label(row, text=comfy_path, fg="#666", anchor="w").pack(side=LEFT, fill=X, expand=True)

        # Row: optional LoRA file picker + strength
        row = Frame(self.root)
        row.pack(fill=X, **pad)
        Label(row, text="LoRA:", width=12, anchor="w").pack(side=LEFT)
        self.lora_var = StringVar(value=s.get("lora", ""))
        Entry(row, textvariable=self.lora_var).pack(side=LEFT, fill=X, expand=True, padx=4)
        Button(row, text="Browse…", command=self._on_browse_lora).pack(side=LEFT)
        Button(row, text="Clear", command=lambda: self.lora_var.set("")).pack(side=LEFT, padx=4)

        row = Frame(self.root)
        row.pack(fill=X, **pad)
        Label(row, text="LoRA strength:", width=12, anchor="w").pack(side=LEFT)
        self.lora_strength_var = StringVar(value=s.get("lora_strength", "1.0"))
        ttk.Spinbox(
            row, from_=-2.0, to=2.0, increment=0.1,
            textvariable=self.lora_strength_var, width=8,
        ).pack(side=LEFT)
        Label(row, text="(only applies to layers this server holds — front-half + tail singles ignored)",
              fg="#666", anchor="w").pack(side=LEFT, padx=8)

        # Row: start/stop + status
        row = Frame(self.root)
        row.pack(fill=X, padx=8, pady=10)
        # Start/Restart button — width is adjusted at state-flip time
        # so the longer "Restart server to apply new settings" text
        # fits readably without permanently bloating the idle button.
        self.start_btn = Button(
            row, text="Start Server", command=self._on_start,
            width=16, font=("Segoe UI", 9),
        )
        self.start_btn.pack(side=LEFT)
        self.stop_btn = Button(row, text="Stop", command=self._on_stop, width=8, state=DISABLED)
        self.stop_btn.pack(side=LEFT, padx=4)
        self.status_label = Label(row, text="idle", fg="#666", anchor="w")
        self.status_label.pack(side=LEFT, padx=12)
        Button(row, text="Clear log", command=self._clear_log).pack(side=RIGHT)

        # Log output
        self.log = ScrolledText(self.root, wrap="word", font=("Consolas", 9), height=18)
        self.log.pack(fill=BOTH, expand=True, padx=8, pady=(0, 8))
        self.log.configure(state=DISABLED)

        # Hook file-change to update n_blocks_max
        self.weights_var.trace_add("write", lambda *_: self._refresh_after_file_change())

        # Watch every setting var so we can flip the Start button to
        # "Restart server to apply new settings" when the form drifts
        # from the snapshot captured at subprocess launch time.
        for var in (
            self.weights_var, self.n_blocks_var, self.port_var, self.bind_var,
            self.device_var, self.dtype_var, self.lora_var, self.lora_strength_var,
        ):
            var.trace_add("write", lambda *_: self._update_start_button_state())

    # ----- Window-ready handshake -----

    def _on_window_ready(self):
        """First event-loop tick after _build_ui returns. Signal the
        launcher splash that the user can see the GUI now."""
        _log_event("window painted, signalling splash to close")
        _signal_ready()

    # ----- Settings persistence -----

    def _capture_settings(self) -> dict:
        try:
            n_blocks = int(self.n_blocks_var.get())
        except Exception:
            n_blocks = 4
        try:
            port = int(self.port_var.get())
        except Exception:
            port = 7777
        return {
            "weights": self.weights_var.get().strip(),
            "n_blocks": n_blocks,
            "port": port,
            "bind": self.bind_var.get().strip(),
            "device": self.device_var.get().split(" ", 1)[0],
            "dtype": self.dtype_var.get(),
            "lora": self.lora_var.get().strip(),
            "lora_strength": self.lora_strength_var.get().strip(),
        }

    def _persist_settings(self) -> None:
        _save_settings(self._capture_settings())

    # ----- Restart-required button state -----

    def _update_start_button_state(self):
        """Flip the Start button between three states based on subprocess
        + form state: idle (Start, enabled), running with no drift
        (Start, disabled), running with form drift (Restart, enabled).
        Width / font flip with state so the long Restart text stays
        readable without bloating the idle button."""
        if self.proc is None:
            # Idle state is handled by _on_server_exit; nothing to do here.
            return
        baseline = self._running_baseline
        if baseline is None:
            return
        if self._capture_settings() != baseline:
            self.start_btn.config(
                text="Restart server to apply new settings",
                state=NORMAL,
                command=self._on_restart,
                width=38,
                font=("Segoe UI", 10, "bold"),
            )
        else:
            self.start_btn.config(
                text="Start Server",
                state=DISABLED,
                command=self._on_start,
                width=16,
                font=("Segoe UI", 9),
            )

    def _on_restart(self):
        # Surface the client-must-restart-too gotcha BEFORE we kill
        # the server, so the warning sits in the log right next to the
        # action that triggered it.
        try:
            old_n = int((self._running_baseline or {}).get("n_blocks", -1))
            new_n = int(self.n_blocks_var.get())
        except Exception:
            old_n = new_n = -1
        if old_n >= 0 and new_n >= 0 and new_n < old_n:
            self._append(
                f"[gui] *** NOTE *** lowering n_blocks ({old_n} -> {new_n}) "
                "requires the CLIENT to restart ComfyUI too — the client "
                "stripped its back-half block weights for the previous run "
                "and can only reload them from disk by a fresh ComfyUI launch.\n"
            )
        self._append("[gui] restarting server with new settings...\n")
        self._on_stop()
        # _on_stop blocks on wait/kill, then _on_server_exit resets
        # widget state. Start fresh with the new form values.
        self._on_start()

    # ----- Async device detection -----

    def _populate_devices_async(self):
        def worker():
            _log_event("nvidia-smi worker started")
            devices = detect_gpus() + ["cpu"]
            _log_event(f"nvidia-smi worker returned {len(devices)} entries")
            self.root.after(0, lambda: self._on_devices_ready(devices))
        threading.Thread(target=worker, daemon=True).start()

    def _on_devices_ready(self, devices: list[str]):
        # Prefer the saved device prefix from last run; if no saved value,
        # fall back to whatever the placeholder combobox currently shows.
        saved = (self._saved_device_prefix or "").strip()
        current_prefix = saved or self.device_var.get().split(" ", 1)[0]
        self.device_combo.config(values=devices)
        for d in devices:
            if d.split(" ", 1)[0] == current_prefix:
                self.device_var.set(d)
                _log_event(f"device combobox populated; selected '{d}'")
                return
        self.device_combo.current(0)
        _log_event("device combobox populated; default selection")

    # ----- Actions -----

    def _on_browse(self):
        path = filedialog.askopenfilename(
            title="Pick the FLUX safetensors file",
            initialdir=str(HERE),
            filetypes=[("Safetensors", "*.safetensors"), ("All files", "*.*")],
        )
        if path:
            self.weights_var.set(path)

    def _on_browse_lora(self):
        # Prefer the user's typical loras folder if we can find one
        lora_dirs = [
            HERE / "loras",
            Path("S:/Auto/ComfyUI_SEC/ComfyUI/models/loras"),
            HERE,
        ]
        initial = next((str(p) for p in lora_dirs if p.is_dir()), str(HERE))
        path = filedialog.askopenfilename(
            title="Pick a LoRA safetensors file (optional — Cancel to skip)",
            initialdir=initial,
            filetypes=[("Safetensors", "*.safetensors"), ("All files", "*.*")],
        )
        if path:
            self.lora_var.set(path)

    def _refresh_after_file_change(self):
        path = self.weights_var.get().strip()
        if not path or not Path(path).is_file():
            self.n_blocks_info.config(text="(pick a model file to see max)")
            return
        _log_event(f"reading safetensors header for n_blocks_max ({Path(path).name})")
        info = detect_n_blocks_max(Path(path))
        if info is None:
            self.n_blocks_info.config(text="(could not read header)")
            _log_event("n_blocks_max read failed")
            return
        n_double, n_single = info
        n_max = n_double + n_single
        self.n_blocks_spin.config(to=n_max)
        cur = self.n_blocks_var.get()
        if cur > n_max:
            self.n_blocks_var.set(n_max)
        self.n_blocks_info.config(
            text=(f"(checkpoint has {n_double} doubles + {n_single} singles = {n_max} blocks; "
                  f"1-{n_double}=last N doubles, "
                  f"{n_double + 1}-{n_max}=all doubles + first (N-{n_double}) singles)")
        )
        _log_event(f"n_blocks_max ready: {n_double}+{n_single}={n_max}")

    def _on_start(self):
        weights = self.weights_var.get().strip()
        if not weights or not Path(weights).is_file():
            messagebox.showerror("comfyui-mesh", "Pick a valid safetensors file first.")
            return

        device_label = self.device_var.get()
        # "cuda:0 (NVIDIA …)" → "cuda:0"
        device = device_label.split(" ", 1)[0]
        n_blocks = int(self.n_blocks_var.get())
        port = int(self.port_var.get())
        bind = self.bind_var.get().strip() or "0.0.0.0"
        dtype = self.dtype_var.get()

        env = os.environ.copy()
        if device.startswith("cuda:"):
            env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
            forwarded_device = "cuda:0"  # after CUDA_VISIBLE_DEVICES, our card is cuda:0
        elif device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
            forwarded_device = "cpu"
        else:
            forwarded_device = device

        # Make sure mesh_server.py can find ComfyUI even if launched
        # without the parent shell's COMFYUI_PATH
        comfy_path = find_comfyui_path()
        if comfy_path and "COMFYUI_PATH" not in env:
            env["COMFYUI_PATH"] = comfy_path

        py = find_python()
        cmd = [
            py, "-u", str(HERE / "mesh_server.py"),
            "--weights", weights,
            "--port", str(port),
            "--bind", bind,
            "--device", forwarded_device,
            "--dtype", dtype,
        ]
        if n_blocks > 0:
            cmd += ["--n-blocks", str(n_blocks)]

        # Optional LoRA + strength
        lora_path = self.lora_var.get().strip()
        if lora_path:
            if not Path(lora_path).is_file():
                messagebox.showerror("comfyui-mesh", f"LoRA file not found:\n{lora_path}")
                return
            try:
                lora_strength = float(self.lora_strength_var.get())
            except ValueError:
                messagebox.showerror("comfyui-mesh", "LoRA strength must be a number.")
                return
            cmd += ["--lora", lora_path, "--lora-strength", str(lora_strength)]

        # Persist settings on a successful launch attempt so next open
        # restores the same configuration.
        self._persist_settings()

        self._append(f"\n[gui] launching: {' '.join(cmd)}\n")
        self._append(f"[gui] CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '(unset)')}\n\n")

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
                env=env,
                cwd=str(HERE),
                creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
            )
        except FileNotFoundError as e:
            messagebox.showerror("comfyui-mesh", f"Could not launch python: {e}")
            return

        self.reader_thread = threading.Thread(
            target=self._reader_loop, args=(self.proc,), daemon=True,
        )
        self.reader_thread.start()

        # Snapshot the form values that drove this launch so future
        # edits can be detected as drift.
        self._running_baseline = self._capture_settings()
        self.start_btn.config(
            text="Start Server", state=DISABLED, command=self._on_start,
            width=16, font=("Segoe UI", 9),
        )
        self.stop_btn.config(state=NORMAL)
        self.status_label.config(text=f"running (pid {self.proc.pid})", fg="#080")

    def _on_stop(self):
        if self.proc is None:
            return
        self._append("[gui] stopping server…\n")
        try:
            if os.name == "nt":
                # Send Ctrl+Break to the process group so Python's signal
                # handlers fire and the socket closes cleanly.
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.proc.send_signal(signal.SIGINT)
        except Exception as e:
            self._append(f"[gui] send_signal failed: {e}\n")
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._append("[gui] still alive after 5s, killing…\n")
            self.proc.kill()
        self._on_server_exit()

    def _on_server_exit(self):
        self.proc = None
        self._running_baseline = None
        self.start_btn.config(
            text="Start Server", state=NORMAL, command=self._on_start,
            width=16, font=("Segoe UI", 9),
        )
        self.stop_btn.config(state=DISABLED)
        self.status_label.config(text="idle", fg="#666")

        # If the server exited because it received a `reconfigure`
        # message from the client, it left a small handoff file
        # telling us what to relaunch with. Apply that to the GUI's
        # form (so the user sees the new value visually) and immediately
        # restart the subprocess.
        handoff = HERE / "mesh_server_reconfig.tmp"
        if handoff.exists():
            try:
                data = json.loads(handoff.read_text(encoding="utf-8"))
                new_n = int(data.get("n_blocks", -1))
            except Exception:
                new_n = -1
            try:
                handoff.unlink()
            except Exception:
                pass
            if new_n >= 0:
                self._append(
                    f"\n[gui] server requested reconfigure to "
                    f"n_blocks={new_n} — applying + restarting...\n\n"
                )
                self.n_blocks_var.set(new_n)
                # Refresh the n_blocks_max info label etc.
                try:
                    self._refresh_after_file_change()
                except Exception:
                    pass
                # Relaunch the subprocess with the new n_blocks via
                # the same Start path the user would click.
                self._on_start()

    def _reader_loop(self, proc: subprocess.Popen):
        try:
            for line in iter(proc.stdout.readline, ""):
                self.output_q.put(line)
        finally:
            self.output_q.put(None)

    def _poll_output(self):
        try:
            while True:
                line = self.output_q.get_nowait()
                if line is None:
                    # The None sentinel may belong to a PREVIOUSLY-active
                    # reader thread (e.g. just hit Restart, the old
                    # subprocess's reader is finishing up, but self.proc
                    # is already the new live process). Only treat the
                    # None as an exit if the CURRENT proc has actually
                    # died — otherwise it's a stale leftover and would
                    # incorrectly grey out the Stop button.
                    if self.proc is None or self.proc.poll() is not None:
                        if self.proc is not None:
                            rc = self.proc.poll()
                            self._append(f"\n[gui] server exited (rc={rc})\n")
                        self._on_server_exit()
                else:
                    self._append(line)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_output)

    def _append(self, text: str):
        self.log.configure(state=NORMAL)
        self.log.insert(END, text)
        self.log.see(END)
        self.log.configure(state=DISABLED)

    def _clear_log(self):
        self.log.configure(state=NORMAL)
        self.log.delete("1.0", END)
        self.log.configure(state=DISABLED)

    def _on_close(self):
        if self.proc is not None:
            if not messagebox.askyesno("comfyui-mesh", "Server is running. Stop it and quit?"):
                return
            self._on_stop()
        self._persist_settings()
        self.root.destroy()


def main():
    _log_event("Tk() about to construct")
    root = Tk()
    _log_event("Tk() constructed")
    MeshServerGUI(root)
    _log_event("entering Tk mainloop")
    root.mainloop()
    _log_event("Tk mainloop returned (window closed)")


if __name__ == "__main__":
    main()
