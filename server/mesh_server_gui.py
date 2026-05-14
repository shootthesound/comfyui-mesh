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
        self.root = root
        root.title("comfyui-mesh — back-half server")
        root.geometry("780x560")

        self.proc: subprocess.Popen | None = None
        self.output_q: queue.Queue = queue.Queue()
        self.reader_thread: threading.Thread | None = None

        self._build_ui()
        self._refresh_after_file_change()
        self.root.after(100, self._poll_output)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ----- UI build -----

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        # Row: weights file picker
        row = Frame(self.root)
        row.pack(fill=X, **pad)
        Label(row, text="Model:", width=12, anchor="w").pack(side=LEFT)
        self.weights_var = StringVar(value="")
        self.weights_entry = Entry(row, textvariable=self.weights_var)
        self.weights_entry.pack(side=LEFT, fill=X, expand=True, padx=4)
        Button(row, text="Browse…", command=self._on_browse).pack(side=LEFT)

        # Row: n_blocks
        row = Frame(self.root)
        row.pack(fill=X, **pad)
        Label(row, text="n_blocks:", width=12, anchor="w").pack(side=LEFT)
        self.n_blocks_var = IntVar(value=4)
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
        self.port_var = IntVar(value=7777)
        ttk.Spinbox(row, from_=1, to=65535, increment=1, textvariable=self.port_var, width=8).pack(side=LEFT)
        self.bind_var = StringVar(value="0.0.0.0")
        Label(row, text="  Bind:", anchor="w").pack(side=LEFT, padx=(16, 4))
        Entry(row, textvariable=self.bind_var, width=18).pack(side=LEFT)

        # Row: device + dtype
        row = Frame(self.root)
        row.pack(fill=X, **pad)
        Label(row, text="Device:", width=12, anchor="w").pack(side=LEFT)
        self.device_var = StringVar()
        device_choices = detect_gpus() + ["cpu"]
        self.device_combo = ttk.Combobox(
            row, textvariable=self.device_var, values=device_choices, state="readonly", width=42,
        )
        self.device_combo.current(0)
        self.device_combo.pack(side=LEFT)

        row = Frame(self.root)
        row.pack(fill=X, **pad)
        Label(row, text="dtype:", width=12, anchor="w").pack(side=LEFT)
        self.dtype_var = StringVar(value="bfloat16")
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

        # Row: start/stop + status
        row = Frame(self.root)
        row.pack(fill=X, padx=8, pady=10)
        self.start_btn = Button(row, text="Start Server", command=self._on_start, width=16)
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

    # ----- Actions -----

    def _on_browse(self):
        path = filedialog.askopenfilename(
            title="Pick the FLUX safetensors file",
            initialdir=str(HERE),
            filetypes=[("Safetensors", "*.safetensors"), ("All files", "*.*")],
        )
        if path:
            self.weights_var.set(path)

    def _refresh_after_file_change(self):
        path = self.weights_var.get().strip()
        if not path or not Path(path).is_file():
            self.n_blocks_info.config(text="(pick a model file to see max)")
            return
        info = detect_n_blocks_max(Path(path))
        if info is None:
            self.n_blocks_info.config(text="(could not read header)")
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

        self.start_btn.config(state=DISABLED)
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
        self.start_btn.config(state=NORMAL)
        self.stop_btn.config(state=DISABLED)
        self.status_label.config(text="idle", fg="#666")

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
        self.root.destroy()


def main():
    root = Tk()
    MeshServerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
