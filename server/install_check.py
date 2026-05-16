"""Pre-flight environment check for the comfyui-mesh server.

Run this on the 4090 machine before mesh_server.py to confirm the
environment is ready:

    python install_check.py

Reports OK / MISSING / BROKEN for every dependency and prints the
fix command where applicable.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def _try(name: str, version_attr: str = "__version__") -> tuple[bool, str]:
    try:
        m = importlib.import_module(name)
        v = getattr(m, version_attr, "?")
        return True, str(v)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main():
    print("=" * 70)
    print("comfyui-mesh server install check")
    print("=" * 70)

    # 1. Python version
    py_v = ".".join(map(str, sys.version_info[:3]))
    print(f"  python:           {py_v}")

    # 2. Required deps
    for pkg in ["torch", "safetensors", "einops"]:
        ok, info = _try(pkg)
        sym = "OK     " if ok else "MISSING"
        print(f"  {pkg:18s}{sym} {info}")
        if not ok:
            print(f"                    pip install {pkg}")

    # 3. CUDA
    ok, _ = _try("torch")
    if ok:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            for i in range(n):
                print(f"  cuda:{i}            OK      {torch.cuda.get_device_name(i)}")
        else:
            print(f"  cuda              MISSING torch.cuda.is_available() == False")

    # 4. ComfyUI on path
    here = Path(__file__).parent
    candidates = [
        os.environ.get("COMFYUI_PATH"),
        str(here / "ComfyUI"),     # what install.bat clones
        "C:/ComfyUI",
        "/opt/ComfyUI",
    ]
    found = None
    for c in candidates:
        if c and Path(c).is_dir() and (Path(c) / "comfy").is_dir():
            found = c
            break
    if found:
        print(f"  ComfyUI source    OK      {found}")
        # Try the actual import we'll need
        sys.path.insert(0, found)
        ok, info = _try("comfy.ldm.flux.model")
        sym = "OK     " if ok else "BROKEN"
        print(f"  comfy.ldm.flux    {sym} {info}")
    else:
        print(f"  ComfyUI source    MISSING checked: {[c for c in candidates if c]}")
        print(f"                    git clone https://github.com/comfyanonymous/ComfyUI ./ComfyUI")
        print(f"                    set COMFYUI_PATH=/path/to/ComfyUI")

    # 5. nvenc-pframe — bundled in this folder; needs cuda-bindings on PATH
    sys.path.insert(0, str(here))  # make the bundled subfolder importable
    ok, info = _try("nvenc_pframe")
    sym = "OK     " if ok else "BROKEN "
    print(f"  nvenc-pframe      {sym} {info} (bundled at ./nvenc_pframe/)")
    if not ok:
        print(f"                    (the bundled package imports cuda-bindings —")
        print(f"                     install with: pip install cuda-bindings)")

    # 6. Model file
    weights = here / "flux-2-klein-9b-fp8.safetensors"
    if weights.exists():
        sz = weights.stat().st_size / 1024**3
        print(f"  weights file      OK      {weights.name} ({sz:.2f} GB)")
    else:
        print(f"  weights file      MISSING expected at {weights}")

    print("=" * 70)


if __name__ == "__main__":
    main()
