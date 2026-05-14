"""Build + load the C encode-loop helper.

Tries to build the helper DLL via setuptools' MSVC compiler on first import.
If the build fails or the DLL can't load, exposes `available = False` so
DirectBackend can fall back to the pure-Python loop.
"""

from __future__ import annotations

import ctypes
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional


_SRC = Path(__file__).parent / "_encode_loop.c"


class EncodeContext(ctypes.Structure):
    """Mirror of EncodeContext in _encode_loop.c — must match field-for-field."""
    _fields_ = [
        ("fn_map", ctypes.c_void_p),
        ("fn_unmap", ctypes.c_void_p),
        ("fn_encode", ctypes.c_void_p),
        ("fn_lock", ctypes.c_void_p),
        ("fn_unlock", ctypes.c_void_p),
        ("fn_memcpy_dtod", ctypes.c_void_p),
        ("fn_memcpy_dtod_async", ctypes.c_void_p),

        ("encoder", ctypes.c_void_p),
        ("cuda_stream", ctypes.c_void_p),

        ("pool_size", ctypes.c_int),
        ("per_frame_bytes", ctypes.c_int),

        ("slot_dst_ptrs", ctypes.POINTER(ctypes.c_uint64)),
        ("out_buffers", ctypes.POINTER(ctypes.c_void_p)),

        ("map_struct_ptrs", ctypes.POINTER(ctypes.c_void_p)),
        ("map_mapped_resource_offset", ctypes.c_int),

        ("pic_struct_ptr", ctypes.c_void_p),
        ("pic_inputBuffer_offset", ctypes.c_int),
        ("pic_outputBitstream_offset", ctypes.c_int),
        ("pic_encodePicFlags_offset", ctypes.c_int),

        ("lock_struct_ptr", ctypes.c_void_p),
        ("lock_outputBitstream_offset", ctypes.c_int),
        ("lock_bitstreamSizeInBytes_offset", ctypes.c_int),
        ("lock_bitstreamBufferPtr_offset", ctypes.c_int),

        ("flags_idr", ctypes.c_uint32),
    ]


_lib: Optional[ctypes.CDLL] = None
_build_error: Optional[str] = None


def _build_dll() -> Path:
    """Compile _encode_loop.c into a DLL we can load via ctypes.

    Returns the path to the built DLL. Caches under ~/.cache/nvenc-pframe
    keyed by the source file's mtime so we don't rebuild on every import."""
    cache_dir = Path(os.path.expanduser("~/.cache/nvenc-pframe"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    src_mtime = int(_SRC.stat().st_mtime)
    suffix = ".dll" if sys.platform == "win32" else ".so"
    out_dll = cache_dir / f"_encode_loop_{src_mtime}{suffix}"
    if out_dll.exists():
        return out_dll

    # Use setuptools' MSVC wrapper on Windows; fall back to system gcc/clang elsewhere
    if sys.platform == "win32":
        from setuptools._distutils import _msvccompiler
        cc = _msvccompiler.MSVCCompiler()
        cc.initialize()
        with tempfile.TemporaryDirectory() as tmp:
            objs = cc.compile([str(_SRC)], output_dir=tmp,
                              extra_postargs=["/O2"])
            cc.link_shared_object(objs, str(out_dll))
    else:
        # Linux / macOS — gcc or clang
        import subprocess
        cflags = ["-O2", "-fPIC", "-shared"]
        subprocess.run(["cc", *cflags, str(_SRC), "-o", str(out_dll)], check=True)

    return out_dll


def _try_load() -> tuple[Optional[ctypes.CDLL], Optional[str]]:
    try:
        path = _build_dll()
        lib = ctypes.CDLL(str(path))
        lib.encode_batch.argtypes = [
            ctypes.POINTER(EncodeContext),     # ctx
            ctypes.c_int,                      # n_frames
            ctypes.POINTER(ctypes.c_uint64),   # src_ptrs
            ctypes.c_void_p,                   # packet_dest
            ctypes.c_size_t,                   # packet_dest_cap
            ctypes.POINTER(ctypes.c_uint32),   # packet_offsets
            ctypes.POINTER(ctypes.c_uint32),   # packet_sizes
        ]
        lib.encode_batch.restype = ctypes.c_int
        return lib, None
    except Exception as e:
        return None, repr(e)


def get_lib() -> Optional[ctypes.CDLL]:
    """Lazy-load the helper DLL on first call. Returns None if unavailable."""
    global _lib, _build_error
    if _lib is not None:
        return _lib
    if _build_error is not None:
        return None
    _lib, _build_error = _try_load()
    return _lib


def is_available() -> bool:
    return get_lib() is not None


def last_error() -> Optional[str]:
    return _build_error
