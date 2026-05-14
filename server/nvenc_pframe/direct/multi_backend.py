"""MultiEngineDirectBackend — DirectBackend distributed across N NVENC engines.

Modern NVIDIA GPUs ship with multiple independent NVENC encoder engines on
the same die — the RTX 5090 has 3, the H100 has 4, the A100 has 1. They
run in true hardware parallel. This class holds one DirectBackend per
engine and dispatches encode calls across them via Python threads.

Composes with each backend's existing 8-deep output bitstream pool, so
the in-flight depth across the whole multi-backend is `n_engines * pool_depth`
(default 24 frames in flight on the 5090).

Threading model: one Python thread per engine during encode_tensor_batch().
The Python GIL is released inside the ctypes calls (cuMemcpyDtoDAsync,
nvEncEncodePicture, nvEncLockBitstream) and inside cuMemcpyDtoH on decode,
so threads actually run in parallel on the NVENC hardware engines.

Mirrors MultiEngineCodecSession's compress_batch / decompress_batch
interface where applicable, plus a tensor-based encode_tensor_batch for
the zero-copy path.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import torch
from cuda.bindings import driver as cuda

from .backend import DirectBackend


def _attach_cuda_ctx(ctx_handle: int) -> None:
    """Bind the given CUDA context to the calling (worker) thread.

    Threads spawned with `threading.Thread` have no CUDA context attached
    by default, so any cuMemcpy/cuvid call inside them fails with
    CUDA_ERROR_INVALID_CONTEXT (201). The main-thread context handle is
    captured during DirectBackend.__init__; workers re-attach it here."""
    if ctx_handle == 0:
        return
    err, = cuda.cuCtxSetCurrent(ctx_handle)
    if int(err) != 0:
        raise RuntimeError(f"cuCtxSetCurrent({ctx_handle:#x}) failed: {err}")


class MultiEngineDirectBackend:
    """Wraps N DirectBackend instances and distributes encodes across them.

    Each underlying DirectBackend opens its own NVENC encoder session,
    which the driver binds to a hardware engine. With N=3 on the 5090,
    NVENC engines run truly in parallel.
    """

    def __init__(self, height: int, width: int, qp: int = 18,
                  n_engines: int = 3, output_pool_size: int = 8,
                  cuda_stream: Optional[int] = None):
        if n_engines < 1:
            raise ValueError("n_engines must be >= 1")
        self.height = height
        self.width = width
        self.qp = qp
        self.n_engines = n_engines

        # Build N independent DirectBackend sessions.
        # Note: sharing a single CUDA stream across all backends would
        # serialize their work. If a stream is requested, we pass it to
        # all backends (caller is opting into shared pacing) — but the
        # default and the parallel-fast path uses None, which lets each
        # backend run on its own driver-internal stream.
        self.backends: list[DirectBackend] = [
            DirectBackend(height=height, width=width, qp=qp,
                          cuda_stream=cuda_stream,
                          output_pool_size=output_pool_size)
            for _ in range(n_engines)
        ]

    # ---- numpy frames path (matches CodecSession.encode_frames) -----------

    def encode_frames_batch(self, frames_list: list[np.ndarray]) -> list[list[bytes]]:
        """Encode a list of [Ni, 3, H, W] uint8 frame batches in parallel
        across engines. Returns list of packet-lists, one per input batch,
        in input order."""
        n = len(frames_list)
        if n == 0:
            return []

        results: list[Optional[list[bytes]]] = [None] * n
        assignments: list[list[int]] = [[] for _ in range(self.n_engines)]
        for i in range(n):
            assignments[i % self.n_engines].append(i)

        def worker(engine_idx: int, indices: list[int]):
            backend = self.backends[engine_idx]
            _attach_cuda_ctx(backend._cuda_ctx)
            for i in indices:
                results[i] = backend.encode_frames(frames_list[i])

        threads = [
            threading.Thread(target=worker, args=(eid, idx_list))
            for eid, idx_list in enumerate(assignments) if idx_list
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return [r for r in results if r is not None]

    # ---- torch CUDA-tensor path (zero-copy across engines) ----------------

    def encode_tensor_batch(self, tensors: list[torch.Tensor]) -> list[list[bytes]]:
        """Encode a list of [Ni, 3, H, W] uint8 CUDA tensors in parallel
        across engines. Each tensor must be on CUDA (the zero-copy path
        requires it). Returns list of packet-lists in input order."""
        n = len(tensors)
        if n == 0:
            return []

        results: list[Optional[list[bytes]]] = [None] * n
        errors: list[Optional[BaseException]] = [None] * self.n_engines
        assignments: list[list[int]] = [[] for _ in range(self.n_engines)]
        for i in range(n):
            assignments[i % self.n_engines].append(i)

        def worker(engine_idx: int, indices: list[int]):
            try:
                backend = self.backends[engine_idx]
                _attach_cuda_ctx(backend._cuda_ctx)
                for i in indices:
                    results[i] = backend.encode_tensor_frames(tensors[i])
            except BaseException as e:
                errors[engine_idx] = e

        threads = [
            threading.Thread(target=worker, args=(eid, idx_list))
            for eid, idx_list in enumerate(assignments) if idx_list
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for e in errors:
            if e is not None:
                raise e
        return [r for r in results if r is not None]

    # ---- decode batch (parallel across engines) ---------------------------

    def decode_frames_batch(self, packets_list: list[list[bytes]],
                              n_frames_per: list[int]) -> list[np.ndarray]:
        """Decode a list of (packets, n_frames) pairs in parallel across
        engines."""
        n = len(packets_list)
        if n == 0:
            return []
        if n != len(n_frames_per):
            raise ValueError("packets_list and n_frames_per must be same length")

        results: list[Optional[np.ndarray]] = [None] * n
        errors: list[Optional[BaseException]] = [None] * self.n_engines
        assignments: list[list[int]] = [[] for _ in range(self.n_engines)]
        for i in range(n):
            assignments[i % self.n_engines].append(i)

        def worker(engine_idx: int, indices: list[int]):
            try:
                backend = self.backends[engine_idx]
                _attach_cuda_ctx(backend._cuda_ctx)
                for i in indices:
                    results[i] = backend.decode_frames(packets_list[i], n_frames_per[i])
            except BaseException as e:
                errors[engine_idx] = e

        threads = [
            threading.Thread(target=worker, args=(eid, idx_list))
            for eid, idx_list in enumerate(assignments) if idx_list
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for e in errors:
            if e is not None:
                raise e
        return [r for r in results if r is not None]

    # ---- zero-copy decode batch (returns torch CUDA tensors) -------------

    def decode_frames_cuda_batch(self, packets_list: list[list[bytes]],
                                   n_frames_per: list[int]) -> list[torch.Tensor]:
        """Decode N bitstreams in parallel across engines, returning each
        as a torch CUDA tensor [Ni, 3, H, W] uint8 — no host round-trip."""
        n = len(packets_list)
        if n == 0:
            return []
        if n != len(n_frames_per):
            raise ValueError("packets_list and n_frames_per must be same length")

        results: list[Optional[torch.Tensor]] = [None] * n
        errors: list[Optional[BaseException]] = [None] * self.n_engines
        assignments: list[list[int]] = [[] for _ in range(self.n_engines)]
        for i in range(n):
            assignments[i % self.n_engines].append(i)

        def worker(engine_idx: int, indices: list[int]):
            try:
                backend = self.backends[engine_idx]
                _attach_cuda_ctx(backend._cuda_ctx)
                for i in indices:
                    results[i] = backend.decode_frames_cuda(packets_list[i],
                                                              n_frames_per[i])
            except BaseException as e:
                errors[engine_idx] = e

        threads = [
            threading.Thread(target=worker, args=(eid, idx_list))
            for eid, idx_list in enumerate(assignments) if idx_list
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for e in errors:
            if e is not None:
                raise e
        return [r for r in results if r is not None]

    # ---- lifecycle --------------------------------------------------------

    def close(self) -> None:
        for b in self.backends:
            try:
                b.close()
            except Exception:
                pass
        self.backends = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
