"""nvenc_pframe — NVENC's P-frame chain as a free delta codec for
temporally-coherent GPU state (CFD, FEM, weather, MD, progressive renders).

This package exposes the direct Video Codec SDK bindings — `DirectBackend`
and `MultiEngineDirectBackend` — that PoC `01_vortex_street_pframe.py`
uses to demonstrate temporal compression of a Navier-Stokes vorticity
trajectory.

The bindings themselves were originally developed in a sibling repo
(`torch-nvenc-compress`, https://github.com/shootthesound/torch-nvenc-compress)
that focuses on the ML / activation / KV-cache application of NVENC's
intra-frame mode. This repo reuses the same bindings and exclusively
exercises the inter-frame (P-frame) mode, which is where most of video's
compression magic actually lives.

Typical use:

    import torch
    from nvenc_pframe.direct.backend import DirectBackend

    backend = DirectBackend(height=H, width=W, qp=18)
    # frames: torch CUDA tensor [N, 3, H, W] uint8 (YUV444 layout)
    # frame 0 forced IDR; frames 1..N-1 default to P-frames
    packets = backend.encode_tensor_frames(frames)
    decoded = backend.decode_frames_cuda(packets, N)
    backend.close()
"""

__version__ = "0.1.0"
