"""Per-channel uint8 quantize + NVENC HEVC YUV444 codec for FLUX
activation tensors crossing the wire between split-rig nodes.

This is the "wire codec" layer of the comfyui-mesh node. It accepts a
torch CUDA tensor of arbitrary shape, packs it into NVENC-friendly
YUV444 frames, encodes via the nvenc-pframe DirectBackend, and returns
bytes plus metadata sufficient for the receiver to reverse the
operation.

Two modes:
    "raw"    — just bf16/fp16 bytes, no compression. Useful as a wire-
               protocol sanity check and as a fallback if nvenc-pframe
               isn't installed on the receiving end.
    "nvenc"  — codec round-trip. Per-channel global min/max -> uint8 ->
               NVENC HEVC YUV444 -> bytes.

For FLUX-class activation tensors (shape [B, T, H], e.g. [1, 4096, 4096]
fp16 = 33.5 MB) the nvenc mode typically gives ~5-15x compression at QP
high enough to be visually indistinguishable in the final image.

Packing strategy: reshape [B, T, H] -> rearrange so that H channels
become the "frame axis" and (B*T) becomes the spatial axis of one
square-ish frame. For [1, 4096, 4096]: 4096 channels -> pack into
frame triplets (Y,U,V) = 1366 codec frames, each 64x64 -> too small
for NVENC. So we tile H into n_frames frames of larger spatial size.
Concrete: pack 3 channels per Y/U/V -> n_frames = ceil(H/3); spatial
shape per frame = some pixel-arrangement of T tokens. For 4096 tokens
we use 64x64.

For the first prototype we use the same level_triplets-style packing
proven on the ECMWF and CFD demos: 3 adjacent channels per YUV frame,
spatial axis is whatever rectangle 0.._T_ tokens fit into.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


HEVC_ALIGN = 16
HEVC_MIN_DIM = 144


def _align_up(x: int) -> int:
    target = max(x, HEVC_MIN_DIM)
    return ((target + HEVC_ALIGN - 1) // HEVC_ALIGN) * HEVC_ALIGN


# Persistent DirectBackend cache, keyed by (height, width, qp, lossless).
# Constructing a DirectBackend invokes nvEncOpenEncodeSessionEx +
# nvEncInitializeEncoder + buffer allocation — measured at ~300-500ms
# per call. The hot path on a real workflow only ever uses 1-2 distinct
# configs (one for img encode at the chosen QP), so caching turns N
# misses into N-1 hits and saves multiple hundreds of ms per timestep.
#
# We deliberately never close() these. Process exit reclaims them. Long-
# running server / client processes hold them for the process lifetime.
_BACKEND_CACHE: dict = {}


def _get_or_create_backend(height: int, width: int, qp: int = 0, lossless: bool = False):
    """Return a cached DirectBackend for the given config, creating one
    if needed. Caller MUST NOT close() the returned backend."""
    from nvenc_pframe.direct.backend import DirectBackend  # type: ignore

    key = (int(height), int(width), int(qp), bool(lossless))
    backend = _BACKEND_CACHE.get(key)
    if backend is None:
        backend = DirectBackend(
            height=height, width=width,
            qp=qp, lossless=lossless,
        )
        _BACKEND_CACHE[key] = backend
    return backend


def _get_decode_backend(height: int, width: int):
    """Decode-side lookup. The decoder doesn't care about qp/lossless
    (it just consumes the bitstream), so any cached backend with matching
    (height, width) works. Falls back to creating one with default config
    if no existing backend matches."""
    height = int(height)
    width = int(width)
    for (h, w, _qp, _ll), backend in _BACKEND_CACHE.items():
        if h == height and w == width:
            return backend
    return _get_or_create_backend(height, width, qp=0, lossless=False)


@dataclass
class WireTensor:
    """Result of encoding a tensor for the wire."""
    name: str
    encoding: str        # "raw" | "nvenc"
    bytes_payload: bytes
    dtype_str: str       # original tensor dtype as string ("torch.float16" etc)
    shape: tuple[int, ...]
    # nvenc-mode metadata
    n_codec_frames: int = 0
    h_padded: int = 0
    w_padded: int = 0
    h_data: int = 0      # per-channel data height (before tiling)
    w_data: int = 0      # per-channel data width  (before tiling)
    n_channels: int = 0
    # Channel-tile dim. tile_dim=1 -> one channel per Y plane (legacy
    # behaviour). tile_dim=4 -> 4x4 grid of channels in each Y plane,
    # so 16 channels per plane and 48 per codec frame. Tiling reduces
    # per-frame NVENC overhead — see README "frame-count reduction".
    tile_dim: int = 1
    # Per-channel min/max for the uint8 quant step (lists of length n_channels).
    # Global-min/max quant is too coarse for ML activations — see codec_mode
    # discussion in the README; per-channel preserves ~bf16 precision per channel.
    var_mins: list = None
    var_maxs: list = None

    def to_header(self) -> dict:
        return {
            "name": self.name,
            "encoding": self.encoding,
            "size": len(self.bytes_payload),
            "dtype": self.dtype_str,
            "shape": list(self.shape),
            "extra": {
                "n_codec_frames": self.n_codec_frames,
                "h_padded": self.h_padded,
                "w_padded": self.w_padded,
                "h_data": self.h_data,
                "w_data": self.w_data,
                "n_channels": self.n_channels,
                "tile_dim": self.tile_dim,
                "var_mins": self.var_mins if self.var_mins is not None else [],
                "var_maxs": self.var_maxs if self.var_maxs is not None else [],
            },
        }


def _pick_2d_layout(n_tokens: int) -> tuple[int, int]:
    """Pick a near-square (h, w) such that h*w >= n_tokens, h*w is a
    plausible image dimension for a video codec, and w is a multiple
    of 16 after padding."""
    h = int(math.isqrt(n_tokens))
    while h > 0 and (n_tokens % h) != 0:
        h -= 1
    if h == 0:
        h = int(math.isqrt(n_tokens)) or 1
    w = (n_tokens + h - 1) // h
    return h, w


def encode_raw(name: str, tensor: torch.Tensor) -> WireTensor:
    """Wire encoding without compression. Bytes are the tensor's
    contiguous CPU buffer in its native dtype.

    NumPy has no bfloat16 dtype, so for bf16 tensors we reinterpret the
    underlying bits as uint16 before going through numpy. The reverse
    happens in decode_raw via `tensor.view(torch.bfloat16)`.
    """
    cpu = tensor.detach().contiguous().cpu()
    orig_dtype = tensor.dtype
    if cpu.dtype == torch.bfloat16:
        cpu = cpu.view(torch.uint16)
    payload = cpu.numpy().tobytes()
    return WireTensor(
        name=name,
        encoding="raw",
        bytes_payload=payload,
        dtype_str=str(orig_dtype),
        shape=tuple(tensor.shape),
    )


def decode_raw(wire: dict, payload: bytes, device: torch.device) -> torch.Tensor:
    """Inverse of encode_raw.

    For bfloat16 the wire bytes are actually uint16 (same bit layout);
    we use `.view(torch.bfloat16)` to *reinterpret* the bits, not cast
    them, because a value cast uint16->bf16 would corrupt the data.
    """
    dtype = _torch_dtype_from_str(wire["dtype"])
    shape = tuple(wire["shape"])
    np_dtype = _numpy_dtype_for(dtype)
    arr = np.frombuffer(payload, dtype=np_dtype).reshape(shape).copy()
    t = torch.from_numpy(arr)
    if dtype == torch.bfloat16:
        # Bit-reinterpret uint16 -> bf16, then move to device
        return t.view(torch.bfloat16).to(device=device)
    return t.to(device=device, dtype=dtype)


def _torch_dtype_from_str(s: str) -> torch.dtype:
    return {
        "torch.float32": torch.float32,
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.int8": torch.int8,
        "torch.uint8": torch.uint8,
    }[s]


def _numpy_dtype_for(t: torch.dtype):
    if t == torch.bfloat16:
        # numpy has no bf16 — round-trip via uint16 with a view
        return np.uint16
    return {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.int8: np.int8,
        torch.uint8: np.uint8,
    }[t]


def encode_nvenc(
    name: str,
    tensor: torch.Tensor,
    qp: int = 18,
    lossless: bool = False,
    tile_dim: int = 4,
) -> WireTensor:
    """Per-channel uint8 quantize + NVENC HEVC YUV444 round-trip.

    Tensor is expected shape [B, T, H] or [B, T1, T2, H] (FLUX-like).
    Internally flattens to [B*T_total, H] and treats H as the channel axis.

    `tile_dim` controls channel tiling within each codec frame:
      - tile_dim=1: one channel per Y plane, one frame per 3 channels
                    (legacy; small frames; many of them)
      - tile_dim=4: 4x4 grid of channels per Y plane, 16 channels per
                    plane, 48 per codec frame. Cuts NVENC frame count
                    by ~16x and eliminates the 144x144 minimum-frame
                    padding overhead. Default.
      - tile_dim=8: 8x8 grid, 64 channels per plane, 192 per frame.
                    More aggressive; bigger frames; fewer of them.

    Pixel correctness is preserved across any tile_dim — same total data
    is encoded; we just rearrange spatially so NVENC sees fewer, larger
    frames.
    """
    from nvenc_pframe.direct.backend import DirectBackend  # type: ignore

    if tile_dim < 1:
        raise ValueError(f"tile_dim must be >= 1, got {tile_dim}")

    orig_shape = tuple(tensor.shape)
    orig_dtype = tensor.dtype
    flat = tensor.detach().contiguous().reshape(-1, orig_shape[-1])  # [N, H]
    n_tokens, n_channels = flat.shape

    h_data, w_data = _pick_2d_layout(n_tokens)
    pad = h_data * w_data - n_tokens
    if pad > 0:
        flat = torch.cat([flat, flat[:1].expand(pad, -1)], dim=0)

    # rearrange to [n_channels, h_data, w_data]
    arr = flat.reshape(h_data, w_data, n_channels).permute(2, 0, 1).contiguous()  # [C, H, W]

    # Per-channel min/max -> uint8. Global min/max is far too lossy
    # for ML activations because a handful of outlier channels widen
    # the range so much that typical channels get only a few of 256
    # bins. Per-channel keeps full uint8 precision per channel; metadata
    # cost is 2 * 4 bytes * n_channels (e.g. 32 KB for 4096 channels).
    arr_f32 = arr.to(torch.float32)  # [C, H, W]
    flat_per_c = arr_f32.reshape(n_channels, -1)
    mins = flat_per_c.amin(dim=1)  # [C]
    maxs = flat_per_c.amax(dim=1)  # [C]
    ranges = (maxs - mins).clamp(min=1e-12)
    scale = (255.0 / ranges).view(-1, 1, 1)
    mn_b = mins.view(-1, 1, 1)
    u8 = ((arr_f32 - mn_b) * scale).clamp(0, 255).to(torch.uint8)
    # Constant-valued channels (max-min ~= 0) become 128; we'll round-
    # trip them back to mins[c] via dequant which is the right answer.
    const_mask = (maxs - mins) < 1e-12
    if const_mask.any():
        u8[const_mask] = 128
    var_mins = mins.tolist()
    var_maxs = maxs.tolist()

    # ----- Channel tiling -----
    # We pack tile_dim*tile_dim channels into each Y/U/V plane as a
    # spatial grid, so each codec frame carries 3*tile_dim^2 channels.
    # Pad the channel count so it's an exact multiple of that group size.
    chans_per_plane = tile_dim * tile_dim
    chans_per_frame = 3 * chans_per_plane
    n_codec_frames = (n_channels + chans_per_frame - 1) // chans_per_frame
    target_channels = n_codec_frames * chans_per_frame
    pad_channels = target_channels - n_channels
    if pad_channels > 0:
        # Replicate last real channel into the tail slots — cleaner for
        # codec efficiency than zero pad.
        u8_padded = torch.cat(
            [u8, u8[-1:].expand(pad_channels, h_data, w_data)],
            dim=0,
        )
    else:
        u8_padded = u8

    # u8_padded shape: [target_channels, h_data, w_data]
    # Reshape into the (frames, planes, tile_row, tile_col, h, w) view,
    # then permute to interleave tile_row with h and tile_col with w,
    # then collapse to a flat 2D plane.
    tiled = u8_padded.reshape(
        n_codec_frames, 3, tile_dim, tile_dim, h_data, w_data
    )
    # [N, 3, tr, tc, h, w] -> [N, 3, tr, h, tc, w]
    tiled = tiled.permute(0, 1, 2, 4, 3, 5).contiguous()
    tile_h = tile_dim * h_data
    tile_w = tile_dim * w_data
    yuv_data = tiled.reshape(n_codec_frames, 3, tile_h, tile_w)

    # Pad spatially up to NVENC alignment + min-dim. With tile_dim>=2
    # the tile is already past 144 in most realistic cases, so this
    # is usually a no-op.
    h_padded = _align_up(tile_h)
    w_padded = _align_up(tile_w)
    if h_padded == tile_h and w_padded == tile_w:
        yuv = yuv_data
    else:
        yuv = torch.zeros(
            (n_codec_frames, 3, h_padded, w_padded),
            dtype=torch.uint8, device=tensor.device,
        )
        yuv[:, :, :tile_h, :tile_w] = yuv_data

    backend = _get_or_create_backend(h_padded, w_padded, qp=qp, lossless=lossless)
    packets = backend.encode_tensor_frames(yuv)
    codec_bytes = b"".join(packets)

    return WireTensor(
        name=name,
        encoding="nvenc",
        bytes_payload=codec_bytes,
        dtype_str=str(orig_dtype),
        shape=orig_shape,
        n_codec_frames=n_codec_frames,
        h_padded=h_padded,
        w_padded=w_padded,
        h_data=h_data,
        w_data=w_data,
        n_channels=n_channels,
        tile_dim=tile_dim,
        var_mins=var_mins,
        var_maxs=var_maxs,
    )


def decode_nvenc(wire: dict, payload: bytes, device: torch.device) -> torch.Tensor:
    """Inverse of encode_nvenc."""
    from nvenc_pframe.direct.backend import DirectBackend  # type: ignore

    extra = wire["extra"]
    n_frames = int(extra["n_codec_frames"])
    h_padded = int(extra["h_padded"])
    w_padded = int(extra["w_padded"])
    h_data = int(extra["h_data"])
    w_data = int(extra["w_data"])
    n_channels = int(extra["n_channels"])
    tile_dim = int(extra.get("tile_dim", 1))  # legacy frames default to 1
    var_mins = extra.get("var_mins", [])
    var_maxs = extra.get("var_maxs", [])
    if not var_mins or not var_maxs or len(var_mins) != n_channels:
        raise ValueError(
            f"decode_nvenc: per-channel var_mins/var_maxs missing or wrong length "
            f"({len(var_mins) if var_mins else 0} vs n_channels {n_channels}). "
            f"This codec.py requires the per-channel quant scheme; both ends must "
            f"be running the same version."
        )
    if tile_dim < 1:
        raise ValueError(f"tile_dim must be >= 1, got {tile_dim}")

    backend = _get_decode_backend(h_padded, w_padded)
    # decode_frames_cuda expects a list of bytes packets — but we
    # concatenated them on encode. The decoder takes the full
    # bitstream as a single packet.
    decoded_t = backend.decode_frames_cuda([payload], n_frames)

    yuv = decoded_t.to(device=device)  # [n_frames, 3, h_padded, w_padded] uint8

    # Crop spatial padding to the tile size, then untile back to per-
    # channel slices. Inverse of the encode-side reshape+permute.
    tile_h = tile_dim * h_data
    tile_w = tile_dim * w_data
    yuv_cropped = yuv[:, :, :tile_h, :tile_w]                                  # [N, 3, tile_h, tile_w]
    untiled = yuv_cropped.reshape(n_frames, 3, tile_dim, h_data, tile_dim, w_data)
    # Inverse of permute(0, 1, 2, 4, 3, 5) is the same permutation
    # (it swaps dims (3,4))
    untiled = untiled.permute(0, 1, 2, 4, 3, 5).contiguous()                   # [N, 3, tr, tc, h, w]
    flat = untiled.reshape(n_frames * 3 * tile_dim * tile_dim, h_data, w_data)
    out_channels = flat[:n_channels].contiguous()                              # [n_channels, h_data, w_data]

    # Per-channel dequant. mins/maxs are float32 lists shipped in the header.
    mins_t = torch.tensor(var_mins, dtype=torch.float32, device=device).view(-1, 1, 1)
    maxs_t = torch.tensor(var_maxs, dtype=torch.float32, device=device).view(-1, 1, 1)
    ranges = (maxs_t - mins_t).clamp(min=1e-12)
    inv_scale = ranges / 255.0
    f = out_channels.to(torch.float32) * inv_scale + mins_t
    # For constant channels (range was ~0), the codec stored uint8=128.
    # Dequant above gives mins + 128*(0)/255 ≈ mins, which equals the
    # constant value — correct.

    # [C, H, W] -> [H, W, C] -> [H*W, C] -> trim padding -> reshape to original
    flat = f.permute(1, 2, 0).reshape(h_data * w_data, n_channels)
    n_tokens = 1
    for d in wire["shape"][:-1]:
        n_tokens *= int(d)
    flat = flat[:n_tokens]
    target_dtype = _torch_dtype_from_str(wire["dtype"])
    return flat.reshape(tuple(wire["shape"])).to(dtype=target_dtype)


def encode_nvenc_clipsparse(
    name: str, tensor: torch.Tensor,
    qp: int = 18, tile_dim: int = 4, outlier_pct: float = 0.5,
) -> WireTensor:
    """Per-channel percentile-clip quant + NVENC HEVC + sparse outlier correction.

    Same NVENC pipeline as `encode_nvenc` (single encode pass, same per-frame
    latency) but with two changes:
      1. Per-channel quant range is `[p_lo, p_hi]` where
         `p_lo = quantile(outlier_pct%)` and `p_hi = quantile(100-outlier_pct%)`.
         Heavy-tailed channels (the LTX contrast-crush culprit) no longer let
         a few extreme values widen the range and squash everyone else into
         a handful of uint8 bins.
      2. Values outside `[p_lo, p_hi]` get clipped to 0/255 by the quant
         step but are also captured separately as `(channel, position, exact
         float16 value)` triples. The decoder scatter-overwrites them after
         dequant, so outliers come back exactly.

    Default `outlier_pct=0.5` clips 0.5% of each tail → 1% outliers total.
    For a 50 MB activation tensor that's ~500K outliers × 10 bytes ≈ 5 MB
    of sparse correction on top of the ~5 MB nvenc bitstream — still ~5×
    smaller than raw, with much better within-channel precision than plain
    nvenc on heavy-tailed distributions.

    Added for the LTX path. FLUX paths never select this mode.
    """
    import struct as _struct

    if tile_dim < 1:
        raise ValueError(f"tile_dim must be >= 1, got {tile_dim}")
    if not (0.0 < outlier_pct < 50.0):
        raise ValueError(f"outlier_pct must be in (0, 50), got {outlier_pct}")

    orig_shape = tuple(tensor.shape)
    orig_dtype = tensor.dtype
    flat = tensor.detach().contiguous().reshape(-1, orig_shape[-1])
    n_tokens, n_channels = flat.shape

    h_data, w_data = _pick_2d_layout(n_tokens)
    pad = h_data * w_data - n_tokens
    if pad > 0:
        flat = torch.cat([flat, flat[:1].expand(pad, -1)], dim=0)

    arr = flat.reshape(h_data, w_data, n_channels).permute(2, 0, 1).contiguous()  # [C, H, W]
    arr_f32 = arr.to(torch.float32)

    # Per-channel percentile clip points. torch.quantile works on the last dim
    # by default; we want per-channel so reshape to [C, H*W] first.
    flat_per_c = arr_f32.reshape(n_channels, -1)
    pct_lo = float(outlier_pct) / 100.0
    pct_hi = 1.0 - pct_lo
    p_lo = torch.quantile(flat_per_c, q=pct_lo, dim=1)  # [C]
    p_hi = torch.quantile(flat_per_c, q=pct_hi, dim=1)  # [C]

    ranges = (p_hi - p_lo).clamp(min=1e-12)
    scale = (255.0 / ranges).view(-1, 1, 1)
    mn_b = p_lo.view(-1, 1, 1)
    hi_b = p_hi.view(-1, 1, 1)
    u8 = ((arr_f32 - mn_b) * scale).clamp(0, 255).to(torch.uint8)
    # Constant-valued channels (range ~ 0) → 128; dequant lands back at mean.
    const_mask = (p_hi - p_lo) < 1e-12
    if const_mask.any():
        u8[const_mask] = 128

    # Find outliers (positions clipped by the percentile cap).
    outlier_mask = (arr_f32 < mn_b) | (arr_f32 > hi_b)  # [C, H, W] bool
    if const_mask.any():
        outlier_mask[const_mask] = False  # constant channels have nothing to correct
    outlier_idx = outlier_mask.nonzero(as_tuple=False)  # [N, 3] (c, h, w) int64
    n_outliers = int(outlier_idx.shape[0])
    if n_outliers > 0:
        ch_idx = outlier_idx[:, 0].to(torch.int32).contiguous().cpu().numpy()
        pos_idx = (outlier_idx[:, 1] * w_data + outlier_idx[:, 2]).to(torch.int32).contiguous().cpu().numpy()
        outlier_vals_f16 = arr_f32[outlier_mask].to(torch.float16).contiguous().cpu().numpy()
        # Pack: [chans:int32 * N][positions:int32 * N][values:float16 * N]
        sparse_bytes = (
            ch_idx.tobytes() + pos_idx.tobytes() + outlier_vals_f16.tobytes()
        )
    else:
        sparse_bytes = b""

    var_mins = p_lo.tolist()
    var_maxs = p_hi.tolist()

    # ----- Channel tiling + NVENC encode (identical to encode_nvenc) -----
    chans_per_plane = tile_dim * tile_dim
    chans_per_frame = 3 * chans_per_plane
    n_codec_frames = (n_channels + chans_per_frame - 1) // chans_per_frame
    target_channels = n_codec_frames * chans_per_frame
    pad_channels = target_channels - n_channels
    if pad_channels > 0:
        u8_padded = torch.cat(
            [u8, u8[-1:].expand(pad_channels, h_data, w_data)],
            dim=0,
        )
    else:
        u8_padded = u8

    tiled = u8_padded.reshape(
        n_codec_frames, 3, tile_dim, tile_dim, h_data, w_data
    ).permute(0, 1, 2, 4, 3, 5).contiguous()
    tile_h = tile_dim * h_data
    tile_w = tile_dim * w_data
    yuv_data = tiled.reshape(n_codec_frames, 3, tile_h, tile_w)
    h_padded = _align_up(tile_h)
    w_padded = _align_up(tile_w)
    if h_padded == tile_h and w_padded == tile_w:
        yuv = yuv_data
    else:
        yuv = torch.zeros(
            (n_codec_frames, 3, h_padded, w_padded),
            dtype=torch.uint8, device=tensor.device,
        )
        yuv[:, :, :tile_h, :tile_w] = yuv_data

    backend = _get_or_create_backend(h_padded, w_padded, qp=qp, lossless=False)
    codec_bytes = b"".join(backend.encode_tensor_frames(yuv))

    # Wire layout: [u32 n_outliers][u32 codec_size][sparse_bytes][codec_bytes]
    header = _struct.pack(">II", n_outliers, len(codec_bytes))
    payload = header + sparse_bytes + codec_bytes

    return WireTensor(
        name=name,
        encoding="Nvenc LTX (5090 optimized)",
        bytes_payload=payload,
        dtype_str=str(orig_dtype),
        shape=orig_shape,
        n_codec_frames=n_codec_frames,
        h_padded=h_padded,
        w_padded=w_padded,
        h_data=h_data,
        w_data=w_data,
        n_channels=n_channels,
        tile_dim=tile_dim,
        var_mins=var_mins,
        var_maxs=var_maxs,
    )


def decode_nvenc_clipsparse(wire: dict, payload: bytes, device: torch.device) -> torch.Tensor:
    """Inverse of encode_nvenc_clipsparse. NVDEC + per-channel dequant
    + scatter-overwrite the sparse outliers exactly."""
    import struct as _struct

    extra = wire["extra"]
    n_frames = int(extra["n_codec_frames"])
    h_padded = int(extra["h_padded"])
    w_padded = int(extra["w_padded"])
    h_data = int(extra["h_data"])
    w_data = int(extra["w_data"])
    n_channels = int(extra["n_channels"])
    tile_dim = int(extra.get("tile_dim", 1))
    var_mins = extra.get("var_mins", [])
    var_maxs = extra.get("var_maxs", [])
    if not var_mins or not var_maxs or len(var_mins) != n_channels:
        raise ValueError(
            f"decode_nvenc_clipsparse: per-channel quant tables missing "
            f"({len(var_mins) if var_mins else 0} vs n_channels {n_channels})"
        )
    if tile_dim < 1:
        raise ValueError(f"tile_dim must be >= 1, got {tile_dim}")

    # Split wire payload
    n_outliers, codec_size = _struct.unpack(">II", payload[:8])
    sparse_total = 4 * n_outliers + 4 * n_outliers + 2 * n_outliers  # chans + positions + values
    sparse_bytes = payload[8 : 8 + sparse_total]
    codec_bytes = bytes(payload[8 + sparse_total : 8 + sparse_total + codec_size])

    # ----- NVDEC + untile (identical to decode_nvenc) -----
    backend = _get_decode_backend(h_padded, w_padded)
    decoded_t = backend.decode_frames_cuda([codec_bytes], n_frames)
    yuv = decoded_t.to(device=device)

    tile_h = tile_dim * h_data
    tile_w = tile_dim * w_data
    yuv_cropped = yuv[:, :, :tile_h, :tile_w]
    untiled = yuv_cropped.reshape(
        n_frames, 3, tile_dim, h_data, tile_dim, w_data
    ).permute(0, 1, 2, 4, 3, 5).contiguous()
    flat = untiled.reshape(n_frames * 3 * tile_dim * tile_dim, h_data, w_data)
    out_channels = flat[:n_channels].contiguous()  # [C, H, W] uint8

    # Per-channel dequant using p_lo / p_hi as the quant range.
    mins_t = torch.tensor(var_mins, dtype=torch.float32, device=device).view(-1, 1, 1)
    maxs_t = torch.tensor(var_maxs, dtype=torch.float32, device=device).view(-1, 1, 1)
    ranges = (maxs_t - mins_t).clamp(min=1e-12)
    inv_scale = ranges / 255.0
    f = out_channels.to(torch.float32) * inv_scale + mins_t  # [C, H, W] float32

    # Scatter-overwrite outliers with their exact float16 values.
    if n_outliers > 0:
        sz_int = 4 * n_outliers
        sz_val = 2 * n_outliers
        chans = np.frombuffer(sparse_bytes[:sz_int], dtype=np.int32).copy()
        positions = np.frombuffer(sparse_bytes[sz_int : 2 * sz_int], dtype=np.int32).copy()
        values_f16 = np.frombuffer(sparse_bytes[2 * sz_int : 2 * sz_int + sz_val], dtype=np.float16).copy()
        chans_t = torch.from_numpy(chans).to(device=device, dtype=torch.long)
        positions_t = torch.from_numpy(positions).to(device=device, dtype=torch.long)
        values_t = torch.from_numpy(values_f16).to(device=device, dtype=torch.float32)
        # Decompose flat positions into (h, w)
        h_idx = positions_t // w_data
        w_idx = positions_t % w_data
        f[chans_t, h_idx, w_idx] = values_t

    # Reshape back to original tensor shape.
    flat = f.permute(1, 2, 0).reshape(h_data * w_data, n_channels)
    n_tokens = 1
    for d in wire["shape"][:-1]:
        n_tokens *= int(d)
    flat = flat[:n_tokens]
    target_dtype = _torch_dtype_from_str(wire["dtype"])
    return flat.reshape(tuple(wire["shape"])).to(dtype=target_dtype)


def encode(name: str, tensor: torch.Tensor, mode: str, qp: int = 18, lossless: bool = False, tile_dim: int = 4) -> WireTensor:
    if mode == "raw":
        return encode_raw(name, tensor)
    elif mode == "nvenc":
        return encode_nvenc(name, tensor, qp=qp, lossless=lossless, tile_dim=tile_dim)
    elif mode == "Nvenc LTX (5090 optimized)":
        return encode_nvenc_clipsparse(name, tensor, qp=qp, tile_dim=tile_dim)
    else:
        raise ValueError(f"unknown codec mode {mode!r}")


def decode(wire: dict, payload: bytes, device: torch.device) -> torch.Tensor:
    enc = wire["encoding"]
    if enc == "raw":
        return decode_raw(wire, payload, device)
    elif enc == "nvenc":
        return decode_nvenc(wire, payload, device)
    elif enc == "Nvenc LTX (5090 optimized)":
        return decode_nvenc_clipsparse(wire, payload, device)
    else:
        raise ValueError(f"unknown wire encoding {enc!r}")
