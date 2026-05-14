"""ctypes bindings for NVDEC (cuviddec / nvcuvid).

Maps cuviddec.h + nvcuvid.h from the NVIDIA Video Codec SDK 13. Unlike
NVENC, NVDEC has no `CreateInstance` table — functions are loaded directly
as DLL/SO exports.

Usage flow for a one-shot HEVC decode:

  1. Open a CUDA context and create a CUvideoctxlock (cuvidCtxLockCreate)
  2. Build CUVIDPARSERPARAMS with three callback function pointers:
       pfnSequenceCallback — fires once with CUVIDEOFORMAT; create decoder here
       pfnDecodePicture   — fires per picture with CUVIDPICPARAMS; pass through
       pfnDisplayPicture  — fires when a frame is ready; map+copy+unmap
  3. cuvidCreateVideoParser
  4. cuvidParseVideoData (pass HEVC bitstream packet)
  5. cuvidParseVideoData with CUVID_PKT_ENDOFSTREAM to flush
  6. Copy frames out of CUDA device memory to torch tensors
  7. cuvidDestroyVideoParser, cuvidDestroyDecoder, cuvidCtxLockDestroy

Platform notes:
- Some fields in cuviddec.h declare `unsigned long` which differs by platform
  (LLP64 Windows = 4 bytes, LP64 Linux = 8 bytes). On Windows we use c_uint32;
  porting to Linux requires per-field review. Documented inline.
"""

from __future__ import annotations

import ctypes
import platform
from ctypes import (
    Structure, POINTER, byref, c_int, c_uint, c_uint32, c_int32,
    c_short, c_uint8, c_ubyte, c_uint64, c_void_p, c_char_p, CFUNCTYPE,
)

from .api import load_nvcuvid


# ---- types & enums --------------------------------------------------------

# CUresult is int (CUDA error code); we map to c_int32
CUresult = c_int32

# Opaque handles
CUvideodecoder = c_void_p
CUvideoparser = c_void_p
CUvideoctxlock = c_void_p
CUcontext = c_void_p
CUstream = c_void_p
CUvideotimestamp = c_uint64

# cudaVideoCodec_enum
cudaVideoCodec_HEVC = 8
cudaVideoCodec_AV1 = 11

# cudaVideoChromaFormat_enum
cudaVideoChromaFormat_Monochrome = 0
cudaVideoChromaFormat_420 = 1
cudaVideoChromaFormat_422 = 2
cudaVideoChromaFormat_444 = 3

# cudaVideoSurfaceFormat_enum
cudaVideoSurfaceFormat_NV12 = 0
cudaVideoSurfaceFormat_P016 = 1
cudaVideoSurfaceFormat_YUV444 = 2
cudaVideoSurfaceFormat_YUV444_16Bit = 3

# cudaVideoCreateFlags
cudaVideoCreate_Default = 0
cudaVideoCreate_PreferCUDA = 1
cudaVideoCreate_PreferDXVA = 2
cudaVideoCreate_PreferCUVID = 4

# cudaVideoDeinterlaceMode (ignored for progressive)
cudaVideoDeinterlaceMode_Weave = 0

# CUVID_PKT_xxx flags
CUVID_PKT_ENDOFSTREAM = 0x01
CUVID_PKT_TIMESTAMP   = 0x02
CUVID_PKT_DISCONTINUITY = 0x04
CUVID_PKT_ENDOFPICTURE = 0x08
CUVID_PKT_NOTIFY_EOS  = 0x10


# ---- structures -----------------------------------------------------------
# Note: `unsigned long` fields in the C header are 4 bytes on Windows LLP64
# and 8 bytes on Linux LP64. We assume Windows here.


class _DisplayArea(Structure):
    _fields_ = [
        ("left", c_short),
        ("top", c_short),
        ("right", c_short),
        ("bottom", c_short),
    ]


class _TargetRect(Structure):
    _fields_ = [
        ("left", c_short),
        ("top", c_short),
        ("right", c_short),
        ("bottom", c_short),
    ]


class CUVIDDECODECREATEINFO(Structure):
    """Args struct for cuvidCreateDecoder."""
    _fields_ = [
        # On Windows, `unsigned long` is 32 bits.
        ("ulWidth", c_uint32),
        ("ulHeight", c_uint32),
        ("ulNumDecodeSurfaces", c_uint32),
        ("CodecType", c_uint32),               # cudaVideoCodec
        ("ChromaFormat", c_uint32),             # cudaVideoChromaFormat
        ("ulCreationFlags", c_uint32),
        ("bitDepthMinus8", c_uint32),
        ("ulIntraDecodeOnly", c_uint32),
        ("ulMaxWidth", c_uint32),
        ("ulMaxHeight", c_uint32),
        ("Reserved1", c_uint32),
        ("display_area", _DisplayArea),         # 4 short = 8 bytes
        ("OutputFormat", c_uint32),             # cudaVideoSurfaceFormat
        ("DeinterlaceMode", c_uint32),          # cudaVideoDeinterlaceMode
        ("ulTargetWidth", c_uint32),
        ("ulTargetHeight", c_uint32),
        ("ulNumOutputSurfaces", c_uint32),
        ("vidLock", c_void_p),                  # CUvideoctxlock
        ("target_rect", _TargetRect),
        ("enableHistogram", c_uint32),
        ("Reserved2", c_uint32 * 4),
    ]


class _FrameRate(Structure):
    _fields_ = [("numerator", c_uint32), ("denominator", c_uint32)]


class _DispRect(Structure):
    _fields_ = [
        ("left", c_int32), ("top", c_int32),
        ("right", c_int32), ("bottom", c_int32),
    ]


class _DispAR(Structure):
    _fields_ = [("x", c_int32), ("y", c_int32)]


class _VideoSignalDesc(Structure):
    _fields_ = [
        # video_format:3 + video_full_range_flag:1 + reserved_zero_bits:4 packed in one byte
        ("packed_byte", c_uint8),
        ("color_primaries", c_uint8),
        ("transfer_characteristics", c_uint8),
        ("matrix_coefficients", c_uint8),
    ]


class CUVIDEOFORMAT(Structure):
    """Video format — passed to pfnSequenceCallback."""
    _fields_ = [
        ("codec", c_uint32),                    # cudaVideoCodec
        ("frame_rate", _FrameRate),
        ("progressive_sequence", c_uint8),
        ("bit_depth_luma_minus8", c_uint8),
        ("bit_depth_chroma_minus8", c_uint8),
        ("min_num_decode_surfaces", c_uint8),
        ("coded_width", c_uint32),
        ("coded_height", c_uint32),
        ("display_area", _DispRect),            # 4 int = 16 bytes
        ("chroma_format", c_uint32),            # cudaVideoChromaFormat
        ("bitrate", c_uint32),
        ("display_aspect_ratio", _DispAR),
        ("video_signal_description", _VideoSignalDesc),
        ("seqhdr_data_length", c_uint32),
    ]


class CUVIDSOURCEDATAPACKET(Structure):
    """Args struct for cuvidParseVideoData."""
    _fields_ = [
        ("flags", c_uint32),                    # tcu_ulong on Windows = 4 bytes
        ("payload_size", c_uint32),
        ("payload", POINTER(c_uint8)),
        ("timestamp", CUvideotimestamp),
    ]


class CUVIDPARSERDISPINFO(Structure):
    """Display info — passed to pfnDisplayPicture."""
    _fields_ = [
        ("picture_index", c_int32),
        ("progressive_frame", c_int32),
        ("top_field_first", c_int32),
        ("repeat_first_field", c_int32),
        ("timestamp", CUvideotimestamp),
    ]


class CUVIDPICPARAMS(Structure):
    """Per-picture decode args — passed through from parser to cuvidDecodePicture.

    The CodecSpecific union holds the codec-specific picture params. We don't
    fill it in (the parser does), but we need the union to be the right size:
    CodecReserved[1024] (uint32) = 4096 bytes — the SDK's documented max.
    """
    _fields_ = [
        ("PicWidthInMbs", c_int32),
        ("FrameHeightInMbs", c_int32),
        ("CurrPicIdx", c_int32),
        ("field_pic_flag", c_int32),
        ("bottom_field_flag", c_int32),
        ("second_field", c_int32),
        ("nBitstreamDataLen", c_uint32),
        ("pBitstreamData", POINTER(c_uint8)),
        ("nNumSlices", c_uint32),
        ("pSliceDataOffsets", POINTER(c_uint32)),
        ("ref_pic_flag", c_int32),
        ("intra_pic_flag", c_int32),
        ("Reserved", c_uint32 * 30),
        ("CodecSpecific", c_uint32 * 1024),     # union sized to CodecReserved[1024]
    ]


class CUVIDPROCPARAMS(Structure):
    """Args struct for cuvidMapVideoFrame64 — post-processing options."""
    _fields_ = [
        ("progressive_frame", c_int32),
        ("second_field", c_int32),
        ("top_field_first", c_int32),
        ("unpaired_field", c_int32),
        ("reserved_flags", c_uint32),
        ("reserved_zero", c_uint32),
        ("raw_input_dptr", c_uint64),
        ("raw_input_pitch", c_uint32),
        ("raw_input_format", c_uint32),
        ("raw_output_dptr", c_uint64),
        ("raw_output_pitch", c_uint32),
        ("Reserved1", c_uint32),
        ("output_stream", c_void_p),            # CUstream
        ("Reserved", c_uint32 * 46),
        ("histogram_dptr", POINTER(c_uint64)),
        ("Reserved2", c_void_p * 1),
    ]


# Callback function pointer types
PFNVIDSEQUENCECALLBACK = CFUNCTYPE(c_int, c_void_p, POINTER(CUVIDEOFORMAT))
PFNVIDDECODECALLBACK = CFUNCTYPE(c_int, c_void_p, POINTER(CUVIDPICPARAMS))
PFNVIDDISPLAYCALLBACK = CFUNCTYPE(c_int, c_void_p, POINTER(CUVIDPARSERDISPINFO))
PFNVIDOPPOINTCALLBACK = CFUNCTYPE(c_int, c_void_p, c_void_p)  # AV1-specific
PFNVIDSEIMSGCALLBACK = CFUNCTYPE(c_int, c_void_p, c_void_p)


class CUVIDPARSERPARAMS(Structure):
    """Args struct for cuvidCreateVideoParser."""
    _fields_ = [
        ("CodecType", c_uint32),
        ("ulMaxNumDecodeSurfaces", c_uint32),
        ("ulClockRate", c_uint32),
        ("ulErrorThreshold", c_uint32),
        ("ulMaxDisplayDelay", c_uint32),
        ("flag_bitfield", c_uint32),            # bAnnexb:1 / bMemoryOptimize:1 / uReserved:30
        ("uReserved1", c_uint32 * 4),
        ("pUserData", c_void_p),
        ("pfnSequenceCallback", PFNVIDSEQUENCECALLBACK),
        ("pfnDecodePicture", PFNVIDDECODECALLBACK),
        ("pfnDisplayPicture", PFNVIDDISPLAYCALLBACK),
        ("pfnGetOperatingPoint", PFNVIDOPPOINTCALLBACK),
        ("pfnGetSEIMsg", PFNVIDSEIMSGCALLBACK),
        ("pvReserved2", c_void_p * 5),
        ("pExtVideoInfo", c_void_p),
    ]


# ---- function pointers --------------------------------------------------

_nvcuvid: ctypes.CDLL | None = None


def _lib() -> ctypes.CDLL:
    global _nvcuvid
    if _nvcuvid is None:
        _nvcuvid = load_nvcuvid()
    return _nvcuvid


# Function signatures
def _bind():
    L = _lib()

    L.cuvidCtxLockCreate.argtypes = [POINTER(c_void_p), c_void_p]
    L.cuvidCtxLockCreate.restype = CUresult

    L.cuvidCtxLockDestroy.argtypes = [c_void_p]
    L.cuvidCtxLockDestroy.restype = CUresult

    L.cuvidCreateVideoParser.argtypes = [POINTER(c_void_p), POINTER(CUVIDPARSERPARAMS)]
    L.cuvidCreateVideoParser.restype = CUresult

    L.cuvidParseVideoData.argtypes = [c_void_p, POINTER(CUVIDSOURCEDATAPACKET)]
    L.cuvidParseVideoData.restype = CUresult

    L.cuvidDestroyVideoParser.argtypes = [c_void_p]
    L.cuvidDestroyVideoParser.restype = CUresult

    L.cuvidCreateDecoder.argtypes = [POINTER(c_void_p), POINTER(CUVIDDECODECREATEINFO)]
    L.cuvidCreateDecoder.restype = CUresult

    L.cuvidDestroyDecoder.argtypes = [c_void_p]
    L.cuvidDestroyDecoder.restype = CUresult

    L.cuvidDecodePicture.argtypes = [c_void_p, POINTER(CUVIDPICPARAMS)]
    L.cuvidDecodePicture.restype = CUresult

    L.cuvidMapVideoFrame64.argtypes = [c_void_p, c_int, POINTER(c_uint64),
                                         POINTER(c_uint32), POINTER(CUVIDPROCPARAMS)]
    L.cuvidMapVideoFrame64.restype = CUresult

    L.cuvidUnmapVideoFrame64.argtypes = [c_void_p, c_uint64]
    L.cuvidUnmapVideoFrame64.restype = CUresult

    return L


# ---- helpers --------------------------------------------------------------

def ctx_lock_create(cuda_context: int) -> c_void_p:
    """Create a CUvideoctxlock for the given CUDA context."""
    L = _bind()
    lock = c_void_p()
    s = L.cuvidCtxLockCreate(byref(lock), cuda_context)
    if int(s) != 0:
        raise RuntimeError(f"cuvidCtxLockCreate failed: CUresult={s}")
    return lock


def ctx_lock_destroy(lock: c_void_p) -> None:
    L = _bind()
    s = L.cuvidCtxLockDestroy(lock)
    if int(s) != 0:
        raise RuntimeError(f"cuvidCtxLockDestroy failed: CUresult={s}")


def create_parser(parser_params: CUVIDPARSERPARAMS) -> c_void_p:
    L = _bind()
    parser = c_void_p()
    s = L.cuvidCreateVideoParser(byref(parser), byref(parser_params))
    if int(s) != 0:
        raise RuntimeError(f"cuvidCreateVideoParser failed: CUresult={s}")
    return parser


def parse_video_data(parser: c_void_p, packet: CUVIDSOURCEDATAPACKET) -> None:
    L = _bind()
    s = L.cuvidParseVideoData(parser, byref(packet))
    if int(s) != 0:
        raise RuntimeError(f"cuvidParseVideoData failed: CUresult={s}")


def destroy_parser(parser: c_void_p) -> None:
    L = _bind()
    s = L.cuvidDestroyVideoParser(parser)
    if int(s) != 0:
        raise RuntimeError(f"cuvidDestroyVideoParser failed: CUresult={s}")


def create_decoder(create_info: CUVIDDECODECREATEINFO) -> c_void_p:
    L = _bind()
    decoder = c_void_p()
    s = L.cuvidCreateDecoder(byref(decoder), byref(create_info))
    if int(s) != 0:
        raise RuntimeError(f"cuvidCreateDecoder failed: CUresult={s}")
    return decoder


def destroy_decoder(decoder: c_void_p) -> None:
    L = _bind()
    s = L.cuvidDestroyDecoder(decoder)
    if int(s) != 0:
        raise RuntimeError(f"cuvidDestroyDecoder failed: CUresult={s}")


def decode_picture(decoder: c_void_p, pic_params_ptr: int) -> None:
    """Pass-through call from inside pfnDecodePicture callback.

    pic_params_ptr is the raw C pointer the parser handed us; we forward it
    directly via ctypes.cast — no copy."""
    L = _bind()
    s = L.cuvidDecodePicture(decoder,
                              ctypes.cast(pic_params_ptr, POINTER(CUVIDPICPARAMS)))
    if int(s) != 0:
        raise RuntimeError(f"cuvidDecodePicture failed: CUresult={s}")


def map_video_frame64(decoder: c_void_p, pic_idx: int,
                       proc_params: CUVIDPROCPARAMS) -> tuple[int, int]:
    """Map a decoded frame; returns (device pointer as uint64, pitch in bytes)."""
    L = _bind()
    dptr = c_uint64()
    pitch = c_uint32()
    s = L.cuvidMapVideoFrame64(decoder, pic_idx, byref(dptr), byref(pitch),
                                byref(proc_params))
    if int(s) != 0:
        raise RuntimeError(f"cuvidMapVideoFrame64 failed: CUresult={s}")
    return int(dptr.value), int(pitch.value)


def unmap_video_frame64(decoder: c_void_p, dptr: int) -> None:
    L = _bind()
    s = L.cuvidUnmapVideoFrame64(decoder, dptr)
    if int(s) != 0:
        raise RuntimeError(f"cuvidUnmapVideoFrame64 failed: CUresult={s}")
