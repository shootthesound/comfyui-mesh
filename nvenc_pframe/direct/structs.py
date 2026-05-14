"""NVENC C struct definitions (ctypes).

Mirrors nvEncodeAPI.h struct layouts. Field order MUST match the header
exactly; getting it wrong silently corrupts the encoder state.

Strategy used here: most fields are zeroed (the C API treats 0 as "use
default"). We only fill in the handful that matter for our use case:
HEVC YUV444 at constQP. The bitfield-packed flags use bit positions
documented inline so they're easy to verify against nvEncodeAPI.h.

References:
- nvEncodeAPI.h from NVIDIA Video Codec SDK 13.0
"""

from __future__ import annotations

from ctypes import (
    Structure, Union, POINTER, byref, c_uint32, c_int32, c_uint16, c_uint8,
    c_int8, c_uint64, c_void_p, CFUNCTYPE,
)

from .api import (
    GUID, NVENCSTATUS, NVENCAPI_STRUCT_VERSION,
    NV_ENC_PARAMS_RC_CONSTQP,
    NV_ENC_TUNING_INFO_HIGH_QUALITY,
)


# Buffer formats from NV_ENC_BUFFER_FORMAT enum
NV_ENC_BUFFER_FORMAT_NV12 = 0x00000001
NV_ENC_BUFFER_FORMAT_YV12 = 0x00000010
NV_ENC_BUFFER_FORMAT_IYUV = 0x00000100
NV_ENC_BUFFER_FORMAT_YUV444 = 0x00001000
NV_ENC_BUFFER_FORMAT_ARGB = 0x01000000

# Struct version constants for SDK 13 (verified against
# FFmpeg/nv-codec-headers/include/ffnvcodec/nvEncodeAPI.h master).
NV_ENC_QP_VER = NVENCAPI_STRUCT_VERSION(1)
NV_ENC_RC_PARAMS_VER = NVENCAPI_STRUCT_VERSION(1)
NV_ENC_CONFIG_HEVC_VER = NVENCAPI_STRUCT_VERSION(1)
NV_ENC_CONFIG_VER = NVENCAPI_STRUCT_VERSION(9) | (1 << 31)         # v9 in SDK 13
NV_ENC_PRESET_CONFIG_VER = NVENCAPI_STRUCT_VERSION(5) | (1 << 31)  # v5 in SDK 13
NV_ENC_INITIALIZE_PARAMS_VER = NVENCAPI_STRUCT_VERSION(7) | (1 << 31)  # v7 in SDK 13
NV_ENC_PIC_PARAMS_VER = NVENCAPI_STRUCT_VERSION(7) | (1 << 31)
NV_ENC_LOCK_BITSTREAM_VER = NVENCAPI_STRUCT_VERSION(2) | (1 << 31)
NV_ENC_CREATE_INPUT_BUFFER_VER = NVENCAPI_STRUCT_VERSION(2)
NV_ENC_CREATE_BITSTREAM_BUFFER_VER = NVENCAPI_STRUCT_VERSION(1)
NV_ENC_REGISTER_RESOURCE_VER = NVENCAPI_STRUCT_VERSION(5)
NV_ENC_MAP_INPUT_RESOURCE_VER = NVENCAPI_STRUCT_VERSION(4)


class NV_ENC_QP(Structure):
    """Quantization parameter triple (per inter-P, inter-B, and intra)."""
    _fields_ = [
        ("qpInterP", c_uint32),
        ("qpInterB", c_uint32),
        ("qpIntra", c_uint32),
    ]


class NV_ENC_RC_PARAMS(Structure):
    """Rate control parameters. We use constQP only; rest are zero defaults.

    Field order verified against SDK 13's nvEncodeAPI.h. Earlier (incorrect)
    versions of this struct had cbQPIndexOffset/crQPIndexOffset as c_int32
    (4 bytes each) when they're actually int8_t (1 byte each), and were
    missing lookaheadLevel, viewBitrateRatios[], reserved3, reserved1.
    Mis-sizing this struct doesn't fail encoder init outright but leaves the
    encoder in a state where nvEncEncodePicture fails with status 12
    (UNSUPPORTED_PARAM).
    """
    _fields_ = [
        ("version", c_uint32),
        ("rateControlMode", c_uint32),    # NV_ENC_PARAMS_RC_MODE enum
        ("constQP", NV_ENC_QP),
        ("averageBitRate", c_uint32),
        ("maxBitRate", c_uint32),
        ("vbvBufferSize", c_uint32),
        ("vbvInitialDelay", c_uint32),
        # 32-bit packed flag field (enableMinQP:1 + enableMaxQP:1 + ...
        # + aqStrength:4 + enableExtLookahead:1 + reservedBitFields:15)
        ("flag_bitfield", c_uint32),
        ("minQP", NV_ENC_QP),
        ("maxQP", NV_ENC_QP),
        ("initialRCQP", NV_ENC_QP),
        ("temporallayerIdxMask", c_uint32),
        ("temporalLayerQP", c_uint8 * 8),
        ("targetQuality", c_uint8),
        ("targetQualityLSB", c_uint8),
        ("lookaheadDepth", c_uint16),
        ("lowDelayKeyFrameScale", c_uint8),
        ("yDcQPIndexOffset", c_int8),
        ("uDcQPIndexOffset", c_int8),
        ("vDcQPIndexOffset", c_int8),
        ("qpMapMode", c_uint32),         # NV_ENC_QP_MAP_MODE enum
        ("multiPass", c_uint32),         # NV_ENC_MULTI_PASS enum
        ("alphaLayerBitrateRatio", c_uint32),
        ("cbQPIndexOffset", c_int8),
        ("crQPIndexOffset", c_int8),
        ("reserved2", c_uint16),
        ("lookaheadLevel", c_uint32),    # NV_ENC_LOOKAHEAD_LEVEL enum
        ("viewBitrateRatios", c_uint8 * 7),  # MAX_NUM_VIEWS_MINUS_1 = 7
        ("reserved3", c_uint8),
        ("reserved1", c_uint32),
    ]


# NV_ENC_CONFIG_HEVC bitfield positions (from nvEncodeAPI.h):
#   bit 0:  useConstrainedIntraPred
#   bit 1:  disableDeblockAcrossSliceBoundary
#   bit 2:  outputBufferingPeriodSEI
#   bit 3:  outputPictureTimingSEI
#   bit 4:  outputAUD
#   bit 5:  enableLTR
#   bit 6:  disableSPSPPS
#   bit 7:  repeatSPSPPS                <-- WE SET THIS for inline SPS/PPS at each IDR
#   bit 8:  enableIntraRefresh
#   bits 9-10: chromaFormatIDC          <-- WE SET TO 3 for YUV 4:4:4
#   bit 11: enableConstrainedEncoding
#   bit 12: enableAlphaLayerEncoding
#   bits 13-15: pixelBitDepthMinus8     <-- 0 for 8-bit
#   bits 16-31: reservedBitFields
HEVC_BITFIELD_REPEAT_SPSPPS = 1 << 7
HEVC_BITFIELD_CHROMA_444 = 3 << 9        # chromaFormatIDC=3 means YUV444


class NV_ENC_CONFIG_HEVC_VUI_PARAMETERS(Structure):
    """VUI (Video Usability Information) for HEVC. All zeros = no VUI."""
    _fields_ = [
        ("overscanInfoPresentFlag", c_uint32),
        ("overscanInfo", c_uint32),
        ("videoSignalTypePresentFlag", c_uint32),
        ("videoFormat", c_uint32),
        ("videoFullRangeFlag", c_uint32),
        ("colourDescriptionPresentFlag", c_uint32),
        ("colourPrimaries", c_uint32),
        ("transferCharacteristics", c_uint32),
        ("colourMatrix", c_uint32),
        ("chromaSampleLocationFlag", c_uint32),
        ("chromaSampleLocationTop", c_uint32),
        ("chromaSampleLocationBot", c_uint32),
        ("bitstreamRestrictionFlag", c_uint32),
        ("timingInfoPresentFlag", c_uint32),
        ("numUnitInTicks", c_uint32),
        ("timeScale", c_uint32),
        ("reserved", c_uint32 * 12),
    ]


class NV_ENC_CONFIG_HEVC(Structure):
    """HEVC-specific encoder config. Only the chromaFormatIDC bits matter for
    our YUV444 use; the rest can stay at preset defaults."""
    _fields_ = [
        ("level", c_uint32),                 # 0 = AUTO_LEVEL
        ("tier", c_uint32),
        ("minCUSize", c_uint32),
        ("maxCUSize", c_uint32),
        ("flag_bitfield", c_uint32),         # bitfield (see HEVC_BITFIELD_* above)
        ("idrPeriod", c_uint32),
        ("intraRefreshPeriod", c_uint32),
        ("intraRefreshCnt", c_uint32),
        ("maxNumRefFramesInDPB", c_uint32),
        ("ltrNumFrames", c_uint32),
        ("vpsId", c_uint32),
        ("spsId", c_uint32),
        ("ppsId", c_uint32),
        ("useBFramesAsRef", c_uint32),       # NV_ENC_BFRAME_REF_MODE enum
        ("numRefL0", c_uint32),              # NV_ENC_NUM_REF_FRAMES enum
        ("numRefL1", c_uint32),
        ("hevcVUIParameters", NV_ENC_CONFIG_HEVC_VUI_PARAMETERS),
        ("ltrTrustMode", c_uint32),
        ("reserved1", c_uint32 * 208),
        ("reserved2", c_void_p * 64),
    ]


class NV_ENC_CONFIG_H264_DUMMY(Structure):
    """Placeholder for H264 config — same size as HEVC for union sizing."""
    _fields_ = [("buf", c_uint32 * 320)]


class NV_ENC_CONFIG_AV1_DUMMY(Structure):
    """Placeholder for AV1 config — same size as HEVC for union sizing."""
    _fields_ = [("buf", c_uint32 * 320)]


class NV_ENC_CODEC_CONFIG(Union):
    """Union of codec-specific configs. Size is max of all members.
    nvEncodeAPI.h reserves 320 uint32s = 1280 bytes for forward compatibility."""
    _fields_ = [
        ("hevcConfig", NV_ENC_CONFIG_HEVC),
        ("h264Config", NV_ENC_CONFIG_H264_DUMMY),
        ("av1Config", NV_ENC_CONFIG_AV1_DUMMY),
        ("reserved", c_uint32 * 320),
    ]


class NV_ENC_CONFIG(Structure):
    """Top-level encoder configuration. Contains rate control + codec-specific."""
    _fields_ = [
        ("version", c_uint32),
        ("profileGUID", GUID),               # 16 bytes
        ("gopLength", c_uint32),
        ("frameIntervalP", c_int32),
        ("monoChromeEncoding", c_uint32),
        ("frameFieldMode", c_uint32),        # NV_ENC_PARAMS_FRAME_FIELD_MODE enum
        ("mvPrecision", c_uint32),           # NV_ENC_MV_PRECISION enum
        ("rcParams", NV_ENC_RC_PARAMS),
        ("encodeCodecConfig", NV_ENC_CODEC_CONFIG),
        ("reserved", c_uint32 * 278),
        ("reserved2", c_void_p * 64),
    ]


class NV_ENC_PRESET_CONFIG(Structure):
    """Wrapper for a preset's pre-filled config — what nvEncGetEncodePresetConfigEx returns."""
    _fields_ = [
        ("version", c_uint32),
        ("presetCfg", NV_ENC_CONFIG),
        ("reserved1", c_uint32 * 255),
        ("reserved2", c_void_p * 64),
    ]


class NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE(Structure):
    """ME hint counts struct (16 bytes total — packed bitfield + 3 reserved u32)."""
    _fields_ = [
        ("packed", c_uint32),       # bitfield: numCands per 16x16/16x8/8x16/8x8 + reserved
        ("reserved1", c_uint32 * 3),
    ]


class NV_ENC_INITIALIZE_PARAMS(Structure):
    """Top-level args struct for nvEncInitializeEncoder.

    Field order verified against SDK 13's nvEncodeAPI.h (struct version 7).
    The SDK 13 layout differs from older docs — encodeConfig moved earlier
    (now after privData, before maxEncodeWidth) and several new fields were
    added (privDataSize, reserved, privData, numStateBuffers, outputStatsLevel).
    """
    _fields_ = [
        ("version", c_uint32),
        ("encodeGUID", GUID),
        ("presetGUID", GUID),
        ("encodeWidth", c_uint32),
        ("encodeHeight", c_uint32),
        ("darWidth", c_uint32),
        ("darHeight", c_uint32),
        ("frameRateNum", c_uint32),
        ("frameRateDen", c_uint32),
        ("enableEncodeAsync", c_uint32),
        ("enablePTD", c_uint32),
        ("flag_bitfield", c_uint32),         # reportSliceOffsets/enableSubFrameWrite/.../splitEncodeMode/etc
        ("privDataSize", c_uint32),
        ("reserved_field", c_uint32),
        ("privData", c_void_p),
        ("encodeConfig", POINTER(NV_ENC_CONFIG)),
        ("maxEncodeWidth", c_uint32),
        ("maxEncodeHeight", c_uint32),
        ("maxMEHintCountsPerBlock", NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE * 2),
        ("tuningInfo", c_uint32),            # NV_ENC_TUNING_INFO enum
        ("bufferFormat", c_uint32),          # NV_ENC_BUFFER_FORMAT enum
        ("numStateBuffers", c_uint32),
        ("outputStatsLevel", c_uint32),      # NV_ENC_OUTPUT_STATS_LEVEL enum
        ("reserved1", c_uint32 * 284),
        ("reserved2", c_void_p * 64),
    ]


# ---- helpers --------------------------------------------------------------

def get_preset_config_ex(table, encoder, codec_guid, preset_guid,
                          tuning=NV_ENC_TUNING_INFO_HIGH_QUALITY) -> NV_ENC_PRESET_CONFIG:
    """Query the SDK for a preset's default config (pre-filled with sensible values)."""
    fn_type = CFUNCTYPE(NVENCSTATUS, c_void_p, GUID, GUID, c_uint32, POINTER(NV_ENC_PRESET_CONFIG))
    fn = fn_type(table.nvEncGetEncodePresetConfigEx)
    pc = NV_ENC_PRESET_CONFIG()
    pc.version = NV_ENC_PRESET_CONFIG_VER
    pc.presetCfg.version = NV_ENC_CONFIG_VER
    s = fn(encoder, codec_guid, preset_guid, tuning, byref(pc))
    if s != 0:
        raise RuntimeError(f"nvEncGetEncodePresetConfigEx failed: status={s}")
    return pc


def initialize_encoder_hevc_yuv444(table, encoder, codec_guid, preset_guid,
                                     width: int, height: int, qp: int,
                                     tuning=NV_ENC_TUNING_INFO_HIGH_QUALITY):
    """Initialise the encoder for HEVC YUV444 at the given QP.

    Pulls preset defaults from the SDK, overrides chromaFormatIDC + constQP,
    then calls nvEncInitializeEncoder. Returns the (init_params, config)
    structs (you must keep references alive until destroy).
    """
    # 1. Get preset defaults
    preset_config = get_preset_config_ex(table, encoder, codec_guid, preset_guid, tuning)
    config = preset_config.presetCfg
    config.version = NV_ENC_CONFIG_VER

    # 2. Set rate control to constQP
    config.rcParams.version = NV_ENC_RC_PARAMS_VER
    config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP
    config.rcParams.constQP.qpInterP = qp
    config.rcParams.constQP.qpInterB = qp
    config.rcParams.constQP.qpIntra = qp

    # 2b. Disable B-frames (frameIntervalP=1 means I-P-P-P, no B-frames),
    # disable lookahead, and force one IDR per frame group. With B-frames the
    # encoder buffers inputs across calls (returns NV_ENC_ERR_NEED_MORE_INPUT)
    # which breaks the per-call-input-buffer-reuse pattern DirectBackend uses.
    config.frameIntervalP = 1
    config.rcParams.lookaheadDepth = 0
    # Clear enableLookahead bit (bit 5) in the RC flag bitfield
    config.rcParams.flag_bitfield &= ~(1 << 5)

    # 3. Override codec-specific HEVC config: chromaFormatIDC=3 for YUV444,
    #    plus repeatSPSPPS so each IDR is self-decodable.
    hevc = config.encodeCodecConfig.hevcConfig
    # Preserve any preset-set bits and add ours
    hevc.flag_bitfield = (
        hevc.flag_bitfield
        | HEVC_BITFIELD_CHROMA_444
        | HEVC_BITFIELD_REPEAT_SPSPPS
    )

    # 4. Build init params
    init = NV_ENC_INITIALIZE_PARAMS()
    init.version = NV_ENC_INITIALIZE_PARAMS_VER
    init.encodeGUID = codec_guid
    init.presetGUID = preset_guid
    init.encodeWidth = width
    init.encodeHeight = height
    init.darWidth = width
    init.darHeight = height
    init.frameRateNum = 30
    init.frameRateDen = 1
    init.enablePTD = 1                       # let NVENC pick frame types
    init.tuningInfo = tuning
    init.bufferFormat = NV_ENC_BUFFER_FORMAT_YUV444
    init.encodeConfig = POINTER(NV_ENC_CONFIG)(config)

    # 5. Call nvEncInitializeEncoder
    fn_type = CFUNCTYPE(NVENCSTATUS, c_void_p, POINTER(NV_ENC_INITIALIZE_PARAMS))
    fn = fn_type(table.nvEncInitializeEncoder)
    s = fn(encoder, byref(init))
    if s != 0:
        raise RuntimeError(f"nvEncInitializeEncoder failed: status={s}")
    return init, config


# ---- Input/output buffer creation ----------------------------------------

class NV_ENC_CREATE_INPUT_BUFFER(Structure):
    """Args for nvEncCreateInputBuffer (CPU-allocated input frame buffer)."""
    _fields_ = [
        ("version", c_uint32),
        ("width", c_uint32),
        ("height", c_uint32),
        ("memoryHeap", c_uint32),       # deprecated
        ("bufferFmt", c_uint32),        # NV_ENC_BUFFER_FORMAT
        ("reserved", c_uint32),
        ("inputBuffer", c_void_p),       # [out] handle to created buffer
        ("pSysMemBuffer", c_void_p),
        ("reserved1", c_uint32 * 58),
        ("reserved2", c_void_p * 63),
    ]


class NV_ENC_CREATE_BITSTREAM_BUFFER(Structure):
    """Args for nvEncCreateBitstreamBuffer (output bitstream buffer)."""
    _fields_ = [
        ("version", c_uint32),
        ("size", c_uint32),               # deprecated
        ("memoryHeap", c_uint32),         # deprecated
        ("reserved", c_uint32),
        ("bitstreamBuffer", c_void_p),    # [out] handle
        ("bitstreamBufferPtr", c_void_p), # [out] reserved
        ("reserved1", c_uint32 * 58),
        ("reserved2", c_void_p * 64),
    ]


class NV_ENC_LOCK_INPUT_BUFFER(Structure):
    """Args for nvEncLockInputBuffer."""
    _fields_ = [
        ("version", c_uint32),
        ("flags_bitfield", c_uint32),     # bit 0 = doNotWait, rest reserved
        ("inputBuffer", c_void_p),
        ("bufferDataPtr", c_void_p),      # [out] mapped CPU pointer
        ("pitch", c_uint32),              # [out]
        ("reserved1", c_uint32 * 251),
        ("reserved2", c_void_p * 64),
    ]


NV_ENC_LOCK_INPUT_BUFFER_VER = NVENCAPI_STRUCT_VERSION(1)


# ---- Resource registration (for CUDA buffer interop / zero-copy encode) ---

# NV_ENC_INPUT_RESOURCE_TYPE enum
NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX = 0
NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR = 1
NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY = 2
NV_ENC_INPUT_RESOURCE_TYPE_OPENGL_TEX = 3

# NV_ENC_BUFFER_USAGE enum
NV_ENC_INPUT_IMAGE = 0
NV_ENC_OUTPUT_BITSTREAM = 2
NV_ENC_OUTPUT_RECON = 4


class NV_ENC_REGISTER_RESOURCE(Structure):
    """Args for nvEncRegisterResource — register a CUDA pointer (or other
    cross-API surface) so the encoder can use it as an input."""
    _fields_ = [
        ("version", c_uint32),
        ("resourceType", c_uint32),       # NV_ENC_INPUT_RESOURCE_TYPE enum
        ("width", c_uint32),
        ("height", c_uint32),
        ("pitch", c_uint32),
        ("subResourceIndex", c_uint32),
        ("resourceToRegister", c_void_p),
        ("registeredResource", c_void_p),  # [out]
        ("bufferFormat", c_uint32),
        ("bufferUsage", c_uint32),         # NV_ENC_BUFFER_USAGE enum
        ("pInputFencePoint", c_void_p),    # NV_ENC_FENCE_POINT_D3D12*
        ("chromaOffset", c_uint32 * 2),    # [out]
        ("chromaOffsetIn", c_uint32 * 2),  # [in]
        ("reserved1", c_uint32 * 244),
        ("reserved2", c_void_p * 61),
    ]


class NV_ENC_MAP_INPUT_RESOURCE(Structure):
    """Args for nvEncMapInputResource — produces an inputBuffer-shaped
    handle from a registered resource so it can be passed to encode_picture."""
    _fields_ = [
        ("version", c_uint32),
        ("subResourceIndex", c_uint32),    # deprecated
        ("inputResource", c_void_p),        # deprecated
        ("registeredResource", c_void_p),
        ("mappedResource", c_void_p),       # [out]
        ("mappedBufferFmt", c_uint32),      # [out]
        ("reserved1", c_uint32 * 251),
        ("reserved2", c_void_p * 63),
    ]


def register_cuda_resource(table, encoder, cuda_ptr: int,
                            width: int, height: int, pitch: int,
                            buffer_format: int = NV_ENC_BUFFER_FORMAT_YUV444):
    """Register a CUDA device pointer as an NVENC input resource.

    Returns the registeredResource handle (opaque) which must be passed to
    map_input_resource for each encode and finally unregister_resource at
    teardown.

    For YUV444 the registered region is 3*height stacked rows of `pitch`
    bytes (Y plane, U plane, V plane) — caller is responsible for the
    underlying allocation.
    """
    rr = NV_ENC_REGISTER_RESOURCE()
    rr.version = NV_ENC_REGISTER_RESOURCE_VER
    rr.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR
    rr.width = width
    rr.height = height
    rr.pitch = pitch
    rr.resourceToRegister = cuda_ptr
    rr.bufferFormat = buffer_format
    rr.bufferUsage = NV_ENC_INPUT_IMAGE
    fn = CFUNCTYPE(NVENCSTATUS, c_void_p, POINTER(NV_ENC_REGISTER_RESOURCE))(
        table.nvEncRegisterResource
    )
    s = fn(encoder, byref(rr))
    if s != 0:
        raise RuntimeError(f"nvEncRegisterResource failed: status={s}")
    return c_void_p(rr.registeredResource)


def unregister_resource(table, encoder, registered_resource):
    fn = CFUNCTYPE(NVENCSTATUS, c_void_p, c_void_p)(table.nvEncUnregisterResource)
    s = fn(encoder, registered_resource)
    if s != 0:
        raise RuntimeError(f"nvEncUnregisterResource failed: status={s}")


def map_input_resource(table, encoder, registered_resource):
    """Produce a per-frame mappedResource handle from a registeredResource.
    The returned handle is what gets passed to encode_picture as input_buffer.
    Must be paired with unmap_input_resource after the encode completes."""
    mp = NV_ENC_MAP_INPUT_RESOURCE()
    mp.version = NV_ENC_MAP_INPUT_RESOURCE_VER
    mp.registeredResource = registered_resource.value if hasattr(registered_resource, "value") else registered_resource
    fn = CFUNCTYPE(NVENCSTATUS, c_void_p, POINTER(NV_ENC_MAP_INPUT_RESOURCE))(
        table.nvEncMapInputResource
    )
    s = fn(encoder, byref(mp))
    if s != 0:
        raise RuntimeError(f"nvEncMapInputResource failed: status={s}")
    return c_void_p(mp.mappedResource)


def unmap_input_resource(table, encoder, mapped_resource):
    fn = CFUNCTYPE(NVENCSTATUS, c_void_p, c_void_p)(table.nvEncUnmapInputResource)
    s = fn(encoder, mapped_resource)
    if s != 0:
        raise RuntimeError(f"nvEncUnmapInputResource failed: status={s}")


def set_io_cuda_streams(table, encoder, input_stream_ptr: int, output_stream_ptr: int) -> None:
    """Bind input-fetch + output-bitstream copies to user-provided CUstreams.

    `input_stream_ptr` and `output_stream_ptr` are *addresses of* CUstream
    handles (NV_ENC_CUSTREAM_PTR is documented as `CUstream*`). The encoder
    will queue its input fetch on inputStream and bitstream copy on
    outputStream, allowing the caller to interleave with other GPU work
    on the same or other streams.
    """
    fn = CFUNCTYPE(NVENCSTATUS, c_void_p, c_void_p, c_void_p)(table.nvEncSetIOCudaStreams)
    s = fn(encoder, input_stream_ptr, output_stream_ptr)
    if s != 0:
        raise RuntimeError(f"nvEncSetIOCudaStreams failed: status={s}")


def create_input_buffer(table, encoder, width: int, height: int,
                         buffer_format: int = NV_ENC_BUFFER_FORMAT_YUV444):
    """Allocate an input buffer for a single frame. Returns the buffer handle."""
    params = NV_ENC_CREATE_INPUT_BUFFER()
    params.version = NV_ENC_CREATE_INPUT_BUFFER_VER
    params.width = width
    params.height = height
    params.bufferFmt = buffer_format
    fn = CFUNCTYPE(NVENCSTATUS, c_void_p, POINTER(NV_ENC_CREATE_INPUT_BUFFER))(
        table.nvEncCreateInputBuffer
    )
    s = fn(encoder, byref(params))
    if s != 0:
        raise RuntimeError(f"nvEncCreateInputBuffer failed: status={s}")
    return c_void_p(params.inputBuffer)


def destroy_input_buffer(table, encoder, buffer):
    fn = CFUNCTYPE(NVENCSTATUS, c_void_p, c_void_p)(table.nvEncDestroyInputBuffer)
    s = fn(encoder, buffer)
    if s != 0:
        raise RuntimeError(f"nvEncDestroyInputBuffer failed: status={s}")


def create_bitstream_buffer(table, encoder):
    """Allocate an output bitstream buffer. Returns the buffer handle."""
    params = NV_ENC_CREATE_BITSTREAM_BUFFER()
    params.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER
    fn = CFUNCTYPE(NVENCSTATUS, c_void_p, POINTER(NV_ENC_CREATE_BITSTREAM_BUFFER))(
        table.nvEncCreateBitstreamBuffer
    )
    s = fn(encoder, byref(params))
    if s != 0:
        raise RuntimeError(f"nvEncCreateBitstreamBuffer failed: status={s}")
    return c_void_p(params.bitstreamBuffer)


def destroy_bitstream_buffer(table, encoder, buffer):
    fn = CFUNCTYPE(NVENCSTATUS, c_void_p, c_void_p)(table.nvEncDestroyBitstreamBuffer)
    s = fn(encoder, buffer)
    if s != 0:
        raise RuntimeError(f"nvEncDestroyBitstreamBuffer failed: status={s}")


def write_input_buffer(table, encoder, buffer, frame_yuv444_bytes: bytes,
                        width: int, height: int) -> None:
    """Lock an input buffer, write YUV444 planar bytes (Y|U|V each H*W), unlock."""
    import ctypes
    lock = NV_ENC_LOCK_INPUT_BUFFER()
    lock.version = NV_ENC_LOCK_INPUT_BUFFER_VER
    lock.inputBuffer = buffer
    fn_lock = CFUNCTYPE(NVENCSTATUS, c_void_p, POINTER(NV_ENC_LOCK_INPUT_BUFFER))(
        table.nvEncLockInputBuffer
    )
    s = fn_lock(encoder, byref(lock))
    if s != 0:
        raise RuntimeError(f"nvEncLockInputBuffer failed: status={s}")

    expected = width * height * 3
    fn_unlock = CFUNCTYPE(NVENCSTATUS, c_void_p, c_void_p)(table.nvEncUnlockInputBuffer)
    if len(frame_yuv444_bytes) != expected:
        fn_unlock(encoder, buffer)
        raise ValueError(f"expected {expected} bytes (3*{width}*{height}), got {len(frame_yuv444_bytes)}")

    pitch = lock.pitch
    dst = lock.bufferDataPtr
    if pitch == width:
        # Hot path: planes are tightly packed in the input buffer too — single
        # memmove instead of 3 * height row-at-a-time copies.
        ctypes.memmove(dst, frame_yuv444_bytes, expected)
    else:
        plane_stride_in = width * height
        for plane_idx in range(3):
            src_plane = frame_yuv444_bytes[plane_idx * plane_stride_in : (plane_idx + 1) * plane_stride_in]
            for row in range(height):
                row_offset = (plane_idx * height + row) * pitch
                ctypes.memmove(dst + row_offset, src_plane[row * width : (row + 1) * width], width)

    s = fn_unlock(encoder, buffer)
    if s != 0:
        raise RuntimeError(f"nvEncUnlockInputBuffer failed: status={s}")


# ---- Encode picture / read bitstream -------------------------------------

class NV_ENC_PIC_PARAMS(Structure):
    """Per-picture encode args. Mostly zeros for default behaviour.

    Field order verified against SDK 13's nvEncodeAPI.h.

    Notes:
    - inputTimeStamp/inputDuration are uint64_t (8-byte aligned). Encoding
      them as `c_uint32 * 2` mis-aligns everything after them and the encoder
      rejects the call with NV_ENC_ERR_UNSUPPORTED_PARAM (status 12).
    - SDK 13 added `outputReconBuffer` (void*) after `stateBufferIdx`, and
      a `reserved4` uint32 padding slot between `meHintRefPicDist[2]` and
      `alphaBuffer`. Tail reserved sizes are 284 (uint32) + 57 (void*).
    """
    _fields_ = [
        ("version", c_uint32),
        ("inputWidth", c_uint32),
        ("inputHeight", c_uint32),
        ("inputPitch", c_uint32),
        ("encodePicFlags", c_uint32),     # NV_ENC_PIC_FLAGS bitfield
        ("frameIdx", c_uint32),
        ("inputTimeStamp", c_uint64),
        ("inputDuration", c_uint64),
        ("inputBuffer", c_void_p),
        ("outputBitstream", c_void_p),
        ("completionEvent", c_void_p),
        ("bufferFmt", c_uint32),
        ("pictureStruct", c_uint32),
        ("pictureType", c_uint32),
        # codecPicParams is a UNION (NV_ENC_CODEC_PIC_PARAMS) of H264/HEVC/AV1
        # pic-params structs + uint32_t reserved[256]. The header's reserved[256]
        # member is only 1024 bytes, but the actual union sizeof is the size of
        # the LARGEST member. NV_ENC_PIC_PARAMS_HEVC computes to 1536 bytes
        # (with 8-byte alignment due to its pointer fields). Using a smaller or
        # 4-byte-aligned type here shifts every following field, causing
        # nvEncEncodePicture to fail with NV_ENC_ERR_UNSUPPORTED_PARAM (status 12).
        # c_uint64 * 192 = 1536 bytes, 8-byte aligned.
        ("codecPicParams", c_uint64 * 192),
        ("meHintCountsPerBlock", NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE * 2),
        ("meExternalHints", c_void_p),
        ("reserved2", c_uint32 * 7),
        ("reserved5", c_void_p * 2),
        ("qpDeltaMap", c_void_p),
        ("qpDeltaMapSize", c_uint32),
        ("reservedBitFields", c_uint32),
        ("meHintRefPicDist", c_uint16 * 2),
        ("reserved4", c_uint32),
        ("alphaBuffer", c_void_p),
        ("meExternalSbHints", c_void_p),
        ("meSbHintsCount", c_uint32),
        ("stateBufferIdx", c_uint32),
        ("outputReconBuffer", c_void_p),
        ("reserved3", c_uint32 * 284),
        ("reserved6", c_void_p * 57),
    ]


class NV_ENC_LOCK_BITSTREAM(Structure):
    """Args for nvEncLockBitstream — many output fields."""
    _fields_ = [
        ("version", c_uint32),
        ("flags_bitfield", c_uint32),     # doNotWait/ltrFrame/getRCStats/reserved
        ("outputBitstream", c_void_p),
        ("sliceOffsets", c_void_p),
        ("frameIdx", c_uint32),
        ("hwEncodeStatus", c_uint32),
        ("numSlices", c_uint32),
        ("bitstreamSizeInBytes", c_uint32),
        ("outputTimeStamp", c_uint64),
        ("outputDuration", c_uint64),
        ("bitstreamBufferPtr", c_void_p),   # [out] pointer to encoded bytes
        ("pictureType", c_uint32),
        ("pictureStruct", c_uint32),
        ("frameAvgQP", c_uint32),
        ("frameSatd", c_uint32),
        ("ltrFrameIdx", c_uint32),
        ("ltrFrameBitmap", c_uint32),
        ("temporalId", c_uint32),
        ("intraMBCount", c_uint32),
        ("interMBCount", c_uint32),
        ("averageMVX", c_int32),
        ("averageMVY", c_int32),
        ("alphaLayerSizeInBytes", c_uint32),
        ("outputStatsPtrSize", c_uint32),
        ("reserved", c_uint32),
        ("outputStatsPtr", c_void_p),
        ("frameIdxDisplay", c_uint32),
        ("reserved1", c_uint32 * 219),
        ("reserved2", c_void_p * 63),
        ("reservedInternal", c_uint32 * 8),
    ]


def encode_picture(table, encoder, input_buffer, output_buffer,
                    width: int, height: int, pic_flags: int = 0,
                    buffer_format: int = NV_ENC_BUFFER_FORMAT_YUV444) -> int:
    """Submit one frame. Returns NVENCSTATUS (0 = success, may also be
    NV_ENC_ERR_NEED_MORE_INPUT meaning the encoder is buffering)."""
    pic = NV_ENC_PIC_PARAMS()
    pic.version = NV_ENC_PIC_PARAMS_VER
    pic.inputWidth = width
    pic.inputHeight = height
    pic.inputPitch = width
    pic.inputBuffer = input_buffer
    pic.outputBitstream = output_buffer
    pic.bufferFmt = buffer_format
    pic.pictureStruct = 1                # NV_ENC_PIC_STRUCT_FRAME
    pic.encodePicFlags = pic_flags
    fn = CFUNCTYPE(NVENCSTATUS, c_void_p, POINTER(NV_ENC_PIC_PARAMS))(
        table.nvEncEncodePicture
    )
    return fn(encoder, byref(pic))


def lock_and_read_bitstream(table, encoder, output_buffer) -> bytes:
    """Lock the bitstream buffer, copy out the encoded bytes, unlock."""
    import ctypes
    lock = NV_ENC_LOCK_BITSTREAM()
    lock.version = NV_ENC_LOCK_BITSTREAM_VER
    lock.outputBitstream = output_buffer
    fn = CFUNCTYPE(NVENCSTATUS, c_void_p, POINTER(NV_ENC_LOCK_BITSTREAM))(
        table.nvEncLockBitstream
    )
    s = fn(encoder, byref(lock))
    if s != 0:
        raise RuntimeError(f"nvEncLockBitstream failed: status={s}")
    n = lock.bitstreamSizeInBytes
    data = ctypes.string_at(lock.bitstreamBufferPtr, n)
    fn_unlock = CFUNCTYPE(NVENCSTATUS, c_void_p, c_void_p)(table.nvEncUnlockBitstream)
    s = fn_unlock(encoder, output_buffer)
    if s != 0:
        raise RuntimeError(f"nvEncUnlockBitstream failed: status={s}")
    return data
