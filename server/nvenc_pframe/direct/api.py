"""ctypes bindings for the NVIDIA Video Codec SDK NVENC C API.

Maps to nvEncodeAPI.h from the NVIDIA Video Codec SDK. The struct layouts
and ordering here are derived from the publicly-published header (the SDK
itself is freely downloadable from developer.nvidia.com but we don't ship it
— at runtime we only need the DLL/SO that ships with the NVIDIA driver).

Tested against driver shipping NVENCAPI version 13.0 (Blackwell-era).

This module is the FOUNDATION for the direct backend. It provides the
ctypes types, the function pointer table struct, and entry-point loaders.
Higher-level encoder lifecycle code lives in encoder.py (TBD).
"""

from __future__ import annotations

import ctypes
import platform
import uuid
from ctypes import (
    Structure, POINTER, byref, c_uint, c_int, c_uint32, c_int32, c_void_p,
    c_char_p, c_uint64, c_uint16, c_uint8, c_ubyte, CFUNCTYPE,
)


# ---- platform DLL loading -------------------------------------------------

def load_nvenc() -> ctypes.CDLL:
    """Load the driver-shipped NVENC shared library."""
    sys_name = platform.system()
    if sys_name == "Windows":
        return ctypes.CDLL("nvEncodeAPI64.dll")
    elif sys_name == "Linux":
        try:
            return ctypes.CDLL("libnvidia-encode.so.1")
        except OSError:
            return ctypes.CDLL("libnvidia-encode.so")
    raise RuntimeError(f"NVENC not supported on {sys_name}")


def load_nvcuvid() -> ctypes.CDLL:
    """Load the driver-shipped NVCUVID (NVDEC) shared library."""
    sys_name = platform.system()
    if sys_name == "Windows":
        return ctypes.CDLL("nvcuvid.dll")
    elif sys_name == "Linux":
        try:
            return ctypes.CDLL("libnvcuvid.so.1")
        except OSError:
            return ctypes.CDLL("libnvcuvid.so")
    raise RuntimeError(f"NVCUVID not supported on {sys_name}")


# ---- API version macros (from nvEncodeAPI.h) ------------------------------

NVENCAPI_MAJOR_VERSION = 13
NVENCAPI_MINOR_VERSION = 0
NVENCAPI_VERSION = NVENCAPI_MAJOR_VERSION | (NVENCAPI_MINOR_VERSION << 24)


def NVENCAPI_STRUCT_VERSION(struct_ver: int) -> int:
    """Construct a NVENC struct version field value."""
    return NVENCAPI_VERSION | (struct_ver << 16) | (0x7 << 28)


# struct version constants
NV_ENCODE_API_FUNCTION_LIST_VER = NVENCAPI_STRUCT_VERSION(2)
NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER = NVENCAPI_STRUCT_VERSION(1)
NV_ENC_INITIALIZE_PARAMS_VER = NVENCAPI_STRUCT_VERSION(5)
NV_ENC_CONFIG_VER = NVENCAPI_STRUCT_VERSION(8)
NV_ENC_PIC_PARAMS_VER = NVENCAPI_STRUCT_VERSION(6)
NV_ENC_LOCK_BITSTREAM_VER = NVENCAPI_STRUCT_VERSION(2)


# ---- function pointer signatures ------------------------------------------
# Each NVENC API function returns NVENCSTATUS (an int) and takes specific args.
# We declare the types here so the function table struct can declare its fields
# with proper signatures.

NVENCSTATUS = c_int32
NV_ENC_OUTPUT_PTR = c_void_p

# Forward declarations (we'll fill in real signatures as we implement each call)
ENCODE_SESSION_HANDLE = c_void_p
INPUT_BUFFER_HANDLE = c_void_p
OUTPUT_BUFFER_HANDLE = c_void_p


# Generic "void* function" type for pointers we haven't typed yet.
# Replace with proper CFUNCTYPE signatures as we implement each.
GenericFn = CFUNCTYPE(NVENCSTATUS, c_void_p)


# ---- NV_ENCODE_API_FUNCTION_LIST ------------------------------------------
# The order MUST match nvEncodeAPI.h exactly. As of NVENCAPI 13.0:

class NV_ENCODE_API_FUNCTION_LIST(Structure):
    """The function pointer table NVENC populates after CreateInstance.

    Order of fields below MUST match nvEncodeAPI.h:NV_ENCODE_API_FUNCTION_LIST.
    """
    _fields_ = [
        ("version", c_uint32),
        ("reserved", c_uint32),
        # Encoder session lifecycle
        ("nvEncOpenEncodeSession", c_void_p),
        # Capability queries
        ("nvEncGetEncodeGUIDCount", c_void_p),
        ("nvEncGetEncodeProfileGUIDCount", c_void_p),
        ("nvEncGetEncodeProfileGUIDs", c_void_p),
        ("nvEncGetEncodeGUIDs", c_void_p),
        ("nvEncGetInputFormatCount", c_void_p),
        ("nvEncGetInputFormats", c_void_p),
        ("nvEncGetEncodeCaps", c_void_p),
        ("nvEncGetEncodePresetCount", c_void_p),
        ("nvEncGetEncodePresetGUIDs", c_void_p),
        ("nvEncGetEncodePresetConfig", c_void_p),
        ("nvEncInitializeEncoder", c_void_p),
        # Buffer management
        ("nvEncCreateInputBuffer", c_void_p),
        ("nvEncDestroyInputBuffer", c_void_p),
        ("nvEncCreateBitstreamBuffer", c_void_p),
        ("nvEncDestroyBitstreamBuffer", c_void_p),
        # Encoding
        ("nvEncEncodePicture", c_void_p),
        ("nvEncLockBitstream", c_void_p),
        ("nvEncUnlockBitstream", c_void_p),
        ("nvEncLockInputBuffer", c_void_p),
        ("nvEncUnlockInputBuffer", c_void_p),
        ("nvEncGetEncodeStats", c_void_p),
        ("nvEncGetSequenceParams", c_void_p),
        # Async support
        ("nvEncRegisterAsyncEvent", c_void_p),
        ("nvEncUnregisterAsyncEvent", c_void_p),
        # Resource registration (CUDA / D3D / etc. interop)
        ("nvEncMapInputResource", c_void_p),
        ("nvEncUnmapInputResource", c_void_p),
        ("nvEncDestroyEncoder", c_void_p),
        ("nvEncInvalidateRefFrames", c_void_p),
        # Session-Ex (modern session opener)
        ("nvEncOpenEncodeSessionEx", c_void_p),
        ("nvEncRegisterResource", c_void_p),
        ("nvEncUnregisterResource", c_void_p),
        ("nvEncReconfigureEncoder", c_void_p),
        ("reserved1", c_void_p),
        # Motion estimation only
        ("nvEncCreateMVBuffer", c_void_p),
        ("nvEncDestroyMVBuffer", c_void_p),
        ("nvEncRunMotionEstimationOnly", c_void_p),
        ("nvEncGetLastErrorString", c_void_p),
        ("nvEncSetIOCudaStreams", c_void_p),
        # Get last error
        ("nvEncGetEncodePresetConfigEx", c_void_p),
        ("nvEncGetSequenceParamEx", c_void_p),
        # Reserved tail (NVIDIA reserves space for future API additions)
        ("reserved2", c_void_p * 64),
    ]


# ---- entry points ---------------------------------------------------------

def create_instance(nvenc: ctypes.CDLL = None) -> NV_ENCODE_API_FUNCTION_LIST:
    """Call NvEncodeAPICreateInstance and return the populated function table.

    Raises RuntimeError on failure (e.g., API version mismatch).
    """
    if nvenc is None:
        nvenc = load_nvenc()
    NvEncodeAPICreateInstance = nvenc.NvEncodeAPICreateInstance
    NvEncodeAPICreateInstance.argtypes = [POINTER(NV_ENCODE_API_FUNCTION_LIST)]
    NvEncodeAPICreateInstance.restype = NVENCSTATUS

    table = NV_ENCODE_API_FUNCTION_LIST()
    table.version = NV_ENCODE_API_FUNCTION_LIST_VER
    status = NvEncodeAPICreateInstance(byref(table))
    if status != 0:
        raise RuntimeError(
            f"NvEncodeAPICreateInstance failed: NVENCSTATUS={status}. "
            f"Check that NVENCAPI_VERSION ({NVENCAPI_VERSION:#010x}) matches "
            f"the installed driver's supported version."
        )
    return table


# ---- specific function callers (filled in as we implement them) -----------

def nvenc_get_encode_guid_count(table: NV_ENCODE_API_FUNCTION_LIST,
                                  encoder_handle: c_void_p) -> int:
    """Call nvEncGetEncodeGUIDCount. Demonstrates the pattern of calling a
    function pointer from the table.

    Signature: NVENCSTATUS(void* encoder, uint32_t* encodeGUIDCount)
    """
    fn_ptr = table.nvEncGetEncodeGUIDCount
    if not fn_ptr:
        raise RuntimeError("nvEncGetEncodeGUIDCount not in API table")
    fn_type = CFUNCTYPE(NVENCSTATUS, c_void_p, POINTER(c_uint32))
    fn = fn_type(fn_ptr)
    count = c_uint32(0)
    status = fn(encoder_handle, byref(count))
    if status != 0:
        raise RuntimeError(f"nvEncGetEncodeGUIDCount failed: NVENCSTATUS={status}")
    return count.value


# ---- Open/close encoder session -------------------------------------------

# NV_ENC_DEVICE_TYPE enum values
NV_ENC_DEVICE_TYPE_DIRECTX = 0
NV_ENC_DEVICE_TYPE_CUDA = 1
NV_ENC_DEVICE_TYPE_OPENGL = 2


class NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS(Structure):
    """Args struct for nvEncOpenEncodeSessionEx."""
    _fields_ = [
        ("version", c_uint32),
        ("deviceType", c_uint32),       # NV_ENC_DEVICE_TYPE
        ("device", c_void_p),            # CUcontext for CUDA mode
        ("reserved", c_void_p),
        ("apiVersion", c_uint32),
        ("reserved1", c_uint32 * 253),
        ("reserved2", c_void_p * 64),
    ]


def open_encode_session_cuda(table: NV_ENCODE_API_FUNCTION_LIST,
                              cuda_context: int) -> c_void_p:
    """Open a NVENC encoder session bound to a CUDA context.

    Args:
        table: function pointer table from create_instance()
        cuda_context: integer CUcontext handle (e.g., from
                      cuda.bindings.driver.cuCtxGetCurrent())

    Returns:
        ctypes c_void_p encoder handle. Must be passed to destroy_encoder()
        at the end of the session's life.
    """
    params = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS()
    params.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER
    params.deviceType = NV_ENC_DEVICE_TYPE_CUDA
    params.device = cuda_context
    params.apiVersion = NVENCAPI_VERSION

    fn_type = CFUNCTYPE(NVENCSTATUS,
                        POINTER(NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS),
                        POINTER(c_void_p))
    fn = fn_type(table.nvEncOpenEncodeSessionEx)
    encoder_handle = c_void_p()
    status = fn(byref(params), byref(encoder_handle))
    if status != 0:
        raise RuntimeError(f"nvEncOpenEncodeSessionEx failed: NVENCSTATUS={status}")
    return encoder_handle


def destroy_encoder(table: NV_ENCODE_API_FUNCTION_LIST,
                     encoder_handle: c_void_p) -> None:
    """Destroy a NVENC encoder session opened by open_encode_session_cuda."""
    fn_type = CFUNCTYPE(NVENCSTATUS, c_void_p)
    fn = fn_type(table.nvEncDestroyEncoder)
    status = fn(encoder_handle)
    if status != 0:
        raise RuntimeError(f"nvEncDestroyEncoder failed: NVENCSTATUS={status}")


def get_last_error_string(table: NV_ENCODE_API_FUNCTION_LIST,
                           encoder_handle: c_void_p) -> str:
    """Call nvEncGetLastErrorString — returns the driver's human-readable
    explanation of the most recent error from this encoder session.

    Signature: const char* NVENCAPI nvEncGetLastErrorString(void* encoder)
    """
    fn_type = CFUNCTYPE(c_char_p, c_void_p)
    fn = fn_type(table.nvEncGetLastErrorString)
    raw = fn(encoder_handle)
    if not raw:
        return "(no error string)"
    return raw.decode("utf-8", errors="replace")


# ---- GUID type (NVENC uses Microsoft-style 128-bit GUIDs) -----------------

class GUID(Structure):
    """Mirrors Windows GUID / nvEncodeAPI.h:GUID struct layout exactly."""
    _fields_ = [
        ("Data1", c_uint32),
        ("Data2", c_uint16),
        ("Data3", c_uint16),
        ("Data4", c_ubyte * 8),
    ]

    def __eq__(self, other):
        if not isinstance(other, GUID):
            return False
        return (self.Data1 == other.Data1 and self.Data2 == other.Data2
                and self.Data3 == other.Data3
                and bytes(self.Data4) == bytes(other.Data4))

    def __hash__(self):
        return hash((self.Data1, self.Data2, self.Data3, bytes(self.Data4)))

    def to_uuid(self) -> uuid.UUID:
        """Convert to a Python uuid.UUID for printing / comparison."""
        b = bytes(
            [
                (self.Data1 >> 24) & 0xFF, (self.Data1 >> 16) & 0xFF,
                (self.Data1 >> 8) & 0xFF, self.Data1 & 0xFF,
                (self.Data2 >> 8) & 0xFF, self.Data2 & 0xFF,
                (self.Data3 >> 8) & 0xFF, self.Data3 & 0xFF,
                *bytes(self.Data4),
            ]
        )
        return uuid.UUID(bytes=b)

    def __repr__(self):
        return f"GUID({self.to_uuid()})"

    @classmethod
    def from_uuid(cls, u: uuid.UUID) -> "GUID":
        b = u.bytes
        g = cls()
        g.Data1 = int.from_bytes(b[0:4], "big")
        g.Data2 = int.from_bytes(b[4:6], "big")
        g.Data3 = int.from_bytes(b[6:8], "big")
        for i in range(8):
            g.Data4[i] = b[8 + i]
        return g


# Codec GUIDs from nvEncodeAPI.h (constants — these never change across SDK versions)
NV_ENC_CODEC_H264_GUID = GUID.from_uuid(uuid.UUID("6bc82762-4e63-4ca4-aa85-1e50f321f6bf"))
NV_ENC_CODEC_HEVC_GUID = GUID.from_uuid(uuid.UUID("790cdc88-4522-4d7b-9425-bda9975f7603"))
NV_ENC_CODEC_AV1_GUID  = GUID.from_uuid(uuid.UUID("0a352289-0aa7-4759-862d-5d15cd16d254"))

# Preset GUIDs (P1 fastest .. P7 slowest/best quality)
NV_ENC_PRESET_P1_GUID = GUID.from_uuid(uuid.UUID("fc0a8d3e-45f8-4cf8-80c7-298871590ebf"))
NV_ENC_PRESET_P2_GUID = GUID.from_uuid(uuid.UUID("f581cfb8-88d6-4381-93f0-df13f9c27dab"))
NV_ENC_PRESET_P3_GUID = GUID.from_uuid(uuid.UUID("36850110-3a07-441f-94d5-3670631f91f6"))
NV_ENC_PRESET_P4_GUID = GUID.from_uuid(uuid.UUID("90a7b826-df06-4862-b9d2-cd6d73a08681"))
NV_ENC_PRESET_P5_GUID = GUID.from_uuid(uuid.UUID("21c6e6b4-297a-4cba-998f-b6cbde72ade3"))
NV_ENC_PRESET_P6_GUID = GUID.from_uuid(uuid.UUID("8e75c279-6299-4ab6-8302-0b215a335cf5"))
NV_ENC_PRESET_P7_GUID = GUID.from_uuid(uuid.UUID("84848c12-6f71-4c13-931b-53e283f57974"))

# Tuning info (NV_ENC_TUNING_INFO enum)
NV_ENC_TUNING_INFO_HIGH_QUALITY = 1
NV_ENC_TUNING_INFO_LOW_LATENCY = 2
NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY = 3
NV_ENC_TUNING_INFO_LOSSLESS = 4

# Rate control mode (NV_ENC_PARAMS_RC_MODE bitfield)
NV_ENC_PARAMS_RC_CONSTQP = 0x00000000
NV_ENC_PARAMS_RC_VBR = 0x00000001
NV_ENC_PARAMS_RC_CBR = 0x00000002


# ---- enumerate codec / preset GUIDs ---------------------------------------

def get_encode_guids(table: NV_ENCODE_API_FUNCTION_LIST,
                      encoder: c_void_p) -> list[GUID]:
    """Return the list of codec GUIDs the encoder supports.

    Two-step query: first call to get the count, then call to fill an array.
    """
    count_fn = CFUNCTYPE(NVENCSTATUS, c_void_p, POINTER(c_uint32))(
        table.nvEncGetEncodeGUIDCount
    )
    count = c_uint32()
    s = count_fn(encoder, byref(count))
    if s != 0:
        raise RuntimeError(f"nvEncGetEncodeGUIDCount failed: status={s}")

    arr_type = GUID * count.value
    arr = arr_type()
    actual = c_uint32()
    list_fn = CFUNCTYPE(NVENCSTATUS, c_void_p, POINTER(GUID), c_uint32, POINTER(c_uint32))(
        table.nvEncGetEncodeGUIDs
    )
    s = list_fn(encoder, arr, count.value, byref(actual))
    if s != 0:
        raise RuntimeError(f"nvEncGetEncodeGUIDs failed: status={s}")
    return [arr[i] for i in range(actual.value)]


def get_preset_guids(table: NV_ENCODE_API_FUNCTION_LIST,
                      encoder: c_void_p,
                      codec_guid: GUID) -> list[GUID]:
    """Return the list of preset GUIDs supported for a given codec."""
    count_fn = CFUNCTYPE(NVENCSTATUS, c_void_p, GUID, POINTER(c_uint32))(
        table.nvEncGetEncodePresetCount
    )
    count = c_uint32()
    s = count_fn(encoder, codec_guid, byref(count))
    if s != 0:
        raise RuntimeError(f"nvEncGetEncodePresetCount failed: status={s}")

    arr_type = GUID * count.value
    arr = arr_type()
    actual = c_uint32()
    list_fn = CFUNCTYPE(NVENCSTATUS, c_void_p, GUID, POINTER(GUID), c_uint32, POINTER(c_uint32))(
        table.nvEncGetEncodePresetGUIDs
    )
    s = list_fn(encoder, codec_guid, arr, count.value, byref(actual))
    if s != 0:
        raise RuntimeError(f"nvEncGetEncodePresetGUIDs failed: status={s}")
    return [arr[i] for i in range(actual.value)]
