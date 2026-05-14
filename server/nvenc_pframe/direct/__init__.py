"""Direct Video Codec SDK backend (work in progress, multi-session project).

## Status as of session 3 (END-TO-END ENCODE WORKING)
- DLL loading via ctypes: DONE
- NvEncodeAPICreateInstance + 39-function table mapping: DONE
- CUDA context integration via cuda.bindings.driver: DONE
- nvEncOpenEncodeSessionEx + nvEncDestroyEncoder lifecycle: DONE
- nvEncGetEncodeGUIDs / nvEncGetEncodePresetGUIDs: DONE
- HEVC + 7 preset GUIDs verified against driver
- NV_ENC_CONFIG / NV_ENC_PRESET_CONFIG / NV_ENC_CONFIG_HEVC: DONE
  (nvEncGetEncodePresetConfigEx returns sane preset defaults)
- nvEncInitializeEncoder for HEVC YUV444 constQP: DONE
  (struct version constants + SDK 13 field layouts verified against
   FFmpeg/nv-codec-headers' nvEncodeAPI.h master)
- nvEncCreateInputBuffer + nvEncCreateBitstreamBuffer: DONE
- nvEncLockInputBuffer + write YUV planes + nvEncUnlockInputBuffer: DONE
- nvEncEncodePicture + nvEncLockBitstream + readout: DONE
  (poc/14_direct_nvenc_first_frame.py emits a real ~2.4 KB HEVC bitstream
   from a synthetic 256x256 YUV444 frame at QP 18)
- nvEncGetLastErrorString helper for diagnostics: DONE

## Pitfalls discovered (worth documenting for the next maintainer)
- NV_ENC_RC_PARAMS in SDK 13 has int8_t cb/crQPIndexOffset (not int32) and
  several trailing fields (lookaheadLevel, viewBitrateRatios, reserved3,
  reserved1) that older docs omit. Wrong layout doesn't fail init — the
  encoder later rejects nvEncEncodePicture with status 12 UNSUPPORTED_PARAM
  with no error string.
- NV_ENC_PIC_PARAMS has 64-bit timestamp fields (8-byte aligned), and
  codecPicParams is a UNION sized to its largest member (HEVC pic params,
  1536 bytes — bigger than the header's nominal `reserved[256]` of 1024).
  Both shift the offsets of every subsequent field.
- NV_ENC_INITIALIZE_PARAMS in SDK 13 added several fields (privDataSize,
  reserved, privData, numStateBuffers, outputStatsLevel) and reordered
  encodeConfig to come BEFORE maxEncodeWidth — different from older SDKs.

## Status as of session 4 (END-TO-END ROUND-TRIP WORKING)
- NVDEC bindings via nvcuvid.dll (decoder.py): DONE
  - cuvidCtxLockCreate / cuvidCtxLockDestroy
  - cuvidCreateVideoParser / cuvidParseVideoData / cuvidDestroyVideoParser
  - cuvidCreateDecoder / cuvidDestroyDecoder
  - cuvidDecodePicture
  - cuvidMapVideoFrame64 / cuvidUnmapVideoFrame64
- CUVIDPARSERPARAMS + 5 CFUNCTYPE callback signatures: DONE
- CUVIDDECODECREATEINFO / CUVIDEOFORMAT / CUVIDPICPARAMS / CUVIDPROCPARAMS
  / CUVIDSOURCEDATAPACKET / CUVIDPARSERDISPINFO structs: DONE
- poc/15_direct_nvdec_round_trip.py: encodes a 256x256 YUV444 frame via
  the direct NVENC path (PoC 14), feeds the HEVC bitstream into NVDEC,
  decodes it back, copies via cuMemcpyDtoH and verifies PSNR (~58 dB at
  QP=18 — visually lossless, max abs diff 7, mean abs diff 0.08).

## Status as of session 5 (DIRECTBACKEND WRAPPER WORKING + BENCHED)
- DirectBackend class (backend.py): matches CodecSession's encode_frames /
  decode_frames signature; persistent NVENC encoder, parser-per-decode-call.
- B-frames disabled in initialize_encoder_hevc_yuv444 (frameIntervalP=1 +
  lookaheadDepth=0 + clear lookahead bit) so per-call encode returns a
  packet immediately instead of buffering.
- Fast-path memmove in write_input_buffer when pitch == width.
- poc/16_direct_backend_bench.py compares DirectBackend vs PyAV CodecSession
  on 16 frames of 256x256 YUV444 at QP=18:
    * encode: 0.50 ms/frame (DirectBackend) vs 0.48 ms/frame (PyAV) — parity
    * decode: 1.58 ms/frame (DirectBackend) vs 5.42 ms/frame (PyAV) — 3.44x
    * init: ~100 ms both sides
- Encode at parity is expected: write_input_buffer still copies through
  host RAM, dominating frame time. Zero-copy is session 6.

## Status as of session 6 (ZERO-COPY ENCODE WORKING)
- NV_ENC_REGISTER_RESOURCE / NV_ENC_MAP_INPUT_RESOURCE structs + helpers
  (register_cuda_resource, map_input_resource, etc.)
- DirectBackend.encode_tensor_frames(tensor) takes a CUDA torch tensor
  [N, 3, H, W] uint8 and encodes in-place via cuMemcpyDtoD into a
  registered staging buffer — no host-side copy on the per-frame hot path.
- poc/16 extended with the zero-copy bench:
    * encode (host buf):    0.51 ms/frame
    * encode (zero-copy):   0.30 ms/frame  — 1.7x over host buf, 1.49x over PyAV
    * decode:               1.55 ms/frame  — 3.42x over PyAV
    * Bitstream bytes identical between host buf and zero-copy paths
      (3841 vs 3841), round-trip diff identical (max=7 mean=0.027) —
      confirms zero-copy doesn't degrade quality.

## Status as of session 7 (PARALLEL-PATH EMPIRICALLY VALIDATED)
- nvEncSetIOCudaStreams binding (set_io_cuda_streams in structs.py).
- DirectBackend(cuda_stream=...) ctor option binds the encoder's input
  fetch + bitstream copy to a user-provided CUDA stream. cuMemcpyDtoDAsync
  on the same stream replaces the sync DtoD when a stream is bound.
- poc/17_parallel_path_demo.py demonstrates that NVENC encode runs
  concurrently with SM compute when bound to a separate stream.
  Result on RTX 5090, 64 frames + 30x4096^2 fp16 GEMM:
    GEMM only:    20.9 ms
    Encode only:  19.9 ms
    Serialized:   40.1 ms (sum, no overlap)
    Parallel:     32.2 ms (1.25x speedup, 43% of theoretical 1.95x ceiling)
  The gap to the ceiling is per-frame lock_and_read_bitstream serialization
  — addressing it requires multiple output bitstream buffers (session 8+).
  The headline claim (NVENC silicon is independent of SM compute) is
  confirmed empirically.

## Status as of session 8 (OUTPUT POOL + ASYNC ENCODE PIPELINE)
- DirectBackend(output_pool_size=K, default 8) allocates a ring of K output
  bitstream buffers AND K registered CUDA staging buffers — each in-flight
  frame owns its slot's input + output, so K-1 frames can be encoding
  concurrently before lock_and_read_bitstream blocks.
- Standalone encode improved 0.30 -> 0.22 ms/frame (zero-copy path) — NVENC
  silicon stays busy instead of idling per-frame between locks.
- poc/17 parallel-path demo improved:
    * Encode-only 14.3 ms (was 19.9 ms with single buffer)
    * Parallel:   26.0 ms (was 32.2 ms)
    * Realized overlap: 67% of theoretical max (was 43%)
- Full bench (poc/16): zero-copy at 0.22 ms/frame is now 2.0x faster than
  PyAV CodecSession (was 1.49x in session 6).

## Status as of session 9 (REAL-WORKLOAD BENCH)
- poc/18_real_activation_bench.py runs the full pipeline (load FLUX
  capture → PCA → quantize → pack → encode → decode → unpack → unproject)
  on N=4 holdout activations from ring0/data/ with K=500 LOO-PCA basis
  built from 16 calibration captures. 668 frames @ 256x256 YUV444 QP=18:
    * pyav-single:    0.489 ms/f enc, 0.917 ms/f dec, 4.39 MB, cos 0.9731
    * pyav-multi*:    0.443 ms/f enc, 0.949 ms/f dec, 4.39 MB, cos 0.9731
    * direct(pool=8): 0.237 ms/f enc, 0.499 ms/f dec, 4.37 MB, cos 0.9881
  Headline: 2.07x faster encode, 1.84x faster decode, 1.91x end-to-end,
  AT EQUAL OR BETTER reconstruction quality and equal or smaller bitstream.
  (*pyav-multi falls back to a single engine in this bench because the test
  serializes by activation; the multi-engine bench in MultiEngineCodecSession
  parallelizes across activations, which we don't do here yet.)

## Open question
- Why does DirectBackend produce HIGHER quality (0.9881 vs 0.9731 mean cos)
  than PyAV at LOWER bitstream size? Same hevc_nvenc, same yuv444p, same
  QP=18. Suspect SEI / VUI / frame-rate metadata or PyAV-side lookahead
  override differs from our struct-level config. Worth diffing SPS/PPS bytes.

## Status as of session 10 (MULTI-ENGINE DIRECTBACKEND)
- MultiEngineDirectBackend (multi_backend.py) holds N independent
  DirectBackend sessions and dispatches encode/decode round-robin across
  Python threads. Each backend keeps its own 8-deep output bitstream pool
  + zero-copy CUDA staging buffers, so total in-flight depth is N*8
  (default 24 on the 5090).
- Pitfall: Python threads have no CUDA context attached by default —
  cuMemcpyDtoDAsync fails with CUDA_ERROR_INVALID_CONTEXT (201). Workers
  call cuCtxSetCurrent(backend._cuda_ctx) before any CUDA work.
- poc/18 extended to bench against MultiEngineDirectBackend (3 engines):
    * pyav-single:           0.469 ms/f enc, 0.887 ms/f dec, cos 0.9731
    * direct(1eng,pool=8):   0.243 ms/f enc, 0.493 ms/f dec, cos 0.9881
    * direct-multi(3engx8):  0.179 ms/f enc, 0.301 ms/f dec, cos 0.9881
  End-to-end speedup vs pyav-single: 2.83x. Multi-engine adds 1.35x enc /
  1.64x dec over single-engine direct (sub-3x because N=4 activations
  splits unevenly across 3 engines + per-frame Python ctypes overhead).

## Side-quest resolved (poc/19)
- Diagnosed the DirectBackend-vs-PyAV cos-sim / bitstream-size gap:
  PyAV's `f.pict_type = PictureType.I` does NOT propagate to NVENC's
  NV_ENC_PIC_FLAG_FORCEIDR through ffmpeg-NVENC, even with
  forced_idr=1 set in the options dict. PyAV emits TRAIL_R (P-frame
  referencing the warmup zero-frame) where DirectBackend emits IDR_W_RADL
  (proper keyframe). ffprobe surfaces this as "Could not find ref with
  POC 0" warnings on PyAV bitstreams. PyAV ends up encoding
  "delta-from-zeros" — bigger bitstream, worse quality on activation data.
  Not a DirectBackend bug — DirectBackend is doing the right thing for
  independent-tensor compression. Could be raised upstream in PyAV.

## Sessions 11+ (ROADMAP)
- Move the per-frame ctypes loop to a C helper to close the remaining
  Python call overhead (the only meaningful residual serializer).
- Cross-GPU PCIe peer-to-peer integration: the encoder zero-copy +
  parallel-path are validated, but the multi-GPU NVLink-class bandwidth
  claim still needs the actual cross-GPU transfer wired through (blocked
  on getting a second GPU into the validation rig).

## Linux portability note
- cuviddec.h uses `unsigned long` for several fields (LLP64 = 4 bytes on
  Windows, LP64 = 8 bytes on Linux). decoder.py uses c_uint32 — fine on
  Windows; Linux port requires per-field review (CUVIDDECODECREATEINFO,
  CUVIDSOURCEDATAPACKET).
"""

from .backend import DirectBackend  # noqa: F401

from .api import (
    NV_ENCODE_API_FUNCTION_LIST,
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS,
    NVENCAPI_VERSION,
    NV_ENCODE_API_FUNCTION_LIST_VER,
    NV_ENC_DEVICE_TYPE_CUDA,
    GUID,
    NV_ENC_CODEC_HEVC_GUID,
    NV_ENC_CODEC_H264_GUID,
    NV_ENC_CODEC_AV1_GUID,
    NV_ENC_PRESET_P1_GUID,
    NV_ENC_PRESET_P4_GUID,
    NV_ENC_PRESET_P7_GUID,
    NV_ENC_TUNING_INFO_HIGH_QUALITY,
    create_instance,
    open_encode_session_cuda,
    destroy_encoder,
    get_last_error_string,
    nvenc_get_encode_guid_count,
    get_encode_guids,
    get_preset_guids,
    load_nvenc,
    load_nvcuvid,
)

__all__ = [
    "DirectBackend",
    "NV_ENCODE_API_FUNCTION_LIST",
    "NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS",
    "NVENCAPI_VERSION",
    "NV_ENCODE_API_FUNCTION_LIST_VER",
    "NV_ENC_DEVICE_TYPE_CUDA",
    "GUID",
    "NV_ENC_CODEC_HEVC_GUID",
    "NV_ENC_CODEC_H264_GUID",
    "NV_ENC_CODEC_AV1_GUID",
    "NV_ENC_PRESET_P1_GUID",
    "NV_ENC_PRESET_P4_GUID",
    "NV_ENC_PRESET_P7_GUID",
    "NV_ENC_TUNING_INFO_HIGH_QUALITY",
    "create_instance",
    "open_encode_session_cuda",
    "destroy_encoder",
    "get_last_error_string",
    "nvenc_get_encode_guid_count",
    "get_encode_guids",
    "get_preset_guids",
    "load_nvenc",
    "load_nvcuvid",
]
