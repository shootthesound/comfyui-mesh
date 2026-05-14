/* Direct NVENC encode hot loop in C — eliminates per-frame Python overhead.
 *
 * The Python side does setup (open encoder, init, register CUDA buffers,
 * pre-allocate the NV_ENC_PIC_PARAMS / NV_ENC_LOCK_BITSTREAM /
 * NV_ENC_MAP_INPUT_RESOURCE structs). It then hands all the function
 * pointers + handles + struct addresses to encode_batch(), which runs
 * the entire per-frame loop without going through Python between frames.
 *
 * The C side does NOT include any NVENC / CUDA headers. We typedef the
 * minimum function-pointer signatures we need and treat the structs
 * opaquely — Python pre-allocated them with the correct layout, and we
 * just stamp the changing fields at the offsets Python tells us.
 *
 * Output bitstream payloads are written into a caller-provided
 * pre-allocated u8 buffer + size array, in original frame order.
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#define CALLCONV __stdcall
#else
#define EXPORT __attribute__((visibility("default")))
#define CALLCONV
#endif

/* NVENC API uses NVENCAPI calling convention which on Windows is __stdcall.
 * We typedef it accordingly. */
typedef int (CALLCONV *FN_MAP)(void* enc, void* params);
typedef int (CALLCONV *FN_UNMAP)(void* enc, void* mapped);
typedef int (CALLCONV *FN_ENCODE)(void* enc, void* params);
typedef int (CALLCONV *FN_LOCK)(void* enc, void* params);
typedef int (CALLCONV *FN_UNLOCK)(void* enc, void* output);

/* CUDA driver API uses CUDAAPI which is __stdcall on Windows. */
typedef int (CALLCONV *FN_MEMCPY_DTOD)(uint64_t dst, uint64_t src, size_t bytes);
typedef int (CALLCONV *FN_MEMCPY_DTOD_ASYNC)(uint64_t dst, uint64_t src,
                                              size_t bytes, void* stream);

/* Field offsets (in bytes) that Python tells us so we can stamp the
 * changing fields without knowing the full struct layout in C.
 *
 * For NV_ENC_MAP_INPUT_RESOURCE we read mappedResource (output field).
 * For NV_ENC_PIC_PARAMS we write inputBuffer + outputBitstream + encodePicFlags.
 * For NV_ENC_LOCK_BITSTREAM we write outputBitstream + read bitstreamSizeInBytes
 *   and bitstreamBufferPtr.
 */

typedef struct {
    /* Function pointers */
    FN_MAP                fn_map;
    FN_UNMAP              fn_unmap;
    FN_ENCODE             fn_encode;
    FN_LOCK               fn_lock;
    FN_UNLOCK             fn_unlock;
    FN_MEMCPY_DTOD        fn_memcpy_dtod;
    FN_MEMCPY_DTOD_ASYNC  fn_memcpy_dtod_async;

    /* Encoder handle */
    void* encoder;

    /* CUDA stream (or NULL for sync memcpy) */
    void* cuda_stream;

    /* Pool config */
    int   pool_size;
    int   per_frame_bytes;

    /* Per-slot CUDA staging buffers (pool_size pointers) */
    uint64_t* slot_dst_ptrs;       /* [pool_size] CUdeviceptr */

    /* Per-slot output bitstream handles (pool_size pointers) */
    void**    out_buffers;          /* [pool_size] NV_ENC_OUTPUT_PTR */

    /* Per-slot pre-allocated NV_ENC_MAP_INPUT_RESOURCE structs */
    void**    map_struct_ptrs;      /* [pool_size] address of struct */
    /* Offset of mappedResource field in NV_ENC_MAP_INPUT_RESOURCE */
    int       map_mapped_resource_offset;

    /* Pre-allocated NV_ENC_PIC_PARAMS struct */
    void*     pic_struct_ptr;
    int       pic_inputBuffer_offset;
    int       pic_outputBitstream_offset;
    int       pic_encodePicFlags_offset;

    /* Pre-allocated NV_ENC_LOCK_BITSTREAM struct */
    void*     lock_struct_ptr;
    int       lock_outputBitstream_offset;
    int       lock_bitstreamSizeInBytes_offset;
    int       lock_bitstreamBufferPtr_offset;

    /* IDR flag value (combined FORCEIDR | OUTPUT_SPSPPS) */
    uint32_t  flags_idr;
} EncodeContext;


/* Encode one batch.
 *
 * Inputs:
 *   ctx              — pre-set EncodeContext
 *   n_frames         — number of frames
 *   src_ptrs         — n_frames CUdeviceptrs into the input tensor
 *
 * Outputs:
 *   packet_data_ptrs — n_frames pointers (filled with internal NVENC
 *                      bitstream pointers; valid until lock/unlock cycle
 *                      ends — caller must copy out before next call)
 *
 *   Wait — that doesn't work because we unlock the bitstream at the end
 *   of the loop. Let me change the design: caller pre-allocates a
 *   single big destination buffer, we copy each frame's bitstream into
 *   it sequentially and report offsets/sizes.
 *
 *   packet_dest      — pre-allocated buffer (caller-sized)
 *   packet_dest_cap  — capacity of packet_dest in bytes
 *   packet_offsets   — [n_frames] uint32_t (filled with offset in packet_dest)
 *   packet_sizes     — [n_frames] uint32_t (filled with size of each frame's payload)
 *
 * Returns 0 on success, negative NVENC error status on failure (or -1000+
 * for our own errors), or +1 if packet_dest was too small.
 */
EXPORT int encode_batch(
    EncodeContext* ctx,
    int n_frames,
    const uint64_t* src_ptrs,
    uint8_t* packet_dest,
    size_t packet_dest_cap,
    uint32_t* packet_offsets,
    uint32_t* packet_sizes
) {
    int K = ctx->pool_size;
    int per_frame_bytes = ctx->per_frame_bytes;

    /* Per-slot in-flight tracking: in_flight[slot] = frame_index, -1 if empty */
    int in_flight_static[64];
    int* in_flight = in_flight_static;
    if (K > 64) {
        return -1001; /* pool too big for stack; caller should pass smaller */
    }
    for (int i = 0; i < K; i++) in_flight[i] = -1;

    size_t dest_off = 0;

    /* Inline drain-slot helper (manual inlining since it's called from two places) */
    #define DRAIN_SLOT(slot) do {                                                 \
        int frame_idx = in_flight[(slot)];                                         \
        if (frame_idx >= 0) {                                                       \
            void* lock_struct = ctx->lock_struct_ptr;                              \
            *(void**)((char*)lock_struct + ctx->lock_outputBitstream_offset)      \
                = ctx->out_buffers[(slot)];                                         \
            int s = ctx->fn_lock(ctx->encoder, lock_struct);                      \
            if (s != 0) return s;                                                  \
            uint32_t sz = *(uint32_t*)((char*)lock_struct                          \
                + ctx->lock_bitstreamSizeInBytes_offset);                          \
            void* src = *(void**)((char*)lock_struct                               \
                + ctx->lock_bitstreamBufferPtr_offset);                            \
            if (dest_off + sz > packet_dest_cap) {                                 \
                ctx->fn_unlock(ctx->encoder, ctx->out_buffers[(slot)]);            \
                return 1; /* dest buffer overflow */                               \
            }                                                                       \
            memcpy(packet_dest + dest_off, src, sz);                               \
            packet_offsets[frame_idx] = (uint32_t)dest_off;                        \
            packet_sizes[frame_idx] = sz;                                           \
            dest_off += sz;                                                         \
            s = ctx->fn_unlock(ctx->encoder, ctx->out_buffers[(slot)]);            \
            if (s != 0) return s;                                                  \
            void* mapped = *(void**)((char*)ctx->map_struct_ptrs[(slot)]          \
                + ctx->map_mapped_resource_offset);                                 \
            s = ctx->fn_unmap(ctx->encoder, mapped);                               \
            if (s != 0) return s;                                                  \
            in_flight[(slot)] = -1;                                                 \
        }                                                                           \
    } while (0)

    for (int i = 0; i < n_frames; i++) {
        int slot = i % K;

        /* Drain slot if it still holds an in-flight frame */
        DRAIN_SLOT(slot);

        /* GPU memcpy: src_ptrs[i] -> slot_dst_ptrs[slot] */
        int err;
        if (ctx->cuda_stream != NULL) {
            err = ctx->fn_memcpy_dtod_async(
                ctx->slot_dst_ptrs[slot], src_ptrs[i],
                (size_t)per_frame_bytes, ctx->cuda_stream);
        } else {
            err = ctx->fn_memcpy_dtod(
                ctx->slot_dst_ptrs[slot], src_ptrs[i],
                (size_t)per_frame_bytes);
        }
        if (err != 0) return -2000 - err;

        /* Map input resource for this slot */
        int s = ctx->fn_map(ctx->encoder, ctx->map_struct_ptrs[slot]);
        if (s != 0) return s;

        /* Stamp pic_params: inputBuffer = mapped, outputBitstream = out, flags */
        void* mapped = *(void**)((char*)ctx->map_struct_ptrs[slot]
            + ctx->map_mapped_resource_offset);
        char* pic = (char*)ctx->pic_struct_ptr;
        *(void**)(pic + ctx->pic_inputBuffer_offset) = mapped;
        *(void**)(pic + ctx->pic_outputBitstream_offset) = ctx->out_buffers[slot];
        *(uint32_t*)(pic + ctx->pic_encodePicFlags_offset) =
            (i == 0) ? ctx->flags_idr : 0;

        s = ctx->fn_encode(ctx->encoder, ctx->pic_struct_ptr);
        if (s != 0 && s != 14) {
            /* unmap on error */
            ctx->fn_unmap(ctx->encoder, mapped);
            return s;
        }
        if (s == 0) {
            in_flight[slot] = i;
        } else {
            /* NEED_MORE_INPUT — unmap and continue */
            ctx->fn_unmap(ctx->encoder, mapped);
        }
    }

    /* Drain any remaining in-flight frames */
    for (int slot = 0; slot < K; slot++) {
        DRAIN_SLOT(slot);
    }

    #undef DRAIN_SLOT
    return 0;
}
