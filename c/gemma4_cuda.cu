#include "gemma4_cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>

/*
 * Gemma 4 CUDA tier.
 *
 *  - Dense matrices (attention projections, dense MLP, LM head) are uploaded once
 *    and stay in VRAM; a matvec ships only the activation vector.
 *  - Routed experts live in a VRAM slab managed as an LRU of fixed-size record
 *    slots. A layer is one input upload, up to top-k expert launches, one
 *    output download. Resident experts need no host data at all.
 *
 * Every kernel is a warp-per-row dot product over the GGML block layouts the CPU
 * reference decodes (Q4_0: 18-byte blocks of 32; Q6_K: 210-byte blocks of 256).
 * Accumulation order differs from the CPU, so results agree to float rounding,
 * not bit for bit.
 */

struct gemma4_cuda_context {
    int device;
    /* single-expert path (coli_gemma4_cuda_run) */
    uint8_t *payload;
    float *input, *gate, *up, *hidden, *output;
    size_t payload_capacity;
    uint32_t model_capacity, expert_capacity;
    /* layer path */
    uint32_t layers, experts, model_width, expert_width;
    size_t record_bytes, stride, budget_bytes;
    bool configured, slab_tried;
    uint8_t *slab;
    uint32_t slots;
    int32_t *map;            /* layers * experts -> slot or -1 */
    int64_t *slot_key;       /* slot -> key or -1 */
    uint64_t *slot_stamp;
    float *slot_scale;
    uint64_t clock, hits, misses, evictions;
    uint32_t resident;
    float *ex_in, *ex_out, *ex_hidden;
};

static gemma4_cuda_context *g_ctx = nullptr;
static size_t g_dense_bytes = 0;
static char g_error[256] = "";
static float *g_d_in = nullptr, *g_d_out = nullptr, *g_h_in = nullptr, *g_h_out = nullptr;
static size_t g_in_cap = 0, g_out_cap = 0;

struct gemma4_cuda_matrix {
    uint8_t *data;
    int type;
    uint32_t rows, columns;
};

static size_t env_mb(const char *name, size_t fallback_mb) {
    const char *text = std::getenv(name);
    if (!text || !*text) return fallback_mb << 20;
    char *end = nullptr;
    unsigned long long value = std::strtoull(text, &end, 10);
    return (end && *end == '\0') ? static_cast<size_t>(value) << 20 : fallback_mb << 20;
}

static void set_error(char *error, size_t capacity, const char *operation,
                      cudaError_t result) {
    std::snprintf(g_error, sizeof(g_error), "%s: %s", operation, cudaGetErrorString(result));
    if (error && capacity) std::snprintf(error, capacity, "%s", g_error);
}

/* ---- kernels ------------------------------------------------------------ */

__device__ __forceinline__ float half_from_bits(uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ uint16_t load_u16(const uint8_t *pointer) {
    return *reinterpret_cast<const uint16_t *>(pointer);
}

/* Four lanes cooperate on one Q4_0 block (8 weights each), so a 32-lane warp
 * walks 8 blocks per step. All loads are 2-byte aligned: blocks are 18 bytes. */
__device__ __forceinline__ float q4_0_row_partial(const uint8_t *row, uint32_t blocks,
                                                  const float *x, uint32_t lane) {
    float accumulator = 0.0F;
    for (uint32_t unit = lane; unit < blocks * 4U; unit += 32U) {
        const uint32_t block = unit >> 2U, quarter = unit & 3U;
        const uint8_t *encoded = row + static_cast<size_t>(block) * 18U;
        const float scale = half_from_bits(load_u16(encoded));
        const uint16_t w0 = load_u16(encoded + 2U + quarter * 4U);
        const uint16_t w1 = load_u16(encoded + 4U + quarter * 4U);
        const uint32_t packed = static_cast<uint32_t>(w0) | (static_cast<uint32_t>(w1) << 16U);
        const float *low = x + static_cast<size_t>(block) * 32U + quarter * 4U;
        const float *high = low + 16;
        float sum = 0.0F;
#pragma unroll
        for (uint32_t j = 0; j < 4U; ++j) {
            const uint32_t byte = (packed >> (8U * j)) & 0xFFU;
            sum += static_cast<float>(static_cast<int>(byte & 15U) - 8) * __ldg(low + j);
            sum += static_cast<float>(static_cast<int>(byte >> 4U) - 8) * __ldg(high + j);
        }
        accumulator += scale * sum;
    }
    return accumulator;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xFFFFFFFFU, value, offset);
    return value;
}

__global__ static void q4_0_matvec(const uint8_t *weights, uint32_t rows,
                                   uint32_t columns, const float *input,
                                   float multiplier, float accumulate_into_output,
                                   float *output) {
    const uint32_t warp = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const uint32_t row = blockIdx.x * (blockDim.x >> 5U) + warp;
    if (row >= rows) return;
    const uint32_t blocks = columns / 32U;
    const float sum = warp_sum(q4_0_row_partial(
        weights + static_cast<size_t>(row) * blocks * 18U, blocks, input, lane));
    if (lane == 0) {
        if (accumulate_into_output != 0.0F) output[row] += multiplier * sum;
        else output[row] = multiplier * sum;
    }
}

/* hidden[i] = gelu_tanh(gate_i . x) * (up_i . x) */
__global__ static void q4_0_gate_up_gelu(const uint8_t *gate_weights,
                                         const uint8_t *up_weights, uint32_t rows,
                                         uint32_t columns, const float *input,
                                         float *hidden) {
    const uint32_t warp = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const uint32_t row = blockIdx.x * (blockDim.x >> 5U) + warp;
    if (row >= rows) return;
    const uint32_t blocks = columns / 32U;
    const size_t offset = static_cast<size_t>(row) * blocks * 18U;
    const float gate = warp_sum(q4_0_row_partial(gate_weights + offset, blocks, input, lane));
    const float up = warp_sum(q4_0_row_partial(up_weights + offset, blocks, input, lane));
    if (lane == 0) {
        const float activated = 0.5F * gate *
            (1.0F + tanhf(0.7978845608028654F *
                          (gate + 0.044715F * gate * gate * gate)));
        hidden[row] = activated * up;
    }
}

/* Q6_K: 256 weights per 210-byte block. Lane = position inside a 32-wide row of
 * each half; a lane decodes its four quarters. Same indexing as q6_k_value(). */
__global__ static void q6_k_matvec(const uint8_t *weights, uint32_t rows,
                                   uint32_t columns, const float *input,
                                   float *output) {
    const uint32_t warp = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const uint32_t row = blockIdx.x * (blockDim.x >> 5U) + warp;
    if (row >= rows) return;
    const uint32_t blocks = columns / 256U;
    const uint8_t *row_data = weights + static_cast<size_t>(row) * blocks * 210U;
    float accumulator = 0.0F;
    for (uint32_t block_index = 0; block_index < blocks; ++block_index) {
        const uint8_t *block = row_data + static_cast<size_t>(block_index) * 210U;
        const float super_scale = half_from_bits(load_u16(block + 208));
        const float *x = input + static_cast<size_t>(block_index) * 256U;
        float sum = 0.0F;
#pragma unroll
        for (uint32_t half = 0; half < 2U; ++half) {
            const uint8_t *ql = block + half * 64U;
            const uint8_t *qh = block + 128U + half * 32U;
            const uint8_t low_a = ql[lane], low_b = ql[lane + 32U], high = qh[lane];
            const int q0 = static_cast<int>((low_a & 15U) | (((high >> 0U) & 3U) << 4U)) - 32;
            const int q1 = static_cast<int>((low_b & 15U) | (((high >> 2U) & 3U) << 4U)) - 32;
            const int q2 = static_cast<int>((low_a >> 4U) | (((high >> 4U) & 3U) << 4U)) - 32;
            const int q3 = static_cast<int>((low_b >> 4U) | (((high >> 6U) & 3U) << 4U)) - 32;
            const uint32_t scale_base = half * 8U + (lane >> 4U);
            const int8_t *scales = reinterpret_cast<const int8_t *>(block + 192U);
            const float *xh = x + half * 128U + lane;
            sum += static_cast<float>(scales[scale_base + 0U]) * static_cast<float>(q0) * __ldg(xh);
            sum += static_cast<float>(scales[scale_base + 2U]) * static_cast<float>(q1) * __ldg(xh + 32);
            sum += static_cast<float>(scales[scale_base + 4U]) * static_cast<float>(q2) * __ldg(xh + 64);
            sum += static_cast<float>(scales[scale_base + 6U]) * static_cast<float>(q3) * __ldg(xh + 96);
        }
        accumulator += super_scale * sum;
    }
    const float total = warp_sum(accumulator);
    if (lane == 0) output[row] = total;
}

__global__ static void clear_floats(float *values, uint32_t count) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) values[index] = 0.0F;
}

static const uint32_t kWarpsPerBlock = 4;
static uint32_t grid_for(uint32_t rows) { return (rows + kWarpsPerBlock - 1U) / kWarpsPerBlock; }

/* ---- context ------------------------------------------------------------ */

static void release_buffers(gemma4_cuda_context *context) {
    if (!context) return;
    cudaFree(context->payload); cudaFree(context->input);
    cudaFree(context->gate); cudaFree(context->up);
    cudaFree(context->hidden); cudaFree(context->output);
    context->payload = nullptr; context->input = nullptr;
    context->gate = nullptr; context->up = nullptr;
    context->hidden = nullptr; context->output = nullptr;
    context->payload_capacity = 0;
    context->model_capacity = 0;
    context->expert_capacity = 0;
}

static cudaError_t reserve(gemma4_cuda_context *context,
                           size_t payload_bytes, uint32_t model_width,
                           uint32_t expert_width) {
    if (context->payload_capacity >= payload_bytes &&
        context->model_capacity >= model_width &&
        context->expert_capacity >= expert_width) return cudaSuccess;
    release_buffers(context);
    cudaError_t result = cudaMalloc(&context->payload, payload_bytes);
    if (result == cudaSuccess)
        result = cudaMalloc(&context->input, static_cast<size_t>(model_width) * sizeof(float));
    if (result == cudaSuccess)
        result = cudaMalloc(&context->gate, static_cast<size_t>(expert_width) * sizeof(float));
    if (result == cudaSuccess)
        result = cudaMalloc(&context->up, static_cast<size_t>(expert_width) * sizeof(float));
    if (result == cudaSuccess)
        result = cudaMalloc(&context->hidden, static_cast<size_t>(expert_width) * sizeof(float));
    if (result == cudaSuccess)
        result = cudaMalloc(&context->output, static_cast<size_t>(model_width) * sizeof(float));
    if (result != cudaSuccess) {
        release_buffers(context);
        return result;
    }
    context->payload_capacity = payload_bytes;
    context->model_capacity = model_width;
    context->expert_capacity = expert_width;
    return cudaSuccess;
}

extern "C" int coli_gemma4_cuda_create(void **opaque, int device,
                                        char *error, size_t error_capacity) {
    if (!opaque || device < 0) return -1;
    *opaque = nullptr;
    int count = 0;
    cudaError_t result = cudaGetDeviceCount(&count);
    if (result != cudaSuccess || device >= count) {
        set_error(error, error_capacity, "cannot select CUDA device", result);
        return -1;
    }
    result = cudaSetDevice(device);
    if (result != cudaSuccess) {
        set_error(error, error_capacity, "cannot activate CUDA device", result);
        return -1;
    }
    gemma4_cuda_context *context =
        static_cast<gemma4_cuda_context *>(std::calloc(1, sizeof(*context)));
    if (!context) return -1;
    context->device = device;
    *opaque = context;
    g_ctx = context;
    return 0;
}

static void free_expert_cache(gemma4_cuda_context *context) {
    cudaFree(context->slab); cudaFree(context->ex_in);
    cudaFree(context->ex_out); cudaFree(context->ex_hidden);
    std::free(context->map); std::free(context->slot_key);
    std::free(context->slot_stamp); std::free(context->slot_scale);
    context->slab = nullptr; context->ex_in = context->ex_out = context->ex_hidden = nullptr;
    context->map = nullptr; context->slot_key = nullptr;
    context->slot_stamp = nullptr; context->slot_scale = nullptr;
    context->slots = 0; context->configured = false; context->slab_tried = false;
}

extern "C" void coli_gemma4_cuda_destroy(void *opaque) {
    gemma4_cuda_context *context = static_cast<gemma4_cuda_context *>(opaque);
    if (!context) return;
    cudaSetDevice(context->device);
    release_buffers(context);
    free_expert_cache(context);
    if (g_ctx == context) {
        cudaFree(g_d_in); cudaFree(g_d_out); cudaFreeHost(g_h_in); cudaFreeHost(g_h_out);
        g_d_in = g_d_out = g_h_in = g_h_out = nullptr;
        g_in_cap = g_out_cap = 0;
        g_ctx = nullptr;
    }
    std::free(context);
}

extern "C" int coli_gemma4_cuda_run(
    void *opaque, const uint8_t *payload, size_t payload_bytes,
    size_t gate_bytes, size_t up_bytes, uint32_t model_width,
    uint32_t expert_width, const float *input, float scale, float *output,
    char *error, size_t error_capacity) {
    gemma4_cuda_context *context = static_cast<gemma4_cuda_context *>(opaque);
    if (!context || !payload || !input || !output || !model_width ||
        !expert_width || model_width % 32U || expert_width % 32U ||
        gate_bytes + up_bytes > payload_bytes) return -1;
    cudaError_t result = cudaSetDevice(context->device);
    if (result == cudaSuccess)
        result = reserve(context, payload_bytes, model_width, expert_width);
    if (result == cudaSuccess)
        result = cudaMemcpy(context->payload, payload, payload_bytes, cudaMemcpyHostToDevice);
    if (result == cudaSuccess)
        result = cudaMemcpy(context->input, input,
                            static_cast<size_t>(model_width) * sizeof(float),
                            cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        set_error(error, error_capacity, "CUDA expert upload failed", result);
        return -1;
    }
    q4_0_gate_up_gelu<<<grid_for(expert_width), kWarpsPerBlock * 32U>>>(
        context->payload, context->payload + gate_bytes, expert_width, model_width,
        context->input, context->hidden);
    q4_0_matvec<<<grid_for(model_width), kWarpsPerBlock * 32U>>>(
        context->payload + gate_bytes + up_bytes, model_width, expert_width,
        context->hidden, scale, 0.0F, context->output);
    result = cudaGetLastError();
    if (result == cudaSuccess)
        result = cudaMemcpy(output, context->output,
                            static_cast<size_t>(model_width) * sizeof(float),
                            cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
        set_error(error, error_capacity, "CUDA expert execution failed", result);
        return -1;
    }
    return 0;
}

/* ---- resident dense matrices -------------------------------------------- */

extern "C" int coli_gemma4_cuda_dense_enabled(void) {
    const char *off = std::getenv("GEMMA4_CUDA_DENSE");
    return g_ctx != nullptr && !(off && off[0] == '0');
}

extern "C" const char *coli_gemma4_cuda_last_error(void) { return g_error; }
extern "C" size_t coli_gemma4_cuda_dense_bytes(void) { return g_dense_bytes; }

extern "C" void *coli_gemma4_cuda_matrix_upload(const uint8_t *data, size_t bytes,
                                                 int type, uint32_t rows,
                                                 uint32_t columns) {
    if (!coli_gemma4_cuda_dense_enabled() || !data || !bytes || !rows || !columns) return nullptr;
    const uint32_t block_elems = type == COLI_GEMMA4_CUDA_Q6_K ? 256U : 32U;
    const size_t block_bytes = type == COLI_GEMMA4_CUDA_Q6_K ? 210U : 18U;
    if (columns % block_elems ||
        bytes != static_cast<size_t>(rows) * (columns / block_elems) * block_bytes) return nullptr;
    cudaError_t probe = cudaSetDevice(g_ctx->device);
    if (probe != cudaSuccess) { set_error(nullptr, 0, "cudaSetDevice", probe); return nullptr; }
    size_t free_bytes = 0, total_bytes = 0;
    probe = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (probe != cudaSuccess) { set_error(nullptr, 0, "cudaMemGetInfo", probe); return nullptr; }
    /* Dense weights are needed by every token, so they are uploaded first and may use
     * everything except the safety reserve; the expert slab takes what is left. */
    const size_t reserve_bytes = env_mb("GEMMA4_CUDA_RESERVE_MB", 256);
    if (free_bytes < bytes + reserve_bytes) {
        std::snprintf(g_error, sizeof(g_error),
                      "dense matrix of %zu MiB does not fit (free %zu MiB, reserve %zu MiB)",
                      bytes >> 20, free_bytes >> 20, reserve_bytes >> 20);
        return nullptr;
    }
    gemma4_cuda_matrix *matrix = static_cast<gemma4_cuda_matrix *>(std::calloc(1, sizeof(*matrix)));
    if (!matrix) return nullptr;
    cudaError_t result = cudaMalloc(&matrix->data, bytes);
    if (result == cudaSuccess)
        result = cudaMemcpy(matrix->data, data, bytes, cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        set_error(nullptr, 0, "dense matrix upload failed", result);
        cudaFree(matrix->data);
        std::free(matrix);
        return nullptr;
    }
    matrix->type = type;
    matrix->rows = rows;
    matrix->columns = columns;
    g_dense_bytes += bytes;
    return matrix;
}

extern "C" void coli_gemma4_cuda_matrix_free(void *opaque) {
    gemma4_cuda_matrix *matrix = static_cast<gemma4_cuda_matrix *>(opaque);
    if (!matrix) return;
    if (g_ctx) cudaSetDevice(g_ctx->device);
    cudaFree(matrix->data);
    const size_t bytes = static_cast<size_t>(matrix->rows) *
        (matrix->columns / (matrix->type == COLI_GEMMA4_CUDA_Q6_K ? 256U : 32U)) *
        (matrix->type == COLI_GEMMA4_CUDA_Q6_K ? 210U : 18U);
    g_dense_bytes = g_dense_bytes >= bytes ? g_dense_bytes - bytes : 0;
    std::free(matrix);
}

static cudaError_t grow_io(size_t input_floats, size_t output_floats) {
    cudaError_t result = cudaSuccess;
    if (input_floats > g_in_cap) {
        cudaFree(g_d_in); cudaFreeHost(g_h_in);
        g_d_in = nullptr; g_h_in = nullptr; g_in_cap = 0;
        result = cudaMalloc(&g_d_in, input_floats * sizeof(float));
        if (result == cudaSuccess) result = cudaMallocHost(&g_h_in, input_floats * sizeof(float));
        if (result != cudaSuccess) return result;
        g_in_cap = input_floats;
    }
    if (output_floats > g_out_cap) {
        cudaFree(g_d_out); cudaFreeHost(g_h_out);
        g_d_out = nullptr; g_h_out = nullptr; g_out_cap = 0;
        result = cudaMalloc(&g_d_out, output_floats * sizeof(float));
        if (result == cudaSuccess) result = cudaMallocHost(&g_h_out, output_floats * sizeof(float));
        if (result != cudaSuccess) return result;
        g_out_cap = output_floats;
    }
    return cudaSuccess;
}

extern "C" int coli_gemma4_cuda_matrix_matvec(void *opaque, const float *input, float *output) {
    gemma4_cuda_matrix *matrix = static_cast<gemma4_cuda_matrix *>(opaque);
    if (!matrix || !g_ctx || !input || !output) return -1;
    cudaError_t result = cudaSetDevice(g_ctx->device);
    if (result == cudaSuccess) result = grow_io(matrix->columns, matrix->rows);
    if (result != cudaSuccess) {
        set_error(nullptr, 0, "dense matvec buffers", result);
        return -1;
    }
    std::memcpy(g_h_in, input, static_cast<size_t>(matrix->columns) * sizeof(float));
    result = cudaMemcpyAsync(g_d_in, g_h_in, static_cast<size_t>(matrix->columns) * sizeof(float),
                             cudaMemcpyHostToDevice, 0);
    if (result == cudaSuccess) {
        if (matrix->type == COLI_GEMMA4_CUDA_Q6_K)
            q6_k_matvec<<<grid_for(matrix->rows), kWarpsPerBlock * 32U>>>(
                matrix->data, matrix->rows, matrix->columns, g_d_in, g_d_out);
        else
            q4_0_matvec<<<grid_for(matrix->rows), kWarpsPerBlock * 32U>>>(
                matrix->data, matrix->rows, matrix->columns, g_d_in, 1.0F, 0.0F, g_d_out);
        result = cudaGetLastError();
    }
    if (result == cudaSuccess)
        result = cudaMemcpyAsync(g_h_out, g_d_out, static_cast<size_t>(matrix->rows) * sizeof(float),
                                 cudaMemcpyDeviceToHost, 0);
    if (result == cudaSuccess) result = cudaStreamSynchronize(0);
    if (result != cudaSuccess) {
        set_error(nullptr, 0, "dense matvec failed", result);
        return -1;
    }
    std::memcpy(output, g_h_out, static_cast<size_t>(matrix->rows) * sizeof(float));
    return 0;
}

/* ---- pinned staging memory ------------------------------------------------ */

extern "C" void *coli_gemma4_cuda_host_alloc(size_t bytes) {
    void *pointer = nullptr;
    if (g_ctx && cudaSetDevice(g_ctx->device) == cudaSuccess &&
        cudaMallocHost(&pointer, bytes) == cudaSuccess) return pointer;
    cudaGetLastError();
    return std::malloc(bytes);
}

extern "C" void coli_gemma4_cuda_host_free(void *pointer) {
    if (!pointer) return;
    cudaPointerAttributes attributes;
    if (cudaPointerGetAttributes(&attributes, pointer) == cudaSuccess &&
        attributes.type == cudaMemoryTypeHost) {
        cudaFreeHost(pointer);
    } else {
        cudaGetLastError();
        std::free(pointer);
    }
}

/* ---- VRAM expert cache --------------------------------------------------- */

extern "C" int coli_gemma4_cuda_expert_cache_configure(
    void *opaque, uint32_t layers, uint32_t experts, size_t record_bytes,
    uint32_t model_width, uint32_t expert_width, size_t budget_bytes,
    char *error, size_t error_capacity) {
    gemma4_cuda_context *context = static_cast<gemma4_cuda_context *>(opaque);
    if (!context || !layers || !experts || !record_bytes || !model_width || !expert_width ||
        model_width % 32U || expert_width % 32U) return -1;
    cudaError_t result = cudaSetDevice(context->device);
    if (result != cudaSuccess) {
        set_error(error, error_capacity, "cannot activate CUDA device", result);
        return -1;
    }
    free_expert_cache(context);
    context->layers = layers;
    context->experts = experts;
    context->record_bytes = record_bytes;
    context->stride = (record_bytes + 255U) / 256U * 256U;
    context->model_width = model_width;
    context->expert_width = expert_width;
    context->budget_bytes = budget_bytes;
    context->map = static_cast<int32_t *>(std::malloc(sizeof(int32_t) * layers * experts));
    if (!context->map) return -1;
    for (size_t i = 0; i < static_cast<size_t>(layers) * experts; ++i) context->map[i] = -1;
    result = cudaMalloc(&context->ex_in, sizeof(float) * model_width);
    if (result == cudaSuccess) result = cudaMalloc(&context->ex_out, sizeof(float) * model_width);
    if (result == cudaSuccess) result = cudaMalloc(&context->ex_hidden, sizeof(float) * expert_width);
    if (result != cudaSuccess) {
        set_error(error, error_capacity, "expert scratch allocation failed", result);
        free_expert_cache(context);
        return -1;
    }
    context->configured = true;
    return 0;
}

/* The slab is allocated lazily, after the dense matrices, so it gets exactly the VRAM
 * they left behind (minus the reserve), capped by the configured budget. */
static void ensure_slab(gemma4_cuda_context *context) {
    if (!context->configured || context->slab_tried) return;
    context->slab_tried = true;
    if (cudaSetDevice(context->device) != cudaSuccess) return;
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) return;
    const size_t reserve_bytes = env_mb("GEMMA4_CUDA_RESERVE_MB", 256);
    size_t usable = free_bytes > reserve_bytes ? free_bytes - reserve_bytes : 0;
    size_t cap = context->budget_bytes ? context->budget_bytes : usable;   /* 0 = no cap */
    const char *override_text = std::getenv("GEMMA4_CUDA_EXPERT_MB");
    if (override_text && *override_text) {
        cap = env_mb("GEMMA4_CUDA_EXPERT_MB", 0);
        if (cap == 0) return;                                   /* explicit 0 disables the cache */
    }
    if (cap < usable) usable = cap;
    size_t slots = usable / context->stride;
    if (slots > static_cast<size_t>(context->layers) * context->experts)
        slots = static_cast<size_t>(context->layers) * context->experts;
    const size_t total_experts = static_cast<size_t>(context->layers) * context->experts;
    const size_t minimum_slots = total_experts < 16 ? total_experts : 16;
    if (slots < minimum_slots) return;
    while (slots >= minimum_slots) {
        if (cudaMalloc(&context->slab, slots * context->stride) == cudaSuccess) break;
        cudaGetLastError();
        context->slab = nullptr;
        if (slots <= minimum_slots) break;
        slots -= slots / 16 + 1;
        if (slots < minimum_slots) slots = minimum_slots;
    }
    if (!context->slab || slots < minimum_slots) return;
    context->slot_key = static_cast<int64_t *>(std::malloc(sizeof(int64_t) * slots));
    context->slot_stamp = static_cast<uint64_t *>(std::calloc(slots, sizeof(uint64_t)));
    context->slot_scale = static_cast<float *>(std::calloc(slots, sizeof(float)));
    if (!context->slot_key || !context->slot_stamp || !context->slot_scale) {
        cudaFree(context->slab); context->slab = nullptr;
        return;
    }
    for (size_t i = 0; i < slots; ++i) context->slot_key[i] = -1;
    context->slots = static_cast<uint32_t>(slots);
}

extern "C" uint32_t coli_gemma4_cuda_expert_cache_slots(void *opaque) {
    gemma4_cuda_context *context = static_cast<gemma4_cuda_context *>(opaque);
    if (!context) return 0;
    ensure_slab(context);
    return context->slots;
}

extern "C" int coli_gemma4_cuda_expert_resident(void *opaque, uint32_t layer, uint32_t expert) {
    gemma4_cuda_context *context = static_cast<gemma4_cuda_context *>(opaque);
    if (!context || !context->configured) return 0;
    ensure_slab(context);
    if (!context->slots || layer >= context->layers || expert >= context->experts) return 0;
    const int32_t slot = context->map[static_cast<size_t>(layer) * context->experts + expert];
    if (slot < 0) return 0;
    context->slot_stamp[slot] = ++context->clock;     /* keep it from being evicted this layer */
    return 1;
}

extern "C" int coli_gemma4_cuda_layer_begin(void *opaque, const float *input,
                                             char *error, size_t error_capacity) {
    gemma4_cuda_context *context = static_cast<gemma4_cuda_context *>(opaque);
    if (!context || !context->configured || !input) return -1;
    ensure_slab(context);
    if (!context->slots) return -1;
    cudaError_t result = cudaSetDevice(context->device);
    if (result == cudaSuccess)
        result = cudaMemcpyAsync(context->ex_in, input, sizeof(float) * context->model_width,
                                 cudaMemcpyHostToDevice, 0);
    if (result == cudaSuccess) {
        clear_floats<<<(context->model_width + 255U) / 256U, 256>>>(context->ex_out,
                                                                    context->model_width);
        result = cudaGetLastError();
    }
    if (result != cudaSuccess) {
        set_error(error, error_capacity, "CUDA layer begin failed", result);
        return -1;
    }
    return 0;
}

extern "C" int coli_gemma4_cuda_expert_launch(
    void *opaque, uint32_t layer, uint32_t expert, const uint8_t *payload,
    size_t gate_bytes, size_t up_bytes, float scale, float weight,
    char *error, size_t error_capacity) {
    gemma4_cuda_context *context = static_cast<gemma4_cuda_context *>(opaque);
    if (!context || !context->configured || !context->slots || layer >= context->layers ||
        expert >= context->experts || gate_bytes + up_bytes > context->record_bytes) return -1;
    const int64_t key = static_cast<int64_t>(layer) * context->experts + expert;
    int32_t slot = context->map[key];
    if (slot >= 0) {
        ++context->hits;
    } else {
        if (!payload) {
            std::snprintf(g_error, sizeof(g_error), "expert %u of layer %u is not in VRAM", expert, layer);
            if (error && error_capacity) std::snprintf(error, error_capacity, "%s", g_error);
            return -1;
        }
        uint32_t victim = 0;
        for (uint32_t i = 0; i < context->slots; ++i) {
            if (context->slot_key[i] < 0) { victim = i; break; }
            if (context->slot_stamp[i] < context->slot_stamp[victim]) victim = i;
        }
        if (context->slot_key[victim] >= 0) {
            context->map[context->slot_key[victim]] = -1;
            ++context->evictions;
        } else {
            ++context->resident;
        }
        const cudaError_t copied = cudaMemcpyAsync(
            context->slab + static_cast<size_t>(victim) * context->stride, payload,
            context->record_bytes, cudaMemcpyHostToDevice, 0);
        if (copied != cudaSuccess) {
            context->slot_key[victim] = -1;
            set_error(error, error_capacity, "CUDA expert upload failed", copied);
            return -1;
        }
        context->slot_key[victim] = key;
        context->slot_scale[victim] = scale;
        context->map[key] = static_cast<int32_t>(victim);
        slot = static_cast<int32_t>(victim);
        ++context->misses;
    }
    context->slot_stamp[slot] = ++context->clock;
    const uint8_t *record = context->slab + static_cast<size_t>(slot) * context->stride;
    q4_0_gate_up_gelu<<<grid_for(context->expert_width), kWarpsPerBlock * 32U>>>(
        record, record + gate_bytes, context->expert_width, context->model_width,
        context->ex_in, context->ex_hidden);
    q4_0_matvec<<<grid_for(context->model_width), kWarpsPerBlock * 32U>>>(
        record + gate_bytes + up_bytes, context->model_width, context->expert_width,
        context->ex_hidden, weight * context->slot_scale[slot], 1.0F, context->ex_out);
    const cudaError_t launched = cudaGetLastError();
    if (launched != cudaSuccess) {
        set_error(error, error_capacity, "CUDA expert launch failed", launched);
        return -1;
    }
    return 0;
}

extern "C" int coli_gemma4_cuda_layer_finish(void *opaque, float *output,
                                              char *error, size_t error_capacity) {
    gemma4_cuda_context *context = static_cast<gemma4_cuda_context *>(opaque);
    if (!context || !context->configured || !output) return -1;
    cudaError_t result = cudaMemcpyAsync(output, context->ex_out,
                                         sizeof(float) * context->model_width,
                                         cudaMemcpyDeviceToHost, 0);
    if (result == cudaSuccess) result = cudaStreamSynchronize(0);
    if (result != cudaSuccess) {
        set_error(error, error_capacity, "CUDA layer finish failed", result);
        return -1;
    }
    return 0;
}

extern "C" void coli_gemma4_cuda_expert_cache_stats(
    void *opaque, uint32_t *slots, uint32_t *resident, uint64_t *hits,
    uint64_t *misses, uint64_t *evictions) {
    gemma4_cuda_context *context = static_cast<gemma4_cuda_context *>(opaque);
    if (slots) *slots = context ? context->slots : 0;
    if (resident) *resident = context ? context->resident : 0;
    if (hits) *hits = context ? context->hits : 0;
    if (misses) *misses = context ? context->misses : 0;
    if (evictions) *evictions = context ? context->evictions : 0;
}

/* ======================================================================================
 * Whole-layer decode on the device (one token).
 *
 * Almost everything below is adapted from kernels that already exist:
 *   k_rms_norm        <- llama.cpp ggml-cuda/norm.cu  rms_norm_f32 (non-vector path, with the
 *                        fused multiply / add / scale variants); the per-head QK norm is the same
 *                        kernel with one block per head and a broadcast weight
 *   k_rope_neox       <- llama.cpp ggml-cuda/rope.cu  rope_neox (freq_factors path; the YaRN terms
 *                        are dropped because this model uses plain RoPE)
 *   k_topk_softmax    <- llama.cpp ggml-cuda/topk-moe.cu  topk_moe_cuda (softmax, iterative
 *                        warp argmax, renormalise); bias / sigmoid / group-mask paths removed
 *   k_attention       <- the score / max / sum / value passes of colibri's
 *                        attention_absorb_batch_kernel, with colibri's ring-slot indexing from
 *                        attention_window_kernel; grouped-query head mapping, no 1/sqrt(d) scale
 *                        (Gemma 4 normalises Q and K instead), matching the CPU reference
 *   k_kv_store        <- colibri cache_store
 *   k_router_logits   <- colibri pipe_router_logits (softmax variant)
 * The quantised matvecs are the kernels above in this file.
 * ====================================================================================== */

#define G4_WARP 32

__device__ __forceinline__ float g4_warp_max(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_xor_sync(0xFFFFFFFFU, value, offset));
    return value;
}

/* ggml-cuda block_reduce<SUM / MAX>: every thread receives the block result. */
__device__ __forceinline__ float g4_block_sum(float value, float *shared) {
    value = warp_sum(value);
    if (blockDim.x > G4_WARP) {
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        __syncthreads();
        if (lane == 0) shared[warp] = value;
        __syncthreads();
        value = lane < (int)(blockDim.x >> 5) ? shared[lane] : 0.0F;
        value = warp_sum(value);
    }
    return value;
}

__device__ __forceinline__ float g4_block_max(float value, float *shared) {
    value = g4_warp_max(value);
    if (blockDim.x > G4_WARP) {
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        __syncthreads();
        if (lane == 0) shared[warp] = value;
        __syncthreads();
        value = lane < (int)(blockDim.x >> 5) ? shared[lane] : -FLT_MAX;
        value = g4_warp_max(value);
    }
    return value;
}

/* out[row] = (norm(x[row]) * w + add[row]) * scale_out. w is shared by every row
 * (null = unweighted), add is per row (null = none). x == out is allowed. */
__global__ static void k_rms_norm(const float *x, const float *w, const float *add,
                                  float *out, int n, float eps, float scale_out) {
    __shared__ float shared[32];
    const size_t base = (size_t)blockIdx.x * n;
    const float *xc = x + base;
    float partial = 0.0F;
    for (int i = threadIdx.x; i < n; i += blockDim.x) partial += xc[i] * xc[i];
    const float total = g4_block_sum(partial, shared);
    const float scale = 1.0F / sqrtf(total / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = scale * xc[i];
        if (w) v *= w[i];
        if (add) v += add[base + i];
        out[base + i] = v * scale_out;
    }
}

/* NEOX rotation of pairs (k, k + half) with per-pair frequency factors. */
__global__ static void k_rope_neox(float *x, int heads, int head_dim, int position,
                                   float base, const float *factors) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x, half = head_dim / 2;
    if (index >= heads * half) return;
    const int head = index / half, k = index % half;
    const float factor = factors ? factors[k] : 1.0F;
    const float frequency = powf(base, -2.0F * (float)k / (float)head_dim) / factor;
    const float angle = (float)position * frequency;
    const float c = cosf(angle), s = sinf(angle);
    float *row = x + (size_t)head * head_dim;
    const float first = row[k], second = row[k + half];
    row[k] = first * c - second * s;
    row[k + half] = second * c + first * s;
}

__global__ static void k_kv_store(float *key_cache, float *value_cache, int slot,
                                  const float *key, const float *value, int kv_width,
                                  float *exported) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kv_width) return;
    key_cache[(size_t)slot * kv_width + i] = key[i];
    value_cache[(size_t)slot * kv_width + i] = value[i];
    exported[i] = key[i];
    exported[kv_width + i] = value[i];
}

/* One block per query head: scores (a warp per cached position), softmax with the
 * block max/sum reductions, then the probability-weighted sum of V. */
__global__ static void k_attention(const float *q, const float *key_cache,
                                   const float *value_cache, float *context,
                                   float *scores_all, int head_dim, int queries_per_kv,
                                   int kv_width, int capacity, int sliding,
                                   unsigned long long first_position, int history,
                                   int score_stride) {
    __shared__ float shared[32];
    const int head = blockIdx.x, kv_head = head / queries_per_kv;
    const int tid = threadIdx.x, warps = blockDim.x >> 5, warp = tid >> 5, lane = tid & 31;
    const float *query_head = q + (size_t)head * head_dim;
    float *scores = scores_all + (size_t)head * score_stride;
    const int first_slot = sliding ? (int)(first_position % (unsigned long long)capacity)
                                   : (int)first_position;
    for (int t = warp; t < history; t += warps) {
        int slot = first_slot + t;
        if (sliding && slot >= capacity) slot -= capacity;
        const float *key_head = key_cache + (size_t)slot * kv_width + (size_t)kv_head * head_dim;
        float score = 0.0F;
        for (int d = lane; d < head_dim; d += 32) score += query_head[d] * key_head[d];
        score = warp_sum(score);
        if (lane == 0) scores[t] = score;
    }
    __syncthreads();
    float local = -FLT_MAX;
    for (int t = tid; t < history; t += blockDim.x) local = fmaxf(local, scores[t]);
    const float maximum = g4_block_max(local, shared);
    local = 0.0F;
    for (int t = tid; t < history; t += blockDim.x) {
        const float e = expf(scores[t] - maximum);
        scores[t] = e;
        local += e;
    }
    const float denominator = g4_block_sum(local, shared);
    __syncthreads();
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float accumulator = 0.0F;
        for (int t = 0; t < history; ++t) {
            int slot = first_slot + t;
            if (sliding && slot >= capacity) slot -= capacity;
            accumulator += (scores[t] / denominator) *
                value_cache[(size_t)slot * kv_width + (size_t)kv_head * head_dim + d];
        }
        context[(size_t)head * head_dim + d] = accumulator;
    }
}

__global__ static void k_router_logits(const float *projection, const float *input,
                                       int width, float *logits) {
    __shared__ float shared[32];
    const float *row = projection + (size_t)blockIdx.x * width;
    float partial = 0.0F;
    for (int i = threadIdx.x; i < width; i += blockDim.x) partial += row[i] * input[i];
    const float total = g4_block_sum(partial, shared);
    if (threadIdx.x == 0) logits[blockIdx.x] = total;
}

__global__ static void k_gelu_mul(const float *gate, const float *up, int count,
                                  float *hidden) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float v = gate[i];
    hidden[i] = 0.5F * v * (1.0F + tanhf(0.7978845608028654F *
                                          (v + 0.044715F * v * v * v))) * up[i];
}

/* Softmax -> top-k -> renormalise, one warp. Lifted from topk_moe_cuda (softmax,
 * with_norm path); NaN logits are sanitised exactly as there so the iterative argmax
 * always yields distinct experts. Ties go to the lower expert index. */
template <int n_experts>
__global__ static void k_topk_softmax(const float *logits, float *weights, int32_t *ids,
                                      int n_expert_used, float clamp_val) {
    constexpr int experts_per_thread = (n_experts > G4_WARP) ? n_experts / G4_WARP : 1;
    const int lane = threadIdx.x;
    float wt[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) wt[i] = -INFINITY;
#pragma unroll
    for (int i = 0; i < n_experts; i += G4_WARP) {
        const int expert = i + lane;
        wt[i / G4_WARP] = (n_experts % G4_WARP == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }
    float max_val = -INFINITY;
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) max_val = fmaxf(max_val, wt[i]);
    max_val = g4_warp_max(max_val);
    float sum = 0.0F;
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const float v = expf(wt[i] - max_val);
        wt[i] = v;
        sum += v;
    }
    sum = warp_sum(sum);
    const float inv_sum = 1.0F / sum;
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) wt[i] *= inv_sum;
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++)
        if (isnan(wt[i])) wt[i] = -FLT_MAX;

    float wt_sum = 0.0F;
    float output_weights[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) output_weights[i] = 0.0F;
    for (int k = 0; k < n_expert_used; k++) {
        float best = wt[0];
        int best_expert = lane;
#pragma unroll
        for (int i = 1; i < experts_per_thread; i++) {
            const int expert = lane + i * G4_WARP;
            if ((n_experts % G4_WARP == 0 || expert < n_experts) && wt[i] > best) {
                best = wt[i];
                best_expert = expert;
            }
        }
#pragma unroll
        for (int mask = G4_WARP / 2; mask > 0; mask /= 2) {
            const float other = __shfl_xor_sync(0xFFFFFFFFU, best, mask, G4_WARP);
            const int other_expert = __shfl_xor_sync(0xFFFFFFFFU, best_expert, mask, G4_WARP);
            if (other > best || (other == best && other_expert < best_expert)) {
                best = other;
                best_expert = other_expert;
            }
        }
        if ((best_expert & (G4_WARP - 1)) == lane) wt[best_expert / G4_WARP] = -INFINITY;
        if ((k & (G4_WARP - 1)) == lane) output_weights[k / G4_WARP] = best;
        if ((best_expert & (G4_WARP - 1)) == lane) {
            ids[k] = best_expert;
            wt_sum += best;
        }
    }
    wt_sum = warp_sum(wt_sum);
    wt_sum = fmaxf(wt_sum, clamp_val);
    const float inv_selected = 1.0F / wt_sum;
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * G4_WARP + lane;
        if (idx < n_expert_used) weights[idx] = output_weights[i] * inv_selected;
    }
}

/* ---- host side ------------------------------------------------------------------- */

#define G4_MAX_SELECTED 32

struct gemma4_gpu_scratch {
    size_t model_width, query_width, kv_width, intermediate, experts, score_floats;
    float *res, *norm, *q, *k, *v, *kv_export, *context, *attn, *after, *dense_in,
          *router_norm, *router_logits, *router_weights, *hidden_gate, *hidden_up, *hidden,
          *dense_out, *dense_norm, *combined, *out, *scores;
    int32_t *router_ids;
    float *host_kv, *host_weights;
    int32_t *host_ids;
};
static gemma4_gpu_scratch g_scratch;

static void scratch_free(void) {
    float **device[] = {&g_scratch.res, &g_scratch.norm, &g_scratch.q, &g_scratch.k,
        &g_scratch.v, &g_scratch.kv_export, &g_scratch.context, &g_scratch.attn,
        &g_scratch.after, &g_scratch.dense_in, &g_scratch.router_norm,
        &g_scratch.router_logits, &g_scratch.router_weights, &g_scratch.hidden_gate,
        &g_scratch.hidden_up, &g_scratch.hidden, &g_scratch.dense_out, &g_scratch.dense_norm,
        &g_scratch.combined, &g_scratch.out, &g_scratch.scores};
    for (size_t i = 0; i < sizeof(device) / sizeof(device[0]); ++i) {
        cudaFree(*device[i]);
        *device[i] = nullptr;
    }
    cudaFree(g_scratch.router_ids);
    g_scratch.router_ids = nullptr;
    cudaFreeHost(g_scratch.host_kv); cudaFreeHost(g_scratch.host_weights);
    cudaFreeHost(g_scratch.host_ids);
    g_scratch.host_kv = g_scratch.host_weights = nullptr;
    g_scratch.host_ids = nullptr;
}

static bool scratch_reserve(size_t model_width, size_t query_width, size_t kv_width,
                            size_t intermediate, size_t experts, size_t score_floats) {
    gemma4_gpu_scratch &s = g_scratch;
    if (s.res && s.model_width >= model_width && s.query_width >= query_width &&
        s.kv_width >= kv_width && s.intermediate >= intermediate &&
        s.experts >= experts && s.score_floats >= score_floats) return true;
    if (model_width < s.model_width) model_width = s.model_width;
    if (query_width < s.query_width) query_width = s.query_width;
    if (kv_width < s.kv_width) kv_width = s.kv_width;
    if (intermediate < s.intermediate) intermediate = s.intermediate;
    if (experts < s.experts) experts = s.experts;
    if (score_floats < s.score_floats) score_floats = s.score_floats;
    scratch_free();
    struct Item { float **pointer; size_t floats; };
    Item items[] = {{&s.res, model_width}, {&s.norm, model_width}, {&s.q, query_width},
        {&s.k, kv_width}, {&s.v, kv_width}, {&s.kv_export, 2 * kv_width},
        {&s.context, query_width}, {&s.attn, model_width}, {&s.after, model_width},
        {&s.dense_in, model_width}, {&s.router_norm, model_width},
        {&s.router_logits, experts}, {&s.router_weights, G4_MAX_SELECTED},
        {&s.hidden_gate, intermediate}, {&s.hidden_up, intermediate},
        {&s.hidden, intermediate}, {&s.dense_out, model_width},
        {&s.dense_norm, model_width}, {&s.combined, model_width}, {&s.out, model_width},
        {&s.scores, score_floats}};
    bool ok = true;
    for (size_t i = 0; ok && i < sizeof(items) / sizeof(items[0]); ++i)
        ok = cudaMalloc(items[i].pointer, items[i].floats * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMalloc(&s.router_ids, experts * sizeof(int32_t)) == cudaSuccess;
    ok = ok && cudaMallocHost(&s.host_kv, 2 * kv_width * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMallocHost(&s.host_weights, G4_MAX_SELECTED * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMallocHost(&s.host_ids, G4_MAX_SELECTED * sizeof(int32_t)) == cudaSuccess;
    if (!ok) {
        cudaGetLastError();
        scratch_free();
        s.model_width = s.query_width = s.kv_width = s.intermediate = s.experts = 0;
        s.score_floats = 0;
        return false;
    }
    s.model_width = model_width; s.query_width = query_width; s.kv_width = kv_width;
    s.intermediate = intermediate; s.experts = experts; s.score_floats = score_floats;
    return true;
}

struct gemma4_gpu_layer {
    uint32_t model_width, query_heads, kv_heads, head_dim, query_width, kv_width, capacity;
    uint32_t intermediate, experts, selected;
    int sliding, key_equals_value;
    float epsilon, rope_base, output_scale;
    gemma4_cuda_matrix *query, *key, *value, *output, *gate, *up, *down;
    float *input_norm, *query_norm, *key_norm, *rope_factors;
    float *post_attention_norm, *ffn_norm, *post_ffw_norm_1, *pre_ffw_norm_2,
          *post_ffw_norm_2, *post_ffw_norm, *router_scale, *router_projection;
    float *key_cache, *value_cache;
};

static float *upload_floats(const float *host, size_t count, float scale = 1.0F) {
    float *device = nullptr;
    if (!host || !count || cudaMalloc(&device, count * sizeof(float)) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    bool ok;
    if (scale == 1.0F) {
        ok = cudaMemcpy(device, host, count * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
    } else {
        float *scaled = static_cast<float *>(std::malloc(count * sizeof(float)));
        ok = scaled != nullptr;
        if (ok) {
            for (size_t i = 0; i < count; ++i) scaled[i] = host[i] * scale;
            ok = cudaMemcpy(device, scaled, count * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
        }
        std::free(scaled);
    }
    if (!ok) {
        cudaGetLastError();
        cudaFree(device);
        return nullptr;
    }
    return device;
}

static void layer_free(gemma4_gpu_layer *layer) {
    if (!layer) return;
    float *buffers[] = {layer->input_norm, layer->query_norm, layer->key_norm,
        layer->rope_factors, layer->post_attention_norm, layer->ffn_norm,
        layer->post_ffw_norm_1, layer->pre_ffw_norm_2, layer->post_ffw_norm_2,
        layer->post_ffw_norm, layer->router_scale, layer->router_projection,
        layer->key_cache, layer->value_cache};
    for (size_t i = 0; i < sizeof(buffers) / sizeof(buffers[0]); ++i) cudaFree(buffers[i]);
    std::free(layer);
}

static bool q4_matrix_ok(const void *opaque, uint32_t rows, uint32_t columns) {
    const gemma4_cuda_matrix *m = static_cast<const gemma4_cuda_matrix *>(opaque);
    return m && m->type == COLI_GEMMA4_CUDA_Q4_0 && m->rows == rows && m->columns == columns;
}

extern "C" void *coli_gemma4_cuda_layer_create(const coli_gemma4_cuda_layer_desc *d,
                                                char *error, size_t error_capacity) {
    auto fail = [&](const char *message) -> void * {
        std::snprintf(g_error, sizeof(g_error), "%s", message);
        if (error && error_capacity) std::snprintf(error, error_capacity, "%s", message);
        return nullptr;
    };
    if (!d || !g_ctx) return fail("no CUDA context");
    const uint32_t query_width = d->query_heads * d->head_dim, kv_width = d->kv_heads * d->head_dim;
    if (!d->model_width || !d->query_heads || !d->kv_heads || !d->head_dim ||
        d->head_dim % 2 || d->query_heads % d->kv_heads || !d->kv_capacity ||
        !d->intermediate_width || d->selected_count == 0 ||
        d->selected_count > G4_MAX_SELECTED || d->selected_count > d->expert_count ||
        d->expert_count < 2 || d->expert_count > 256 ||
        (d->expert_count & (d->expert_count - 1)) != 0)
        return fail("unsupported Gemma layer geometry for the GPU path");
    if (!q4_matrix_ok(d->query, query_width, d->model_width) ||
        !q4_matrix_ok(d->key, kv_width, d->model_width) ||
        !q4_matrix_ok(d->output, d->model_width, query_width) ||
        (!d->key_equals_value && !q4_matrix_ok(d->value, kv_width, d->model_width)) ||
        !q4_matrix_ok(d->gate, d->intermediate_width, d->model_width) ||
        !q4_matrix_ok(d->up, d->intermediate_width, d->model_width) ||
        !q4_matrix_ok(d->down, d->model_width, d->intermediate_width))
        return fail("a layer matrix is not resident in VRAM as Q4_0");
    if (!d->input_norm || !d->query_norm || !d->key_norm || !d->post_attention_norm ||
        !d->ffn_norm || !d->post_ffw_norm_1 || !d->pre_ffw_norm_2 || !d->post_ffw_norm_2 ||
        !d->post_ffw_norm || !d->router_input_scale || !d->router_projection ||
        (d->rope_factors && d->rope_factor_count != d->head_dim / 2))
        return fail("missing layer weights");
    if (cudaSetDevice(g_ctx->device) != cudaSuccess) return fail("cannot select the CUDA device");
    gemma4_gpu_layer *layer = static_cast<gemma4_gpu_layer *>(std::calloc(1, sizeof(*layer)));
    if (!layer) return fail("out of memory");
    layer->model_width = d->model_width; layer->query_heads = d->query_heads;
    layer->kv_heads = d->kv_heads; layer->head_dim = d->head_dim;
    layer->query_width = query_width; layer->kv_width = kv_width;
    layer->capacity = d->kv_capacity; layer->intermediate = d->intermediate_width;
    layer->experts = d->expert_count; layer->selected = d->selected_count;
    layer->sliding = d->sliding; layer->key_equals_value = d->key_equals_value;
    layer->epsilon = d->epsilon; layer->rope_base = d->rope_base;
    layer->output_scale = d->layer_output_scale;
    layer->query = static_cast<gemma4_cuda_matrix *>(d->query);
    layer->key = static_cast<gemma4_cuda_matrix *>(d->key);
    layer->value = static_cast<gemma4_cuda_matrix *>(d->value);
    layer->output = static_cast<gemma4_cuda_matrix *>(d->output);
    layer->gate = static_cast<gemma4_cuda_matrix *>(d->gate);
    layer->up = static_cast<gemma4_cuda_matrix *>(d->up);
    layer->down = static_cast<gemma4_cuda_matrix *>(d->down);
    const size_t mw = d->model_width;
    const float root_scale = 1.0F / std::sqrt(static_cast<float>(d->model_width));
    layer->input_norm = upload_floats(d->input_norm, mw);
    layer->query_norm = upload_floats(d->query_norm, d->head_dim);
    layer->key_norm = upload_floats(d->key_norm, d->head_dim);
    layer->rope_factors = d->rope_factors ? upload_floats(d->rope_factors, d->head_dim / 2) : nullptr;
    layer->post_attention_norm = upload_floats(d->post_attention_norm, mw);
    layer->ffn_norm = upload_floats(d->ffn_norm, mw);
    layer->post_ffw_norm_1 = upload_floats(d->post_ffw_norm_1, mw);
    layer->pre_ffw_norm_2 = upload_floats(d->pre_ffw_norm_2, mw);
    layer->post_ffw_norm_2 = upload_floats(d->post_ffw_norm_2, mw);
    layer->post_ffw_norm = upload_floats(d->post_ffw_norm, mw);
    layer->router_scale = upload_floats(d->router_input_scale, mw, root_scale);
    layer->router_projection = upload_floats(d->router_projection, mw * d->expert_count);
    const bool small_ok = layer->input_norm && layer->query_norm && layer->key_norm &&
        (!d->rope_factors || layer->rope_factors) && layer->post_attention_norm &&
        layer->ffn_norm && layer->post_ffw_norm_1 && layer->pre_ffw_norm_2 &&
        layer->post_ffw_norm_2 && layer->post_ffw_norm && layer->router_scale &&
        layer->router_projection;
    const size_t cache_bytes = static_cast<size_t>(d->kv_capacity) * kv_width * sizeof(float);
    if (!small_ok || cudaMalloc(&layer->key_cache, cache_bytes) != cudaSuccess ||
        cudaMalloc(&layer->value_cache, cache_bytes) != cudaSuccess ||
        !scratch_reserve(d->model_width, query_width, kv_width, d->intermediate_width,
                         d->expert_count, static_cast<size_t>(d->query_heads) * d->kv_capacity)) {
        cudaGetLastError();
        layer_free(layer);
        return fail("not enough VRAM for the device layer (KV cache or scratch)");
    }
    return layer;
}

extern "C" void coli_gemma4_cuda_layer_destroy(void *opaque) {
    if (g_ctx) cudaSetDevice(g_ctx->device);
    layer_free(static_cast<gemma4_gpu_layer *>(opaque));
}

static void launch_matvec(const gemma4_cuda_matrix *m, const float *input, float *output) {
    q4_0_matvec<<<grid_for(m->rows), kWarpsPerBlock * 32U>>>(
        m->data, m->rows, m->columns, input, 1.0F, 0.0F, output);
}

extern "C" int coli_gemma4_cuda_layer_kv_store(void *opaque, uint32_t slot,
                                                const float *key, const float *value) {
    gemma4_gpu_layer *layer = static_cast<gemma4_gpu_layer *>(opaque);
    if (!layer || !g_ctx || slot >= layer->capacity) return -1;
    const size_t bytes = static_cast<size_t>(layer->kv_width) * sizeof(float);
    if (cudaSetDevice(g_ctx->device) != cudaSuccess ||
        cudaMemcpy(layer->key_cache + static_cast<size_t>(slot) * layer->kv_width, key, bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(layer->value_cache + static_cast<size_t>(slot) * layer->kv_width, value,
                   bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaGetLastError();
        return -1;
    }
    return 0;
}

extern "C" int coli_gemma4_cuda_layer_attend(
    void *opaque, uint64_t position, uint64_t first_position, uint32_t history_count,
    const float *residual, uint32_t *ids, float *weights, float *key, float *value) {
    gemma4_gpu_layer *L = static_cast<gemma4_gpu_layer *>(opaque);
    gemma4_cuda_context *ctx = g_ctx;
    gemma4_gpu_scratch &S = g_scratch;
    if (!L || !ctx || !residual || !ids || !weights || !key || !value || !history_count ||
        history_count > L->capacity || position > 0x7FFFFFFFULL) return -2;
    if (!L->sliding && position >= L->capacity) return -2;
    if (!ctx->configured) return -2;
    ensure_slab(ctx);
    if (!ctx->slots || !S.res) return -2;
    if (cudaSetDevice(ctx->device) != cudaSuccess) return -1;
    const int mw = static_cast<int>(L->model_width), hd = static_cast<int>(L->head_dim);
    const int slot = L->sliding ? static_cast<int>(position % L->capacity)
                                : static_cast<int>(position);
    cudaError_t result = cudaMemcpyAsync(S.res, residual, sizeof(float) * mw,
                                         cudaMemcpyHostToDevice, 0);
    if (result != cudaSuccess) { cudaGetLastError(); return -1; }
    k_rms_norm<<<1, 256>>>(S.res, L->input_norm, nullptr, S.norm, mw, L->epsilon, 1.0F);
    launch_matvec(L->query, S.norm, S.q);
    launch_matvec(L->key, S.norm, S.k);
    if (L->key_equals_value)
        cudaMemcpyAsync(S.v, S.k, sizeof(float) * L->kv_width, cudaMemcpyDeviceToDevice, 0);
    else
        launch_matvec(L->value, S.norm, S.v);
    k_rms_norm<<<L->query_heads, 256>>>(S.q, L->query_norm, nullptr, S.q, hd, L->epsilon, 1.0F);
    k_rms_norm<<<L->kv_heads, 256>>>(S.k, L->key_norm, nullptr, S.k, hd, L->epsilon, 1.0F);
    k_rms_norm<<<L->kv_heads, 256>>>(S.v, nullptr, nullptr, S.v, hd, L->epsilon, 1.0F);
    k_rope_neox<<<(L->query_heads * (hd / 2) + 127) / 128, 128>>>(
        S.q, L->query_heads, hd, static_cast<int>(position), L->rope_base, L->rope_factors);
    k_rope_neox<<<(L->kv_heads * (hd / 2) + 127) / 128, 128>>>(
        S.k, L->kv_heads, hd, static_cast<int>(position), L->rope_base, L->rope_factors);
    k_kv_store<<<(L->kv_width + 255) / 256, 256>>>(
        L->key_cache, L->value_cache, slot, S.k, S.v, static_cast<int>(L->kv_width), S.kv_export);
    k_attention<<<L->query_heads, 256>>>(
        S.q, L->key_cache, L->value_cache, S.context, S.scores, hd,
        static_cast<int>(L->query_heads / L->kv_heads), static_cast<int>(L->kv_width),
        static_cast<int>(L->capacity), L->sliding, first_position,
        static_cast<int>(history_count), static_cast<int>(L->capacity));
    launch_matvec(L->output, S.context, S.attn);
    k_rms_norm<<<1, 256>>>(S.attn, L->post_attention_norm, S.res, S.after, mw, L->epsilon, 1.0F);
    k_rms_norm<<<1, 256>>>(S.after, L->ffn_norm, nullptr, S.dense_in, mw, L->epsilon, 1.0F);
    k_rms_norm<<<1, 256>>>(S.after, L->router_scale, nullptr, S.router_norm, mw, L->epsilon, 1.0F);
    k_router_logits<<<L->experts, 128>>>(L->router_projection, S.router_norm, mw, S.router_logits);
#define G4_TOPK(N) case N: k_topk_softmax<N><<<1, 32>>>(S.router_logits, S.router_weights, \
        S.router_ids, L->selected, 0.0F); break;
    switch (L->experts) {
    G4_TOPK(2) G4_TOPK(4) G4_TOPK(8) G4_TOPK(16) G4_TOPK(32) G4_TOPK(64) G4_TOPK(128)
    default: k_topk_softmax<256><<<1, 32>>>(S.router_logits, S.router_weights,
                                            S.router_ids, L->selected, 0.0F); break;
    }
#undef G4_TOPK
    k_rms_norm<<<1, 256>>>(S.after, L->pre_ffw_norm_2, nullptr, ctx->ex_in, mw, L->epsilon, 1.0F);
    result = cudaGetLastError();
    if (result == cudaSuccess)
        result = cudaMemcpyAsync(S.host_ids, S.router_ids, sizeof(int32_t) * L->selected,
                                 cudaMemcpyDeviceToHost, 0);
    if (result == cudaSuccess)
        result = cudaMemcpyAsync(S.host_weights, S.router_weights, sizeof(float) * L->selected,
                                 cudaMemcpyDeviceToHost, 0);
    if (result == cudaSuccess)
        result = cudaMemcpyAsync(S.host_kv, S.kv_export, sizeof(float) * 2 * L->kv_width,
                                 cudaMemcpyDeviceToHost, 0);
    if (result == cudaSuccess) result = cudaStreamSynchronize(0);
    if (result != cudaSuccess) {
        set_error(nullptr, 0, "device attention failed", result);
        return -1;
    }
    for (uint32_t i = 0; i < L->selected; ++i) {
        if (S.host_ids[i] < 0 || static_cast<uint32_t>(S.host_ids[i]) >= L->experts) {
            std::snprintf(g_error, sizeof(g_error), "device router produced an invalid expert id");
            return -1;
        }
        ids[i] = static_cast<uint32_t>(S.host_ids[i]);
        weights[i] = S.host_weights[i];
    }
    std::memcpy(key, S.host_kv, sizeof(float) * L->kv_width);
    std::memcpy(value, S.host_kv + L->kv_width, sizeof(float) * L->kv_width);
    return 0;
}

extern "C" int coli_gemma4_cuda_layer_dense(void *opaque) {
    gemma4_gpu_layer *L = static_cast<gemma4_gpu_layer *>(opaque);
    gemma4_gpu_scratch &S = g_scratch;
    if (!L || !g_ctx) return -1;
    const int mw = static_cast<int>(L->model_width), inter = static_cast<int>(L->intermediate);
    launch_matvec(L->gate, S.dense_in, S.hidden_gate);
    launch_matvec(L->up, S.dense_in, S.hidden_up);
    k_gelu_mul<<<(inter + 255) / 256, 256>>>(S.hidden_gate, S.hidden_up, inter, S.hidden);
    launch_matvec(L->down, S.hidden, S.dense_out);
    k_rms_norm<<<1, 256>>>(S.dense_out, L->post_ffw_norm_1, nullptr, S.dense_norm, mw,
                           L->epsilon, 1.0F);
    const cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) {
        set_error(nullptr, 0, "device dense MLP failed", result);
        return -1;
    }
    return 0;
}

extern "C" int coli_gemma4_cuda_layer_complete(void *opaque, float *output) {
    gemma4_gpu_layer *L = static_cast<gemma4_gpu_layer *>(opaque);
    gemma4_cuda_context *ctx = g_ctx;
    gemma4_gpu_scratch &S = g_scratch;
    if (!L || !ctx || !output) return -1;
    const int mw = static_cast<int>(L->model_width);
    k_rms_norm<<<1, 256>>>(ctx->ex_out, L->post_ffw_norm_2, S.dense_norm, S.combined, mw,
                           L->epsilon, 1.0F);
    k_rms_norm<<<1, 256>>>(S.combined, L->post_ffw_norm, S.after, S.out, mw, L->epsilon,
                           L->output_scale);
    cudaError_t result = cudaGetLastError();
    if (result == cudaSuccess)
        result = cudaMemcpy(output, S.out, sizeof(float) * mw, cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
        set_error(nullptr, 0, "device layer finish failed", result);
        return -1;
    }
    return 0;
}

extern "C" int coli_gemma4_cuda_layer_begin_resident(void *opaque, char *error,
                                                      size_t error_capacity) {
    gemma4_cuda_context *context = static_cast<gemma4_cuda_context *>(opaque);
    if (!context || !context->configured) return -1;
    ensure_slab(context);
    if (!context->slots) return -1;
    cudaError_t result = cudaSetDevice(context->device);
    if (result == cudaSuccess) {
        clear_floats<<<(context->model_width + 255U) / 256U, 256>>>(context->ex_out,
                                                                    context->model_width);
        result = cudaGetLastError();
    }
    if (result != cudaSuccess) {
        set_error(error, error_capacity, "CUDA layer begin failed", result);
        return -1;
    }
    return 0;
}
