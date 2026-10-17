#include "gemma4_cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
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
    if (cudaSetDevice(g_ctx->device) != cudaSuccess) return nullptr;
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) return nullptr;
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
    if (slots < 16) return;
    while (slots >= 16) {
        if (cudaMalloc(&context->slab, slots * context->stride) == cudaSuccess) break;
        cudaGetLastError();
        context->slab = nullptr;
        slots -= slots / 16 + 1;
    }
    if (!context->slab || slots < 16) return;
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
