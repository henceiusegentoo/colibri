#ifndef COLI_GEMMA4_CUDA_H
#define COLI_GEMMA4_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- context ------------------------------------------------------------ */
int coli_gemma4_cuda_create(void **context, int device,
                            char *error, size_t error_capacity);
void coli_gemma4_cuda_destroy(void *context);

/* One expert, one call: uploads the whole record every time. Kept for the
 * unit tests and as the fallback when the VRAM expert cache is unavailable. */
int coli_gemma4_cuda_run(void *context, const uint8_t *payload,
                         size_t payload_bytes, size_t gate_bytes,
                         size_t up_bytes, uint32_t model_width,
                         uint32_t expert_width, const float *input,
                         float scale, float *output,
                         char *error, size_t error_capacity);

/* ---- resident dense matrices ---------------------------------------------
 * Weights are uploaded once and stay in VRAM. Types: 0 = Q4_0, 1 = Q6_K. An
 * upload that does not fit the dense budget returns NULL and the caller keeps
 * using its CPU copy, so a small card degrades instead of failing.            */
#define COLI_GEMMA4_CUDA_Q4_0 0
#define COLI_GEMMA4_CUDA_Q6_K 1
int coli_gemma4_cuda_dense_enabled(void);
void *coli_gemma4_cuda_matrix_upload(const uint8_t *data, size_t bytes, int type,
                                     uint32_t rows, uint32_t columns);
void coli_gemma4_cuda_matrix_free(void *matrix);
int coli_gemma4_cuda_matrix_matvec(void *matrix, const float *input, float *output);
/* Bytes of VRAM held by uploaded dense matrices. */
size_t coli_gemma4_cuda_dense_bytes(void);
const char *coli_gemma4_cuda_last_error(void);

/* Page-locked host memory for expert staging (falls back to malloc when CUDA is
 * unavailable). Release with coli_gemma4_cuda_host_free. */
void *coli_gemma4_cuda_host_alloc(size_t bytes);
void coli_gemma4_cuda_host_free(void *pointer);

/* ---- VRAM expert cache + batched layer execution --------------------------
 * A layer is executed as:  layer_begin, then one launch per selected expert,
 * then layer_finish. Experts already resident in VRAM are launched without any
 * host data (payload == NULL); the others are copied into the least recently
 * used slot first. All experts of a layer run on one stream with a single input
 * upload and a single output download.                                         */
int coli_gemma4_cuda_expert_cache_configure(
    void *context, uint32_t layers, uint32_t experts_per_layer,
    size_t record_bytes, uint32_t model_width, uint32_t expert_width,
    size_t budget_bytes, char *error, size_t error_capacity);
/* Slots in the VRAM slab (allocates it on first call; 0 = no VRAM cache). */
uint32_t coli_gemma4_cuda_expert_cache_slots(void *context);
/* 1 if resident (and protects it from eviction for the rest of the layer). */
int coli_gemma4_cuda_expert_resident(void *context, uint32_t layer, uint32_t expert);
int coli_gemma4_cuda_layer_begin(void *context, const float *input,
                                 char *error, size_t error_capacity);
int coli_gemma4_cuda_expert_launch(
    void *context, uint32_t layer, uint32_t expert, const uint8_t *payload,
    size_t gate_bytes, size_t up_bytes, float scale, float weight,
    char *error, size_t error_capacity);
int coli_gemma4_cuda_layer_finish(void *context, float *output,
                                  char *error, size_t error_capacity);
void coli_gemma4_cuda_expert_cache_stats(
    void *context, uint32_t *slots, uint32_t *resident, uint64_t *hits,
    uint64_t *misses, uint64_t *evictions);

/* ---- whole-layer decode on the GPU (one token) ----------------------------
 * Stage 1 (attend): input norm, Q/K/V, QK norm, RoPE, KV append, attention, output
 *   projection, post-attention norm + residual, FFN norm, router softmax + top-k and
 *   the expert-input norm, all on the device. Only the selected expert ids/weights
 *   and the new token's K/V (so the host cache stays an exact mirror) come back.
 * Stage 2 (dense + experts + complete): the dense MLP and the VRAM-resident experts run
 *   while missed experts stream in, then the post-norms, combine and layer scale, and
 *   one download of the layer output.
 * Every matrix must be a resident Q4_0 matrix (uploaded with matrix_upload); the
 * device KV cache is a mirror of the host cache: host writes go through kv_store.   */
typedef struct {
    uint32_t model_width, query_heads, kv_heads, head_dim, kv_capacity;
    uint32_t intermediate_width, expert_count, selected_count;
    int sliding, key_equals_value;
    float epsilon, rope_base, layer_output_scale;
    void *query, *key, *value, *output, *gate, *up, *down;   /* value may be NULL */
    const float *input_norm, *query_norm, *key_norm, *rope_factors; /* factors may be NULL */
    uint32_t rope_factor_count;
    const float *post_attention_norm, *ffn_norm, *post_ffw_norm_1, *pre_ffw_norm_2,
                *post_ffw_norm_2, *post_ffw_norm;
    const float *router_input_scale, *router_projection;     /* projection: experts x width */
} coli_gemma4_cuda_layer_desc;

void *coli_gemma4_cuda_layer_create(const coli_gemma4_cuda_layer_desc *desc,
                                    char *error, size_t error_capacity);
void coli_gemma4_cuda_layer_destroy(void *layer);
/* 0 = done; -2 = GPU path unavailable right now (nothing was modified, use the CPU
 * path); -1 = hard error. key/value receive kv_heads * head_dim floats each. */
int coli_gemma4_cuda_layer_attend(void *layer, uint64_t position,
                                  uint64_t first_position, uint32_t history_count,
                                  const float *residual, uint32_t *ids, float *weights,
                                  float *key, float *value);
int coli_gemma4_cuda_layer_kv_store(void *layer, uint32_t slot,
                                    const float *key, const float *value);
int coli_gemma4_cuda_layer_dense(void *layer);
int coli_gemma4_cuda_layer_complete(void *layer, float *output);
/* Expert accumulation for a layer whose input is already on the device (written by
 * attend): clears the accumulator only. No upload, and no download at the end. */
int coli_gemma4_cuda_layer_begin_resident(void *context, char *error,
                                          size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif
