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

#ifdef __cplusplus
}
#endif

#endif
