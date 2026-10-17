#include "gemma4_cuda.h"

#include <stdio.h>
#include <stdlib.h>

static void not_built(char *error, size_t capacity) {
    if (error && capacity)
        snprintf(error, capacity, "Gemma CUDA backend was not built");
}

int coli_gemma4_cuda_create(void **context, int device,
                            char *error, size_t error_capacity) {
    (void)device;
    if (context) *context = NULL;
    not_built(error, error_capacity);
    return -1;
}

void coli_gemma4_cuda_destroy(void *context) { (void)context; }

int coli_gemma4_cuda_run(void *context, const uint8_t *payload,
                         size_t payload_bytes, size_t gate_bytes,
                         size_t up_bytes, uint32_t model_width,
                         uint32_t expert_width, const float *input,
                         float scale, float *output,
                         char *error, size_t error_capacity) {
    (void)context; (void)payload; (void)payload_bytes; (void)gate_bytes;
    (void)up_bytes; (void)model_width; (void)expert_width; (void)input;
    (void)scale; (void)output;
    not_built(error, error_capacity);
    return -1;
}

int coli_gemma4_cuda_dense_enabled(void) { return 0; }

void *coli_gemma4_cuda_matrix_upload(const uint8_t *data, size_t bytes, int type,
                                     uint32_t rows, uint32_t columns) {
    (void)data; (void)bytes; (void)type; (void)rows; (void)columns;
    return NULL;
}

void coli_gemma4_cuda_matrix_free(void *matrix) { (void)matrix; }

int coli_gemma4_cuda_matrix_matvec(void *matrix, const float *input, float *output) {
    (void)matrix; (void)input; (void)output;
    return -1;
}

size_t coli_gemma4_cuda_dense_bytes(void) { return 0; }

const char *coli_gemma4_cuda_last_error(void) { return "Gemma CUDA backend was not built"; }

int coli_gemma4_cuda_expert_cache_configure(
    void *context, uint32_t layers, uint32_t experts_per_layer,
    size_t record_bytes, uint32_t model_width, uint32_t expert_width,
    size_t budget_bytes, char *error, size_t error_capacity) {
    (void)context; (void)layers; (void)experts_per_layer; (void)record_bytes;
    (void)model_width; (void)expert_width; (void)budget_bytes;
    not_built(error, error_capacity);
    return -1;
}

uint32_t coli_gemma4_cuda_expert_cache_slots(void *context) {
    (void)context;
    return 0;
}

int coli_gemma4_cuda_expert_resident(void *context, uint32_t layer, uint32_t expert) {
    (void)context; (void)layer; (void)expert;
    return 0;
}

int coli_gemma4_cuda_layer_begin(void *context, const float *input,
                                 char *error, size_t error_capacity) {
    (void)context; (void)input;
    not_built(error, error_capacity);
    return -1;
}

int coli_gemma4_cuda_expert_launch(
    void *context, uint32_t layer, uint32_t expert, const uint8_t *payload,
    size_t gate_bytes, size_t up_bytes, float scale, float weight,
    char *error, size_t error_capacity) {
    (void)context; (void)layer; (void)expert; (void)payload; (void)gate_bytes;
    (void)up_bytes; (void)scale; (void)weight;
    not_built(error, error_capacity);
    return -1;
}

int coli_gemma4_cuda_layer_finish(void *context, float *output,
                                  char *error, size_t error_capacity) {
    (void)context; (void)output;
    not_built(error, error_capacity);
    return -1;
}

void coli_gemma4_cuda_expert_cache_stats(
    void *context, uint32_t *slots, uint32_t *resident, uint64_t *hits,
    uint64_t *misses, uint64_t *evictions) {
    (void)context;
    if (slots) *slots = 0;
    if (resident) *resident = 0;
    if (hits) *hits = 0;
    if (misses) *misses = 0;
    if (evictions) *evictions = 0;
}

void *coli_gemma4_cuda_host_alloc(size_t bytes) { return malloc(bytes); }

void coli_gemma4_cuda_host_free(void *pointer) { free(pointer); }

void *coli_gemma4_cuda_layer_create(const coli_gemma4_cuda_layer_desc *desc,
                                    char *error, size_t error_capacity) {
    (void)desc;
    not_built(error, error_capacity);
    return NULL;
}

void coli_gemma4_cuda_layer_destroy(void *layer) { (void)layer; }

int coli_gemma4_cuda_layer_attend(void *layer, uint64_t position,
                                  uint64_t first_position, uint32_t history_count,
                                  const float *residual, uint32_t *ids, float *weights,
                                  float *key, float *value) {
    (void)layer; (void)position; (void)first_position; (void)history_count;
    (void)residual; (void)ids; (void)weights; (void)key; (void)value;
    return -2;
}

int coli_gemma4_cuda_layer_kv_store(void *layer, uint32_t slot,
                                    const float *key, const float *value) {
    (void)layer; (void)slot; (void)key; (void)value;
    return 0;
}

int coli_gemma4_cuda_layer_dense(void *layer) { (void)layer; return -1; }

int coli_gemma4_cuda_layer_complete(void *layer, float *output) {
    (void)layer; (void)output;
    return -1;
}

int coli_gemma4_cuda_layer_begin_resident(void *context, char *error,
                                          size_t error_capacity) {
    (void)context;
    not_built(error, error_capacity);
    return -1;
}
