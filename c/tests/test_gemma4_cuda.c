/* GPU-vs-CPU checks for the Gemma 4 CUDA tier. Build with CUDA=1:
 *   make CUDA=1 tests/test_gemma4_cuda && ./tests/test_gemma4_cuda
 * Exits 0 with a SKIP line when no CUDA device is available. */
#define _CRT_SECURE_NO_WARNINGS

#include "../gemma4_backend.h"
#include "../gemma4_cuda.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rng_state = 20260717u;

static uint32_t next_random(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state >> 8;
}

static float next_float(void) { return (float)(next_random() % 2001u) / 1000.0F - 1.0F; }

static void fill_bytes(uint8_t *bytes, size_t count) {
    size_t i;
    for (i = 0; i < count; ++i) bytes[i] = (uint8_t)next_random();
}

/* fp16 scale in [0.0005, 0.05]: always finite and positive. */
static void put_scale(uint8_t *where) {
    uint16_t bits = (uint16_t)(0x1000u + next_random() % 0x1c00u);
    where[0] = (uint8_t)bits;
    where[1] = (uint8_t)(bits >> 8);
}

static void fill_q4_0(uint8_t *data, size_t rows, size_t columns) {
    size_t blocks = rows * (columns / 32), i;
    fill_bytes(data, blocks * 18);
    for (i = 0; i < blocks; ++i) put_scale(data + i * 18);
}

static void fill_q6_k(uint8_t *data, size_t rows, size_t columns) {
    size_t blocks = rows * (columns / 256), i;
    fill_bytes(data, blocks * 210);
    for (i = 0; i < blocks; ++i) put_scale(data + i * 210 + 208);
}

static int close_enough(const float *got, const float *want, size_t count,
                        const char *label) {
    double worst = 0.0, norm = 0.0;
    size_t i, worst_index = 0;
    for (i = 0; i < count; ++i) norm += (double)want[i] * (double)want[i];
    norm = sqrt(norm / (double)count) + 1e-12;
    for (i = 0; i < count; ++i) {
        double difference = fabs((double)got[i] - (double)want[i]) / norm;
        if (!isfinite(got[i])) difference = 1e30;
        if (difference > worst) { worst = difference; worst_index = i; }
    }
    printf("  %-34s worst error %.3g x rms (element %zu)\n", label, worst, worst_index);
    return worst < 2e-4;
}

static int test_dense(int type, size_t rows, size_t columns, const char *label) {
    size_t block_elems = type == COLI_GEMMA4_CUDA_Q6_K ? 256 : 32;
    size_t block_bytes = type == COLI_GEMMA4_CUDA_Q6_K ? 210 : 18;
    size_t bytes = rows * (columns / block_elems) * block_bytes, i;
    uint8_t *weights = (uint8_t *)malloc(bytes);
    float *input = (float *)malloc(columns * sizeof(float));
    float *want = (float *)malloc(rows * sizeof(float));
    float *got = (float *)malloc(rows * sizeof(float));
    void *matrix;
    int ok = 0;
    if (!weights || !input || !want || !got) goto done;
    if (type == COLI_GEMMA4_CUDA_Q6_K) fill_q6_k(weights, rows, columns);
    else fill_q4_0(weights, rows, columns);
    for (i = 0; i < columns; ++i) input[i] = next_float();
    if ((type == COLI_GEMMA4_CUDA_Q6_K
             ? coli_gemma4_q6_k_matvec(weights, rows, columns, input, want)
             : coli_gemma4_q4_0_matvec(weights, rows, columns, input, want)) != 0) goto done;
    matrix = coli_gemma4_cuda_matrix_upload(weights, bytes, type, (uint32_t)rows,
                                            (uint32_t)columns);
    if (!matrix) { printf("  %s: upload failed: %s\n", label, coli_gemma4_cuda_last_error()); goto done; }
    memset(got, 0, rows * sizeof(float));
    if (coli_gemma4_cuda_matrix_matvec(matrix, input, got) != 0) {
        printf("  %s: matvec failed: %s\n", label, coli_gemma4_cuda_last_error());
        coli_gemma4_cuda_matrix_free(matrix);
        goto done;
    }
    ok = close_enough(got, want, rows, label);
    coli_gemma4_cuda_matrix_free(matrix);
done:
    free(weights); free(input); free(want); free(got);
    return ok;
}

/* CPU reference for one expert record, mirroring g4_run_payload. */
static void expert_reference(const uint8_t *record, uint32_t model_width,
                             uint32_t expert_width, const float *input, float scale,
                             float *output) {
    size_t gate_bytes = (size_t)expert_width * (model_width / 32) * 18;
    float *gate = (float *)malloc(expert_width * sizeof(float));
    float *up = (float *)malloc(expert_width * sizeof(float));
    float *hidden = (float *)malloc(expert_width * sizeof(float));
    uint32_t i;
    coli_gemma4_q4_0_matvec(record, expert_width, model_width, input, gate);
    coli_gemma4_q4_0_matvec(record + gate_bytes, expert_width, model_width, input, up);
    for (i = 0; i < expert_width; ++i) hidden[i] = coli_gemma4_gelu_tanh(gate[i]) * up[i];
    coli_gemma4_q4_0_matvec(record + 2 * gate_bytes, model_width, expert_width, hidden, output);
    for (i = 0; i < model_width; ++i) output[i] *= scale;
    free(gate); free(up); free(hidden);
}

static int test_expert_cache(void *context) {
    enum { LAYERS = 3, EXPERTS = 40, TOP_K = 4, STEPS = 300 };
    const uint32_t model_width = 96, expert_width = 64;
    const size_t gate_bytes = (size_t)expert_width * (model_width / 32) * 18;
    const size_t down_bytes = (size_t)model_width * (expert_width / 32) * 18;
    const size_t record_bytes = 2 * gate_bytes + down_bytes;
    const size_t stride = (record_bytes + 255) / 256 * 256;
    uint8_t *records = (uint8_t *)malloc((size_t)LAYERS * EXPERTS * record_bytes);
    float scales[LAYERS * EXPERTS];
    uint32_t slots, step, i, k, launches = 0, bad = 0;
    uint64_t hits, misses, evictions;
    uint32_t resident;
    char error[256];
    int ok = 0;
    if (!records) return 0;
    for (i = 0; i < LAYERS * EXPERTS; ++i) {
        uint8_t *record = records + (size_t)i * record_bytes;
        fill_q4_0(record, expert_width, model_width);
        fill_q4_0(record + gate_bytes, expert_width, model_width);
        fill_q4_0(record + 2 * gate_bytes, model_width, expert_width);
        scales[i] = 0.5F + (float)(next_random() % 100u) / 100.0F;
    }
    /* A budget of 20 slots forces constant eviction: 120 experts compete for 20. */
    if (coli_gemma4_cuda_expert_cache_configure(context, LAYERS, EXPERTS, record_bytes,
                                                model_width, expert_width, 20 * stride,
                                                error, sizeof(error)) != 0) {
        printf("  expert cache configure failed: %s\n", error);
        goto done;
    }
    slots = coli_gemma4_cuda_expert_cache_slots(context);
    printf("  expert cache slots: %u (wanted 20)\n", slots);
    if (slots != 20) goto done;
    for (step = 0; step < STEPS; ++step) {
        uint32_t layer = next_random() % LAYERS;
        uint32_t ids[TOP_K];
        float weights[TOP_K], input[96], want[96], got[96], expert_out[96];
        for (k = 0; k < TOP_K; ++k) {
            int again;
            do {   /* distinct experts within a layer, skewed so some are hot */
                ids[k] = next_random() % 4 ? next_random() % 8 : next_random() % EXPERTS;
                again = 0;
                for (i = 0; i < k; ++i) if (ids[i] == ids[k]) again = 1;
            } while (again);
            weights[k] = 0.1F + (float)(next_random() % 50u) / 100.0F;
        }
        for (i = 0; i < model_width; ++i) input[i] = next_float();
        memset(want, 0, sizeof(want));
        for (k = 0; k < TOP_K; ++k) {
            size_t index = (size_t)layer * EXPERTS + ids[k];
            expert_reference(records + index * record_bytes, model_width, expert_width,
                             input, scales[index], expert_out);
            for (i = 0; i < model_width; ++i) want[i] += weights[k] * expert_out[i];
        }
        if (coli_gemma4_cuda_layer_begin(context, input, error, sizeof(error)) != 0) goto done;
        for (k = 0; k < TOP_K; ++k) {
            size_t index = (size_t)layer * EXPERTS + ids[k];
            /* Exactly what the backend does: payload only when the expert is not resident. */
            const uint8_t *payload = coli_gemma4_cuda_expert_resident(context, layer, ids[k])
                                         ? NULL : records + index * record_bytes;
            if (coli_gemma4_cuda_expert_launch(context, layer, ids[k], payload, gate_bytes,
                                               gate_bytes, scales[index], weights[k],
                                               error, sizeof(error)) != 0) {
                printf("  launch failed: %s\n", error);
                goto done;
            }
            ++launches;
        }
        if (coli_gemma4_cuda_layer_finish(context, got, error, sizeof(error)) != 0) goto done;
        {
            double norm = 0.0, worst = 0.0;
            for (i = 0; i < model_width; ++i) norm += (double)want[i] * want[i];
            norm = sqrt(norm / model_width) + 1e-12;
            for (i = 0; i < model_width; ++i) {
                double difference = fabs((double)got[i] - want[i]) / norm;
                if (difference > worst) worst = difference;
            }
            if (!(worst < 2e-4)) {
                if (!bad) printf("  step %u layer %u: error %.3g x rms\n", step, layer, worst);
                ++bad;
            }
        }
    }
    coli_gemma4_cuda_expert_cache_stats(context, &slots, &resident, &hits, &misses, &evictions);
    printf("  %u launches: hits=%llu misses=%llu evictions=%llu resident=%u/%u\n", launches,
           (unsigned long long)hits, (unsigned long long)misses,
           (unsigned long long)evictions, resident, slots);
    ok = bad == 0 && hits + misses == launches && hits > 0 && misses > 0 &&
         evictions > 0 && resident == slots && evictions + resident == misses;
    if (!ok && !bad) printf("  cache bookkeeping is inconsistent\n");
done:
    free(records);
    return ok;
}

int main(void) {
    void *context = NULL;
    char error[256] = "";
    int failures = 0;
    if (coli_gemma4_cuda_create(&context, 0, error, sizeof(error)) != 0) {
        printf("SKIP: no usable CUDA device (%s)\n", error);
        return 0;
    }
    printf("dense Q4_0 / Q6_K matvec vs CPU:\n");
    failures += !test_dense(COLI_GEMMA4_CUDA_Q4_0, 2112, 2816, "Q4_0 dense MLP 2816->2112");
    failures += !test_dense(COLI_GEMMA4_CUDA_Q4_0, 704, 2816, "Q4_0 expert gate 2816->704");
    failures += !test_dense(COLI_GEMMA4_CUDA_Q4_0, 2816, 704, "Q4_0 expert down 704->2816");
    failures += !test_dense(COLI_GEMMA4_CUDA_Q6_K, 4099, 2816, "Q6_K LM head slice 2816->4099");
    printf("VRAM expert cache vs CPU:\n");
    failures += !test_expert_cache(context);
    coli_gemma4_cuda_destroy(context);
    if (failures) {
        printf("FAIL: %d check(s) failed\n", failures);
        return 1;
    }
    printf("PASS\n");
    return 0;
}
