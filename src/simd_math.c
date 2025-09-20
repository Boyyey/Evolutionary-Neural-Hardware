#include <immintrin.h>
#include <math.h>
#include <string.h>
#include "../include/simd_math.h"

/* Helper functions */
static inline int is_aligned(const void* ptr, size_t alignment) {
    return ((uintptr_t)ptr & (alignment - 1)) == 0;
}

/* Vector addition: dst = a + b */
void simd_vector_add_f32(float* dst, const float* a, const float* b, size_t count) {
    size_t i = 0;
    
    /* Process 8 elements at a time with AVX */
    #ifdef __AVX__
    if (is_aligned(a, 32) && is_aligned(b, 32) && is_aligned(dst, 32)) {
        for (; i + 7 < count; i += 8) {
            __m256 va = _mm256_load_ps(a + i);
            __m256 vb = _mm256_load_ps(b + i);
            __m256 vc = _mm256_add_ps(va, vb);
            _mm256_store_ps(dst + i, vc);
        }
    } else {
        for (; i + 7 < count; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vc = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(dst + i, vc);
        }
    }
    #endif
    
    /* Process 4 elements at a time with SSE */
    #ifdef __SSE__
    for (; i + 3 < count; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 vc = _mm_add_ps(va, vb);
        _mm_storeu_ps(dst + i, vc);
    }
    #endif
    
    /* Process remaining elements */
    for (; i < count; i++) {
        dst[i] = a[i] + b[i];
    }
}

/* Vector multiplication: dst = a * b */
void simd_vector_mul_f32(float* dst, const float* a, const float* b, size_t count) {
    size_t i = 0;
    
    /* Process 8 elements at a time with AVX */
    #ifdef __AVX__
    if (is_aligned(a, 32) && is_aligned(b, 32) && is_aligned(dst, 32)) {
        for (; i + 7 < count; i += 8) {
            __m256 va = _mm256_load_ps(a + i);
            __m256 vb = _mm256_load_ps(b + i);
            __m256 vc = _mm256_mul_ps(va, vb);
            _mm256_store_ps(dst + i, vc);
        }
    } else {
        for (; i + 7 < count; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vc = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(dst + i, vc);
        }
    }
    #endif
    
    /* Process 4 elements at a time with SSE */
    #ifdef __SSE__
    for (; i + 3 < count; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 vc = _mm_mul_ps(va, vb);
        _mm_storeu_ps(dst + i, vc);
    }
    #endif
    
    /* Process remaining elements */
    for (; i < count; i++) {
        dst[i] = a[i] * b[i];
    }
}

/* Fused multiply-add: dst = a + b * c */
void simd_vector_add_mul_f32(float* dst, const float* a, const float* b, float c, size_t count) {
    __m256 vc = _mm256_set1_ps(c);
    size_t i = 0;
    
    /* Process 8 elements at a time with AVX */
    #ifdef __AVX__
    if (is_aligned(a, 32) && is_aligned(b, 32) && is_aligned(dst, 32)) {
        for (; i + 7 < count; i += 8) {
            __m256 va = _mm256_load_ps(a + i);
            __m256 vb = _mm256_load_ps(b + i);
            __m256 vd = _mm256_fmadd_ps(vb, vc, va);
            _mm256_store_ps(dst + i, vd);
        }
    } else {
        for (; i + 7 < count; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vd = _mm256_fmadd_ps(vb, vc, va);
            _mm256_storeu_ps(dst + i, vd);
        }
    }
    #endif
    
    /* Process remaining elements */
    for (; i < count; i++) {
        dst[i] = a[i] + b[i] * c;
    }
}

/* Matrix-vector multiplication: dst = matrix * vector */
void simd_matrix_vector_mul_f32(float* dst, const float* matrix, const float* vector, 
                               size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        const float* row = matrix + i * cols;
        float sum = 0.0f;
        size_t j = 0;
        
        /* Process 8 elements at a time with AVX */
        #ifdef __AVX__
        __m256 vsum = _mm256_setzero_ps();
        for (; j + 7 < cols; j += 8) {
            __m256 v = _mm256_loadu_ps(vector + j);
            __m256 m = _mm256_loadu_ps(row + j);
            vsum = _mm256_fmadd_ps(m, v, vsum);
        }
        /* Horizontal sum of vsum */
        __m128 vlow = _mm256_castps256_ps128(vsum);
        __m128 vhigh = _mm256_extractf128_ps(vsum, 1);
        vlow = _mm_add_ps(vlow, vhigh);
        __m128 shuf = _mm_movehdup_ps(vlow);
        __m128 sums = _mm_add_ps(vlow, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        sum = _mm_cvtss_f32(sums);
        #endif
        
        /* Process remaining elements */
        for (; j < cols; j++) {
            sum += row[j] * vector[j];
        }
        
        dst[i] = sum;
    }
}

/* Activation functions */
void simd_sigmoid_f32(float* dst, const float* src, size_t count) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    size_t i = 0;
    
    /* Process 8 elements at a time with AVX */
    #ifdef __AVX__
    for (; i + 7 < count; i += 8) {
        __m256 x = _mm256_loadu_ps(src + i);
        x = _mm256_max_ps(x, _mm256_set1_ps(-100.0f));  // Avoid underflow
        x = _mm256_min_ps(x, _mm256_set1_ps(100.0f));   // Avoid overflow
        __m256 exp_x = _mm256_exp_ps(x);
        __m256 result = _mm256_div_ps(one, _mm256_add_ps(one, exp_x));
        _mm256_storeu_ps(dst + i, result);
    }
    #endif
    
    /* Process remaining elements */
    for (; i < count; i++) {
        float x = src[i];
        x = (x < -100.0f) ? -100.0f : (x > 100.0f) ? 100.0f : x;  // Clamp to avoid overflow
        dst[i] = 1.0f / (1.0f + expf(-x));
    }
}

void simd_tanh_f32(float* dst, const float* src, size_t count) {
    const __m256 two = _mm256_set1_ps(2.0f);
    const __m256 one = _mm256_set1_ps(1.0f);
    size_t i = 0;
    
    /* Process 8 elements at a time with AVX */
    #ifdef __AVX__
    for (; i + 7 < count; i += 8) {
        __m256 x = _mm256_loadu_ps(src + i);
        x = _mm256_max_ps(x, _mm256_set1_ps(-100.0f));  // Avoid underflow
        x = _mm256_min_ps(x, _mm256_set1_ps(100.0f));   // Avoid overflow
        __m256 exp_2x = _mm256_exp_ps(_mm256_mul_ps(two, x));
        __m256 result = _mm256_div_ps(
            _mm256_sub_ps(exp_2x, one),
            _mm256_add_ps(exp_2x, one)
        );
        _mm256_storeu_ps(dst + i, result);
    }
    #endif
    
    /* Process remaining elements */
    for (; i < count; i++) {
        float x = src[i];
        x = (x < -100.0f) ? -100.0f : (x > 100.0f) ? 100.0f : x;  // Clamp to avoid overflow
        float exp_2x = expf(2.0f * x);
        dst[i] = (exp_2x - 1.0f) / (exp_2x + 1.0f);
    }
}

void simd_relu_f32(float* dst, const float* src, size_t count) {
    const __m256 zero = _mm256_setzero_ps();
    size_t i = 0;
    
    /* Process 8 elements at a time with AVX */
    #ifdef __AVX__
    for (; i + 7 < count; i += 8) {
        __m256 x = _mm256_loadu_ps(src + i);
        __m256 result = _mm256_max_ps(x, zero);
        _mm256_storeu_ps(dst + i, result);
    }
    #endif
    
    /* Process remaining elements */
    for (; i < count; i++) {
        dst[i] = (src[i] > 0.0f) ? src[i] : 0.0f;
    }
}

/* Generic activation function */
void simd_activate_f32(float* dst, const float* src, int activation, size_t count) {
    switch (activation) {
        case NEAT_ACTIVATION_SIGMOID:
            simd_sigmoid_f32(dst, src, count);
            break;
        case NEAT_ACTIVATION_TANH:
            simd_tanh_f32(dst, src, count);
            break;
        case NEAT_ACTIVATION_RELU:
            simd_relu_f32(dst, src, count);
            break;
        case NEAT_ACTIVATION_LINEAR:
            if (dst != src) {
                memcpy(dst, src, count * sizeof(float));
            }
            break;
        default:
            /* Default to ReLU for unknown activation */
            simd_relu_f32(dst, src, count);
            break;
    }
}

/* Random number generation */
void simd_rand_fill_f32(float* dst, float min, float max, size_t count) {
    const float scale = (max - min) / (float)RAND_MAX;
    for (size_t i = 0; i < count; i++) {
        dst[i] = min + scale * (float)rand();
    }
}

/* CPU feature detection */
int simd_supports_avx() {
    #ifdef __AVX__
    return 1;
    #else
    return 0;
    #endif
}

int simd_supports_avx2() {
    #ifdef __AVX2__
    return 1;
    #else
    return 0;
    #endif
}

int simd_supports_avx512() {
    #ifdef __AVX512F__
    return 1;
    #else
    return 0;
    #endif
}

/* Vector reduction operations */
float simd_vector_sum_f32(const float* src, size_t count) {
    float sum = 0.0f;
    size_t i = 0;
    
    /* Process 8 elements at a time with AVX */
    #ifdef __AVX__
    __m256 vsum = _mm256_setzero_ps();
    for (; i + 7 < count; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        vsum = _mm256_add_ps(vsum, v);
    }
    /* Horizontal sum of vsum */
    __m128 vlow = _mm256_castps256_ps128(vsum);
    __m128 vhigh = _mm256_extractf128_ps(vsum, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    sum = _mm_cvtss_f32(sums);
    #endif
    
    /* Process remaining elements */
    for (; i < count; i++) {
        sum += src[i];
    }
    
    return sum;
}

/* Vector dot product */
float simd_vector_dot_f32(const float* a, const float* b, size_t count) {
    float sum = 0.0f;
    size_t i = 0;
    
    /* Process 8 elements at a time with AVX */
    #ifdef __AVX__
    __m256 vsum = _mm256_setzero_ps();
    for (; i + 7 < count; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        vsum = _mm256_fmadd_ps(va, vb, vsum);
    }
    /* Horizontal sum of vsum */
    __m128 vlow = _mm256_castps256_ps128(vsum);
    __m128 vhigh = _mm256_extractf128_ps(vsum, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    sum = _mm_cvtss_f32(sums);
    #endif
    
    /* Process remaining elements */
    for (; i < count; i++) {
        sum += a[i] * b[i];
    }
    
    return sum;
}

/* Vector normalization */
void simd_normalize_l2_f32(float* dst, const float* src, size_t count) {
    float norm = sqrtf(simd_vector_dot_f32(src, src, count));
    if (norm > 1e-10f) {
        float inv_norm = 1.0f / norm;
        simd_vector_mul_scalar_f32(dst, src, inv_norm, count);
    } else {
        memcpy(dst, src, count * sizeof(float));
    }
}

/* Vector-scalar operations */
void simd_vector_add_scalar_f32(float* dst, const float* src, float scalar, size_t count) {
    __m256 vscalar = _mm256_set1_ps(scalar);
    size_t i = 0;
    
    /* Process 8 elements at a time with AVX */
    #ifdef __AVX__
    for (; i + 7 < count; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m256 result = _mm256_add_ps(v, vscalar);
        _mm256_storeu_ps(dst + i, result);
    }
    #endif
    
    /* Process remaining elements */
    for (; i < count; i++) {
        dst[i] = src[i] + scalar;
    }
}

void simd_vector_mul_scalar_f32(float* dst, const float* src, float scalar, size_t count) {
    __m256 vscalar = _mm256_set1_ps(scalar);
    size_t i = 0;
    
    /* Process 8 elements at a time with AVX */
    #ifdef __AVX__
    for (; i + 7 < count; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m256 result = _mm256_mul_ps(v, vscalar);
        _mm256_storeu_ps(dst + i, result);
    }
    #endif
    
    /* Process remaining elements */
    for (; i < count; i++) {
        dst[i] = src[i] * scalar;
    }
}

/* Vector initialization */
void simd_vector_set_f32(float* dst, float value, size_t count) {
    __m256 vvalue = _mm256_set1_ps(value);
    size_t i = 0;
    
    /* Process 8 elements at a time with AVX */
    #ifdef __AVX__
    for (; i + 7 < count; i += 8) {
        _mm256_storeu_ps(dst + i, vvalue);
    }
    #endif
    
    /* Process remaining elements */
    for (; i < count; i++) {
        dst[i] = value;
    }
}

void simd_vector_zero_f32(float* dst, size_t count) {
    simd_vector_set_f32(dst, 0.0f, count);
}

/* Vector copy */
void simd_vector_copy_f32(float* dst, const float* src, size_t count) {
    /* Use memcpy for large arrays */
    if (count > 64) {
        memcpy(dst, src, count * sizeof(float));
        return;
    }
    
    /* For small arrays, use SIMD */
    size_t i = 0;
    
    /* Process 8 elements at a time with AVX */
    #ifdef __AVX__
    for (; i + 7 < count; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, v);
    }
    #endif
    
    /* Process remaining elements */
    for (; i < count; i++) {
        dst[i] = src[i];
    }
}

/* Vector statistics */
void simd_vector_mean_stddev_f32(float* mean, float* stddev, const float* src, size_t count) {
    if (count == 0) {
        *mean = 0.0f;
        *stddev = 0.0f;
        return;
    }
    
    /* Calculate mean */
    *mean = simd_vector_sum_f32(src, count) / (float)count;
    
    /* Calculate variance */
    float variance = 0.0f;
    size_t i = 0;
    
    /* Process 8 elements at a time with AVX */
    #ifdef __AVX__
    __m256 vmean = _mm256_set1_ps(*mean);
    __m256 vsum = _mm256_setzero_ps();
    
    for (; i + 7 < count; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m256 diff = _mm256_sub_ps(v, vmean);
        vsum = _mm256_fmadd_ps(diff, diff, vsum);
    }
    
    /* Horizontal sum of vsum */
    __m128 vlow = _mm256_castps256_ps128(vsum);
    __m128 vhigh = _mm256_extractf128_ps(vsum, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    variance = _mm_cvtss_f32(sums);
    #endif
    
    /* Process remaining elements */
    for (; i < count; i++) {
        float diff = src[i] - *mean;
        variance += diff * diff;
    }
    
    /* Calculate standard deviation */
    variance /= (float)count;
    *stddev = sqrtf(variance);
}
