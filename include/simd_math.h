#ifndef SIMD_MATH_H
#define SIMD_MATH_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>

/* SIMD vector types */
#if defined(__AVX2__)
    #include <immintrin.h>
    #define SIMD_ALIGNMENT 32
    typedef __m256 simd_float8_t;
    #define SIMD_WIDTH 8
#elif defined(__SSE4_2__)
    #include <immintrin.h>
    #include <smmintrin.h>
    #define SIMD_ALIGNMENT 16
    typedef __m128 simd_float4_t;
    #define SIMD_WIDTH 4
#else
    #define SIMD_ALIGNMENT 16
    #define SIMD_WIDTH 1
#endif

/* SIMD math functions */

/**
 * @brief Add two arrays using SIMD instructions
 * @param a First input array
 * @param b Second input array
 * @param result Output array
 * @param size Size of the arrays (must be a multiple of SIMD_WIDTH)
 */
void simd_add(const float* a, const float* b, float* result, size_t size);

/**
 * @brief Multiply two arrays using SIMD instructions
 * @param a First input array
 * @param b Second input array
 * @param result Output array
 * @param size Size of the arrays (must be a multiple of SIMD_WIDTH)
 */
void simd_mul(const float* a, const float* b, float* result, size_t size);

/**
 * @brief Fused multiply-add operation using SIMD
 * @param a First input array
 * @param b Second input array
 * @param c Third input array (to be added)
 * @param result Output array
 * @param size Size of the arrays (must be a multiple of SIMD_WIDTH)
 */
void simd_fma(const float* a, const float* b, const float* c, float* result, size_t size);

/**
 * @brief Compute sigmoid activation using SIMD
 * @param input Input array
 * @param output Output array
 * @param size Size of the arrays (must be a multiple of SIMD_WIDTH)
 */
void simd_sigmoid(const float* input, float* output, size_t size);

/**
 * @brief Compute ReLU activation using SIMD
 * @param input Input array
 * @param output Output array
 * @param size Size of the arrays (must be a multiple of SIMD_WIDTH)
 */
void simd_relu(const float* input, float* output, size_t size);

/**
 * @brief Compute tanh activation using SIMD
 * @param input Input array
 * @param output Output array
 * @param size Size of the arrays (must be a multiple of SIMD_WIDTH)
 */
void simd_tanh(const float* input, float* output, size_t size);

/**
 * @brief Dot product of two vectors using SIMD
 * @param a First input vector
 * @param b Second input vector
 * @param size Size of the vectors (must be a multiple of SIMD_WIDTH)
 * @return Dot product result
 */
float simd_dot(const float* a, const float* b, size_t size);

/**
 * @brief Matrix-vector multiplication using SIMD
 * @param matrix Input matrix (row-major order)
 * @param vector Input vector
 * @param result Output vector
 * @param rows Number of rows in the matrix
 * @param cols Number of columns in the matrix (must be a multiple of SIMD_WIDTH)
 */
void simd_matvec_mul(const float* matrix, const float* vector, float* result, 
                     size_t rows, size_t cols);

/**
 * @brief Matrix-matrix multiplication using SIMD
 * @param a First input matrix (row-major order)
 * @param b Second input matrix (row-major order)
 * @param result Output matrix (row-major order)
 * @param m Rows of matrix a
 * @param n Columns of matrix a / Rows of matrix b
 * @param p Columns of matrix b (must be a multiple of SIMD_WIDTH)
 */
void simd_matmul(const float* a, const float* b, float* result, 
                 size_t m, size_t n, size_t p);

#endif /* SIMD_MATH_H */
