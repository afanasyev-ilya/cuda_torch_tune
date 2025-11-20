// Build example (adjust paths & arch as needed):
//   nvcc -O3 -std=c++17 sgemm.cu -o gemm_bench -I ~/cutlass/include -lcublas -arch=sm_86

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include <cmath>

// ----------------- Helpers & macros -----------------

#define CHECK_CUDA(call)                                               \
  do                                                                   \
  {                                                                    \
    cudaError_t status_ = (call);                                      \
    if (status_ != cudaSuccess)                                        \
    {                                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(status_)       \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
      std::exit(EXIT_FAILURE);                                         \
    }                                                                  \
  } while (0)

#define CHECK_CUBLAS(call)                                             \
  do                                                                   \
  {                                                                    \
    cublasStatus_t status_ = (call);                                   \
    if (status_ != CUBLAS_STATUS_SUCCESS)                              \
    {                                                                  \
      std::cerr << "cuBLAS error: " << status_                         \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
      std::exit(EXIT_FAILURE);                                         \
    }                                                                  \
  } while (0)

#define CEIL_DIV(a, b) ( ((a) - 1) / (b) + 1 )

// -------------------------- cublas performance reference --------------------------

void run_cublas_gemm(cublasHandle_t handle, int M, int N, int K, const float* dA, const float* dB, float* dC, int iters)
{
    // We store row-major A, B, C in C/C++.
    // cuBLAS assumes column-major. To avoid data copies, we use the well-known trick:
    //
    // Row-major C = A * B  (A[M,K], B[K,N], C[M,N])
    //
    // If we treat these same buffers as column-major, they represent the transposed
    // matrices:
    //   A_row (M x K, RM)  <->  A_col^T (K x M, CM)
    //   B_row (K x N, RM)  <->  B_col^T (N x K, CM)
    //   C_row (M x N, RM)  <->  C_col^T (N x M, CM)
    //
    // We want C_row = A_row * B_row.
    // In column-major:
    //   C_col = B_col * A_col   (no transposes), where:
    //     B_col = B_row^T (N x K)
    //     A_col = A_row^T (K x M)
    // Then:
    //   C_col = B_row^T * A_row^T  =>  C_col^T = A_row * B_row
    // And C_col^T is exactly C_row.
    //
    // So we call cublasSgemm with:
    //   m = N, n = M, k = K
    //   A = dB (B_row), lda = N
    //   B = dA (A_row), ldb = K
    //   C = dC, ldc = N

    const float alpha = 1.0f;
    const float beta = 0.0f;

    int m = N;
    int n = M;
    int k = K;
    int lda = N;
    int ldb = K;
    int ldc = N;

    // Warm-up
    CHECK_CUBLAS(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &alpha,
        dB, lda,
        dA, ldb,
        &beta,
        dC, ldc));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
    {
        CHECK_CUBLAS(cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            dB, lda,
            dA, ldb,
            &beta,
            dC, ldc));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    double avg_ms = total_ms / iters;
    double flops = 2.0 * double(M) * double(N) * double(K);
    double tflops = flops / (avg_ms * 1e-3) / 1e12;

    std::cout << "[cuBLAS SGEMM] avg time: " << avg_ms << " ms,  TFLOP/s: " << tflops << std::endl << std::endl;
}

// -------------------------- wrappers and verification --------------------------

void cublas_ref_gemm(
    cublasHandle_t handle,
    int M, int N, int K,
    const float* dA,  // row-major A[M,K]
    const float* dB,  // row-major B[K,N]
    float* dC_ref)    // row-major C[M,N]
{
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    int m = N;
    int n = M;
    int k = K;
    int lda = N;
    int ldb = K;
    int ldc = N;

    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dB, lda, dA, ldb, &beta, dC_ref, ldc));
}

using KernelFunc = void(*)(const float*,
                           const float*,
                           float*,
                           int, int, int, float, float);

template <KernelFunc GemmKernel>
void run_custom_sgemm(cublasHandle_t handle,
                      const float* dA, 
                      const float* dB, 
                      float* dC,
                      int M,
                      int N,
                      int K,
                      int iters,
                      dim3 block_size,
                      dim3 grid_size,
                      std::string name,
                      bool verify = true)
{
    float alpha = 1.0;
    float betta = 1.0;

    CHECK_CUDA(cudaMemset(dC, 0, M * N * sizeof(float)));

    // Warm-up
    GemmKernel<<<grid_size, block_size>>> (dA, dB, dC, M, N, K, alpha, betta);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Verify
    if(verify) {
        int64_t sizeC = int64_t(M) * N;

        // Compute reference FP32 GEMM with cuBLAS
        float* dC_ref = nullptr;
        CHECK_CUDA(cudaMalloc(&dC_ref, sizeC * sizeof(float)));
        CHECK_CUDA(cudaMemset(dC_ref, 0, sizeC * sizeof(float)));

        cublas_ref_gemm(handle, M, N, K, dA, dB, dC_ref);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Copy both results back to host
        std::vector<float> hC(sizeC);
        std::vector<float> hC_ref(sizeC);

        CHECK_CUDA(cudaMemcpy(hC.data(),     dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hC_ref.data(), dC_ref, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaFree(dC_ref));

        double max_abs = 0.0;
        double max_rel = 0.0;
        const int max_print_count = 0;
        for (int64_t i = 0; i < sizeC; ++i) {
            double ref = hC_ref[i];
            double val = hC[i];
            double diff = std::abs(val - ref);
            double rel  = (std::abs(ref) > 0.0) ? diff / std::abs(ref) : diff;

            if(i < max_print_count) {
                std::cout << ref << " vs " << val << std::endl;
            }

            max_abs = std::max(diff, max_abs);
            max_rel = std::max(rel, max_rel);
        }

        std::cout << "check: max_abs = " << max_abs << ", max_rel = " << max_rel << "   ";

        double tol = 1e-3;  // BF16-level accuracy, adjust if you like
        if (max_abs < tol) {
            std::cout << "  (OK)" << std::endl;
        } else {
            std::cout << "  (WARN)" << std::endl;
        }
    }

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start)); 
    for (int i = 0; i < iters; ++i)
    {
        GemmKernel<<<grid_size, block_size>>> (dA, dB, dC, M, N, K, alpha, betta);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    double avg_ms = total_ms / iters;
    double flops = 2.0 * double(M) * double(N) * double(K);
    double tflops = flops / (avg_ms * 1e-3) / 1e12;

    std::cout << "[" << name << "] avg time: " << avg_ms << " ms,  TFLOP/s: " << tflops << std::endl << std::endl;
}

// --------------------------------------- Main -------------------------------------------

__global__ void sgemm_naive(const float *A,
                            const float *B, 
                            float *C, 
                            int M, int N, int K,
                            float alpha, float beta) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= M || y >= N)
        return;
    
    float tmp = 0.0;

    for (int i = 0; i < K; ++i) {
        tmp += A[x * K + i] * B[i * N + y];
    }

    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
}

__global__ void sgemm_coalcesed(const float *A,
                                const float *B, 
                                float *C, 
                                int M, int N, int K,
                                float alpha, float beta) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= M || y >= N)
        return;
    
    float tmp = 0.0;

    for (int i = 0; i < K; ++i) {
        // we improved memory access patter here, 
        tmp += A[y * K + i] * B[i * N + x];
    }

    C[y * N + x] = alpha * tmp + beta * C[y * N + x];
}

// --------------------------------------- Main -------------------------------------------

int main(int argc, char** argv)
{
    const bool verify = true;
    const bool run_slow = false;

    int size_sq = 4096;

    if (argc > 1)
        size_sq = std::atoi(argv[1]);
    int M = size_sq;
    int N = size_sq;
    int K = size_sq;

    int iters = 10;
    if (argc > 2)
        iters = std::atoi(argv[1]);

    std::cout << "SGEMM benchmark: C = A * B (row-major FP32)\n";
    std::cout << "  M = " << M << ", N = " << N << ", K = " << K
        << ", iters = " << iters << std::endl << std::endl;

    size_t sizeA = size_t(M) * K;
    size_t sizeB = size_t(K) * N;
    size_t sizeC = size_t(M) * N;

    size_t bytesA = sizeA * sizeof(float);
    size_t bytesB = sizeB * sizeof(float);
    size_t bytesC = sizeC * sizeof(float);

    // Host buffers
    std::vector<float> hA(sizeA), hB(sizeB), hC(sizeC);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < sizeA; ++i)
        hA[i] = dist(rng);
    for (size_t i = 0; i < sizeB; ++i)
        hB[i] = dist(rng);
    std::fill(hC.begin(), hC.end(), 0.0f);

    // Device buffers
    float* dA, * dB, * dC;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, bytesC));

    // cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // cublas reference perf
    {
        run_cublas_gemm(handle, M, N, K, dA, dB, dC, iters);
    }

    if(run_slow)
    {
        dim3 grid_size(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
        dim3 block_size(32, 32, 1);
        run_custom_sgemm<sgemm_naive>(handle, dA, dB, dC, M, N, K, iters, block_size, grid_size, "naive global", verify);
    }

    {
        dim3 grid_size(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
        dim3 block_size(32, 32, 1);
        run_custom_sgemm<sgemm_coalcesed>(handle, dA, dB, dC, M, N, K, iters, block_size, grid_size, "coalcesed global", verify);
    }
    
    CHECK_CUBLAS(cublasDestroy(handle));

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}