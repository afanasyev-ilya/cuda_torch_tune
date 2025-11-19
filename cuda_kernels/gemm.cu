// gemm_bench.cu
//
// Benchmark three FP32 GEMMs (row-major):
//   1) cuBLAS sgemm
//   2) Naive CUDA kernel
//   3) CUTLASS SIMT FP32 (device::Gemm)
//
// Problem: C = A * B
// A: M x K, B: K x N, C: M x N, all row-major.
//
// Command line:
//   ./gemm_bench [M] [N] [K] [iters]
// Defaults:
//   M = N = K = 2048, iters = 10
//
// Build example (adjust paths & arch as needed):
//   nvcc -O3 -std=c++17 gemm.cu -o gemm_bench -I ~/cutlass/include -lcublas -arch=sm_86

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include <cmath>

// CUTLASS headers
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/arch/arch.h"

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

#define CHECK_CUTLASS(status)                                            \
  do                                                                     \
  {                                                                      \
    if ((status) != cutlass::Status::kSuccess)                           \
    {                                                                    \
      std::cerr << "CUTLASS error: "                                     \
                << cutlass::cutlassGetStatusString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                                           \
    }                                                                    \
  } while (0)


// ----------------- Naive CUDA GEMM -----------------

// Row-major: A[M,K], B[K,N], C[M,N]
__global__ void naive_gemm_kernel(
    int M, int N, int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N)
        return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
    {
        float a = A[row * K + k];
        float b = B[k * N + col];
        acc += a * b;
    }
    C[row * N + col] = acc;
}

void run_naive_gemm(
    int M, int N, int K,
    const float* dA, const float* dB, float* dC,
    int iters)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
        (M + blockDim.y - 1) / blockDim.y);

    // Warm-up
    naive_gemm_kernel << <gridDim, blockDim >> > (M, N, K, dA, dB, dC);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
    {
        naive_gemm_kernel << <gridDim, blockDim >> > (M, N, K, dA, dB, dC);
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

    std::cout << "[Naive CUDA]   avg time: " << avg_ms
        << " ms,  TFLOP/s: " << tflops << std::endl;
}

// ----------------- cuBLAS GEMM (row-major via trick) -----------------

void run_cublas_gemm(
    cublasHandle_t handle,
    int M, int N, int K,
    const float* dA, const float* dB, float* dC,
    int iters)
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

    std::cout << "[cuBLAS SGEMM] avg time: " << avg_ms
        << " ms,  TFLOP/s: " << tflops << std::endl;
}

// ----------------- CUTLASS SIMT FP32 GEMM -----------------

using CutlassLayout = cutlass::layout::RowMajor;

// CUTLASS device GEMM type: SIMT FP32, row-major A/B/C, FP32 accumulator.
// Arch tag: Sm80 (works fine on Ampere-family GPUs; adjust if you like).
using GemmCutlass = cutlass::gemm::device::Gemm<
    float, CutlassLayout, // A
    float, CutlassLayout, // B
    float, CutlassLayout, // C
    float,                // accumulator
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80>;

void run_cutlass_gemm(
    int M, int N, int K,
    const float* dA, const float* dB, float* dC,
    int iters)
{
    GemmCutlass gemm_op;

    int lda = K;
    int ldb = N;
    int ldc = N;

    float alpha = 1.0f;
    float beta = 0.0f;

    typename GemmCutlass::Arguments args(
        { M, N, K }, // problem size (m, n, k)
        { dA, lda }, // A
        { dB, ldb }, // B
        { dC, ldc }, // C (input)
        { dC, ldc }, // D (output)
        { alpha, beta });

    CHECK_CUTLASS(gemm_op.can_implement(args));

    size_t workspace_size = gemm_op.get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_size > 0)
    {
        CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
    }

    // Warm-up
    CHECK_CUTLASS(gemm_op.initialize(args, workspace));
    CHECK_CUTLASS(gemm_op());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
    {
        CHECK_CUTLASS(gemm_op.initialize(args, workspace));
        CHECK_CUTLASS(gemm_op());
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    if (workspace)
    {
        CHECK_CUDA(cudaFree(workspace));
    }
    double avg_ms = total_ms / iters;
    double flops = 2.0 * double(M) * double(N) * double(K);
    double tflops = flops / (avg_ms * 1e-3) / 1e12;

    std::cout << "[CUTLASS SIMT] avg time: " << avg_ms
        << " ms,  TFLOP/s: " << tflops << std::endl;
}

// ----------------- Main -----------------

int main(int argc, char** argv)
{
    int size_sq = 4096;
    int M = size_sq;
    int N = size_sq;
    int K = size_sq;
    int iters = 10;

    if (argc > 1)
        M = std::atoi(argv[1]);
    if (argc > 2)
        N = std::atoi(argv[2]);
    if (argc > 3)
        K = std::atoi(argv[3]);
    if (argc > 4)
        iters = std::atoi(argv[4]);

    std::cout << "GEMM benchmark: C = A * B (row-major FP32)\n";
    std::cout << "  M = " << M << ", N = " << N << ", K = " << K
        << ", iters = " << iters << "\n";

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

    std::cout << "----------------------------------------\n";
    run_cublas_gemm(handle, M, N, K, dA, dB, dC, iters);

    CHECK_CUDA(cudaMemset(dC, 0, bytesC));
    run_naive_gemm(M, N, K, dA, dB, dC, iters);

    CHECK_CUDA(cudaMemset(dC, 0, bytesC));
    run_cutlass_gemm(M, N, K, dA, dB, dC, iters);

    CHECK_CUBLAS(cublasDestroy(handle));

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}