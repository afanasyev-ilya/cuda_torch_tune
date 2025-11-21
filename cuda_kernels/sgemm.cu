// Build example (adjust paths & arch as needed):
//   nvcc -O3 -std=c++20 sgemm.cu -o gemm_bench -I ~/cutlass/include -lcublas -arch=sm_86

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include <cmath>
#include <cassert>

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

    std::cout << "[cuBLAS SGEMM]\navg time: " << avg_ms << " ms, \n   TFLOP/s: " << tflops << std::endl << std::endl;
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

    std::cout << "[" << name << "]\navg time: " << avg_ms << " ms,\n   " << tflops << " TFLOP/s" << std::endl << std::endl;
}

// --------------------------------------- Main -------------------------------------------

__global__ void sgemm_naive(const float *A,
                            const float *B, 
                            float *C, 
                            int M, int N, int K,
                            float alpha, float beta) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= M || y >= N)
        return;
    
    float tmp = 0.0;

    for (int i = 0; i < K; ++i) {
        tmp += A[x * K + i] * B[i * N + y];
    }

    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
}

// ---------------------------------------

__global__ void sgemm_coalcesed(const float *A,
                                const float *B, 
                                float *C, 
                                int M, int N, int K,
                                float alpha, float beta) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= N || row >= M)
        return;
    
    float sum = 0.0;

    for (int i = 0; i < K; ++i) {
        // we improved memory access patter here, 
        sum += A[row * K + i] * B[i * N + col];
    }

    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

// ---------------------------------------

template <int TILE_SIZE>
__global__ void sgemm_shared(const float *A,
                             const float *B, 
                             float *C, 
                             int M, int N, int K,
                             float alpha, float beta) {
    // Thread position within block
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;
    
    // Global output position
    const int row = blockIdx.y * blockDim.y + thread_row;
    const int col = blockIdx.x * blockDim.x + thread_col;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    const int num_tiles = (K - 1) / TILE_SIZE + 1;

    float sum = 0.0f;
    
    for (int t = 0; t < num_tiles; t++) {
        As[thread_row][thread_col] = A[row * K + (thread_col + t*TILE_SIZE)];
        Bs[thread_row][thread_col] = B[(thread_row + t*TILE_SIZE) * N + col];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[thread_row][k] * Bs[k][thread_col];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// ---------------------------------------

// idea is https://siboehm.com/assets/img/CUDA-MMM/kernel_4_1D_blocktiling.png
// each thread calculates small TM size column of matrix C elements
template<int BM, int BN, int BK, int ELEM_PER_THREAD>
__global__ void sgemm_1D_blocking(const float *A,
                                  const float *B, 
                                  float *C, 
                                  int M, int N, int K,
                                  float alpha, float beta) {
    // Thread position within block
    const int thread_row = threadIdx.y; // should be 0, 8
    const int thread_col = threadIdx.x; // should be 0, 64
    
    // Starting coords of 64x64 output tile for matrix C
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // we multiply 64x8 * 8x64 to get 64x64 block. 
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // This results into more K steps then previosly (we used 32), but does not matter much
    const int num_tiles = (K - 1) / BK + 1;

    // ELEM_PER_THREAD = 64 / 8 = BM / block_size.y
    float sums[ELEM_PER_THREAD] = {0};
    
    for (int t = 0; t < num_tiles; t++) {
        int tile_offset = t * BK;
        // since we have 64*8 = 512 threads and shared memory size is 512, we can copy in one pass without loop
        As[thread_col][thread_row] = A[(block_row + thread_col) * K + (thread_row + tile_offset)];
        // B access is coalcesed
        Bs[thread_row][thread_col] = B[(thread_row + tile_offset) * N + (block_col + thread_col)];

        __syncthreads();

        for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {
            float B_val = Bs[dot_idx][thread_col];
            #pragma unroll
            for (int elt_idx = 0; elt_idx < ELEM_PER_THREAD; ++elt_idx) {
                sums[elt_idx] += As[ELEM_PER_THREAD * thread_row + elt_idx][dot_idx] * B_val;
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for(int elt_idx = 0; elt_idx < ELEM_PER_THREAD; elt_idx++) {
        // each threads writes back TM elements of matrix C of the same col (adj rows)
        int row = block_row + thread_row * ELEM_PER_THREAD + elt_idx;
        int col = block_col + thread_col;
        if (row < M && col < N) {
            C[row * N + col] = alpha * sums[elt_idx] + beta * C[row * N + col];
        }
    }
}

// ---------------------------------------

// MICRO_M * MICRO_K = elements processed by each thread
template<int TILE_M, int TILE_N, int TILE_K, int MICRO_M, int MICRO_N>
__global__ void sgemm_2D_blocking(const float *A,
                                  const float *B, 
                                  float *C, 
                                  int M, int N, int K,
                                  float alpha, float beta) {
    // Thread position within block
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;
    
    // Starting coords of output tile for matrix C
    const int block_row = blockIdx.y * TILE_M;
    const int block_col = blockIdx.x * TILE_N;

    // leading dims for simplicity
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    // indexes for loading
    const int tid = threadIdx.x + blockDim.x * threadIdx.y;
    const int block_size = blockDim.x * blockDim.y;

    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    const int num_tiles = (K - 1) / TILE_K + 1;

    float reg_sums[MICRO_M][MICRO_N] = {0};
    float reg_a[MICRO_M] = {0};
    float reg_b[MICRO_N] = {0};
    
    for (int tile_id = 0; tile_id < num_tiles; tile_id++) {
        int tile_offset = tile_id * TILE_K;
        // copy A
        #pragma unroll
        for(int i = tid; i < TILE_M*TILE_K; i += block_size) {
            int shared_col = i % TILE_K; // changes in range of [0, TILE_K]
            int shared_row = i / TILE_K; // changes in range of [0, TILE_M]

            int global_col = tile_offset + shared_col;
            int global_row = block_row + shared_row;
            As[shared_row][shared_col] = A[global_row * lda + global_col];
        }

        // copy B
        #pragma unroll
        for(int i = tid; i < TILE_K*TILE_N; i += block_size) {
            int shared_col = i % TILE_N; // changes in range of [0, TILE_N]
            int shared_row = i / TILE_N; // changes in range of [0, TILE_K]

            int global_col = block_row + shared_col;
            int global_row = tile_offset + shared_row;
            Bs[shared_row][shared_col] = B[global_row * ldb + global_col];
        }

        __syncthreads();

        for(int dot_idx = 0; dot_idx < TILE_K; dot_idx++) {
            #pragma unroll
            for(int elt_idx = 0; elt_idx < MICRO_M; elt_idx++) {
                // this is actually most complext thing here
                // similar to 1d, dot_idx runs among cols A
                // and for rows, we just copy MICRO_M elements (row elements per thread)
                reg_a[elt_idx] = As[thread_row * MICRO_M + elt_idx][dot_idx];
            }
            
            #pragma unroll
            for(int elt_idx = 0; elt_idx < MICRO_N; elt_idx++) {
                // vise versa but for B matrix
                reg_b[elt_idx] = Bs[dot_idx][thread_col * MICRO_N + elt_idx];
            }

            // actual matmul on registers here
            #pragma unroll
            for(int a_idx = 0; a_idx < MICRO_M; a_idx++) {
                #pragma unroll
                for(int b_idx = 0; b_idx < MICRO_N; b_idx++) {
                    reg_sums[a_idx][b_idx] += reg_a[a_idx] * reg_b[b_idx];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for(int a_idx = 0; a_idx < MICRO_M; a_idx++) {
        #pragma unroll
        for(int b_idx = 0; b_idx < MICRO_N; b_idx++) {
            int row = block_row + thread_row * MICRO_M + a_idx;
            int col = block_col + thread_col * MICRO_N + b_idx;
            if (row < M && col < N) {
                C[row * ldc + col] = alpha * reg_sums[a_idx][b_idx] + beta * C[row * ldc + col];
            }
        }
    }
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
        iters = std::atoi(argv[2]);

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
        dim3 block_size(32, 32, 1);
        dim3 grid_size(CEIL_DIV(N, block_size.x), CEIL_DIV(M, block_size.y), 1);
        run_custom_sgemm<sgemm_naive>(handle, dA, dB, dC, M, N, K, iters, block_size, grid_size, "naive global", verify);
    }

    {
        dim3 block_size(32, 32, 1);
        dim3 grid_size(CEIL_DIV(N, block_size.x), CEIL_DIV(M, block_size.y), 1);
        run_custom_sgemm<sgemm_coalcesed>(handle, dA, dB, dC, M, N, K, iters, block_size, grid_size, "coalcesed global", verify);
    }

    {
        const int TILE_SIZE = 32;
        dim3 block_size(TILE_SIZE, TILE_SIZE, 1);
        dim3 grid_size(CEIL_DIV(N, block_size.x), CEIL_DIV(M, block_size.y), 1);
        run_custom_sgemm<sgemm_shared<TILE_SIZE>>(handle, dA, dB, dC, M, N, K, iters, block_size, grid_size, "shared", verify);
    }

    auto run_1d_blocking = [&]<int ELEM_PER_THREAD>() {
        std::cout << "using " << ELEM_PER_THREAD << " element per thread:\n";
        const int BM = 64;
        const int BN = 64;
        assert(BM == BN);
        const int BK = BM / ELEM_PER_THREAD;
        
        dim3 block_size(BM, BK, 1);
        dim3 grid_size(CEIL_DIV(N, block_size.x), CEIL_DIV(M, block_size.y * ELEM_PER_THREAD), 1);
        run_custom_sgemm<sgemm_1D_blocking<BM, BN, BK, ELEM_PER_THREAD>>(
            handle, dA, dB, dC, M, N, K, iters, block_size, grid_size, "1D blocking", verify);
    };

    run_1d_blocking.operator()<4>();
    run_1d_blocking.operator()<8>();
    run_1d_blocking.operator()<16>();
    run_1d_blocking.operator()<32>();

    {
        const int MICRO_M = 4;
        const int MICRO_N = 4;
        dim3 block_size(16, 16, 1);
        dim3 grid_size(CEIL_DIV(N, block_size.x * MICRO_N), CEIL_DIV(M, block_size.y * MICRO_M), 1);
        run_custom_sgemm<sgemm_2D_blocking<64, 64, 8, MICRO_M, MICRO_N>>(
            handle, dA, dB, dC, M, N, K, iters, block_size, grid_size, "2D blocking", verify);
    }
    
    CHECK_CUBLAS(cublasDestroy(handle));

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}