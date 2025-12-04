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

double BASELINE_TFLOPS = 1;

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

#define SAFE_KERNEL_CALL( KernelCallInstruction ){ \
    KernelCallInstruction; \
    cudaError_t cuerr = cudaGetLastError(); \
    if(cuerr != cudaSuccess) { \
        printf("CUDA error in kernel launch: %s at kernel \"" #KernelCallInstruction "\"\n", cudaGetErrorString(cuerr)); \
		throw "error in CUDA kernel launch, aborting..."; \
    } \
    cuerr = cudaDeviceSynchronize(); \
    if(cuerr != cudaSuccess) { \
        printf("CUDA error in kernel execution: %s at kernel \"" #KernelCallInstruction "\"\n", cudaGetErrorString(cuerr)); \
		throw "error in CUDA kernel execution, aborting..."; \
    } \
}

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
    BASELINE_TFLOPS = tflops;

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
double run_custom_sgemm(cublasHandle_t handle,
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
                        bool verify = true,
                        bool verbose = true)
    {
    float alpha = 1.0;
    float betta = 1.0;

    CHECK_CUDA(cudaMemset(dC, 0, M * N * sizeof(float)));

    // Warm-up
    SAFE_KERNEL_CALL((GemmKernel<<<grid_size, block_size>>> (dA, dB, dC, M, N, K, alpha, betta)));
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
        for (int64_t i = 0; i < sizeC; ++i) {
            double ref = hC_ref[i];
            double val = hC[i];
            double diff = std::abs(val - ref);
            double rel  = (std::abs(ref) > 0.0) ? diff / std::abs(ref) : diff;

            max_abs = std::max(diff, max_abs);
            max_rel = std::max(rel, max_rel);
        }

        if(verbose)
            std::cout << "check: max_abs = " << max_abs << ", max_rel = " << max_rel << "   ";

        double tol = 1e-3;  // BF16-level accuracy, adjust if you like
        bool correct = true;
        if (max_abs < tol) {
            if(verbose)
                std::cout << "  (OK)" << std::endl;
        } else {
            if(verbose)
                std::cout << "  (ERROR)" << std::endl;
            correct = false;
        }
        /*if(!correct) {
            return 0.0;
        }*/
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

    if(verbose) {
        std::cout << "[" << name << "]\n";
        std::cout << "avg time: " << avg_ms << " ms\n";
        std::cout << tflops << " TFLOP/s (" << 100.0*(tflops/BASELINE_TFLOPS) << "%)" << std::endl << std::endl;
    }
    return tflops;
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
        if(row < M && (thread_col + t*TILE_SIZE) < K)
            As[thread_row][thread_col] = A[row * K + (thread_col + t*TILE_SIZE)];
        else
            As[thread_row][thread_col] = 0;
        if((thread_row + t*TILE_SIZE) < K && col < N)
            Bs[thread_row][thread_col] = B[(thread_row + t*TILE_SIZE) * N + col];
        else
            Bs[thread_row][thread_col] = 0;

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
        if((block_row + thread_col) < M && (thread_row + tile_offset) < K)
            As[thread_col][thread_row] = A[(block_row + thread_col) * K + (thread_row + tile_offset)];
        else 
            As[thread_col][thread_row] = 0;
        // B access is coalcesed
        if((thread_row + tile_offset) < K && (block_col + thread_col) < N)
            Bs[thread_row][thread_col] = B[(thread_row + tile_offset) * N + (block_col + thread_col)];
        else
            Bs[thread_row][thread_col] = 0;
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
            float val = 0.0f;
            if(global_col < K && global_row < M)
                val = A[global_row * lda + global_col];
            As[shared_row][shared_col] = val;
        }

        // copy B
        #pragma unroll
        for(int i = tid; i < TILE_K*TILE_N; i += block_size) {
            int shared_col = i % TILE_N; // changes in range of [0, TILE_N]
            int shared_row = i / TILE_N; // changes in range of [0, TILE_K]

            int global_col = block_col + shared_col;
            int global_row = tile_offset + shared_row;
            float val = 0.0f;
            if(global_col < N && global_row < K)
                val = B[global_row * ldb + global_col];
            Bs[shared_row][shared_col] = val;
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

// ---------------------------------------

template<int TILE_M, int TILE_N, int TILE_K, int MICRO_M, int MICRO_N>
__global__ void sgemm_vectorize_smem(const float *A,
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

    // change: we transposed As here
    __shared__ float As[TILE_K * TILE_M];
    __shared__ float Bs[TILE_K * TILE_N];

    const int num_tiles = (K - 1) / TILE_K + 1;

    float reg_sums[MICRO_M][MICRO_N] = {0};
    float reg_a[MICRO_M] = {0};
    float reg_b[MICRO_N] = {0};
    
    for (int tile_id = 0; tile_id < num_tiles; tile_id++) {
        int tile_offset = tile_id * TILE_K;
        // copy A
        constexpr int VECTOR_LENGTH = 4;
        
        // we do less loop step now, because of copy is done using vectors
        #pragma unroll
        for(int i = tid; i < (TILE_M*TILE_K) / VECTOR_LENGTH; i += block_size) {
            int linear_idx = i * VECTOR_LENGTH;
            // shared cols go with stride 4 (VECTOR_LENGTH) now
            int shared_col = linear_idx % TILE_K; // changes in range of [0, TILE_K], with strides
            int shared_row = linear_idx / TILE_K; // changes in range of [0, TILE_M]

            int global_col = tile_offset + shared_col;
            int global_row = block_row + shared_row;
            
            float4 tmp = reinterpret_cast<const float4 *>(&A[global_row * lda + global_col])[0];
            As[(shared_col + 0) * TILE_M + shared_row] = tmp.x;
            As[(shared_col + 1) * TILE_M + shared_row] = tmp.y;
            As[(shared_col + 2) * TILE_M + shared_row] = tmp.z;
            As[(shared_col + 3) * TILE_M + shared_row] = tmp.w;
        }

        // copy B
        #pragma unroll
        for(int i = tid; i < (TILE_K*TILE_N) / VECTOR_LENGTH; i += block_size) {
            int linear_idx = i * VECTOR_LENGTH;
            int shared_col = linear_idx % TILE_N; // changes in range of [0, TILE_N], with strides
            int shared_row = linear_idx / TILE_N; // changes in range of [0, TILE_K]

            int global_col = block_col + shared_col;
            int global_row = tile_offset + shared_row;

            float4 tmp = reinterpret_cast<const float4 *>(&B[global_row * ldb + global_col])[0];
            Bs[(shared_row) * TILE_N + shared_col + 0] = tmp.x;
            Bs[(shared_row) * TILE_N + shared_col + 1] = tmp.y;
            Bs[(shared_row) * TILE_N + shared_col + 2] = tmp.z;
            Bs[(shared_row) * TILE_N + shared_col + 3] = tmp.w;
        }

        __syncthreads();

        for(int dot_idx = 0; dot_idx < TILE_K; dot_idx++) {
            #pragma unroll
            for(int elt_idx = 0; elt_idx < MICRO_M; elt_idx++) {
                // this is actually most complext thing here
                // similar to 1d, dot_idx runs among cols A
                // and for rows, we just copy MICRO_M elements (row elements per thread)
                reg_a[elt_idx] = As[dot_idx * TILE_M + thread_row * MICRO_M + elt_idx];
            }
            
            #pragma unroll
            for(int elt_idx = 0; elt_idx < MICRO_N; elt_idx++) {
                // vise versa but for B matrix
                reg_b[elt_idx] = Bs[dot_idx * TILE_N + thread_col * MICRO_N + elt_idx];
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

template<int TILE_M, int TILE_N, int TILE_K, int MICRO_M, int MICRO_N>
__global__ void sgemm_db(const float *A,
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
    const int tid        = threadIdx.x + blockDim.x * threadIdx.y;
    const int block_size = blockDim.x * blockDim.y;

    // double-buffered shared memory: we transpose As
    __shared__ float As[2][TILE_K * TILE_M];  // [buffer][k * TILE_M + m]
    __shared__ float Bs[2][TILE_K * TILE_N];  // [buffer][k * TILE_N + n]

    const int num_tiles = (K - 1) / TILE_K + 1;

    float reg_sums[MICRO_M][MICRO_N] = {0.0f};
    float reg_a[MICRO_M];
    float reg_b[MICRO_N];

    constexpr int VECTOR_LENGTH = 4;

    // -------------------------
    // Preload tile 0 into buffer 0
    // -------------------------
    int buf = 0;
    {
        int tile_offset = 0;

        // copy A for tile 0
        #pragma unroll
        for (int i = tid; i < (TILE_M * TILE_K) / VECTOR_LENGTH; i += block_size) {
            int linear_idx = i * VECTOR_LENGTH;
            int shared_col = linear_idx % TILE_K;   // [0, TILE_K), stride 4
            int shared_row = linear_idx / TILE_K;   // [0, TILE_M)

            int global_col = tile_offset + shared_col;
            int global_row = block_row + shared_row;

            float4 tmp = reinterpret_cast<const float4 *>(
                &A[global_row * lda + global_col]
            )[0];

            float *As_buf = &As[buf][0];

            As_buf[(shared_col + 0) * TILE_M + shared_row] = tmp.x;
            As_buf[(shared_col + 1) * TILE_M + shared_row] = tmp.y;
            As_buf[(shared_col + 2) * TILE_M + shared_row] = tmp.z;
            As_buf[(shared_col + 3) * TILE_M + shared_row] = tmp.w;
        }

        // copy B for tile 0
        #pragma unroll
        for (int i = tid; i < (TILE_K * TILE_N) / VECTOR_LENGTH; i += block_size) {
            int linear_idx = i * VECTOR_LENGTH;
            int shared_col = linear_idx % TILE_N;   // [0, TILE_N), stride 4
            int shared_row = linear_idx / TILE_N;   // [0, TILE_K)

            int global_col = block_col + shared_col;
            int global_row = tile_offset + shared_row;

            float4 tmp = reinterpret_cast<const float4 *>(
                &B[global_row * ldb + global_col]
            )[0];

            float *Bs_buf = &Bs[buf][0];
            int base = shared_row * TILE_N + shared_col;
            Bs_buf[base + 0] = tmp.x;
            Bs_buf[base + 1] = tmp.y;
            Bs_buf[base + 2] = tmp.z;
            Bs_buf[base + 3] = tmp.w;
        }

        __syncthreads(); // tile 0 ready
    }

    // -------------------------
    // Main loop with double buffering
    // -------------------------
    for (int tile_id = 0; tile_id < num_tiles; ++tile_id) {
        int next_tile = tile_id + 1;
        int next_buf  = buf ^ 1;

        // Start loading next tile into the other buffer (if any)
        if (next_tile < num_tiles) {
            int tile_offset = next_tile * TILE_K;

            // copy A for next tile into As[next_buf]
            #pragma unroll
            for (int i = tid; i < (TILE_M * TILE_K) / VECTOR_LENGTH; i += block_size) {
                int linear_idx = i * VECTOR_LENGTH;
                int shared_col = linear_idx % TILE_K;
                int shared_row = linear_idx / TILE_K;

                int global_col = tile_offset + shared_col;
                int global_row = block_row + shared_row;

                float4 tmp = reinterpret_cast<const float4 *>(
                    &A[global_row * lda + global_col]
                )[0];

                float *As_buf = &As[next_buf][0];

                As_buf[(shared_col + 0) * TILE_M + shared_row] = tmp.x;
                As_buf[(shared_col + 1) * TILE_M + shared_row] = tmp.y;
                As_buf[(shared_col + 2) * TILE_M + shared_row] = tmp.z;
                As_buf[(shared_col + 3) * TILE_M + shared_row] = tmp.w;
            }

            // copy B for next tile into Bs[next_buf]
            #pragma unroll
            for (int i = tid; i < (TILE_K * TILE_N) / VECTOR_LENGTH; i += block_size) {
                int linear_idx = i * VECTOR_LENGTH;
                int shared_col = linear_idx % TILE_N;
                int shared_row = linear_idx / TILE_N;

                int global_col = block_col + shared_col;
                int global_row = tile_offset + shared_row;

                float4 tmp = reinterpret_cast<const float4 *>(
                    &B[global_row * ldb + global_col]
                )[0];

                float *Bs_buf = &Bs[next_buf][0];
                int base = shared_row * TILE_N + shared_col;
                Bs_buf[base + 0] = tmp.x;
                Bs_buf[base + 1] = tmp.y;
                Bs_buf[base + 2] = tmp.z;
                Bs_buf[base + 3] = tmp.w;
            }
        }

        // ---- compute using current buffer (As[buf], Bs[buf]) ----
        float *As_cur = &As[buf][0];
        float *Bs_cur = &Bs[buf][0];

        for (int dot_idx = 0; dot_idx < TILE_K; ++dot_idx) {
            #pragma unroll
            for (int elt_idx = 0; elt_idx < MICRO_M; ++elt_idx) {
                reg_a[elt_idx] =
                    As_cur[dot_idx * TILE_M +
                           thread_row * MICRO_M + elt_idx];
            }

            #pragma unroll
            for (int elt_idx = 0; elt_idx < MICRO_N; ++elt_idx) {
                reg_b[elt_idx] =
                    Bs_cur[dot_idx * TILE_N +
                           thread_col * MICRO_N + elt_idx];
            }

            #pragma unroll
            for (int a_idx = 0; a_idx < MICRO_M; ++a_idx) {
                #pragma unroll
                for (int b_idx = 0; b_idx < MICRO_N; ++b_idx) {
                    reg_sums[a_idx][b_idx] +=
                        reg_a[a_idx] * reg_b[b_idx];
                }
            }
        }

        __syncthreads(); // ensure next tile (if any) is fully loaded
        buf = next_buf;
    }

    #pragma unroll
    for (int a_idx = 0; a_idx < MICRO_M; ++a_idx) {
        #pragma unroll
        for (int b_idx = 0; b_idx < MICRO_N; ++b_idx) {
            int row = block_row + thread_row * MICRO_M + a_idx;
            int col = block_col + thread_col * MICRO_N + b_idx;
            if (row < M && col < N) {
                C[row * ldc + col] =
                    alpha * reg_sums[a_idx][b_idx] +
                    beta  * C[row * ldc + col];
            }
        }
    }
}

// ---------------------------------------

/*constexpr int WARPSIZE = 32;

template<
    int TILE_M, int TILE_N, int TILE_K,
    int MICRO_M, int MICRO_N,
    int WM, int WN,
    int WMITER, int WNITER
>
__global__ void sgemm_warp_tiling(const float *A,
                                  const float *B,
                                  float *C,
                                  int M, int N, int K,
                                  float alpha, float beta)
{
    // Map template aliases to the notation from the article
    constexpr int BM = TILE_M;
    constexpr int BN = TILE_N;
    constexpr int BK = TILE_K;
    constexpr int TM = MICRO_M;
    constexpr int TN = MICRO_N;

    // Derived warp-subtile sizes
    static_assert(WMITER > 0 && WNITER > 0, "WMITER and WNITER must be > 0");
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;

    // Consistency checks with the scheme from the blog:
    //   WSUBM * WSUBN must equal WARPSIZE * TM * TN
    static_assert(WSUBM * WSUBN == WARPSIZE * TM * TN,
                  "WSUBM * WSUBN must equal WARPSIZE * TM * TN");

    // Each block computes a BM x BN tile of C
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    // Linear thread id within block
    const int tid = threadIdx.x + blockDim.x * threadIdx.y;
    const int block_size = blockDim.x * blockDim.y;

    // Warp / lane indices based on linear tid
    const int warpIdx = tid / WARPSIZE;           // [0 .. #warps-1]
    const int lane    = tid % WARPSIZE;           // [0 .. 31]

    // Warp grid inside the block tile: (#warp rows) x (#warp cols)
    constexpr int warpColsPerBlock = BN / WN;
    constexpr int warpRowsPerBlock = BM / WM;
    static_assert(warpColsPerBlock * warpRowsPerBlock * WARPSIZE ==
                  (BM * BN) / (TM * TN * WMITER * WNITER),
                  "Block tiling / warp tiling configuration inconsistent");

    const int warpRow = warpIdx / warpColsPerBlock;  // warp row within block tile
    const int warpCol = warpIdx % warpColsPerBlock;  // warp col within block tile

    // Threads layout inside a warp-subtile (WSUBM x WSUBN)
    constexpr int threadsPerRow = WSUBN / TN;        // how many threads along N
    static_assert(threadsPerRow * (WSUBM / TM) == WARPSIZE,
                  "Bad (WSUBM,WSUBN,TM,TN) combination");

    const int threadColInWarp = lane % threadsPerRow;
    const int threadRowInWarp = lane / threadsPerRow;

    // Shared memory: As is laid out as [k][m], Bs as [k][n]
    __shared__ float As[TILE_K * TILE_M];  // [BK][BM]  index: k*BM + m
    __shared__ float Bs[TILE_K * TILE_N];  // [BK][BN]  index: k*BN + n

    const int num_tiles = (K + TILE_K - 1) / TILE_K;

    // Per-thread registers
    float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
    float regM[WMITER * TM] = {0.0f};
    float regN[WNITER * TN] = {0.0f};

    for (int tile_id = 0; tile_id < num_tiles; ++tile_id) {
        const int tile_offset = tile_id * TILE_K;

        // Load A tile into shared memory, transposed to [k][m]
        // Global: A is [M][K] row-major: A[global_row * lda + global_col]
        for (int i = tid; i < TILE_M * TILE_K; i += block_size) {
            int shared_row_m = i / TILE_K;   // 0..BM-1   (m)
            int shared_col_k = i % TILE_K;   // 0..BK-1   (k)

            int global_row = block_row + shared_row_m; // m index
            int global_col = tile_offset + shared_col_k; // k index

            float val = 0.0f;
            if (global_row < M && global_col < K) {
                val = A[global_row * lda + global_col];
            }

            // store as [k][m]
            As[shared_col_k * TILE_M + shared_row_m] = val;
        }

        // Load B tile into shared memory, layout [k][n]
        // Global: B is [K][N] row-major: B[global_row * ldb + global_col]
        for (int i = tid; i < TILE_K * TILE_N; i += block_size) {
            int shared_row_k = i / TILE_N;   // 0..BK-1 (k)
            int shared_col_n = i % TILE_N;   // 0..BN-1 (n)

            int global_row = tile_offset + shared_row_k; // k index
            int global_col = block_col + shared_col_n;   // n index

            float val = 0.0f;
            if (global_row < K && global_col < N) {
                val = B[global_row * ldb + global_col];
            }

            Bs[shared_row_k * TILE_N + shared_col_n] = val; // [k][n]
        }

        __syncthreads();

        // dotIdx runs along K within this BK tile
        for (int dotIdx = 0; dotIdx < TILE_K; ++dotIdx) {
            // 1) Load a full warp-tile row-chunk from As into regM
            for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                for (int i = 0; i < TM; ++i) {
                    int rowInBlock =
                        warpRow * WM +
                        wSubRowIdx * WSUBM +
                        threadRowInWarp * TM +
                        i; // 0..BM-1

                    regM[wSubRowIdx * TM + i] =
                        As[dotIdx * BM + rowInBlock];
                }
            }

            // 2) Load a full warp-tile column-chunk from Bs into regN
            for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                for (int i = 0; i < TN; ++i) {
                    int colInBlock =
                        warpCol * WN +
                        wSubColIdx * WSUBN +
                        threadColInWarp * TN +
                        i; // 0..BN-1

                    regN[wSubColIdx * TN + i] =
                        Bs[dotIdx * BN + colInBlock];
                }
            }

            // 3) Accumulate outer-products into threadResults
            for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                        for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                            int idx =
                                (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                (wSubColIdx * TN) + resIdxN;

                            threadResults[idx] +=
                                regM[wSubRowIdx * TM + resIdxM] *
                                regN[wSubColIdx * TN + resIdxN];
                        }
                    }
                }
            }
        }

        __syncthreads();
    }

    // Write back C
    for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    int localIdx =
                        (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        (wSubColIdx * TN) + resIdxN;

                    int row =
                        block_row +
                        warpRow * WM +
                        wSubRowIdx * WSUBM +
                        threadRowInWarp * TM +
                        resIdxM;

                    int col =
                        block_col +
                        warpCol * WN +
                        wSubColIdx * WSUBN +
                        threadColInWarp * TN +
                        resIdxN;

                    if (row < M && col < N) {
                        float old = C[row * ldc + col];
                        C[row * ldc + col] =
                            alpha * threadResults[localIdx] + beta * old;
                    }
                }
            }
        }
    }
}*/

template<
    int BM, int BN, int BK, // Block Tile: 128x128
    int TM, int TN, // Thread Tile: 8x8 elements per thread
// Warp Configuration
// We have 8 warps (256/32). We need to arrange them to cover the 128x128 block.
// A good arrangement for 8 warps is 4 warps in M (vertical) and 2 in N (horizontal).
    int WARPS_M, int WARPS_N
>
__global__ void sgemm_warp_tiling(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    int M, int N, int K,
    float alpha, float beta) 
{
    // Derived: Threads per block = (BM*BN) / (TM*TN) = 4096 / 16 = 256 threads
    const int NUM_THREADS = (BM * BN) / (TM * TN);
    const int WARP_SIZE = 32;

    // Warp Tile Size: The area ONE warp covers
    const int WM = BM / WARPS_M;
    const int WN = BN / WARPS_N;

    // 1. Thread & Warp ID calculations
    const int tid = threadIdx.x; // Linear thread ID (0..255)
    const int warpId = tid / WARP_SIZE;
    const int laneId = tid % WARP_SIZE;

    // Map Warp ID to Warp Row/Col in the Block Grid
    const int warpRow = warpId / WARPS_N;
    const int warpCol = warpId % WARPS_N;

    // Map Lane ID (0..31) to Thread Row/Col within the Warp Tile
    // A warp covers WM x WN (16 x 32). Each thread covers TM x TN (8x8).
    // Threads in Warp (Rows x Cols) = (WM/TM) x (WN/TN) = 4 x 8 = 32 threads.
    const int numThreadsWarpN = WN / TN; // 8 threads wide
    const int threadRowInWarp = laneId / numThreadsWarpN; // 0..3
    const int threadColInWarp = laneId % numThreadsWarpN; // 0..7

    // 2. Global Memory Coordinates
    // The top-left corner of the Block's tile in C
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    // Allocate Shared Memory
    // Padding +4 to avoid bank conflicts when accessing columns
    __shared__ float As[BK][BM]; 
    __shared__ float Bs[BK][BN]; 

    // Move A and B pointers to the start of this block's row/col
    const float* A_ptr = A + blockRow * K;
    const float* B_ptr = B + blockCol; // B is assumed Row-Major (KxN) based on standard sgemm

    // 3. Registers for Double Buffering / Accumulation
    float threadResults[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    // 4. Main Loop over K blocks
    for (int k_step = 0; k_step < K; k_step += BK) {
        
        // --- Loading Global -> Shared (Vectorized) ---
        
        // We need to load BK*BM elements for A, and BK*BN elements for B.
        // Threads collaborate. 
        // We use float4 to load 4 floats at once.
        const int totalElementsA = BM * BK;
        const int threadsPerBlock = NUM_THREADS;
        
        // Load A (Transposed in Shared Memory for faster access later)
        // Global A is [M x K]. We want As[k][m].
        // To vectorize A loads from global (Row Major), we load contiguous K.
        // A_ptr points to A[blockRow][k_step].
        
        // Parallel copy logic for A
        // Each thread loads 4 floats (float4) if possible
        const int floatsPerThread = 4;
        const int numVectorLoadsA = totalElementsA / (threadsPerBlock * floatsPerThread);
        
        // This is a simplified loader assuming aligned memory and dimensions. 
        // In robust code, you handle boundaries.
        // For A: we want to load rows of A.
        // Mapping: tid maps to specific pixels in the BKxBM tile.
        // We iterate flatly over the tile.
        
        for (int i = 0; i < totalElementsA; i += threadsPerBlock) {
            int idx = i + tid;
            int row = idx / BK; // row in A tile (0..BM)
            int col = idx % BK; // col in A tile (0..BK)
            
            // Standard scalar load for A to allow easy transpose in shared
            // (Vectorized load for A is harder if we transpose, stick to scalar for A correctness/simplicity first)
            // Or, strictly follow Siboehm's vector load pattern:
            
            if (row < BM && col < BK) {
                // Global index: (blockRow + row)*K + (k_step + col)
                 As[col][row] = A[(blockRow + row) * K + (k_step + col)]; 
                 // Stored as [k][m] to reduce bank conflicts during compute
            }
        }

        // Load B (Standard)
        // Global B is [K x N]. We want Bs[k][n].
        // Vectorized float4 load for B is essential as we read row-major chunks.
        for (int i = tid * 4; i < BK * BN; i += threadsPerBlock * 4) {
            int row = i / BN; // 0..BK
            int col = i % BN; // 0..BN
            
            if (row < BK && col < BN) {
               // Reinterpret cast for float4 load
               float4 vec = reinterpret_cast<const float4*>(&B[(k_step + row) * N + (blockCol + col)])[0];
               Bs[row][col + 0] = vec.x;
               Bs[row][col + 1] = vec.y;
               Bs[row][col + 2] = vec.z;
               Bs[row][col + 3] = vec.w;
            }
        }

        __syncthreads();

        // --- Compute Phase ---
        
        // Iterate over the BK dimension (dot product axis)
        for (int k = 0; k < BK; ++k) {
            // 1. Load fragments into registers
            // We want As[k, threadRow...] and Bs[k, threadCol...]
            // Thanks to Warps, we only need to calculate our offsets once.
            
            // Calculate absolute M and N indices for this thread within the Block
            // M index = warpRow * WM + threadRowInWarp * TM
            // N index = warpCol * WN + threadColInWarp * TN
            
            int threadPixelRow = warpRow * WM + threadRowInWarp * TM;
            int threadPixelCol = warpCol * WN + threadColInWarp * TN;

            // Load TM values from As (Column k, Rows threadPixelRow...)
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                regM[i] = As[k][threadPixelRow + i];
            }
            
            // Load TN values from Bs (Row k, Cols threadPixelCol...)
            #pragma unroll
            for (int i = 0; i < TN; ++i) {
                regN[i] = Bs[k][threadPixelCol + i];
            }

            // Outer Product
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    threadResults[m * TN + n] += regM[m] * regN[n];
                }
            }
        }
        
        __syncthreads();
    }

    // 5. Write back results to Global Memory
    int threadPixelRow = warpRow * WM + threadRowInWarp * TM;
    int threadPixelCol = warpCol * WN + threadColInWarp * TN;

    for (int m = 0; m < TM; ++m) {
        for (int n = 0; n < TN; ++n) {
            int globalRow = blockRow + threadPixelRow + m;
            int globalCol = blockCol + threadPixelCol + n;

            if (globalRow < M && globalCol < N) {
                int idx = globalRow * N + globalCol;
                float val = threadResults[m * TN + n];
                // Beta handling
                if (beta != 0.0f) {
                    val = alpha * val + beta * C[idx];
                } else {
                    val = alpha * val;
                }
                C[idx] = val;
            }
        }
    }
}

// ---------------------------------------

struct Config {
    int BM, BN, BK, MICRO_M, MICRO_N;
    double flops;
};

std::ostream& operator<<(std::ostream& os, const Config& c) {
    os << "Config{"
       << "BM=" << c.BM
       << ", BN=" << c.BN
       << ", BK=" << c.BK
       << ", MICRO_M=" << c.MICRO_M
       << ", MICRO_N=" << c.MICRO_N
       << ", flops=" << c.flops
       << "}";
    return os;
}

template<int BM, int BN, int BK, int MICRO_M, int MICRO_N>
double bench_config(cublasHandle_t handle,
                    const float* dA, const float* dB, float* dC,
                    int M, int N, int K, int iters, bool verify)
{
    static_assert(BN % MICRO_N == 0, "BN % MICRO_N != 0");
    static_assert(BM % MICRO_M == 0, "BM % MICRO_M != 0");

    dim3 block_size(BN / MICRO_N, BM / MICRO_M, 1);
    int threads = block_size.x * block_size.y;
    if (threads > 1024) return 0.0;  // invalid on many GPUs

    dim3 grid_size(
        CEIL_DIV(N, block_size.x * MICRO_N),
        CEIL_DIV(M, block_size.y * MICRO_M),
        1
    );

    // Optional: shared mem check (BM+BN)*BK*sizeof(float) etc.
    // size_t smem = (BM * BK + BK * BN) * sizeof(float);
    // if (smem > max_smem_per_block) return 0.0;

    bool verbose = false;
    double flops = run_custom_sgemm<sgemm_db<BM, BN, BK, MICRO_M, MICRO_N>>(
        handle, dA, dB, dC, M, N, K, iters,
        block_size, grid_size,
        "autotune", verify, verbose
    );
    return flops;
}

Config autotune_sgemm(cublasHandle_t handle,
                      const float* dA, const float* dB, float* dC,
                      int M, int N, int K, int iters, bool verify)
{
    Config best {0,0,0,0,0,0.0};

    auto update = [&](int BM, int BN, int BK, int MM, int NN, double flops) {
        if (flops > best.flops) {
            best = {BM, BN, BK, MM, NN, flops};
        }
    };

    // macro to instantiate + benchmark one config
    #define TRY(BM, BN, BK, MM, NN) do {                           \
        double g = bench_config<BM, BN, BK, MM, NN>(               \
            handle, dA, dB, dC, M, N, K, iters, verify);           \
        update(BM, BN, BK, MM, NN, g);                             \
    } while (0)

    // square-ish tiles, low BK
    TRY( 64,  64,  8,  4,  4);
    TRY( 64,  64,  8,  8,  4);
    TRY( 64,  64,  8,  4,  8);

    TRY(128, 128,  8,  4,  4);
    TRY(128, 128,  8,  8,  4);
    TRY(128, 128,  8,  4,  8);

    // higher BK
    TRY( 64,  64, 16,  4,  4);
    TRY(128, 128, 16,  4,  4);
    TRY(128, 128, 16,  8,  8);

    // rectangular tiles
    TRY(128,  64,  8,  8,  4);
    TRY( 64, 128,  8,  4,  8);
    TRY(128,  64, 16,  8,  4);
    TRY( 64, 128, 16,  4,  8);

    #undef TRY

    return best;
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
        const int BM = 64;
        const int BN = 64;
        const int BK = 8;
        const int MICRO_M = 4;
        const int MICRO_N = 4;
        dim3 block_size(BN/MICRO_N, BM/MICRO_M, 1);
        dim3 grid_size(CEIL_DIV(N, block_size.x * MICRO_N), CEIL_DIV(M, block_size.y * MICRO_M), 1);
        run_custom_sgemm<sgemm_2D_blocking<BM, BN, BK, MICRO_M, MICRO_N>>(
            handle, dA, dB, dC, M, N, K, iters, block_size, grid_size, "2D blocking", verify);
    }

    {
        const int BM = 64;
        const int BN = 64;
        const int BK = 8;
        const int MICRO_M = 4;
        const int MICRO_N = 4;
        dim3 block_size(BN/MICRO_N, BM/MICRO_M, 1);
        dim3 grid_size(CEIL_DIV(N, block_size.x * MICRO_N), CEIL_DIV(M, block_size.y * MICRO_M), 1);
        run_custom_sgemm<sgemm_vectorize_smem<BM, BN, BK, MICRO_M, MICRO_N>>(
            handle, dA, dB, dC, M, N, K, iters, block_size, grid_size, "vectorize shmem", verify);
    }

    {   
        // Autotuned results for RTX 3060
        const int BM = 64;
        const int BN = 64;
        const int BK = 8;
        const int MICRO_M = 8;
        const int MICRO_N = 4;

        dim3 block_size(BN/MICRO_N, BM/MICRO_M, 1);
        dim3 grid_size(CEIL_DIV(N, block_size.x * MICRO_N), CEIL_DIV(M, block_size.y * MICRO_M), 1);
        run_custom_sgemm<sgemm_vectorize_smem<BM, BN, BK, MICRO_M, MICRO_N>>(
            handle, dA, dB, dC, M, N, K, iters, block_size, grid_size, "autotune", verify);
    }

    /*{   
        std::cout << "Autotuning in process..." << std::endl;
        auto cfg = autotune_sgemm(handle, dA, dB, dC, M, N, K, iters, verify);
        std::cout << "done!" << std::endl;
        std::cout << "best result: " << cfg << std::endl << std::endl;

        // results for RTX 3060
        const int BM = 128;
        const int BN = 128;
        const int BK = 16;
        const int MICRO_M = 8;
        const int MICRO_N = 8;

        dim3 block_size(BN/MICRO_N, BM/MICRO_M, 1);
        dim3 grid_size(CEIL_DIV(N, block_size.x * MICRO_N), CEIL_DIV(M, block_size.y * MICRO_M), 1);
        run_custom_sgemm<sgemm_db<BM, BN, BK, MICRO_M, MICRO_N>>(
            handle, dA, dB, dC, M, N, K, iters, block_size, grid_size, "double buffering(DB)", verify);
    }*/

    {   
        std::cout << "started new test " << std::endl;
        // results for RTX 3060
        const int BM = 128;
        const int BN = 128;
        const int BK = 16;
        const int TM = 8;
        const int TN = 8;

        const int WARPS_M = 4;
        const int WARPS_N = 2;

        dim3 block_size(BN/TN, BM/TM, 1);
        dim3 grid_size(CEIL_DIV(N, block_size.x * TN), CEIL_DIV(M, block_size.y * TM), 1);
        run_custom_sgemm<sgemm_warp_tiling<BM, BN, BK, TM, TN, WARPS_M, WARPS_N>>(
            handle, dA, dB, dC, M, N, K, iters, block_size, grid_size, "warp tiling", verify);
    }
    
    CHECK_CUBLAS(cublasDestroy(handle));

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}