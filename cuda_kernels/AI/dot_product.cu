#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(err_), __FILE__, __LINE__);            \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

#define WARP_SIZE 32

// Optimized-ish dot product kernel: grid-stride loop + warp-level reduction.
__global__ void dot_kernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ result,
                           int n)
{
    extern __shared__ float shared[];  // one float per warp

    int tid = threadIdx.x;
    int globalThreadId = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;

    // Grid-stride loop
    for (int i = globalThreadId; i < n; i += stride) {
        sum += a[i] * b[i];
    }

    // Warp-level reduction using shuffle
    const unsigned mask = 0xffffffff;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Write per-warp sums into shared memory (lane 0 of each warp)
    if ((tid & (WARP_SIZE - 1)) == 0) {
        shared[tid / WARP_SIZE] = sum;
    }

    __syncthreads();

    // First warp reduces the warp-sums
    if (tid < WARP_SIZE) {
        int numWarps = blockDim.x / WARP_SIZE;
        float val = (tid < numWarps) ? shared[tid] : 0.0f;

        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }

        if (tid == 0) {
            atomicAdd(result, val);
        }
    }
}

// Benchmark with pageable host memory (malloc) + cudaMemcpy.
float run_pageable(const float* h_a, const float* h_b, float* h_result,
                   float* d_a, float* d_b, float* d_result,
                   int N, int iters,
                   int blocks, int threads, size_t shmemBytes)
{
    size_t bytes = static_cast<size_t>(N) * sizeof(float);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float total_ms = 0.0f;

    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));

        CUDA_CHECK(cudaEventRecord(start));

        // H2D copies
        CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

        // Kernel
        dot_kernel<<<blocks, threads, shmemBytes>>>(d_a, d_b, d_result, N);
        CUDA_CHECK(cudaGetLastError());

        // D2H copy of scalar result
        CUDA_CHECK(cudaMemcpy(h_result, d_result, sizeof(float),
                              cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / iters;
}

// Benchmark with pinned host memory (cudaMallocHost) + cudaMemcpy.
float run_pinned(const float* h_a, const float* h_b, float* h_result,
                 float* d_a, float* d_b, float* d_result,
                 int N, int iters,
                 int blocks, int threads, size_t shmemBytes)
{
    size_t bytes = static_cast<size_t>(N) * sizeof(float);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float total_ms = 0.0f;

    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));

        CUDA_CHECK(cudaEventRecord(start));

        // Faster H2D copies thanks to pinned memory
        CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

        // Kernel
        dot_kernel<<<blocks, threads, shmemBytes>>>(d_a, d_b, d_result, N);
        CUDA_CHECK(cudaGetLastError());

        // D2H copy of scalar result
        CUDA_CHECK(cudaMemcpy(h_result, d_result, sizeof(float),
                              cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / iters;
}

// Benchmark with pinned *mapped* host memory (zero-copy).
// No explicit H2D copy; the kernel reads host memory over PCIe.
// We still copy the scalar result back to host.
float run_mapped(const float* h_a_mapped, const float* h_b_mapped,
                 float* h_result,
                 float* d_result,
                 const float* d_a_mapped, const float* d_b_mapped,
                 int N, int iters,
                 int blocks, int threads, size_t shmemBytes)
{
    (void)h_a_mapped; // not used directly in timing, but kept for symmetry
    (void)h_b_mapped;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float total_ms = 0.0f;

    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));

        CUDA_CHECK(cudaEventRecord(start));

        // Kernel directly accesses mapped host memory
        dot_kernel<<<blocks, threads, shmemBytes>>>(d_a_mapped, d_b_mapped,
                                                    d_result, N);
        CUDA_CHECK(cudaGetLastError());

        // D2H copy of scalar
        CUDA_CHECK(cudaMemcpy(h_result, d_result, sizeof(float),
                              cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / iters;
}

int main(int argc, char** argv)
{
    int N = 1 << 26;   // default: ~67M elements
    int iters = 10;

    if (argc > 1) {
        N = std::atoi(argv[1]);
    }
    if (argc > 2) {
        iters = std::atoi(argv[2]);
    }

    printf("Dot product benchmark\n");
    printf("N = %d elements, iterations = %d\n", N, iters);

    // Allow mapped memory if possible (must be called early)
    bool mapped_possible = true;
    cudaError_t flagErr = cudaSetDeviceFlags(cudaDeviceMapHost);
    if (flagErr != cudaSuccess) {
        fprintf(stderr,
                "Warning: cudaSetDeviceFlags(cudaDeviceMapHost) failed: %s\n",
                cudaGetErrorString(flagErr));
        mapped_possible = false;
    }

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Using device %d: %s\n", device, prop.name);
    printf("  canMapHostMemory = %d\n", prop.canMapHostMemory);

    if (!prop.canMapHostMemory || !mapped_possible) {
        printf("Mapped (zero-copy) test will be skipped on this device.\n");
        mapped_possible = false;
    }

    size_t bytes = static_cast<size_t>(N) * sizeof(float);

    // 1) Pageable host memory
    float* h_a_pageable = (float*)std::malloc(bytes);
    float* h_b_pageable = (float*)std::malloc(bytes);
    if (!h_a_pageable || !h_b_pageable) {
        fprintf(stderr, "Failed to allocate pageable host memory\n");
        return EXIT_FAILURE;
    }
    float h_result_pageable = 0.0f;

    // 2) Pinned host memory
    float* h_a_pinned = nullptr;
    float* h_b_pinned = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_a_pinned, bytes)); // page-locked
    CUDA_CHECK(cudaMallocHost(&h_b_pinned, bytes));
    float h_result_pinned = 0.0f;

    // 3) Pinned & mapped host memory (zero-copy)
    float* h_a_mapped = nullptr;
    float* h_b_mapped = nullptr;
    float* d_a_mapped = nullptr;
    float* d_b_mapped = nullptr;
    float h_result_mapped = 0.0f;

    if (mapped_possible) {
        CUDA_CHECK(cudaHostAlloc(&h_a_mapped, bytes, cudaHostAllocMapped));
        CUDA_CHECK(cudaHostAlloc(&h_b_mapped, bytes, cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(&d_a_mapped, h_a_mapped, 0));
        CUDA_CHECK(cudaHostGetDevicePointer(&d_b_mapped, h_b_mapped, 0));
    }
    
    // Device buffers used for cases 1 & 2 (and for the scalar result in case 3)
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_result = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));

    // Initialize data: simple pattern so we know the correct dot product.
    // a[i] = 1.0f, b[i] = 2.0f => dot = 2 * N
    for (int i = 0; i < N; ++i) {
        float aVal = rand() % 100 - 50;
        float bVal = rand() % 100 - 50;

        h_a_pageable[i] = aVal;
        h_b_pageable[i] = bVal;

        h_a_pinned[i] = aVal;
        h_b_pinned[i] = bVal;

        if (mapped_possible) {
            h_a_mapped[i] = aVal;
            h_b_mapped[i] = bVal;
        }
    }

    double ref = 0.0;
    for (int i = 0; i < N; ++i) {
        ref += (double)h_a_pageable[i] * (double)h_b_pageable[i];
    }

    const int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 65535) {
        blocks = 65535; // safeguard for 1D grid
    }
    size_t shmemBytes = (threads / WARP_SIZE) * sizeof(float);

    printf("Kernel config: blocks = %d, threads = %d, shmem = %zu bytes\n",
           blocks, threads, shmemBytes);

    // Warm-up to remove first-time jitters
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a_pageable, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b_pageable, bytes, cudaMemcpyHostToDevice));
    dot_kernel<<<blocks, threads, shmemBytes>>>(d_a, d_b, d_result, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 1) pageable benchmark
    float time_pageable_ms = run_pageable(h_a_pageable, h_b_pageable,
                                          &h_result_pageable,
                                          d_a, d_b, d_result,
                                          N, iters,
                                          blocks, threads, shmemBytes);

    // 2) pinned benchmark
    float time_pinned_ms = run_pinned(h_a_pinned, h_b_pinned,
                                      &h_result_pinned,
                                      d_a, d_b, d_result,
                                      N, iters,
                                      blocks, threads, shmemBytes);

    // 3) mapped benchmark
    float time_mapped_ms = 0.0f;
    if (mapped_possible) {
        time_mapped_ms = run_mapped(h_a_mapped, h_b_mapped,
                                    &h_result_mapped,
                                    d_result,
                                    d_a_mapped, d_b_mapped,
                                    N, iters,
                                    blocks, threads, shmemBytes);
    }

    printf("\nReference dot = %.6f (expected %.6f)\n",
           (float)ref, 2.0f * (float)N);

    printf("\n==== Results (avg over %d iters) ====\n", iters);
    printf("1) Pageable   (malloc + cudaMemcpy):\n");
    printf("   time = %.3f ms, result = %.6f\n",
           time_pageable_ms, h_result_pageable);

    printf("2) Pinned     (cudaMallocHost + cudaMemcpy):\n");
    printf("   time = %.3f ms, result = %.6f\n",
           time_pinned_ms, h_result_pinned);

    if (mapped_possible) {
        printf("3) Mapped     (cudaHostAllocMapped, zero-copy):\n");
        printf("   time = %.3f ms, result = %.6f\n",
               time_mapped_ms, h_result_mapped);
    } else {
        printf("3) Mapped     (zero-copy): skipped (not supported).\n");
    }

    // Clean up
    std::free(h_a_pageable);
    std::free(h_b_pageable);

    CUDA_CHECK(cudaFreeHost(h_a_pinned));
    CUDA_CHECK(cudaFreeHost(h_b_pinned));

    if (mapped_possible) {
        CUDA_CHECK(cudaFreeHost(h_a_mapped));
        CUDA_CHECK(cudaFreeHost(h_b_mapped));
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_result));

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}

/*

Dot product benchmark
N = 67108864 elements, iterations = 10
Using device 0: NVIDIA RTX A5000
  canMapHostMemory = 1
Kernel config: blocks = 65535, threads = 256, shmem = 32 bytes

Reference dot = 18278664.000000 (expected 134217728.000000)

==== Results (avg over 10 iters) ====
1) Pageable   (malloc + cudaMemcpy):
   time = 41.227 ms, result = 18278648.000000
2) Pinned     (cudaMallocHost + cudaMemcpy):
   time = 23.760 ms, result = 18278660.000000
3) Mapped     (cudaHostAllocMapped, zero-copy):
   time = 20.407 ms, result = 18278656.000000

*/
