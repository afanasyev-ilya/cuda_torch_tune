// fp32_gemm_80_20.cu (renamed & documented)
// A simplified, high-impact FP32 GEMM (80/20 version) with clearer naming.
//
// Concepts & naming
//   - Each thread block computes one C **block-tile** of size TILE_M x TILE_N
//   - We sweep the K dimension in **K-chunks** of size TILE_K
//   - Each thread accumulates a **micro-tile** (MICRO_M x MICRO_N) of C entirely in registers
//   - We stage one A **panel** (TILE_M x TILE_K) and one B **panel** (TILE_K x TILE_N) in shared memory
//
// Kept optimizations (the 80/20 that matters)
//   • Shared-memory tiling + register micro-tiles (boosts arithmetic intensity)
//   • Cooperative, coalesced loads to shared memory
//   • +1 padding in shared memory to avoid common bank conflicts on inner strides
//   • Straightforward timing harness + small CPU reference for correctness
//
// Not included (nice-to-haves you can add later)
//   • float4 vectorized loads, double-buffering / cp.async, swizzled layouts, split-K
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_75 -Xptxas -O3,-v -lineinfo fp32_gemm_80_20.cu -o gemm80
// Run:
//   ./gemm80 M N K [iters]
// Example:
//   ./gemm80 4096 4096 4096 50

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <iostream>
#include <cstring>

#ifndef CEIL_DIV
#define CEIL_DIV(x,y) (((x) + (y) - 1) / (y))
#endif

// ===================== Tunables (clear names) =====================
// One C block-tile computed per thread block
#ifndef TILE_M
#define TILE_M 128   // rows of C in a block-tile (and rows of A panel)
#endif
#ifndef TILE_N
#define TILE_N 128   // cols of C in a block-tile (and cols of B panel)
#endif
#ifndef TILE_K
#define TILE_K 16    // K-depth per iteration (A/B panel thickness)
#endif

// Per-thread micro-tile accumulated in registers
#ifndef MICRO_M
#define MICRO_M 8    // rows of C per thread
#endif
#ifndef MICRO_N
#define MICRO_N 8    // cols of C per thread
#endif

// Thread block layout (THREADS_X * THREADS_Y = 256 by default)
#ifndef THREADS_X
#define THREADS_X (TILE_N / MICRO_N)   // columns of threads per block
#endif
#ifndef THREADS_Y
#define THREADS_Y (TILE_M / MICRO_M)   // rows of threads per block
#endif

static_assert(TILE_M % MICRO_M == 0 && TILE_N % MICRO_N == 0, "TILE_M must be divisible by MICRO_M and TILE_N by MICRO_N");
static_assert(THREADS_X * THREADS_Y == 256, "Default launch uses 256 threads (adjust if you change tile sizes)");

// ===================== Error handling =====================
static void checkCuda(cudaError_t e, const char* file, int line) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA Error %s:%d: %s\n", file, line, cudaGetErrorString(e));
        std::abort();
    }
}
#define CHECK_CUDA(x) checkCuda((x), __FILE__, __LINE__)

// ===================== The kernel =====================
// Computes C[blockRow : blockRow+TILE_M, blockCol : blockCol+TILE_N]
// from A and B using K-chunked panels in shared memory. Row-major layout.
__global__ void __launch_bounds__(THREADS_X*THREADS_Y, 2)
sgemm_80_20_kernel(
    int M, int N, int K,
    const float* __restrict__ A, int lda,   // A is MxK, row-major, lda = K
    const float* __restrict__ B, int ldb,   // B is KxN, row-major, ldb = N
    float* __restrict__ C, int ldc,         // C is MxN, row-major, ldc = N
    float alpha, float beta)
{
    // Shared memory staging buffers (panels)
    // +1 padding on the innermost dimension reduces bank conflicts when kk sweeps
    __shared__ float shA[TILE_M][TILE_K + 1];  // A panel: TILE_M x TILE_K
    __shared__ float shB[TILE_K][TILE_N + 1];  // B panel: TILE_K x TILE_N

    // Thread coordinates inside the block
    const int tx = threadIdx.x; // [0, THREADS_X)
    const int ty = threadIdx.y; // [0, THREADS_Y)

    // Which C block-tile are we computing?
    const int blockRow = blockIdx.y * TILE_M; // top-left row of C block-tile
    const int blockCol = blockIdx.x * TILE_N; // top-left col of C block-tile

    // Top-left element of this thread's micro-tile within the block-tile
    const int rowBase = blockRow + ty * MICRO_M; // micro-tile origin row
    const int colBase = blockCol + tx * MICRO_N; // micro-tile origin col

    // Register fragment holding the MICRO_M x MICRO_N output submatrix for this thread
    float c_reg[MICRO_M][MICRO_N];
    #pragma unroll
    for (int i = 0; i < MICRO_M; ++i) {
        #pragma unroll
        for (int j = 0; j < MICRO_N; ++j) 
            c_reg[i][j] = 0.f;
    }

    // Flattened thread id for cooperative loads
    const int tId = ty * THREADS_X + tx; // 0..THREADS_X*THREADS_Y-1 (e.g., 0..255)
    const int threadsPerBlock = THREADS_X * THREADS_Y;

    // ---- Sweep the K dimension in chunks of TILE_K ----
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // 1) Load A panel (TILE_M x TILE_K) cooperatively into shA
        //    Each thread pulls multiple elements in a strided loop to cover the panel
        for (int idx = tId; idx < TILE_M * TILE_K; idx += threadsPerBlock) {
            int aRow = idx / TILE_K;      // [0, TILE_M)
            int aCol = idx % TILE_K;      // [0, TILE_K)
            int gRow = blockRow + aRow;   // global row in A
            int gCol = k0 + aCol;         // global col in A
            float v = 0.f;
            if (gRow < M && gCol < K) v = A[gRow * lda + gCol];
            shA[aRow][aCol] = v;          // coalesced across threads for contiguous gCol
        }

        // 2) Load B panel (TILE_K x TILE_N) cooperatively into shB
        for (int idx = tId; idx < TILE_K * TILE_N; idx += threadsPerBlock) {
            int bRow = idx / TILE_N;      // [0, TILE_K)
            int bCol = idx % TILE_N;      // [0, TILE_N)
            int gRow = k0 + bRow;         // global row in B
            int gCol = blockCol + bCol;   // global col in B
            float v = 0.f;
            if (gRow < K && gCol < N) v = B[gRow * ldb + gCol];
            shB[bRow][bCol] = v;          // coalesced across threads for contiguous gCol
        }

        __syncthreads(); // ensure shA/shB are fully populated before compute

        // 3) Compute this K-chunk contribution: outer-product style micro-kernel
        //    For each kk in [0, TILE_K):
        //      - Each thread loads a MICRO_M row vector from shA and a MICRO_N col vector from shB
        //      - Accumulates MICRO_M x MICRO_N FMAs into c_reg
        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            float a_reg[MICRO_M];
            float b_reg[MICRO_N];

            #pragma unroll
            for (int i = 0; i < MICRO_M; ++i) {
                int r = ty * MICRO_M + i;     // row inside the TILE_M block-tile
                a_reg[i] = shA[r][kk];
            }
            #pragma unroll
            for (int j = 0; j < MICRO_N; ++j) {
                int c = tx * MICRO_N + j;     // col inside the TILE_N block-tile
                b_reg[j] = shB[kk][c];
            }

            #pragma unroll
            for (int i = 0; i < MICRO_M; ++i) {
                float aVal = a_reg[i];
                #pragma unroll
                for (int j = 0; j < MICRO_N; ++j) {
                    c_reg[i][j] = fmaf(aVal, b_reg[j], c_reg[i][j]);
                }
            }
        }

        __syncthreads(); // reuse shA/shB for the next K-chunk
    }

    // 4) Write this thread's MICRO_M x MICRO_N results back to C
    #pragma unroll
    for (int i = 0; i < MICRO_M; ++i) {
        int row = rowBase + i;
        if (row < M) {
            #pragma unroll
            for (int j = 0; j < MICRO_N; ++j) {
                int col = colBase + j;
                if (col < N) {
                    int idx = row * ldc + col;
                    float out = alpha * c_reg[i][j];
                    if (beta != 0.f) 
                        out += beta * C[idx];
                    C[idx] = out;
                }
            }
        }
    }
}

// ===================== Host launcher =====================
// Launch one thread block per C block-tile. Grid covers ceil(N/TILE_N) x ceil(M/TILE_M) tiles.
void sgemm_80_20(
    int M, int N, int K,
    const float* A, int lda,    // lda = K for row-major contiguous A
    const float* B, int ldb,    // ldb = N for row-major contiguous B
    float* C, int ldc,          // ldc = N for row-major contiguous C
    float alpha = 1.f, float beta = 0.f,
    cudaStream_t stream = 0)
{
    dim3 block(THREADS_X, THREADS_Y, 1); // e.g., 16x16 = 256 threads
    dim3 grid(CEIL_DIV(N, TILE_N),       // number of C block-tiles across columns
              CEIL_DIV(M, TILE_M),       // number of C block-tiles across rows
              1);
    sgemm_80_20_kernel<<<grid, block, 0, stream>>>(M, N, K, A, lda, B, ldb, C, ldc, alpha, beta);
}

// ===================== Minimal Benchmark & Correctness =====================
void ref_gemm_cpu(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) acc += (double)A[i*lda + k] * (double)B[k*ldb + j];
            C[i*ldc + j] = (float)(alpha * acc + beta * C[i*ldc + j]);
        }
    }
}

float randf(){ return (float)rand() / (float)RAND_MAX - 0.5f; }

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096, iters = 5;
    if (argc >= 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }
    if (argc >= 5) { iters = atoi(argv[4]); }

    printf("(80/20) TILE_M=%d TILE_N=%d TILE_K=%d MICRO_M=%d MICRO_N=%d THREADS_X=%d THREADS_Y=%d\n",
           TILE_M, TILE_N, TILE_K, MICRO_M, MICRO_N, THREADS_X, THREADS_Y);
    printf("GEMM: %dx%dx%d, iters=%d\n", M, N, K, iters);

    const int lda = K, ldb = N, ldc = N; // row-major contiguous layout
    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    float *hA = (float*)malloc(bytesA);
    float *hB = (float*)malloc(bytesB);
    float *hC = (float*)malloc(bytesC);
    float *hC_ref = (float*)malloc(bytesC);

    srand(42);
    for (size_t i=0;i<(size_t)M*K;++i) hA[i]=randf();
    for (size_t i=0;i<(size_t)K*N;++i) hB[i]=randf();
    memset(hC, 0, bytesC);

    float *dA,*dB,*dC;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));
    CHECK_CUDA(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, bytesC));

    // Warmup (also catches config errors when run with CUDA_LAUNCH_BLOCKING=1)
    sgemm_80_20(M,N,K,dA,lda,dB,ldb,dC,ldc,1.f,0.f);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop; CHECK_CUDA(cudaEventCreate(&start)); CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaMemset(dC, 0, bytesC));
    CHECK_CUDA(cudaEventRecord(start));
    for (int it=0; it<iters; ++it) {
        sgemm_80_20(M,N,K,dA,lda,dB,ldb,dC,ldc,1.f,0.f);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms=0.f; CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop));
    std::cout << "ms: " << ms << std::endl;
    double avg_ms = ms / iters;
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops = flops / (avg_ms * 1e-3) / 1e12;
    printf("Our 80/20 kernel: %8.3f ms avg, %6.3f TFLOP/s\n", avg_ms, tflops);

    // Correctness: CPU reference on a smaller problem to avoid long runtimes
    int Mc = M, Nc = N, Kc = K;
    if ((size_t)M*(size_t)N > (size_t)1024*1024) { Mc = 512; Nc = 512; Kc = 512; }
    size_t bytesCc = (size_t)Mc * Nc * sizeof(float);
    float *hAc = hA, *hBc = hB; // reuse prefixes
    float *hCc = (float*)malloc(bytesCc);
    float *dCc; CHECK_CUDA(cudaMalloc(&dCc, bytesCc));
    CHECK_CUDA(cudaMemset(dCc, 0, bytesCc));
    sgemm_80_20(Mc,Nc,Kc,dA,lda,dB,ldb,dCc,Nc,1.f,0.f);
    CHECK_CUDA(cudaMemcpy(hCc, dCc, bytesCc, cudaMemcpyDeviceToHost));

    // CPU ref
    memset(hC_ref, 0, bytesCc);
    ref_gemm_cpu(Mc,Nc,Kc,1.f,hAc,lda,hBc,ldb,0.f,hC_ref,Nc);

    double max_abs=0.0, max_rel=0.0;
    for (size_t i=0;i<(size_t)Mc*Nc;++i){
        double a=hCc[i], b=hC_ref[i];
        double d=fabs(a-b); if(d>max_abs) max_abs=d;
        double r=d/(fabs(b)+1e-7); if(r>max_rel) max_rel=r;
    }
    printf("max |diff| = %.3e, max rel err = %.3e\n", max_abs, max_rel);

    // Cleanup
    free(hA); free(hB); free(hC); free(hC_ref); free(hCc);
    CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dB)); CHECK_CUDA(cudaFree(dC)); CHECK_CUDA(cudaFree(dCc));
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
