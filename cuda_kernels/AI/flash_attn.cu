


// flash_attention_fused.cu
// Single-pass fused O = softmax(Q K^T / sqrt(d)) · V
// - Uses your 128×128×TILE_K GEMM tiling (no double-buffering yet)
// - Streams over key tiles (N) inside the CTA (grid.x tiles Dv, grid.y tiles M)
// - Online softmax with per-row (m, l) in shared
// - 8-wide stripes across the 128-key tile to do E8·V8 micro-GEMMs

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cstdio>
#include <cmath>

// ---- your tunables (keep as-is) ----
#ifndef TILE_M
#define TILE_M 128
#endif
#ifndef TILE_N
#define TILE_N 128
#endif
#ifndef TILE_K
#define TILE_K 16
#endif

#ifndef BX
#define BX 16
#endif
#ifndef BY
#define BY 16
#endif

#ifndef MICRO_M
#define MICRO_M 8
#endif
#ifndef MICRO_N
#define MICRO_N 8
#endif

// Tile over Dv: map THREADS_X * MICRO_N columns of V per CTA (defaults to 128)
#ifndef TILE_DV
#define TILE_DV (BX * MICRO_N)
#endif

static_assert(BX == TILE_N / MICRO_N, "BX must be TILE_N / MICRO_N");
static_assert(BY == TILE_M / MICRO_M, "BY must be TILE_M / MICRO_M");
static_assert(TILE_DV % MICRO_N == 0, "TILE_DV must be multiple of MICRO_N");

#ifndef CEIL_DIV
#define CEIL_DIV(x,y) (((x) + (y) - 1) / (y))
#endif

// -------------------------------- Kernel --------------------------------

template <typename scalar_t>
__global__ void flash_attention_kernel(
    const scalar_t* __restrict__ Q,   // [B, M, K]
    const scalar_t* __restrict__ K,   // [B, N, K]  (row-major)
    const scalar_t* __restrict__ V,   // [B, N, Dv] (row-major)
    scalar_t* __restrict__ O,         // [B, M, Dv]
    size_t B_size,
    size_t M_size,
    size_t N_size,
    size_t K_size,
    size_t DV_size,
    scalar_t scale)                   // usually 1/sqrt((float)K_size)
{
    const int lda = K_size;   // Q: row-major [M x K]
    const int ldk = K_size;   // K: row-major [N x K]
    const int ldv = DV_size;  // V: row-major [N x Dv]
    const int ldo = DV_size;  // O: row-major [M x Dv]

    const int batch_idx = blockIdx.z;

    const size_t Q_offset = (size_t)batch_idx * M_size * K_size;
    const size_t K_offset = (size_t)batch_idx * N_size * K_size;
    const size_t V_offset = (size_t)batch_idx * N_size * DV_size;
    const size_t O_offset = (size_t)batch_idx * M_size * DV_size;

    // Shared panels (same as your GEMM, +1 pad to avoid common bank conflicts)
    __shared__ float A_shared[TILE_M][TILE_K + 1];   // Q panel: 128 x TILE_K
    __shared__ float B_shared[TILE_K][TILE_N + 1];   // K^T panel: TILE_K x 128

    // Online softmax per-row state for the 128 rows owned by this CTA
    __shared__ float row_m[TILE_M];                  // running max
    __shared__ float row_l[TILE_M];                  // running sum of exp(score - m)

    // Reduction scratch across THREADS_X (=BX) for 128 rows
    __shared__ float tile_row_max[TILE_M][BX];       // store per-thread partial maxima
    __shared__ float tile_row_sum[TILE_M][BX];       // store per-thread partial sums

    // 8-wide slices for E8·V8 micro-GEMMs (K=8)
    __shared__ float E8[TILE_M][MICRO_N];            // 128 x 8, exp(score - new_m)
    __shared__ float V8[MICRO_N][TILE_DV + 1];       // 8 x TILE_DV (+1 pad)

    // Init softmax state
    if (threadIdx.x == 0) {
        for (int r = 0; r < TILE_M; ++r) {
            row_m[r] = -INFINITY;
            row_l[r] = 0.f;
        }
    }
    __syncthreads();

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int block_size = blockDim.x * blockDim.y;

    // This CTA writes O for rows [M tile] and Dv columns [dv_start : dv_start+TILE_DV)
    const int dv_start = blockIdx.x * TILE_DV;

    // Per-thread micro-tile for O over Dv (MICRO_N columns)
    float O_reg[MICRO_M][MICRO_N];
    #pragma unroll
    for (int i = 0; i < MICRO_M; ++i)
    #pragma unroll
    for (int j = 0; j < MICRO_N; ++j)
        O_reg[i][j] = 0.f;

    // ---------------- stream over key tiles (128 columns) ----------------
    for (int n_start = 0; n_start < (int)N_size; n_start += TILE_N) {

        // ---- compute scores tile C_reg (128x128) = Q_tile * (K_tile)^T
        float C_reg[MICRO_M][MICRO_N];
        #pragma unroll
        for (int i = 0; i < MICRO_M; ++i)
        #pragma unroll
        for (int j = 0; j < MICRO_N; ++j)
            C_reg[i][j] = 0.f;

        for (int k_start = 0; k_start < (int)K_size; k_start += TILE_K) {
            // Load Q panel [128 x TILE_K]
            #pragma unroll
            for (int ii = tid; ii < TILE_M * TILE_K; ii += block_size) {
                int lr = ii / TILE_K;      // 0..127
                int lc = ii % TILE_K;      // 0..TILE_K-1
                int gk = k_start + lc;
                int gm = blockIdx.y * TILE_M + lr;
                float v = 0.f;
                if (gk < (int)K_size && gm < (int)M_size) {
                    size_t idx = (size_t)gm * lda + gk + Q_offset;
                    v = Q[idx];
                }
                A_shared[lr][lc] = v;
            }

            // Load K panel^T into B_shared [TILE_K x 128]
            #pragma unroll
            for (int ii = tid; ii < TILE_N * TILE_K; ii += block_size) {
                int lk = ii / TILE_N;      // 0..TILE_K-1
                int ln = ii % TILE_N;      // 0..127
                int gk = k_start + lk;
                int gn = n_start + ln;
                float v = 0.f;
                if (gk < (int)K_size && gn < (int)N_size) {
                    size_t idx = (size_t)gn * ldk + gk + K_offset; // row-major [N x K]
                    v = K[idx];
                }
                B_shared[lk][ln] = v;  // transpose-on-store
            }

            __syncthreads();

            // Compute micro-kernel: C_reg += A_shared * B_shared
            #pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk) {
                float a_frag[MICRO_M];
                float b_frag[MICRO_N];

                #pragma unroll
                for (int i = 0; i < MICRO_M; ++i)
                    a_frag[i] = A_shared[i + MICRO_M * threadIdx.y][kk];

                #pragma unroll
                for (int j = 0; j < MICRO_N; ++j)
                    b_frag[j] = B_shared[kk][j + MICRO_N * threadIdx.x];

                #pragma unroll
                for (int i = 0; i < MICRO_M; ++i)
                #pragma unroll
                for (int j = 0; j < MICRO_N; ++j)
                    C_reg[i][j] = fmaf(a_frag[i], b_frag[j], C_reg[i][j]);
            }

            __syncthreads();
        } // end K loop

        // Scale scores by 1/sqrt(d)
        #pragma unroll
        for (int i = 0; i < MICRO_M; ++i)
        #pragma unroll
        for (int j = 0; j < MICRO_N; ++j)
            C_reg[i][j] *= (float)scale;

        // ---- Row-wise tile max → new_m
        float row_max_local[MICRO_M];
        #pragma unroll
        for (int i = 0; i < MICRO_M; ++i) 
            row_max_local[i] = -INFINITY;

        #pragma unroll
        for (int i = 0; i < MICRO_M; ++i)
        #pragma unroll
        for (int j = 0; j < MICRO_N; ++j)
            row_max_local[i] = fmaxf(row_max_local[i], C_reg[i][j]);

        #pragma unroll
        for (int i = 0; i < MICRO_M; ++i) {
            int r = threadIdx.y * MICRO_M + i;
            tile_row_max[r][threadIdx.x] = row_max_local[i];
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            #pragma unroll
            for (int g = 0; g < MICRO_M; ++g) {
                int r = threadIdx.y * MICRO_M + g;
                float mx = -INFINITY;
                #pragma unroll
                for (int c = 0; c < BX; ++c) mx = fmaxf(mx, tile_row_max[r][c]);
                tile_row_max[r][0] = fmaxf(row_m[r], mx); // new_m
            }
        }
        __syncthreads();

        float new_m_local[MICRO_M];
        #pragma unroll
        for (int i = 0; i < MICRO_M; ++i) {
            int r = threadIdx.y * MICRO_M + i;
            new_m_local[i] = tile_row_max[r][0];
        }

        // ---- Rescale Ō by exp(old_m - new_m)
        #pragma unroll
        for (int i = 0; i < MICRO_M; ++i) {
            int r = threadIdx.y * MICRO_M + i;
            float rescale = __expf(row_m[r] - new_m_local[i]);
            #pragma unroll
            for (int j = 0; j < MICRO_N; ++j)
                O_reg[i][j] *= rescale;
        }

        // zero partial row-sum slots
        #pragma unroll
        for (int i = 0; i < MICRO_M; ++i) {
            int r = threadIdx.y * MICRO_M + i;
            tile_row_sum[r][threadIdx.x] = 0.f;
        }
        __syncthreads();

        // ---- Stream the 128-key tile in 8-wide stripes and do E8·V8
        for (int owner = 0; owner < BX; ++owner) {
            // owner writes E8 = exp(score - new_m) for its 8 columns
            if (threadIdx.x == owner) {
                #pragma unroll
                for (int i = 0; i < MICRO_M; ++i) {
                    int r = threadIdx.y * MICRO_M + i;
                    float sum_part = 0.f;
                    #pragma unroll
                    for (int j = 0; j < MICRO_N; ++j) {
                        float e = __expf(C_reg[i][j] - new_m_local[i]);
                        E8[r][j] = e;
                        sum_part += e;
                    }
                    tile_row_sum[r][threadIdx.x] += sum_part; // accumulate per stripe
                }
            }
            __syncthreads();

            // load V8 for these 8 keys into shared: V8[8 x TILE_DV]
            #pragma unroll
            for (int ii = tid; ii < MICRO_N * TILE_DV; ii += block_size) {
                int j8 = ii / TILE_DV;            // 0..7
                int dv = ii % TILE_DV;            // 0..TILE_DV-1
                int gk = n_start + owner * MICRO_N + j8; // global key index
                int gdv = dv_start + dv;                 // global dv column
                float v = 0.f;
                if (gk < (int)N_size && gdv < (int)DV_size) {
                    size_t vidx = (size_t)gk * ldv + gdv + V_offset;
                    v = V[vidx];
                }
                V8[j8][dv] = v;
            }
            __syncthreads();

            // Ō += E8 · V8  (K=8 micro-GEMM)
            #pragma unroll
            for (int i = 0; i < MICRO_M; ++i) {
                int r = threadIdx.y * MICRO_M + i;
                #pragma unroll
                for (int j = 0; j < MICRO_N; ++j) {   // per-thread dv micro-columns
                    int dv_local = j;                 // 0..7
                    int dv_col   = threadIdx.x * MICRO_N + dv_local; // 0..TILE_DV-1
                    if (dv_start + dv_col < (int)DV_size) {
                        float acc = O_reg[i][dv_local];
                        #pragma unroll
                        for (int k8 = 0; k8 < MICRO_N; ++k8) {
                            acc = fmaf(E8[r][k8], V8[k8][dv_col], acc);
                        }
                        O_reg[i][dv_local] = acc;
                    }
                }
            }
            __syncthreads();
        } // end stripes

        // ---- Reduce row sums across BX and update (l, m)
        if (threadIdx.x == 0) {
            #pragma unroll
            for (int g = 0; g < MICRO_M; ++g) {
                int r = threadIdx.y * MICRO_M + g;
                float tsum = 0.f;
                #pragma unroll
                for (int c = 0; c < BX; ++c) tsum += tile_row_sum[r][c];
                float old_m = row_m[r];
                float new_m = new_m_local[g];
                row_l[r] = row_l[r] * __expf(old_m - new_m) + tsum;
                row_m[r] = new_m;
            }
        }
        __syncthreads();

    } // end loop over key tiles

    // ---- finalize: divide by l and store O tile
    #pragma unroll
    for (int i = 0; i < MICRO_M; ++i) {
        int gm = blockIdx.y * TILE_M + threadIdx.y * MICRO_M + i;  // global row
        if (gm >= (int)M_size) continue;
        float denom = row_l[threadIdx.y * MICRO_M + i];
        #pragma unroll
        for (int j = 0; j < MICRO_N; ++j) {
            int dv_col = threadIdx.x * MICRO_N + j;
            int gdv    = dv_start + dv_col;
            if (gdv < (int)DV_size) {
                size_t oidx = (size_t)gm * ldo + gdv + O_offset;
                O[oidx] = (scalar_t)(O_reg[i][j] / denom);
            }
        }
    }
}

// -------------------------------- Wrapper --------------------------------

torch::Tensor flash_attention_forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, float scale)
{
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "inputs must be on cuda");
    TORCH_CHECK(Q.dtype() == torch::kFloat32 &&
                K.dtype() == torch::kFloat32 &&
                V.dtype() == torch::kFloat32, "inputs must be fp32");

    // Shapes: Q[B,M,K], K[B,N,K], V[B,N,Dv]
    const int64_t B  = Q.size(0);
    const int64_t M  = Q.size(1);
    const int64_t Kd = Q.size(2);
    TORCH_CHECK(K.size(0)==B && K.size(2)==Kd, "K must be [B,N,K] matching Q");
    TORCH_CHECK(V.size(0)==B && V.size(1)==K.size(1), "V must share [B,N] with K");
    const int64_t N  = K.size(1);
    const int64_t Dv = V.size(2);

    auto opts = Q.options();
    torch::Tensor O = torch::empty({B, M, Dv}, opts);

    dim3 block(BX, BY, 1);
    dim3 grid( CEIL_DIV((int)Dv, TILE_DV),
               CEIL_DIV((int)M,  TILE_M),
               (int)B );

    auto stream = at::cuda::getCurrentCUDAStream();

    flash_attention_kernel<float><<<grid, block, 0, stream.stream()>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        (size_t)B, (size_t)M, (size_t)N, (size_t)Kd, (size_t)Dv,
        (float)scale
    );

#ifdef TIME_FLOPS
    // Optional timing in your style
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start, stream.stream());
    cudaEventRecord(stop,  stream.stream());
    cudaEventSynchronize(stop);
    float ms=0.f; cudaEventElapsedTime(&ms, start, stop);
    double sec = ms/1e3;
    double flops_qk = 2.0 * (double)B * (double)M * (double)N * (double)Kd;
    double flops_pv = 2.0 * (double)B * (double)M * (double)N * (double)Dv;
    double flops_total = flops_qk + flops_pv;
    std::cout << "FlashAttention shapes: Q["<<M<<"x"<<Kd<<"], K["<<N<<"x"<<Kd<<"], V["<<N<<"x"<<Dv<<"]\n";
    std::cout << "  QK^T FLOPs: " << flops_qk/1e12 << "e12,  P·V FLOPs: " << flops_pv/1e12 << "e12\n";
    std::cout << "  Total TFLOP/s: " << flops_total / (sec*1e12) << std::endl;
    cudaEventDestroy(start); cudaEventDestroy(stop);
#endif

    return O;
}
