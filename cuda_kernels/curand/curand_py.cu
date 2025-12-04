

// pi_monte_carlo.cu
#include <cstdio>
#include <cstdlib>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

// Инициализация Philox для каждого потока.
// subsequence = глобальный ID потока -> независимые подпоследовательности.
__global__ void init_rng(curandStatePhilox4_32_10_t* states, unsigned long long seed) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, /*subsequence*/ (unsigned long long)gid, /*offset*/ 0ULL, &states[gid]);
}

// Основное ядро: каждый поток генерирует samples_per_thread точек.
// Используем curand_uniform4: за итерацию получаем 2 точки (x1,y1) и (x2,y2).
__global__ void monte_carlo_pi(curandStatePhilox4_32_10_t* states,
                               unsigned long long* global_hits,
                               int samples_per_thread) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state = states[gid];

    unsigned int local_hits = 0;
    int i = 0;

    // Батчим по 2 точки на итерацию через uniform4
    for (; i + 1 < samples_per_thread; i += 2) {
        float4 u = curand_uniform4(&state); // (0,1]
        float x1 = u.x, y1 = u.y;
        float x2 = u.z, y2 = u.w;
        local_hits += (x1 * x1 + y1 * y1 <= 1.0f);
        local_hits += (x2 * x2 + y2 * y2 <= 1.0f);
    }

    // Если нечётное число точек — доберём одну
    if (i < samples_per_thread) {
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);
        local_hits += (x * x + y * y <= 1.0f);
    }

    // Блочная редукция в shared memory -> одна атомарка на блок
    __shared__ unsigned int smem[THREADS_PER_BLOCK];
    smem[threadIdx.x] = local_hits;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) smem[threadIdx.x] += smem[threadIdx.x + offset];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(global_hits, (unsigned long long)smem[0]);
    }

    states[gid] = state; // вернули состояние (полезно, если делать несколько прогонов)
}

int main(int argc, char** argv) {
    int blocks = (argc > 1) ? std::atoi(argv[1]) : 256;      // сетка
    int spt    = (argc > 2) ? std::atoi(argv[2]) : 1024;     // samples per thread

    size_t num_threads = (size_t)blocks * THREADS_PER_BLOCK;
    size_t total_samples = num_threads * (size_t)spt;

    printf("grid=%d blocks, block=%d threads, samples/thread=%d\n",
           blocks, THREADS_PER_BLOCK, spt);
    printf("total samples = %zu\n", total_samples);

    // Выделяем память
    curandStatePhilox4_32_10_t* d_states = nullptr;
    unsigned long long* d_hits = nullptr;
    cudaMalloc(&d_states, num_threads * sizeof(curandStatePhilox4_32_10_t));
    cudaMalloc(&d_hits, sizeof(unsigned long long));
    cudaMemset(d_hits, 0, sizeof(unsigned long long));

    // Инициализация RNG
    unsigned long long seed = 123456789ULL;
    init_rng<<<blocks, THREADS_PER_BLOCK>>>(d_states, seed);
    cudaDeviceSynchronize();

    // Запуск вычислений
    monte_carlo_pi<<<blocks, THREADS_PER_BLOCK>>>(d_states, d_hits, spt);
    cudaDeviceSynchronize();

    // Результаты
    unsigned long long h_hits = 0;
    cudaMemcpy(&h_hits, d_hits, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    double pi_est = 4.0 * (double)h_hits / (double)total_samples;
    printf("hits inside quarter circle = %llu\n", (unsigned long long)h_hits);
    printf("pi ≈ %.6f\n", pi_est);

    cudaFree(d_states);
    cudaFree(d_hits);
    return 0;
}