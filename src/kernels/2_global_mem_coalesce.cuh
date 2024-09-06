#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(
    int M, int N, int K, float alpha,
    const float *A, const float *B,
    float beta, float *C
) {
    const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (x < M && y < N){
        float temp = 0.;
        for (int k = 0; k < K; k++)
            temp += A[x * K + k] * B[k * N + y];
        
        C[x * N + y] = alpha * temp + beta * C[x * N + y];
    }
}