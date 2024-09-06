#include <stdio.h>
#include <stdlib.h>
#include "assert.h" 

// CUDA runtime
#include <cuda_runtime.h>

#define OFFSET(row, col, ld) (row * ld + col)
#define FETCH_FLOAT(pointer) (reinterpret_cast<float4*>(&pointer)[0])


// Assume A, B and C are originally stored in global memory
template<
    const int BLOCK_SIZE_M,  // (bm) height of block of C that each  block calculate
    const int BLOCK_SIZE_K,  // (bk) width of block of A that each  block load into shared memory
    const int BLOCK_SIZE_N,  // (bn) width of block of C that each  block calculate
    const int RM, // (rm) height of block of C that each thread calculate
    const int RN,  // (rn) width of block of C that each thread calculate
    const bool ENABLE_DOUBLE_BUFFER // whether enable double buffering or not
>
__global__ void SGEMM(
    float * __restrict__ A, // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#restrict
    float * __restrict__ B,
    float * __restrict__ C, 
    const int M,
    const int N,
    const int K
){
    // ---------------------------------------------------------------------------
    // Part 1: Parameter Settings  
    int bx, by = blockIdx.x, blockIdx.y; // Block index
    int tx, ty = threadIdx.x, threadIdx.y; // Thread Index

    // the threads number in Block of X,Y
    const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / RN; // bn/rn
    const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / RM; // bm/rm
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

    // thread id in cur Block
    const int tid = ty * THREAD_X_PER_BLOCK + tx;

    // shared memory
    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];

    // registers for C
    float accum[RM][RN] = {0};

    // registers for A and B
    float frag_a[2][RM];
    float frag_b[2][RN];

    // ----------------------------------
    // Registers for loading global memory
    // To move data from global memory to shared memory, the data need to pass through registers first
    // For A, totally BLOCK_SIZE_M * BLOCK_SIZE_K float need to be moved in one large iteration
    // Considering each thread can fetch 4 float numbers (in data type float4), divide the summand by (THREAD_NUM_PER_BLOCK * 4)
    const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / (THREAD_NUM_PER_BLOCK * 4);
    const int ldg_num_b = BLOCK_SIZE_K * BLOCK_SIZE_N / (THREAD_NUM_PER_BLOCK * 4);
    float ldg_a_reg[4 * ldg_num_a];
    float ldg_b_reg[4 * ldg_num_b];

    // Threads number in one row
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

    // Row number and col number that needs to be loaded by this thread
    const int A_TILE_START_ROW = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_START_ROW = tid / B_TILE_THREAD_PER_ROW;

    // The column **index** of the starting element in the part to be fetched
    const int A_TILE_START_COL = tid % A_TILE_THREAD_PER_ROW * 4; 
    const int B_TILE_START_COL = tid % B_TILE_THREAD_PER_ROW * 4;

    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;
    

    // ---------------------------------------------------------------------------
    // Part 2: Data Prefetching before each large iteration

    // Part 2.1: Fetching the block of A into the shared memory
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE){
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;

        int orig_row = by * BLOCK_SIZE_M + A_TILE_START_ROW + i;
        // int orig_col = A_TILE_START_COL; 
        int orig_col = bx * BLOCK_SIZE_K + A_TILE_START_COL;
        FETCH_FLOAT(ldg_a_reg[ldg_index]) = FETCH_FLOAT(A[OFFSET(
                orig_row, orig_col, K
            )]
        );

        As[0][A_TILE_START_COL][A_TILE_START_ROW + i] = ldg_a_reg[ldg_index];
        As[0][A_TILE_START_COL+1][A_TILE_START_ROW + i] = ldg_a_reg[ldg_index + 1];
        As[0][A_TILE_START_COL+2][A_TILE_START_ROW + i] = ldg_a_reg[ldg_index + 2];
        As[0][A_TILE_START_COL+3][A_TILE_START_ROW + i] = ldg_a_reg[ldg_index + 3];
    }

    // Part 2.2: Fetching the block of B into the shared memory
    #pragma unroll
    for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
        // int orig_row = B_TILE_START_ROW + i;
        int orig_row = by * BLOCK_SIZE_K + B_TILE_START_ROW + i;
        int orig_col = bx * BLOCK_SIZE_N + B_TILE_START_COL;
        FETCH_FLOAT(Bs[0][B_TILE_START_ROW]) = FETCH_FLOAT(B[OFFSET(
            orig_row, orig_col, N
        )]);
    }
    __syncthreads();
    

    // Part 2.2: Fetching data from shared memory to register
    #pragma unroll
    for (int a_reg_i = 0; a_reg_i < RM; a_reg_i += 4)
        FETCH_FLOAT(frag_a[0][a_reg_i]) = FETCH_FLOAT(As)
}