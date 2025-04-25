#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define MAX_NNZ_PER_ROW 1

double get_time_in_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000.0) + (ts.tv_nsec / 1000000.0);
}


float compare_matrices(float *a, float *b, int n)
{
    int i;
    float max_diff = 0.0, diff;
    int printed = 0;

    for (i = 0; i < n; i++) {
        diff = abs(a[i] - b[i]);
        max_diff = (diff > max_diff ? diff : max_diff);
        if (0 == printed)
            if (max_diff > 0.5f || max_diff < -0.5f) {
            printf("\n error: i %d diff %f  got %f  expect %f ", i, max_diff, b[i], a[i]);
            printed = 1;
            }
    }

    return max_diff;
}

void cpu_spmv_ell(int N, int *colIndex, float *values, float *vector, float *result)
{
    for (int row = 0; row < N; ++row)
    {
        result[row] = 0.0f;
        for (int j = 0; j < MAX_NNZ_PER_ROW; ++j)
        {
            if (j < MAX_NNZ_PER_ROW)
            {
                result[row] += values[row * MAX_NNZ_PER_ROW + j] * vector[colIndex[row * MAX_NNZ_PER_ROW + j]];
            }
        }
    }
}

__global__ void gpu_warm()
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void gpu_spmv_ell(int N, int *colIndex, float *values, float *vector, float *result)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < N)
    {
        result[row] = 0.0f;
        for (int j = 0; j < MAX_NNZ_PER_ROW; ++j)
        {
            if (j < MAX_NNZ_PER_ROW)
            {
                result[row] += values[row * MAX_NNZ_PER_ROW + j] * vector[colIndex[row * MAX_NNZ_PER_ROW + j]];
            }
        }
    }
}

int main (int argc, char argv[]) {
    
    int N = 10000;

    float *cpu_result = (float *)malloc(sizeof(float) * N);
    float *h_vector   = (float *)malloc(sizeof(float) * N);
    float *h_values   = (float *)malloc(sizeof(float) * N * MAX_NNZ_PER_ROW);
    int   *h_colIndex   = (int *)malloc(sizeof(int) * N * MAX_NNZ_PER_ROW);
    for (int i = 0; i < N; i++) {
        h_vector[i] = 1.0 + i;
        h_values[i] = 1.0;
        h_colIndex[i] = i;
    }

    float *h_result = (float *)malloc(sizeof(float) * N);
    float *d_result;
    float *d_vector;
    float *d_values;
    int   *d_colIndex;
    
    cudaMalloc(&d_result, sizeof(float) * N);
    cudaMalloc(&d_vector, sizeof(float) * N);
    cudaMalloc(&d_values, sizeof(float) * N);
    cudaMalloc(&d_colIndex, sizeof(int) * N);

    cudaMemcpy(d_vector, h_vector, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIndex, h_colIndex, sizeof(int) * N, cudaMemcpyHostToDevice);


    //run on CPU
    double start, end;
    start = get_time_in_ms();

    cpu_spmv_ell(N, h_colIndex, h_values, h_vector, cpu_result);
    
    end = get_time_in_ms();
    printf("cpu cost: %f ms\n", end - start);

    //run on GPU
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerBlock((N + BLOCK_SIZE - 1) / (BLOCK_SIZE));

    gpu_warm<<<blocksPerBlock, threadsPerBlock>>>();
    start = get_time_in_ms();

    gpu_spmv_ell<<<blocksPerBlock, threadsPerBlock>>>(N, d_colIndex, d_values, d_vector, d_result);
    
    cudaDeviceSynchronize();
    end = get_time_in_ms();
    printf("gpu cost: %f ms\n", end - start);

    //compare
    cudaMemcpy(h_result, d_result, sizeof(float) * N, cudaMemcpyDeviceToHost);
    double diff;
    diff = compare_matrices(h_result, cpu_result, N);
    if (diff > 0.5f || diff < -0.5f) {
      printf("diff too big !\n");
      exit(-1);
    }
    printf("Correct\n");
    
    for (int i = 0; i < 10; i++)
    {
        printf("i = %d, cpu_result = %f, gpu_result = %f\n", i, cpu_result[i], h_result[i]);
    }
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);
    // printf("Compute capability: %d.%d\n", prop.major, prop.minor);


    return 0;
}
