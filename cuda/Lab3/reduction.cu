#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 512

double get_time_in_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000.0) + (ts.tv_nsec / 1000000.0);
}

void cpu_reduction(int* A, int* sum, int N)
{
	for(int i = 0; i < N; ++i)
	{
        for(int j = 0; j < N; ++j)
        {
            sum[i] += A[i * N + j];
        }
	}
}

int compare_matrices(int *a, int *b, int n)
{
    int i;
    int max_diff = 0.0, diff;
    int printed = 0;

    for (i = 0; i < n; i++) {
        diff = abs(a[i] - b[i]);
        max_diff = (diff > max_diff ? diff : max_diff);
        if (0 == printed)
            if (max_diff > 0.5f || max_diff < -0.5f) {
            printf("\n error: i %d diff %d  got %d  expect %d ", i, max_diff, b[i], a[i]);
            printed = 1;
            }
    }

    return max_diff;
}

__global__ void gpu_warm(int* A, int* sum, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int sum_loc = 0;
}

__global__ void gpu_reduction_basic(int* A, int* sum, int N)
{
    __shared__ int partSum[2 * BLOCK_SIZE];
  	int tx = threadIdx.x;
    int start = 2 * blockDim.x * blockIdx.x;
    partSum[tx] = A[start + tx];
    partSum[tx + blockDim.x] = A[start + tx + blockDim.x];

    for(int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        __syncthreads();
        if(tx % stride == 0)
        {
            partSum[2 * tx] += partSum[2 * tx + stride];
        }
    }
    if(tx == 0)
    {
        sum[blockIdx.x] = partSum[0]; 
    }
}

__global__ void gpu_reduction_better(int* A, int* sum, int N)
{
    __shared__ int partSum[2 * BLOCK_SIZE];
  	int tx = threadIdx.x;
    int start = 2 * blockDim.x * blockIdx.x;
    partSum[tx] = A[start + tx];
    partSum[tx + blockDim.x] = A[start + tx + blockDim.x];

    for(int stride = blockDim.x; stride > 0; stride /= 2)
    {
        __syncthreads();
        if(tx < stride)
        {
            partSum[tx] += partSum[tx + stride];
        }
    }
    if(tx == 0)
    {
        sum[blockIdx.x] = partSum[0]; 
    }
}

int main (int argc, char argv[]) {
    
    int N = 1024;

    size_t size_A = sizeof(int) * N * N;
    size_t size_sum = sizeof(int) * N;

    int *h_A;
    int *h_sum;
    int *d_A;
    int *d_sum;
    int *result;//for GPU sum result

    h_A = (int *)malloc(size_A);
    h_sum = (int *)malloc(size_sum);
    result = (int *)malloc(size_sum);

    for(int i = 0; i < N; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
    	    h_A[i * N + j] = 1;
        }
        h_sum[i] = 0;
    }


    cudaMalloc((int**)&d_A, size_A);
    cudaMalloc((int**)&d_sum, size_sum);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);

    //run on CPU
    double start, end;
    start = get_time_in_ms();

    cpu_reduction(h_A, h_sum, N);
    
    end = get_time_in_ms();
    printf("cpu cost: %f ms\n", end - start);

    //run on GPU
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerBlock((N * N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2));
    gpu_warm<<<blocksPerBlock, threadsPerBlock>>>(d_A, d_sum, N);
    start = get_time_in_ms();

    // gpu_reduction_basic<<<blocksPerBlock, threadsPerBlock>>>(d_A, d_sum, N);

    gpu_reduction_better<<<blocksPerBlock, threadsPerBlock>>>(d_A, d_sum, N);
    
    cudaDeviceSynchronize();
    end = get_time_in_ms();
    printf("gpu cost: %f ms\n", end - start);

    //compare
    cudaMemcpy(result, d_sum, size_sum, cudaMemcpyDeviceToHost);
    
    double diff;
    diff = compare_matrices(h_sum, result, N);
    if (diff > 0.5f || diff < -0.5f) {
      printf("diff too big !\n");
      exit(-1);
    }
    printf("Correct\n");

    free(h_A);
    free(h_sum);
    free(result);

    cudaFree(d_A);
    cudaFree(d_sum);

    return 0;
}

