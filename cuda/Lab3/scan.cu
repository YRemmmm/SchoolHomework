#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

double get_time_in_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000.0) + (ts.tv_nsec / 1000000.0);
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

void cpu_scan(int* A, int N)
{
	for(int i = 0; i < N; ++i)
	{
        for(int j = 1; j < N; ++j)
        {
            A[i * N + j] += A[i * N + j - 1];
        }
	}

    for(int i = 0; i < N; ++i)
    {
        for(int j = 1; j < N; ++j)
        {
            A[j * N + i] += A[(j - 1) * N + i];
        }
    }
}

__global__ void gpu_warm()
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void gpu_scan_basic_x(int* A, int N)
{
    __shared__ int sharedA[BLOCK_SIZE];
  	int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N * N)
    {
        sharedA[threadIdx.x] = A[i];
    }

    for(int stride = 1; stride <= threadIdx.x; stride *= 2)
    {
        __syncthreads();
        int inr = sharedA[threadIdx.x - stride];
        __syncthreads();
        sharedA[threadIdx.x] += inr;
    }       

    if(i < N * N)
    {
        A[i] = sharedA[threadIdx.x]; 
    }
}

__global__ void gpu_scan_basic_y(int* A, int N)
{
    __shared__ int sharedA[BLOCK_SIZE];
  	int i = blockDim.x * threadIdx.x + blockIdx.x;
    if(i < N * N)
    {
        sharedA[threadIdx.x] = A[i];
    }

    for(int stride = 1; stride <= threadIdx.x; stride *= 2)
    {
        __syncthreads();
        int inr = sharedA[threadIdx.x - stride];
        __syncthreads();
        sharedA[threadIdx.x] += inr;
    } 

    if(i < N * N)
    {
        A[i] = sharedA[threadIdx.x]; 
    }
}

__global__ void gpu_scan_better_x(int* A, int N)
{
    __shared__ int sharedA[BLOCK_SIZE];
  	int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N * N)
    {
        sharedA[threadIdx.x] = A[i];
    }

    for(int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index < BLOCK_SIZE) {
            sharedA[index] += sharedA[index - stride];
        }

    }       
    for (int stride = blockDim.x / 2;  stride  >  0; stride /= 2)  {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < BLOCK_SIZE) {
            sharedA[index + stride] += sharedA[index];
        }
    }
    __syncthreads();
    if(i < N * N)
    {
        A[i] = sharedA[threadIdx.x]; 
    }
}

__global__ void gpu_scan_better_y(int* A, int N)
{
    __shared__ int sharedA[BLOCK_SIZE];
  	int i = blockDim.x * threadIdx.x + blockIdx.x;
    if(i < N * N)
    {
        sharedA[threadIdx.x] = A[i];
    }

    for(int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index < BLOCK_SIZE) {
            sharedA[index] += sharedA[index - stride];
        }

    }       
    for (int stride = blockDim.x / 2;  stride  >  0; stride /= 2)  {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < BLOCK_SIZE) {
            sharedA[index + stride] += sharedA[index];
        }
    }
    __syncthreads();
    if(i < N * N)
    {
        A[i] = sharedA[threadIdx.x]; 
    }
}

int main (int argc, char argv[]) {
    
    int N = BLOCK_SIZE;

    size_t size_A = sizeof(int) * N * N;

    int *h_A;
    int *d_A;
    int *result;//for GPU sum result

    h_A = (int *)malloc(size_A);
    result = (int *)malloc(size_A);

    for(int i = 0; i < N; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
    	    h_A[i * N + j] = 1;
        }
    }


    cudaMalloc((int**)&d_A, size_A);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);

    //run on CPU
    double start, end;
    start = get_time_in_ms();

    cpu_scan(h_A, N);
    
    end = get_time_in_ms();
    printf("cpu cost: %f ms\n", end - start);

    //run on GPU
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerBlock((N * N + BLOCK_SIZE - 1) / (BLOCK_SIZE));
    gpu_warm<<<blocksPerBlock, threadsPerBlock>>>();
    start = get_time_in_ms();

    // gpu_scan_basic_x<<<blocksPerBlock, threadsPerBlock>>>(d_A, N);
    // cudaDeviceSynchronize();
    // gpu_scan_basic_y<<<blocksPerBlock, threadsPerBlock>>>(d_A, N);

    gpu_scan_better_x<<<blocksPerBlock, threadsPerBlock>>>(d_A, N);
    cudaDeviceSynchronize();
    gpu_scan_better_y<<<blocksPerBlock, threadsPerBlock>>>(d_A, N);

    cudaDeviceSynchronize();
    end = get_time_in_ms();
    printf("gpu cost: %f ms\n", end - start);

    //compare
    cudaMemcpy(result, d_A, size_A, cudaMemcpyDeviceToHost);
    
    double diff;
    diff = compare_matrices(h_A, result, N*N);
    if (diff > 0.5f || diff < -0.5f) {
      printf("diff too big !\n");
      exit(-1);
    }
    printf("Correct\n");
    
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);
    // printf("Compute capability: %d.%d\n", prop.major, prop.minor);

    free(h_A);
    free(result);

    cudaFree(d_A);

    return 0;
}
