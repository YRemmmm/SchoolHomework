#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 512

double get_time_in_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000.0) + (ts.tv_nsec / 1000000.0);
}

int sort_compare(const void *a, const void *b) {
    return (*(float*)a > *(float*)b) - (*(float*)a < *(float*)b);
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

void cpu_merge(float *A, int m, float *B, int n, float *C)
{
    int i = 0, j = 0, k = 0;
    while (i < m && j < n)
    {
        if (A[i] <= B[j])
        {
            C[k++] = A[i++];
        }
        else
        {
            C[k++] = B[j++];
        }
    }
    while (i < m)
    {
        C[k++] = A[i++];
    }
    while (j < n)
    {
        C[k++] = B[j++];
    }
}

__global__ void gpu_warm()
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
}


__device__ int co_rank(int k, float *A, int m, float *B, int n)
{
    int i = k < m ? k : m;                 // i = min(k, m)
    int j = k - i;                         // j = k
    int i_low = 0 > (k - n) ? 0 : (k - n); // i_low = max(0, k - n)
    int j_low = 0 > (k - m) ? 0 : (k - m); // j_low = max(0, k - m)
    int delta;
    bool active = true;
    while (active)
    {
        if (i > 0 && j < n && A[i - 1] > B[j])
        {
            delta = ((i - i_low + 1) >> 1); // ceil((i - i_low) / 2)
            j_low = j;
            j += delta;
            i -= delta;
        }
        else if (j > 0 && i < m && B[j - 1] >= A[i])
        {
            delta = ((j - j_low  + 1) >> 1); // ceil((j - j_low) / 2)
            i_low = i;
            i += delta;
            j -= delta;
        }
        else
        {
            active = false;
        }
    }

    return i;
}


__device__ void merge_sequential(float *A, int m, float *B, int n, float *C)
{
    int i = 0; // index into A
    int j = 0; // index into B
    int k = 0; // index into C

    // handle the start of A[] and B[]
    while ((i < m) && (j < n))
    {
        if (A[i] <= B[j])
        {
            C[k++] = A[i++];
        }
        else
        {
            C[k++] = B[j++];
        }
    }

    if (i == m)
    {
        for (; j < n; j++)
        {
            C[k++] = B[j];
        }
    }
    else
    {
        for (; i < m; i++)
        {
            C[k++] = A[i];
        }
    }
}

__global__ void gpu_merge_basic(float *A, int m, float *B, int n, float *C)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int k_curr = tid*(int)ceilf((m+n)/(blockDim.x*gridDim.x)); 
    int k_next = min((tid+1)*(int)ceilf((m+n)/(blockDim.x*gridDim.x)), m+n); 
    int i_curr = co_rank(k_curr, A, m, B, n);
    int i_next = co_rank(k_next, A, m, B, n);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;
    merge_sequential(&A[i_curr], i_next-i_curr, &B[j_curr], j_next-j_curr, &C[k_curr] );
}

__global__ void gpu_merge_tiled(float *A, int m, float *B, int n, float *C, int tile_size)
{
    extern __shared__ float shareAB[];
    float* A_S = &shareAB[0];
    float* B_S = &shareAB[tile_size];
    int C_curr = blockIdx.x * (int)ceilf((m + n) / gridDim.x);
    int C_next = min((blockIdx.x + 1) * (int)ceilf((m + n) / gridDim.x), (m + n));

    __syncthreads();

    int A_curr = co_rank(C_curr, A, m, B, n);
    int A_next = co_rank(C_next, A, m, B, n);
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;

    __syncthreads();
    
    int counter = 0;
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int total_iteration = (int)ceilf((C_length)/(tile_size)); 
    
    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;
    while (counter < total_iteration) {
        __syncthreads();
        for (int i = 0; i < tile_size; i += blockDim.x) {
            if (i + threadIdx.x < A_length - A_consumed) {
                A_S[i + threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x];
            }
        }
        __syncthreads();
        for (int j = 0; j < tile_size; j += blockDim.x) {
            if (j + threadIdx.x < B_length - B_consumed) {
                B_S[j + threadIdx.x] = B[B_curr + B_consumed + j + threadIdx.x];
            }
        }
        __syncthreads();

        int c_curr = threadIdx.x * (tile_size / blockDim.x);
        int c_next = (threadIdx.x + 1) * (tile_size / blockDim.x);
        c_curr = (c_curr <= C_length - C_completed) ? c_curr : C_length - C_completed;
        c_next = (c_next <= C_length - C_completed) ? c_next : C_length - C_completed;

        int a_curr = co_rank(c_curr, A_S, min(tile_size, A_length - A_consumed), B_S, min(tile_size, B_length - B_consumed));
        int b_curr = c_curr - a_curr;
        int a_next = co_rank(c_next, A_S, min(tile_size, A_length - A_consumed), B_S, min(tile_size, B_length - B_consumed));
        int b_next = c_next - a_next;
      
        merge_sequential(A_S + a_curr, a_next - a_curr, B_S + b_curr, b_next - b_curr,
            C + C_curr + C_completed + c_curr);
        counter++;
        C_completed += tile_size;
        A_consumed += co_rank(tile_size, A_S, min(tile_size, A_length - A_consumed), B_S, min(tile_size, B_length - B_consumed));
        B_consumed = C_completed - A_consumed;

        __syncthreads();
    }
}

int main (int argc, char argv[]) {
    
    int N = 100000000;

    float *h_A = (float *)malloc((sizeof(float) * N));
    float *h_B = (float *)malloc((sizeof(float) * N));
    float *h_C = (float *)malloc((sizeof(float) * N * 2));

    for (int i = 0; i < N; ++i) {
        h_A[i] = ((float)rand() / (RAND_MAX)) * (10000 - 1) + 1;
    }
    for (int i = 0; i < N; ++i) {
        h_B[i] = ((float)rand() / (RAND_MAX)) * (10000 - 1) + 1;
    }
    qsort(h_A, N, sizeof(float), sort_compare);
    qsort(h_B, N, sizeof(float), sort_compare);
    
    float *d_A;
    float *d_B;
    float *d_C;
    cudaMalloc(&d_A, sizeof(float) * N);
    cudaMalloc(&d_B, sizeof(float) * N);
    cudaMalloc(&d_C, sizeof(float) * N * 2);
    cudaMemcpy(d_A, h_A, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * N, cudaMemcpyHostToDevice);
    
    float *result = (float *)malloc(sizeof(float) * N * 2);

    //run on CPU
    double start, end;
    start = get_time_in_ms();

    cpu_merge(h_A, N, h_B, N, h_C);
    
    end = get_time_in_ms();
    printf("cpu cost: %f ms\n", end - start);

    //run on GPU
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerBlock((2 * N + BLOCK_SIZE - 1) / (BLOCK_SIZE));

    gpu_warm<<<blocksPerBlock, threadsPerBlock>>>();
    start = get_time_in_ms();

    gpu_merge_basic<<<blocksPerBlock, threadsPerBlock>>>(d_A, N, d_B, N, d_C);
    // gpu_merge_tiled<<<blocksPerBlock, threadsPerBlock, sizeof(int)*2*BLOCK_SIZE>>>(d_A, N, d_B, N, d_C, BLOCK_SIZE);
    
    cudaDeviceSynchronize();
    end = get_time_in_ms();
    printf("gpu cost: %f ms\n", end - start);

    //compare
    cudaMemcpy(result, d_C, sizeof(float) * N * 2, cudaMemcpyDeviceToHost);
    double diff;
    diff = compare_matrices(h_C, result, N * 2);
    if (diff > 0.5f || diff < -0.5f) {
      printf("diff too big !\n");
      exit(-1);
    }
    printf("Correct\n");
    
    for (int i = 0; i < 50; i++)
    {
        printf("i = %d, cpu_result = %f, gpu_result = %f\n", i, h_C[i], result[i]);
    }

    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);
    // printf("Compute capability: %d.%d\n", prop.major, prop.minor);

    
    free(result);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
