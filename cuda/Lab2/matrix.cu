#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double CpuMatirxMul(int *C, int *A, int *B, int Aw, int Ah, int Bw, int Bh);
double GpuMatirxMul(int *C, int *A, int *B, int Aw, int Ah, int Bw, int Bh);
double GpuMatirxMulShared(int *C, int *A, int *B, int Aw, int Ah, int Bw, int Bh);
int verify_results(int *C, int *D, int Cw, int Ch);

int main() {
    
    int *A, *B, *C, *D;
    int Aw, Ah, Bw, Bh, Cw, Ch;
    int A_value, B_value;
    
    A_value = 7;
    B_value = 0;
    
    //init A & B
    Aw = 2047;
    Ah = 1023;
    Bw = 511;
    Bh = 2047;
    if (Aw == Bh)
    {
        Ch = Ah;
        Cw = Bw;
    } else {
        return 0;
    }
    A = (int *)malloc(sizeof(int) * Aw * Ah);
    B = (int *)malloc(sizeof(int) * Bw * Bh);
    C = (int *)malloc(sizeof(int) * Cw * Ch);
    D = (int *)malloc(sizeof(int) * Cw * Ch);
    for (int i = 0; i < Aw * Ah; i++)
    {
        A[i] = A_value;
    }
    for (int i = 0; i < Bw * Bh; i++)
    {
        B[i] = B_value;
    }
    for (int i = 0; i < Cw * Ch; i++)
    {
        C[i] = 0;
    }
    for (int i = 0; i < Cw * Ch; i++)
    {
        D[i] = 0;
    }
    
    
    double cpu_time_used, gpu_time_used, gpu_shared_time_used, speedup;
    
    cpu_time_used = CpuMatirxMul(C, A, B, Aw, Ah, Bw, Bh);
    gpu_time_used = GpuMatirxMul(D, A, B, Aw, Ah, Bw, Bh);
    speedup = cpu_time_used / gpu_time_used;
    
    
    // Verify results
    int result_correct = verify_results(C, D, Cw, Ch);
    if (result_correct) {
        printf("Results are correct.\n");
    } else {
        printf("Results are incorrect.\n");
    }
    
    //time
    printf("CPU taken: %f ms\n", cpu_time_used);
    printf("GPU taken: %f ms\n", gpu_time_used);
    printf("G/C Speedup: %f\n", speedup);
    
    
    gpu_shared_time_used = GpuMatirxMulShared(D, A, B, Aw, Ah, Bw, Bh);
    speedup = gpu_time_used / gpu_shared_time_used;
    
    
    // Verify results
    result_correct = verify_results(C, D, Cw, Ch);
    if (result_correct) {
        printf("Results are correct.\n");
    } else {
        printf("Results are incorrect.\n");
    }
    
    //time
    printf("GPU taken: %f ms\n", gpu_time_used);
    printf("GPU Shared taken: %f ms\n", gpu_shared_time_used);
    printf("G/S Speedup: %f\n", speedup);

    return 0;
}


double CpuMatirxMul(int *C, int *A, int *B, int Aw, int Ah, int Bw, int Bh)
{
    clock_t start = clock();
    for (int i = 0; i < Ah; i++)
    {
        for (int k = 0; k < Aw; k++)
        {
            for (int j = 0; j < Bw; j++)
            {
                C[i * Bw + j] += A[i * Aw + k] * B[k * Bw + j];
            }
        }
    }    
    clock_t end = clock();
    double time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    return time_used * 1000;
}

__global__ void GpuMatirxMulKernel(int *C, int *A, int *B, int Aw, int Ah, int Bw, int Bh)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < Ah && j < Bw) 
    {
        for (int k = 0; k < Aw; k++)
        {
            C[i * Bw + j] += A[i * Aw + k] * B[k * Bw + j];
        } 
    }

}


__global__ void GpuMatirxMulKernelShared(int *C, int *A, int *B, int Aw, int Ah, int Bw, int Bh) {
    // 共享内存声明
    __shared__ int As[16][16];
    __shared__ int Bs[16][16];

    int i = blockIdx.y * 16 + threadIdx.y;
    int j = blockIdx.x * 16 + threadIdx.x;
    
    int kBegin = 16 * blockIdx.x;
    int kEnd = kBegin + 16;
    if (i < Ah && j < Bw) {
        int sum = 0;
        
        for (int k = kBegin; k < kEnd && k < Aw; k += 16) {
            As[threadIdx.y][threadIdx.x] = A[i * Aw + k];
            Bs[threadIdx.y][threadIdx.x] = B[k * Bw + j];
            
            __syncthreads();
            
            for (int m = 0; m < 16; ++m) {
                sum += As[threadIdx.y][m] * Bs[m][threadIdx.x];
            }
            __syncthreads();
        }
        
        C[i * Bw + j] = sum;
    }
}

double GpuMatirxMul(int *C, int *A, int *B, int Aw, int Ah, int Bw, int Bh)
{
    int size_A = Aw * Ah * sizeof(int);
    int size_B = Bw * Bh * sizeof(int);
    int size_C = Ah * Bw * sizeof(int);
    
    int *d_A, *d_B, *d_C;
    
    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);
    
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size_C, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((Bw + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (Ah + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    
    //////////////////////////
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    GpuMatirxMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, Aw, Ah, Bw, Bh);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    /////////////////////////
    
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    
    return elapsedTime;
}


double GpuMatirxMulShared(int *C, int *A, int *B, int Aw, int Ah, int Bw, int Bh)
{
    int size_A = Aw * Ah * sizeof(int);
    int size_B = Bw * Bh * sizeof(int);
    int size_C = Ah * Bw * sizeof(int);
    
    int *d_A, *d_B, *d_C;
    
    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);
    
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size_C, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((Bw + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (Ah + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    
    //////////////////////////
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    GpuMatirxMulKernelShared<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, Aw, Ah, Bw, Bh);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    /////////////////////////
    
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    
    return elapsedTime;
}

int verify_results(int *C, int *D, int Cw, int Ch) {
    for (int i = 0; i < Cw * Ch; i++) {
        if (C[i] != D[i]) {
            return 0; // Results do not match
        }
    }
    return 1; // Results match
}