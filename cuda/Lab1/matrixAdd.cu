#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double CpuMatirxAdd(int *C, int *A, int *B, int Aw, int Ah, int Bw, int Bh);
double GpuMatirxAdd(int *C, int *A, int *B, int Aw, int Ah, int Bw, int Bh);
int verify_results(int *C, int *D, int Cw, int Ch);

int main() {
    
    int *A, *B, *C, *D;
    int Aw, Ah, Bw, Bh, Cw, Ch;
    int width, height;
    int A_value, B_value;
    
    width  = 2048;
    height = 1024;
    A_value = 7;
    B_value = 0;
    
    //init A & B
    Aw = width;
    Ah = height;
    Bw = width;
    Bh = height;
    if (Aw == Bw && Ah == Bh)
    {
        Ch = Ah;
        Cw = Aw;
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
    
    
    double cpu_time_used, gpu_time_used;
    
    cpu_time_used = CpuMatirxAdd(C, A, B, Aw, Ah, Bw, Bh);
    gpu_time_used = GpuMatirxAdd(D, A, B, Aw, Ah, Bw, Bh);
    
    
    // Verify results
    int result_correct = verify_results(C, D, Cw, Ch);
    if (result_correct) {
        printf("Results are correct.\n");
    } else {
        printf("Results are incorrect.\n");
    }
    
    //time
    printf("CPU taken: %f seconds\n", cpu_time_used);
    printf("GPU taken: %f seconds\n", gpu_time_used);
    
    return 0;
}


double CpuMatirxAdd(int *C, int *A, int *B, int Aw, int Ah, int Bw, int Bh)
{
    clock_t start = clock();
    for (int i = 0; i < Ah; i++)
    {
        for (int j = 0; j < Aw; j++)
        {
            C[i * Aw + j] = A[i * Aw + j] + B[i * Aw + j];
        }
    }    
    clock_t end = clock();
    double time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    return time_used;
}

__global__ void GpuMatirxAddKernel(int *C, int *A, int *B, int Aw, int Ah, int Bw, int Bh)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < Ah && j < Aw) 
    {
        C[i * Aw + j] = A[i * Aw + j] + B[i * Aw + j];
    }

}

double GpuMatirxAdd(int *C, int *A, int *B, int Aw, int Ah, int Bw, int Bh)
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

    GpuMatirxAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, Aw, Ah, Bw, Bh);

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
    
    
    return elapsedTime / 1000.0;
}

int verify_results(int *C, int *D, int Cw, int Ch) {
    for (int i = 0; i < Cw * Ch; i++) {
        if (C[i] != D[i]) {
            return 0; // Results do not match
        }
    }
    return 1; // Results match
}