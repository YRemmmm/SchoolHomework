#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <cuda_runtime.h>   
#include <algorithm>
typedef unsigned char uchar;

typedef struct{
    int w;
    int h;
    unsigned char * img;
} PGM_IMG;    

//check cuda error and set cuda device
void checkErr(cudaError err,int num);     /*check cuda errors*/
void setCudaDevice(int devNum);

//define gray-scale images
PGM_IMG read_pgm(const char * path);
void write_pgm(PGM_IMG img, const char * path);
void free_pgm(PGM_IMG img);
//edge detection for gray-scale images
PGM_IMG denoising(PGM_IMG img_in, int n);
void blur_3x3(uchar * img_in, uchar * img_out,int img_w, int img_h);
void blur_5x5(uchar * img_in, uchar * img_out,int img_w, int img_h);
void median_3x3(uchar * img_in, uchar * img_out,int img_w, int img_h);
void median_5x5(uchar * img_in, uchar * img_out,int img_w, int img_h);
//call edge detection
void run_cpu_test(PGM_IMG img_in, int n);

void run_gpu_test(PGM_IMG img_in, int n);
PGM_IMG gpu_denoising(PGM_IMG img_in, int n);
__global__ void kernel_blur_3x3(uchar * img_in, uchar * img_out,int img_w, int img_h);
__global__ void kernel_blur_5x5(uchar * img_in, uchar * img_out,int img_w, int img_h);
__global__ void kernel_blur_3x3_shared(uchar * img_in, uchar * img_out,int img_w, int img_h);
__global__ void kernel_blur_5x5_shared(uchar * img_in, uchar * img_out,int img_w, int img_h);
__global__ void kernel_median_3x3(uchar * img_in, uchar * img_out,int img_w, int img_h);
__global__ void kernel_median_5x5(uchar * img_in, uchar * img_out,int img_w, int img_h);
__global__ void kernel_median_3x3_shared(uchar * img_in, uchar * img_out,int img_w, int img_h);
__global__ void kernel_median_5x5_shared(uchar * img_in, uchar * img_out,int img_w, int img_h);
int main()
{
	int devNum=0;   
    setCudaDevice(devNum);  
    PGM_IMG img_ibuf;
	clock_t start, finish;
    double duration; 

	printf("---------------Read PGM-----------------\n");
    img_ibuf = read_pgm("lena_noise.pgm");
    
	printf("--------------------------------\n");
	start = clock();
    run_cpu_test(img_ibuf, 1);
	finish = clock();
	duration = ((double)(finish - start) / CLOCKS_PER_SEC) * 1000; //CLOCKS_PER_SEC=1000,∫¡√Î
    printf( "The time of calculating is :%f ms\n", duration); 

    
	printf("--------------------------------\n");
	start = clock();
    run_cpu_test(img_ibuf, 2);
	finish = clock();
	duration = ((double)(finish - start) / CLOCKS_PER_SEC) * 1000; //CLOCKS_PER_SEC=1000,∫¡√Î
    printf( "The time of calculating is :%f ms\n", duration); 

    
	printf("--------------------------------\n");
	start = clock();
    run_cpu_test(img_ibuf, 3);
	finish = clock();
	duration = ((double)(finish - start) / CLOCKS_PER_SEC) * 1000; //CLOCKS_PER_SEC=1000,∫¡√Î
    printf( "The time of calculating is :%f ms\n", duration); 

    
	printf("--------------------------------\n");
	start = clock();
    run_cpu_test(img_ibuf, 4);
	finish = clock();
	duration = ((double)(finish - start) / CLOCKS_PER_SEC) * 1000; //CLOCKS_PER_SEC=1000,∫¡√Î
    printf( "The time of calculating is :%f ms\n", duration); 



	printf("--------------------------------\n");
    run_gpu_test(img_ibuf, 0);
    printf( "GPU warm-up finished\n"); 
    
    cudaEvent_t start1, stop1;
    float time1;

	printf("--------------------------------\n");
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1,0);
    run_gpu_test(img_ibuf, 1);
    cudaEventRecord(stop1,0);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&time1,start1,stop1);
    printf( "The time of calculating is :%f ms\n", time1); 
    
	printf("--------------------------------\n");
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1,0);
    run_gpu_test(img_ibuf, 2);
    cudaEventRecord(stop1,0);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&time1,start1,stop1);
    printf( "The time of calculating is :%f ms\n", time1); 
    
	printf("--------------------------------\n");
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1,0);
    run_gpu_test(img_ibuf, 3);
    cudaEventRecord(stop1,0);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&time1,start1,stop1);
    printf( "The time of calculating is :%f ms\n", time1); 
    
	printf("--------------------------------\n");
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1,0);
    run_gpu_test(img_ibuf, 4);
    cudaEventRecord(stop1,0);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&time1,start1,stop1);
    printf( "The time of calculating is :%f ms\n", time1); 
    
	printf("--------------------------------\n");
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1,0);
    run_gpu_test(img_ibuf, 5);
    cudaEventRecord(stop1,0);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&time1,start1,stop1);
    printf( "The time of calculating is :%f ms\n", time1); 
    
	printf("--------------------------------\n");
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1,0);
    run_gpu_test(img_ibuf, 6);
    cudaEventRecord(stop1,0);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&time1,start1,stop1);
    printf( "The time of calculating is :%f ms\n", time1); 
    
	printf("--------------------------------\n");
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1,0);
    run_gpu_test(img_ibuf, 7);
    cudaEventRecord(stop1,0);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&time1,start1,stop1);
    printf( "The time of calculating is :%f ms\n", time1); 
    
	printf("--------------------------------\n");
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1,0);
    run_gpu_test(img_ibuf, 8);
    cudaEventRecord(stop1,0);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&time1,start1,stop1);
    printf( "The time of calculating is :%f ms\n", time1); 
    

    free_pgm(img_ibuf); 
    return 0;
}
/*-----------------------------------------------------------*/
/*---------------------two useful functions------------------*/
/*----------------------do not modify------------------------*/
/*-----------------------------------------------------------*/
 void checkErr(cudaError err,int num)     /*check cuda errors*/
{
	 if( cudaSuccess != err) {  
		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        
                __FILE__, num-1, cudaGetErrorString( err) );              
	 }
}
void setCudaDevice(int devNum)
{
	cudaError_t err = cudaSuccess;
	printf("\nCUDA Device #%d\n", devNum);
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, devNum);
	printf("Name:                          %s\n",  devProp.name);
	printf("Total global memory:           %u\n",  devProp.totalGlobalMem);
	printf("Major revision number:         %d\n",  devProp.major);
	printf("Minor revision number:         %d\n",  devProp.minor);
	err=cudaSetDevice(devNum);
	checkErr(err,__LINE__);
}
/*-----------------------------------------------------------*/
/*----------------------read and write pgm picture-----------*/
/*----------------------do not modify------------------------*/
/*-----------------------------------------------------------*/
PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256]; 
    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "rb");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    fscanf(in_file, "%s", sbuf); /*  Skip the magic number,P2/P5   */
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);  
    printf("size:%s\n",sbuf);
    printf("Image size: %d x %d\n", result.w, result.h);
	printf("v_max:%d\n",v_max);
    
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    fread(result.img, sizeof(unsigned char), result.w*result.h, in_file); //to result.img
	printf("Read the picture succeed!\n\n");
    fclose(in_file);
    
    return result;   //PGM_IMG
}

void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(img.img,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
	printf("write the file............\n");
}
void free_pgm(PGM_IMG img)
{
    free(img.img);
}

void run_cpu_test(PGM_IMG img_in, int n)
{
    PGM_IMG img_obuf;
    printf("Starting CPU processing...\n");
    img_obuf = denoising(img_in, n);
    switch (n){
        case 1:
            write_pgm(img_obuf, "cpu_blur_3x3.pgm");
	        printf("Write new file (cpu_blur_3x3) succeed!\n\n");
            break;
        case 2:
            write_pgm(img_obuf, "cpu_blur_5x5.pgm");
	        printf("Write new file (cpu_blur_5x5) succeed!\n\n");
            break;
        case 3:
            write_pgm(img_obuf, "cpu_median_3x3.pgm");
	        printf("Write new file (cpu_median_3x3) succeed!\n\n");
            break;
        case 4:
            write_pgm(img_obuf, "cpu_median_5x5.pgm");
	        printf("Write new file (cpu_median_5x5) succeed!\n\n");
            break;
    }
    free_pgm(img_obuf);
}

PGM_IMG denoising(PGM_IMG img_in, int n)
{
    PGM_IMG result;
    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
	memset(result.img,0,result.w * result.h * sizeof(unsigned char));
    switch (n){
        case 1:
            blur_3x3(img_in.img, result.img,result.w,result.h);
            break;
        case 2:
            blur_5x5(img_in.img, result.img,result.w,result.h);
            break;
        case 3:
            median_3x3(img_in.img, result.img,result.w,result.h);
            break;
        case 4:
            median_5x5(img_in.img, result.img,result.w,result.h);
            break;
    }
    return result;
}

void blur_3x3(uchar * img_in, uchar * img_out,int img_w, int img_h){
    for(int row = 0; row < img_h; row++)  
    {  
        for(int col = 0; col < img_w; col++)  
        {  
            if ((row >= 1) && (row < img_h - 1) && (col >= 1) && (col < img_w - 1)) {
                int sum = 0;
                for (int dy = -1; dy <= 1; ++dy)
                {
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        sum += img_in[(row + dy) * img_w + (col + dx)];
                    }
                }
                img_out[row * img_w + col] = (uchar)(sum / 9);
            } else {
                img_out[row * img_w + col] = img_in[row * img_w + col];
            }

		}  
    }
}

void blur_5x5(uchar * img_in, uchar * img_out,int img_w, int img_h){
    for(int row = 0; row < img_h; row++)  
    {  
        for(int col = 0; col < img_w; col++)  
        {  
            if ((row >= 2) && (row < img_h - 2) && (col >= 2) && (col < img_w - 2)) {
                int sum = 0;
                for (int dy = -2; dy <= 2; ++dy)
                {
                    for (int dx = -2; dx <= 2; ++dx)
                    {
                        sum += img_in[(row + dy) * img_w + (col + dx)];
                    }
                }
                img_out[row * img_w + col] = (uchar)(sum / 25);
            } else {
                img_out[row * img_w + col] = img_in[row * img_w + col];
            }

		}  
    }
}

void cpu_sort(int n, uchar *a){
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1 - i; j++) {
            if(a[j] > a[j + 1]) {
                uchar tmp = a[j];
                a[j] = a[j + 1];
                a[j + 1] = tmp;
            }
        }
    }
}

void median_3x3(uchar * img_in, uchar * img_out,int img_w, int img_h){
    int num = 3;
    for(int row = 0; row < img_h; row++)  
    {  
        for(int col = 0; col < img_w; col++)  
        {  
            if ((row >= (num / 2)) && (row < img_h - (num / 2)) && (col >= (num / 2)) && (col < img_w - (num / 2))) {
                
                uchar windows[num * num];
                int index = 0;
                for (int dy = -(num / 2); dy <= (num / 2); ++dy)
                {
                    for (int dx = -(num / 2); dx <= (num / 2); ++dx)
                    {
                        windows[index++] = img_in[(row + dy) * img_w + (col + dx)];
                    }
                }
                cpu_sort(num * num, &windows[0]);
                img_out[row * img_w + col] = windows[num * num / 2];
            } else {
                img_out[row * img_w + col] = img_in[row * img_w + col];
            }
		}  
    }
}

void median_5x5(uchar * img_in, uchar * img_out,int img_w, int img_h){
    int num = 5;
    for(int row = 0; row < img_h; row++)  
    {  
        for(int col = 0; col < img_w; col++)  
        {  
            if ((row >= (num / 2)) && (row < img_h - (num / 2)) && (col >= (num / 2)) && (col < img_w - (num / 2))) {
                
                uchar windows[num * num];
                int index = 0;
                for (int dy = -(num / 2); dy <= (num / 2); ++dy)
                {
                    for (int dx = -(num / 2); dx <= (num / 2); ++dx)
                    {
                        windows[index++] = img_in[(row + dy) * img_w + (col + dx)];
                    }
                }
                cpu_sort(num * num, &windows[0]);
                img_out[row * img_w + col] = windows[num * num / 2];
            } else {
                img_out[row * img_w + col] = img_in[row * img_w + col];
            }
		}  
    }
}

void run_gpu_test(PGM_IMG img_in, int n)
{
    PGM_IMG img_obuf;
    printf("Starting GPU processing...\n");
    img_obuf = gpu_denoising(img_in, n);  //edge detection
    switch (n){
        case 1:
            write_pgm(img_obuf, "gpu_blur_3x3.pgm");
	        printf("Write new file (gpu_blur_3x3) succeed!\n\n");
            break;
        case 2:
            write_pgm(img_obuf, "gpu_blur_5x5.pgm");
	        printf("Write new file (gpu_blur_5x5) succeed!\n\n");
            break;
        case 3:
            write_pgm(img_obuf, "gpu_shared_blur_3x3.pgm");
	        printf("Write new file (gpu_shared_blur_3x3) succeed!\n\n");
            break;
        case 4:
            write_pgm(img_obuf, "gpu_shared_blur_5x5.pgm");
	        printf("Write new file (gpu_shared_blur_5x5) succeed!\n\n");
            break;
        case 5:
            write_pgm(img_obuf, "gpu_median_3x3.pgm");
	        printf("Write new file (gpu_median_3x3) succeed!\n\n");
            break;
        case 6:
            write_pgm(img_obuf, "gpu_median_5x5.pgm");
	        printf("Write new file (gpu_median_5x5) succeed!\n\n");
            break;
        case 7:
            write_pgm(img_obuf, "gpu_shared_median_3x3.pgm");
	        printf("Write new file (gpu_shared_median_3x3) succeed!\n\n");
            break;
        case 8:
            write_pgm(img_obuf, "gpu_shared_median_5x5.pgm");
	        printf("Write new file (gpu_shared_median_5x5) succeed!\n\n");
            break;
    }
    free_pgm(img_obuf);
}

PGM_IMG gpu_denoising(PGM_IMG img_in, int n)
{
    PGM_IMG img_out;
    img_out.w = img_in.w;
    img_out.h = img_in.h;
	size_t size = img_out.w * img_out.h * sizeof(uchar);
    img_out.img = (uchar *)malloc(size);
	memset(img_out.img,0,size);
	
    // Allocate the device memory for result.img
    uchar *img_temp=NULL, *d_img_in = NULL,*d_img_out = NULL;

	cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **)&d_img_in, size);
    checkErr(err,__LINE__);
	err = cudaMemcpy(d_img_in, img_in.img, size, cudaMemcpyHostToDevice);
    checkErr(err,__LINE__);

	err = cudaMalloc((void **)&d_img_out, size);
    checkErr(err,__LINE__);
	err = cudaMemcpy(d_img_out, img_out.img, size, cudaMemcpyHostToDevice);
    checkErr(err,__LINE__);

	dim3 threadsPerBlock(16,16);
	dim3 blocksPerGrid((img_in.w+15)/16,(img_in.h+15)/16);
    size_t sharedMemSize = (blocksPerGrid.x + 4) * (blocksPerGrid.y + 4) * sizeof(uchar);
    switch (n){
        case 0:
            kernel_blur_3x3<<<blocksPerGrid,threadsPerBlock>>>(d_img_in, d_img_out,img_in.w,img_in.h);
            break;
        case 1:
            kernel_blur_3x3<<<blocksPerGrid,threadsPerBlock>>>(d_img_in, d_img_out,img_in.w,img_in.h);
            break;
        case 2:
            kernel_blur_5x5<<<blocksPerGrid,threadsPerBlock>>>(d_img_in, d_img_out,img_in.w,img_in.h);
            break;
        case 3:
            kernel_blur_3x3_shared<<<blocksPerGrid,threadsPerBlock,sharedMemSize>>>(d_img_in, d_img_out,img_in.w,img_in.h);
            break;
        case 4:
            kernel_blur_5x5_shared<<<blocksPerGrid,threadsPerBlock,sharedMemSize>>>(d_img_in, d_img_out,img_in.w,img_in.h);
            break;
        case 5:
            kernel_median_3x3<<<blocksPerGrid,threadsPerBlock>>>(d_img_in, d_img_out,img_in.w,img_in.h);
            break;
        case 6:
            kernel_median_5x5<<<blocksPerGrid,threadsPerBlock>>>(d_img_in, d_img_out,img_in.w,img_in.h);
            break;
        case 7:
            kernel_median_3x3_shared<<<blocksPerGrid,threadsPerBlock,sharedMemSize>>>(d_img_in, d_img_out,img_in.w,img_in.h);
            break;
        case 8:
            kernel_median_5x5_shared<<<blocksPerGrid,threadsPerBlock,sharedMemSize>>>(d_img_in, d_img_out,img_in.w,img_in.h);
            break;
    }




	err = cudaGetLastError();

	err = cudaMemcpy(img_out.img, d_img_out, size, cudaMemcpyDeviceToHost);
    checkErr(err,__LINE__);
	cudaFree(d_img_in);
	cudaFree(d_img_out);
	//free(img_temp);
    return img_out;
}

//question 1
__global__ void kernel_blur_3x3(uchar * img_in, uchar * img_out,int img_w, int img_h)
{
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

    if ((row >= 1) && (row < img_h - 1) && (col >= 1) && (col < img_w - 1))
    {
        int sum = 0;
        
        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int dx = -1; dx <= 1; ++dx)
            {
                sum += img_in[(row + dy) * img_w + (col + dx)];
            }
        }

        img_out[row * img_w + col] = (uchar)(sum / 9);
    }
    else
    {
        if ((row >= 0) && (row < img_h) && (col >= 0) && (col < img_w))
        {
            img_out[row * img_w + col] = img_in[row * img_w + col];
        }
    }
}


__global__ void kernel_blur_5x5(uchar * img_in, uchar * img_out,int img_w, int img_h)
{
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
    
    if ((row >= 2) && (row < img_h - 2) && (col >= 2) && (col < img_w - 2))
    {
        int sum = 0;
        
        for (int dy = -2; dy <= 2; ++dy)
        {
            for (int dx = -2; dx <= 2; ++dx)
            {
                sum += img_in[(row + dy) * img_w + (col + dx)];
            }
        }

        img_out[row * img_w + col] = (uchar)(sum / 25);
    }
    else
    {
        if ((row >= 0) && (row < img_h) && (col >= 0) && (col < img_w))
        {
            img_out[row * img_w + col] = img_in[row * img_w + col];
        }
    }
}

//question 2
__global__ void kernel_blur_3x3_shared(uchar * img_in, uchar * img_out, int img_w, int img_h)
{
    extern __shared__ uchar shared[];

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    int srow = threadIdx.y + 1;
    int scol = threadIdx.x + 1;

    if ((row < img_h) && (col < img_w))
    {
        shared[srow * (blockDim.x + 2) + scol] = img_in[row * img_w + col];
    }
    else
    {
        shared[srow * (blockDim.x + 2) + scol] = 0;
    }

    if (threadIdx.y == 0 && row >= 1 && row < img_h - 1)
    {
        if (col < img_w)
            shared[(srow - 1) * (blockDim.x + 2) + scol] = img_in[(row - 1) * img_w + col];
    }
    if (threadIdx.y == blockDim.y - 1 && row < img_h - 1)
    {
        if (col < img_w)
            shared[(srow + 1) * (blockDim.x + 2) + scol] = img_in[(row + 1) * img_w + col];
    }
    if (threadIdx.x == 0 && col >= 1 && col < img_w - 1)
    {
        if (row < img_h)
            shared[srow * (blockDim.x + 2) + (scol - 1)] = img_in[row * img_w + col - 1];
    }
    if (threadIdx.x == blockDim.x - 1 && col < img_w - 1)
    {
        if (row < img_h)
            shared[srow * (blockDim.x + 2) + (scol + 1)] = img_in[row * img_w + col + 1];
    }

    __syncthreads();

    if ((row >= 1) && (row < img_h - 1) && (col >= 1) && (col < img_w - 1))
    {
        int sum = 0;
        
        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int dx = -1; dx <= 1; ++dx)
            {
                sum += shared[(srow + dy) * (blockDim.x + 2) + (scol + dx)];
            }
        }

        img_out[row * img_w + col] = (uchar)(sum / 9);
    }
    else
    {
        if ((row >= 0) && (row < img_h) && (col >= 0) && (col < img_w))
        {
            img_out[row * img_w + col] = shared[srow * (blockDim.x + 2) + scol];
        }
    }
}


__global__ void kernel_blur_5x5_shared(uchar * img_in, uchar * img_out, int img_w, int img_h)
{
    extern __shared__ uchar shared[];

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    int srow = threadIdx.y + 2;
    int scol = threadIdx.x + 2;

    if ((row < img_h) && (col < img_w))
    {
        shared[srow * (blockDim.x + 4) + scol] = img_in[row * img_w + col];
    }
    else
    {
        shared[srow * (blockDim.x + 4) + scol] = 0;
    }

    // Load halo elements for the top and bottom rows
    if (threadIdx.y == 0 && row >= 2 && row < img_h - 2)
    {
        if (col < img_w)
            shared[(srow - 2) * (blockDim.x + 4) + scol] = img_in[(row - 2) * img_w + col];
        if (col < img_w)
            shared[(srow - 1) * (blockDim.x + 4) + scol] = img_in[(row - 1) * img_w + col];
    }
    if (threadIdx.y == blockDim.y - 1 && row < img_h - 2)
    {
        if (col < img_w)
            shared[(srow + 1) * (blockDim.x + 4) + scol] = img_in[(row + 1) * img_w + col];
        if (col < img_w)
            shared[(srow + 2) * (blockDim.x + 4) + scol] = img_in[(row + 2) * img_w + col];
    }

    // Load halo elements for the left and right columns
    if (threadIdx.x == 0 && col >= 2 && col < img_w - 2)
    {
        if (row < img_h)
            shared[srow * (blockDim.x + 4) + (scol - 2)] = img_in[row * img_w + col - 2];
        if (row < img_h)
            shared[srow * (blockDim.x + 4) + (scol - 1)] = img_in[row * img_w + col - 1];
    }
    if (threadIdx.x == blockDim.x - 1 && col < img_w - 2)
    {
        if (row < img_h)
            shared[srow * (blockDim.x + 4) + (scol + 1)] = img_in[row * img_w + col + 1];
        if (row < img_h)
            shared[srow * (blockDim.x + 4) + (scol + 2)] = img_in[row * img_w + col + 2];
    }

    __syncthreads();

    if ((row >= 2) && (row < img_h - 2) && (col >= 2) && (col < img_w - 2))
    {
        int sum = 0;
        
        for (int dy = -2; dy <= 2; ++dy)
        {
            for (int dx = -2; dx <= 2; ++dx)
            {
                sum += shared[(srow + dy) * (blockDim.x + 4) + (scol + dx)];
            }
        }

        img_out[row * img_w + col] = (uchar)(sum / 25);
    }
    else
    {
        if ((row >= 0) && (row < img_h) && (col >= 0) && (col < img_w))
        {
            img_out[row * img_w + col] = shared[srow * (blockDim.x + 4) + scol];
        }
    }
}






//question 3

__device__ void gpu_sort(int n, uchar *a){
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1 - i; j++) {
            if(a[j] > a[j + 1]) {
                uchar tmp = a[j];
                a[j] = a[j + 1];
                a[j + 1] = tmp;
            }
        }
    }
}

__global__ void kernel_median_3x3(uchar * img_in, uchar * img_out,int img_w, int img_h)
{
    int num = 3;
    
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

    if ((row >= (num / 2)) && (row < img_h - (num / 2)) && (col >= (num / 2)) && (col < img_w - (num / 2))) {
        
        uchar windows[9];
        int index = 0;
        for (int dy = -(num / 2); dy <= (num / 2); ++dy)
        {
            for (int dx = -(num / 2); dx <= (num / 2); ++dx)
            {
                windows[index++] = img_in[(row + dy) * img_w + (col + dx)];
            }
        }
        gpu_sort(num * num, &windows[0]);
        img_out[row * img_w + col] = windows[num * num / 2];
    } else {
        if ((row >= 0) && (row < img_h) && (col >= 0) && (col < img_w))
        {
            img_out[row * img_w + col] = img_in[row * img_w + col];
        }
    }
}


__global__ void kernel_median_5x5(uchar * img_in, uchar * img_out,int img_w, int img_h)
{
    int num = 5;
    
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

    if ((row >= (num / 2)) && (row < img_h - (num / 2)) && (col >= (num / 2)) && (col < img_w - (num / 2))) {
        uchar windows[25];
        int index = 0;
        for (int dy = -(num / 2); dy <= (num / 2); ++dy)
        {
            for (int dx = -(num / 2); dx <= (num / 2); ++dx)
            {
                windows[index++] = img_in[(row + dy) * img_w + (col + dx)];
            }
        }
        gpu_sort(num * num, &windows[0]);
        img_out[row * img_w + col] = windows[num * num / 2];
    } else {
        if ((row >= 0) && (row < img_h) && (col >= 0) && (col < img_w))
        {
            img_out[row * img_w + col] = img_in[row * img_w + col];
        }
    }
}



__global__ void kernel_median_3x3_shared(uchar * img_in, uchar * img_out,int img_w, int img_h)
{
    int num = 3;
    
    extern __shared__ uchar shared[];

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    int srow = threadIdx.y + 1;
    int scol = threadIdx.x + 1;

    if ((row < img_h) && (col < img_w))
    {
        shared[srow * (blockDim.x + 2) + scol] = img_in[row * img_w + col];
    }
    else
    {
        shared[srow * (blockDim.x + 2) + scol] = 0;
    }

    if (threadIdx.y == 0 && row >= 1 && row < img_h - 1)
    {
        if (col < img_w)
            shared[(srow - 1) * (blockDim.x + 2) + scol] = img_in[(row - 1) * img_w + col];
    }
    if (threadIdx.y == blockDim.y - 1 && row < img_h - 1)
    {
        if (col < img_w)
            shared[(srow + 1) * (blockDim.x + 2) + scol] = img_in[(row + 1) * img_w + col];
    }
    if (threadIdx.x == 0 && col >= 1 && col < img_w - 1)
    {
        if (row < img_h)
            shared[srow * (blockDim.x + 2) + (scol - 1)] = img_in[row * img_w + col - 1];
    }
    if (threadIdx.x == blockDim.x - 1 && col < img_w - 1)
    {
        if (row < img_h)
            shared[srow * (blockDim.x + 2) + (scol + 1)] = img_in[row * img_w + col + 1];
    }

    __syncthreads();

    if ((row >= (num / 2)) && (row < img_h - (num / 2)) && (col >= (num / 2)) && (col < img_w - (num / 2))) {
        
        uchar windows[9];
        int index = 0;
        for (int dy = -(num / 2); dy <= (num / 2); ++dy)
        {
            for (int dx = -(num / 2); dx <= (num / 2); ++dx)
            {
                windows[index++] = shared[(srow + dy) * (blockDim.x + 2) + (scol + dx)];
            }
        }
        gpu_sort(num * num, &windows[0]);
        img_out[row * img_w + col] = windows[num * num / 2];
    } else {
        if ((row >= 0) && (row < img_h) && (col >= 0) && (col < img_w))
        {
            img_out[row * img_w + col] = shared[srow * (blockDim.x + 2) + scol];
        }
    }
}


__global__ void kernel_median_5x5_shared(uchar * img_in, uchar * img_out,int img_w, int img_h)
{
    int num = 5;
    
    extern __shared__ uchar shared[];

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    int srow = threadIdx.y + 2;
    int scol = threadIdx.x + 2;

    if ((row < img_h) && (col < img_w))
    {
        shared[srow * (blockDim.x + 4) + scol] = img_in[row * img_w + col];
    }
    else
    {
        shared[srow * (blockDim.x + 4) + scol] = 0;
    }

    // Load halo elements for the top and bottom rows
    if (threadIdx.y == 0 && row >= 2 && row < img_h - 2)
    {
        if (col < img_w)
            shared[(srow - 2) * (blockDim.x + 4) + scol] = img_in[(row - 2) * img_w + col];
        if (col < img_w)
            shared[(srow - 1) * (blockDim.x + 4) + scol] = img_in[(row - 1) * img_w + col];
    }
    if (threadIdx.y == blockDim.y - 1 && row < img_h - 2)
    {
        if (col < img_w)
            shared[(srow + 1) * (blockDim.x + 4) + scol] = img_in[(row + 1) * img_w + col];
        if (col < img_w)
            shared[(srow + 2) * (blockDim.x + 4) + scol] = img_in[(row + 2) * img_w + col];
    }

    // Load halo elements for the left and right columns
    if (threadIdx.x == 0 && col >= 2 && col < img_w - 2)
    {
        if (row < img_h)
            shared[srow * (blockDim.x + 4) + (scol - 2)] = img_in[row * img_w + col - 2];
        if (row < img_h)
            shared[srow * (blockDim.x + 4) + (scol - 1)] = img_in[row * img_w + col - 1];
    }
    if (threadIdx.x == blockDim.x - 1 && col < img_w - 2)
    {
        if (row < img_h)
            shared[srow * (blockDim.x + 4) + (scol + 1)] = img_in[row * img_w + col + 1];
        if (row < img_h)
            shared[srow * (blockDim.x + 4) + (scol + 2)] = img_in[row * img_w + col + 2];
    }

    __syncthreads();

    if ((row >= (num / 2)) && (row < img_h - (num / 2)) && (col >= (num / 2)) && (col < img_w - (num / 2))) {
        uchar windows[25];
        int index = 0;
        for (int dy = -(num / 2); dy <= (num / 2); ++dy)
        {
            for (int dx = -(num / 2); dx <= (num / 2); ++dx)
            {
                windows[index++] = shared[(srow + dy) * (blockDim.x + 4) + (scol + dx)];
            }
        }
        gpu_sort(num * num, &windows[0]);
        img_out[row * img_w + col] = windows[num * num / 2];
    } else {
        if ((row >= 0) && (row < img_h) && (col >= 0) && (col < img_w))
        {
            img_out[row * img_w + col] = shared[srow * (blockDim.x + 4) + scol];
        }
    }
}