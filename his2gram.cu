#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdio.h>
#include <iostream>  
#include <cmath>
#include <math.h>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "device_functions.h"
#include <cuda.h> 
#include"device_atomic_functions.h"
using namespace std;
using namespace cv;

//block:16*16 thread
//The grid dimension is calculated according to the image size:
//(width+TILE_WIDTH-1)/ TILE_WIDTH, (height+TILE_WIDTH-1)/ TILE_WIDTH£©
#define TILE_WIDTH 16      //thread width
//_Calhistogramkernel kernel function:calculate the histogram of the image
__global__ void _Calhistogramkernel(unsigned char* input_data, unsigned int* histOrg,
    unsigned int step, unsigned int width, unsigned int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int value = 0;
    if (x < width && y < height)
    {
        value = input_data[y * step + x];
        atomicAdd(&(histOrg[value]), 1);
    }
    __syncthreads();
}
//Call the host-side function of the _Calhistogramkernel kernel function    
void Calhistogram(unsigned int* p_hist, unsigned char* srcImagedata,
    unsigned int step, unsigned int width, unsigned int height) {
    //Only pass the data data of the image as a parameter to the device
    unsigned char* devicechar;
    cudaMalloc((void**)(&devicechar), width * height * sizeof(unsigned char));
    cudaMemcpy(devicechar, srcImagedata, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    // Calculate the size of the thread block calling the kernel function and the number of thread blocks
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH); //Thread block dimensions   
    dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH); //Dimension of thread grid
    // Apply a space for the histogram on the Device side
    unsigned int* devhisto;
    cudaMalloc((void**)&devhisto, 256 * sizeof(unsigned int));
    //Initialize the histogram array on the device side
    cudaMemset(devhisto, 0, 256 * sizeof(unsigned int));
    // Call the kernel function to calculate the histogram of the input image
    _Calhistogramkernel << < dimGrid, dimBlock >> > (devicechar, devhisto, step, width, height);  //Call kernel function
    // Copy the result of the histogram back to the host memory
    cudaMemcpy(p_hist, devhisto, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(devicechar);
    cudaFree(devhisto);
}
//_calequhistker calculates the gray value after transformation
__global__ void _Calequhistker(unsigned int* devequhist, float* devicecdfhist, unsigned int size) {
    __shared__ unsigned int sharedequhist[256];
    int Id = threadIdx.x;
    sharedequhist[Id] = (unsigned int)(255.0 * devicecdfhist[Id] + 0.5);
    __syncthreads();
    devequhist[Id] = sharedequhist[Id];
}
//The host-side function to calculate the converted gray value, call _calequhistker
void Calequhist(unsigned int* equhist, float* cdfhist, unsigned int size) {
    float* devicecdfhist;
    cudaMalloc((void**)(&devicecdfhist), 256 * sizeof(float));
    cudaMemcpy(devicecdfhist, cdfhist, 256 * sizeof(float), cudaMemcpyHostToDevice);
    unsigned int* devequhist;
    cudaMalloc((void**)&devequhist, 256 * sizeof(unsigned int));
    //Initialize the histogram array on the device side
    cudaMemset(devequhist, 0, 256 * sizeof(unsigned int));
    _Calequhistker << <1, 256 >> > (devequhist, devicecdfhist, size);
    cudaMemcpy(equhist, devequhist, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(devicecdfhist);
    cudaFree(devequhist);
}
//_MapImagekernel: Kernel function to calculate the output image from the gray mapping relationship
__global__ void _MapImagekernel(unsigned char* devicesrcdata, unsigned char* devicedstdata,
    unsigned int* devhistequ, unsigned int step, unsigned int width, unsigned int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int value = 0;
    if (x < width && y < height)
    {
        value = devicedstdata[y * step + x];
        devicesrcdata[y * step + x] = devhistequ[value];
    }
    __syncthreads();
}
//Compute the host function of the output image: call _MapImagekernel 
void MapImage(unsigned char* dstdata, unsigned char* srcdata, unsigned int* p_histequ,
    unsigned int step, unsigned int width, unsigned int height) {
    //Pass the image data as an argument to the device
    unsigned char* devicesrcdata, * devicedstdata;
    cudaMalloc((void**)(&devicesrcdata), width * height * sizeof(unsigned char));
    cudaMalloc((void**)(&devicedstdata), width * height * sizeof(unsigned char));
    cudaMemcpy(devicesrcdata, srcdata, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(devicedstdata, srcdata, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    // Apply a space for a balanced histogram on the Device side
    unsigned int* devhistequ;
    cudaMalloc((void**)&devhistequ, 256 * sizeof(unsigned int));
    cudaMemcpy(devhistequ, p_histequ, 256 * sizeof(unsigned int), cudaMemcpyHostToDevice);
    // Calculate the size of the thread block calling the kernel function and the number of thread blocks
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH); //Thread block dimensions   
    dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH); //Dimension of thread grid
    // Call the kernel function to calculate the histogram of the input image
    _MapImagekernel << < dimGrid, dimBlock >> > (devicedstdata, devicesrcdata, devhistequ, step, width, height);  //Call kernel function
    // Copy the result of the histogram back to the host memory
    cudaMemcpy(dstdata, devicedstdata, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    //Free memory
    cudaFree(devicesrcdata);
    cudaFree(devicedstdata);
    cudaFree(devhistequ);
}
//Main function
int main(int argc, char** argv)
{
    // Read images using OpenCV
    unsigned int i = 0, step = 0;
    cv::Mat srcImage = cv::imread("picture/DSC02315.JPG", 1);//Picture path
    unsigned int height = srcImage.rows;
    unsigned int width = srcImage.cols;
    unsigned int size = width * height;
    // Graying
    cv::cvtColor(srcImage, srcImage, CV_BGR2GRAY);
    cv::Mat dstImage1 = srcImage;
    unsigned char* hostdata = srcImage.data;
    unsigned char* dstdata = dstImage1.data;
    cv::imshow("original image", srcImage);
    cv::imwrite("picture/result/cpuDSC02315.JPG", srcImage);
    
    
    double t = (double)cvGetTickCount();

    if (width % 4 == 0)
        step = width;
    else
        step = (width / 4) * 4 + 4;
    //printf("step=%d\n", step);
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    //Define the histogram array on the host
    unsigned int hosthist[256] = { 0 };
    unsigned int* p_hist = hosthist;
    Calhistogram(p_hist, hostdata, step, width, height);
    //Normalized histogram (serial)  
    float histPDF[256] = { 0 };
    for (i = 0; i < 255; i++)
    {
        histPDF[i] = (float)hosthist[i] / size;
    }
    //Cumulative histogram (serial) 
    float histCDF[256] = { 0 };
    for (i = 0; i < 256; i++)
    {
        if (0 == i) histCDF[i] = histPDF[i];
        else histCDF[i] = histCDF[i - 1] + histPDF[i];
    }
    //Histogram equalization (parallel)
    unsigned int histEQU[256] = { 0 };
    unsigned int* p_histequ = histEQU;
    Calequhist(p_histequ, histCDF, size);
    //Mapping (parallel)
    MapImage(dstdata, hostdata, p_histequ, step, width, height);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    t = (double)cvGetTickCount() - t;
    printf("\nThe time of CPU Image histogram equalization time is : %gms\n", t / (cvGetTickFrequency() * 1000));
    printf("The time of GPU Image histogram equalization is :%gms\n", time);
    printf("cuda accelerated:%g times\n", t / (cvGetTickFrequency() * 1000) / time);
    // show result
    cv::imwrite("picture/result/cudaDSC02315.JPG", dstImage1);
    cv::imshow("picture/result/cudaDSC02315.JPG", dstImage1);
    waitKey(0);
    return 0;
}
