#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <opencv2\opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

//Sobel算子边缘检测核函数
__global__ void sobelInCuda(unsigned char* dataIn, unsigned char* dataOut, int imgHeight, int imgWidth)
{
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    int index = yIndex * imgWidth + xIndex;
    int Gx = 0;
    int Gy = 0;

    if (xIndex > 0 && xIndex < imgWidth - 1 && yIndex > 0 && yIndex < imgHeight - 1)
    {
        Gx = dataIn[(yIndex - 1) * imgWidth + xIndex + 1] + 2 * dataIn[yIndex * imgWidth + xIndex + 1] + dataIn[(yIndex + 1) * imgWidth + xIndex + 1]
            - (dataIn[(yIndex - 1) * imgWidth + xIndex - 1] + 2 * dataIn[yIndex * imgWidth + xIndex - 1] + dataIn[(yIndex + 1) * imgWidth + xIndex - 1]);
        Gy = dataIn[(yIndex - 1) * imgWidth + xIndex - 1] + 2 * dataIn[(yIndex - 1) * imgWidth + xIndex] + dataIn[(yIndex - 1) * imgWidth + xIndex + 1]
            - (dataIn[(yIndex + 1) * imgWidth + xIndex - 1] + 2 * dataIn[(yIndex + 1) * imgWidth + xIndex] + dataIn[(yIndex + 1) * imgWidth + xIndex + 1]);
        dataOut[index] = (abs(Gx) + abs(Gy)) / 2;
    }
}

//Sobel算子边缘检测CPU函数
void sobel(Mat srcImg, Mat dstImg, int imgHeight, int imgWidth)
{
    int Gx = 0;
    int Gy = 0;
    for (int i = 1; i < imgHeight - 1; i++)
    {
        uchar* dataUp = srcImg.ptr<uchar>(i - 1);
        uchar* data = srcImg.ptr<uchar>(i);
        uchar* dataDown = srcImg.ptr<uchar>(i + 1);
        uchar* out = dstImg.ptr<uchar>(i);
        for (int j = 1; j < imgWidth - 1; j++)
        {
            Gx = (dataUp[j + 1] + 2 * data[j + 1] + dataDown[j + 1]) - (dataUp[j - 1] + 2 * data[j - 1] + dataDown[j - 1]);
            Gy = (dataUp[j - 1] + 2 * dataUp[j] + dataUp[j + 1]) - (dataDown[j - 1] + 2 * dataDown[j] + dataDown[j + 1]);
            out[j] = (abs(Gx) + abs(Gy)) / 2;
        }
    }
}

int main()
{
    Mat grayImg = imread("D:/project/image_segment_with_cuda/test.jpg", 0);

    int imgHeight = grayImg.rows;
    int imgWidth = grayImg.cols;

    Mat gaussImg;
    //高斯滤波
    GaussianBlur(grayImg, gaussImg, Size(3, 3), 0, 0, BORDER_DEFAULT);

    double time1 = static_cast<double>(getTickCount());
    //Sobel算子CPU实现
    Mat dst(imgHeight, imgWidth, CV_8UC1, Scalar(0));
    sobel(gaussImg, dst, imgHeight, imgWidth);
    //计时器结束
    time1 = ((double)getTickCount() - time1) / getTickFrequency();
    //输出运行时间
    cout << "The Run Time is :" << time1<< "s" << endl;


    //CUDA实现后的传回的图像
    Mat dstImg(imgHeight, imgWidth, CV_8UC1, Scalar(0));

    //创建GPU内存
    unsigned char* d_in;
    unsigned char* d_out;

    cudaMalloc((void**)&d_in, imgHeight * imgWidth * sizeof(unsigned char));
    cudaMalloc((void**)&d_out, imgHeight * imgWidth * sizeof(unsigned char));

    //将高斯滤波后的图像从CPU传入GPU
    cudaMemcpy(d_in, gaussImg.data, imgHeight * imgWidth * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((imgWidth + threadsPerBlock.x - 1) / threadsPerBlock.x, (imgHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

    //计时器开始
    double time0 = static_cast<double>(getTickCount()); 
    //调用核函数
    sobelInCuda << <blockspergrid, threadsperblock>>> (d_in, d_out, imgheight, imgwidth);
    //sobelInCuda << <1,512 >> > (d_in, d_out, imgHeight, imgWidth);
    //计时器结束
    time0 = ((double)getTickCount() - time0) / getTickFrequency(); 
    //输出运行时间
    cout << "The Run Time is :" << time0 << "s" << endl; 


    //将图像传回GPU
    cudaMemcpy(dstImg.data, d_out, imgHeight * imgWidth * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    //释放GPU内存
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}