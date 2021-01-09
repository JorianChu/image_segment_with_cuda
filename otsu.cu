#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <device_functions.h>
#include <iostream>
#include <string.h>
#include <opencv2\opencv.hpp>
#include <math.h>
using namespace std;
using namespace cv;

/*
���������䷽��г���
ֻ����CPU�˵��ã���Ҫ��hist���鴫���ſɼ���
���������ʱ����ͼ���ٶȽ���
*/
__host__ int otsuThresh(int* hist, int imgHeight, int imgWidth)
{
    float sum = 0;
    for (int i = 0; i < 256; i++)
    {
        sum += i * hist[i];
    }
    float w0 = 0, u0 = 0;
    float u = sum / (imgHeight * imgWidth);
    float val = 0, maxval = 0;
    float s = 0, n = 0;
    int thresh = 0;
    for (int i = 0; i < 256; i++)
    {
        s += hist[i] * i;
        n += hist[i];
        w0 = n / (imgHeight * imgWidth);
        u0 = s / n;
        val = (u - u0) * (u - u0) * w0 / (1 - w0);
        if (val > maxval)
        {
            maxval = val;
            thresh = i;
        }
    }
    return thresh;
}

//__global__�����ĺ��������߱�������δ��뽻��CPU���ã���GPUִ��
//�Ҷ�ֱ��ͼͳ��
__global__ void imhistInCuda(unsigned char* dataIn, int* hist, int imgHeight, int imgWidth)
{
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

    if (xIndex < imgWidth && yIndex < imgHeight)
    {
        atomicAdd(&hist[dataIn[yIndex * imgWidth + xIndex]], 1);
    }
}

//���������䷽��CUDA�ı����
__global__ void OTSUthresh(const int* hist, float* sum, float* s, float* n, float* val, int imgHeight, int imgWidth, int* OtsuThresh)
{
    if (blockIdx.x == 0)
    {
        int index = threadIdx.x;
        atomicAdd(&sum[0], hist[index] * index);
    }
    else
    {
        int index = threadIdx.x;
        if (index < blockIdx.x)
        {
            atomicAdd(&s[blockIdx.x - 1], hist[index] * index);
            atomicAdd(&n[blockIdx.x - 1], hist[index]);
        }
    }
    __syncthreads(); //�����߳�ͬ��
    if (blockIdx.x > 0)
    {
        int index = blockIdx.x - 1;
        float u = sum[0] / (imgHeight * imgWidth);
        float w0 = n[index] / (imgHeight * imgWidth);
        float u0 = s[index] / n[index];
        if (w0 == 1)
        {
            val[index] = 0;
        }
        else
        {
            val[index] = (u - u0) * (u - u0) * w0 / (1 - w0);
        }
    }
    __syncthreads(); //�����߳�ͬ��
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        float maxval = 0;
        for (int i = 0; i < 256; i++)
        {
            if (val[i] > maxval)
            {
                maxval = val[i];
                OtsuThresh[0] = i;
                OtsuThresh[1] = val[i];
            }
        }
    }
}

//��ֵ��
__global__ void otsuInCuda(unsigned char* dataIn, unsigned char* dataOut, int imgHeight, int imgWidth, int* hThresh)
{
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

    if (xIndex < imgWidth && yIndex < imgHeight)
    {
        if (dataIn[yIndex * imgWidth + xIndex] > hThresh[0])
        {
            dataOut[yIndex * imgWidth + xIndex] = 255;
        }
    }
}

int main()
{
    //����Ҷ�ͼ
    Mat srcImg = imread("D:/project/image_segment_with_cuda/test.jpg", 0);

    int imgHeight = srcImg.rows;
    int imgWidth = srcImg.cols;

    //opencvʵ��OTSU��ֵ��
    double time0 = static_cast<double>(getTickCount());
    
    Mat dstImg1;
    threshold(srcImg, dstImg1, 0, 255, THRESH_OTSU);

    time0 = ((double)getTickCount() - time0) / getTickFrequency();
    cout << "The Run Time is :" << time0 << "s" << endl;



    //CUDA�ı�
    Mat dstImg2(imgHeight, imgWidth, CV_8UC1, Scalar(0));

    //��GPU�˿����ڴ�
    unsigned char* d_in;
    int* d_hist;

    cudaMalloc((void**)&d_in, imgHeight * imgWidth * sizeof(unsigned char));
    cudaMalloc((void**)&d_hist, 256 * sizeof(int));

    //����Ҷ�ͼ��GPU
    cudaMemcpy(d_in, srcImg.data, imgHeight * imgWidth * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock1(32, 32);
    dim3 blocksPerGrid1((imgWidth + 32 - 1) / 32, (imgHeight + 32 - 1) / 32);

    double time1 = static_cast<double>(getTickCount());

    imhistInCuda << <blocksPerGrid1, threadsPerBlock1 >> > (d_in, d_hist, imgHeight, imgWidth);
    
    time1 = ((double)getTickCount() - time1) / getTickFrequency();
    cout << "The Run Time is :" << time1 << "s" << endl;

    cout << "Accelerative Ratio:" << ceil(time0 / time1) << endl;

    float* d_sum;
    float* d_s;
    float* d_n;
    float* d_val;
    int* d_t;

    cudaMalloc((void**)&d_sum, sizeof(float));
    cudaMalloc((void**)&d_s, 256 * sizeof(float));
    cudaMalloc((void**)&d_n, 256 * sizeof(float));
    cudaMalloc((void**)&d_val, 256 * sizeof(float));
    cudaMalloc((void**)&d_t, 2 * sizeof(int));

    //���������䷽����㲢�й������257Ϊ1 + 256��
    //��1��block��������ͼ��Ҷȵ�sum����256��block���ڼ���256���Ҷȶ�Ӧ��s, n
    dim3 threadsPerBlock2(256, 1);
    dim3 blocksPerGrid2(257, 1);

    OTSUthresh << <blocksPerGrid2, threadsPerBlock2 >> > (d_hist, d_sum, d_s, d_n, d_val, imgHeight, imgWidth, d_t);

    unsigned char* d_out;

    cudaMalloc((void**)&d_out, imgHeight * imgWidth * sizeof(unsigned char));

    otsuInCuda << <blocksPerGrid1, threadsPerBlock1 >> > (d_in, d_out, imgHeight, imgWidth, d_t);

    //������ͼ��
    cudaMemcpy(dstImg2.data, d_out, imgHeight * imgWidth * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // ���������
        //int th[2] = { 0, 0 };
        //float n[256];
        //memset(n, 0, sizeof(n));
        //cudaMemcpy(th, d_t, 2 * sizeof(int), cudaMemcpyDeviceToHost);
        //cudaMemcpy(n, d_n, 256 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_hist);
    cudaFree(d_sum);
    cudaFree(d_s);
    cudaFree(d_n);
    cudaFree(d_val);
    cudaFree(d_t);

    imwrite("result1.jpg", dstImg1);
    imwrite("result2.jpg", dstImg2);

    return 0;
}