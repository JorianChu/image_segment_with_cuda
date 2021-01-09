#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <device_functions.h>
#include <iostream>
#include <string.h>
#include <opencv2\opencv.hpp>
#include <math.h>
#include <io.h>
#include <vector>
#include <fstream>
using namespace std;
using namespace cv;

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

//get gray histogram
__global__ void imhistInCuda(unsigned char* dataIn, int* hist, int imgHeight, int imgWidth)
{
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

    if (xIndex < imgWidth && yIndex < imgHeight)
    {
        atomicAdd(&hist[dataIn[yIndex * imgWidth + xIndex]], 1);
    }
}

// Calculate the maximum Between-Class variance with CUDA
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
    //synchronize all threads 
    __syncthreads(); 
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
    __syncthreads(); 
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

//thresholding
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

//implementation of ostu in GPU without Opencv ostu lib
// get global best threshold
int getOstu(const Mat& in){
    int rows = in.rows;
    int cols = in.cols;
    long size = rows * cols;
    float histogram[256] = { 0 };
    for (int i = 0; i < rows; ++i){
        const uchar* p = in.ptr<uchar>(i);
        for (int j = 0; j < cols; ++j){
            histogram[int(*p++)]++;
        }
    }
    int threshold;
    //sum0: sum of the foreground grayscale 
    //sum1:sum of the background grayscale 
    long sum0 = 0, sum1 = 0;
    //cnt0: sum of foreground pixel
    long cnt0 = 0, cnt1 = 0; 
    //w0,w1: The proportion of the foreground gray scale pixels (0~i) and
    //background gray scale pixels(i~255)  to the whole image
    double w0 = 0, w1 = 0; 
    //u0:Average gray level of the foreground pixel
    //u1://Average gray level of the background pixel
    double u0 = 0, u1 = 0;  
    double variance = 0; 
    double maxVariance = 0;
    for (int i = 1; i < 256; i++) {
        sum0 = 0;
        sum1 = 0;
        cnt0 = 0;
        cnt1 = 0;
        w0 = 0;
        w1 = 0;
        for (int j = 0; j < i; j++) {
            cnt0 += histogram[j];
            sum0 += j * histogram[j];
        }
        u0 = (double)sum0 / cnt0;
        w0 = (double)cnt0 / size;
        for (int j = i; j <= 255; j++){
            cnt1 += histogram[j];
            sum1 += j * histogram[j];
        }
        u1 = (double)sum1 / cnt1;
        w1 = 1 - w0; 
        //Calculate the variance of foreground and background pixels
        variance = w0 * w1 * (u0 - u1) * (u0 - u1);
        if (variance > maxVariance){
            maxVariance = variance;
            threshold = i;
        }
    }
    return threshold;
}

// get all files In the specified directory
void getAllFiles(string path, vector<string>& files, vector<string>& files_name)
{
    //文件句柄 
    long  hFile = 0;
    //文件信息 
    struct _finddata_t fileinfo;
    string p;
    if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
    {
        do
        {
            if ((fileinfo.attrib & _A_SUBDIR)) {
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0) {
                    //files.push_back(p.assign(path).append("\\").append(fileinfo.name) );
                    getAllFiles(p.assign(path).append("\\").append(fileinfo.name), files, files_name);
                }
            }else{
                files.push_back(p.assign(path).append("\\").append(fileinfo.name));
                files_name.push_back(fileinfo.name);
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }
}




int main()
{
    //input gray scale image，change path of image in various resolution
    // define double-dimensional array with the result dealt with opencv in raw iamge
    //Mat srcImg = imread("D:/project/image_segment_with_cuda/image/raw/2.jpg", 0);

    vector<string> temp;
    vector<string> files_name;
    getAllFiles("D:\\project\\image_segment_with_cuda\\image\\raw", temp,files_name);
    for (int i = 0; i < temp.size(); ++i){
        cout << temp[i] << endl;
        Mat srcImg = imread(temp[i], 0);

        int imgHeight = srcImg.rows;
        int imgWidth = srcImg.cols;

        //0.implemention of  OTSU binarization in CPU
        double time_cpu = static_cast<double>(getTickCount());
        int bestThreshold = getOstu(srcImg);
        cout << "The best threshold is: " << bestThreshold << endl;
        Mat otsuResultImage = Mat::zeros(imgHeight, imgWidth, CV_8UC1);
        //Binary operation
        for (int i = 0; i < imgHeight; i++) {
            for (int j = 0; j < imgWidth; j++) {
                if (srcImg.at<uchar>(i, j) > bestThreshold) {
                    otsuResultImage.at<uchar>(i, j) = 255;
                }
                else {
                    otsuResultImage.at<uchar>(i, j) = 0;
                }
            }
        }
        time_cpu = ((double)getTickCount() - time_cpu) / getTickFrequency();
        cout << "The Run Time of OSTU Algorithm with cpu is :" << time_cpu * 1000 << "ms" << endl;

        //1.implemention of  OTSU binarization by OpenCv
        double time_opencv = static_cast<double>(getTickCount());
        Mat dstImg1;
        //THRESH_OTSU: use Otsu algorithm to choose the optimal threshold value, default value is 8
        threshold(srcImg, dstImg1, 0, 255, THRESH_OTSU);
        time_opencv = ((double)getTickCount() - time_opencv) / getTickFrequency();
        cout << "The Run Time of OSTU algorithm with OpenCv is :" << time_opencv * 1000 << "ms" << endl;



        //2.implemention of  OTSU binarization by CUDA in GPU
        cudaEvent_t start, stop;
        float time;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        // CV_8UC1: the number of channels:1, Unsigned 8bits image
        Mat dstImg2(imgHeight, imgWidth, CV_8UC1, Scalar(0));
        //create GPU memory 
        unsigned char* d_in;
        int* d_hist;
        cudaMalloc((void**)&d_in, imgHeight * imgWidth * sizeof(unsigned char));
        cudaMalloc((void**)&d_hist, 256 * sizeof(int));

        // transfer the gray scale image  data from CPU(srcImg.data) to GPU(d_in)
        cudaMemcpy(d_in, srcImg.data, imgHeight * imgWidth * sizeof(unsigned char), cudaMemcpyHostToDevice);

        //block dimension:32*32
        dim3 threadsPerBlock1(32, 32);
        //
        dim3 blocksPerGrid1((imgWidth + 32 - 1) / 32, (imgHeight + 32 - 1) / 32);
        imhistInCuda << <blocksPerGrid1, threadsPerBlock1 >> > (d_in, d_hist, imgHeight, imgWidth);

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

        //define the maximum Between-Class variance calculation parallel specification,  where 257 is 1 + 256,
        //The first block is used to calculate the sum of the gray scale image, 
        //and the next 256 blocks are used to calculate the S and N corresponding to the 256 grayscales
        dim3 threadsPerBlock2(256, 1);
        dim3 blocksPerGrid2(257, 1);
        OTSUthresh << <blocksPerGrid2, threadsPerBlock2 >> > (d_hist, d_sum, d_s, d_n, d_val, imgHeight, imgWidth, d_t);

        unsigned char* d_out;
        cudaMalloc((void**)&d_out, imgHeight * imgWidth * sizeof(unsigned char));
        otsuInCuda << <blocksPerGrid1, threadsPerBlock1 >> > (d_in, d_out, imgHeight, imgWidth, d_t);
        // output the image dealt with Ostu method, from GPU to CPU
        cudaMemcpy(dstImg2.data, d_out, imgHeight * imgWidth * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        // calculate the time consumption of program with CUDA in GPU
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cout << "The Run Time of OSTU Algorithm with CUDA is :" << time << "ms" << endl;
        cout << "Accelerative ratio between CUDA and Opencv is  :" << time_opencv * 1000 / time << endl;
        cout << "Accelerative ratio between CUDA and CPU program is  :" << time_cpu * 1000 / time << endl;


        cudaFree(d_in);
        cudaFree(d_out);
        cudaFree(d_hist);
        cudaFree(d_sum);
        cudaFree(d_s);
        cudaFree(d_n);
        cudaFree(d_val);
        cudaFree(d_t);

        // output the image after image segment
        //imwrite("./image/result/2_cpu.jpg", otsuResultImage);
        //imwrite("./image/result/2_opencv.jpg", dstImg1);
        //imwrite("./image/result/2_cuda.jpg", dstImg2);

        imwrite("./image/result/"+files_name[i]+"_cpu.jpg", otsuResultImage);
        imwrite("./image/result/" + files_name[i] + "_opencv.jpg", dstImg1);
        imwrite("./image/result/" + files_name[i] + "_cuda.jpg", dstImg2);

        return 0;
    }
}