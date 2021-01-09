#include<opencv2\opencv.hpp>  
#include<cmath>  
#include<iostream>  
#include <opencv2\imgproc\types_c.h>
#include <opencv2\opencv.hpp>
using namespace cv;
using namespace std;


//OTSU算法函数实现  
int OTSU(Mat& srcImage)
{
    int nRows = srcImage.rows;
    int nCols = srcImage.cols;

    int threshold = 0;
    double max = 0.0;
    double AvePix[256];
    int nSumPix[256];
    double nProDis[256];
    double nSumProDis[256];

    for (int i = 0; i < 256; i++)
    {
        AvePix[i] = 0.0;
        nSumPix[i] = 0;
        nProDis[i] = 0.0;
        nSumProDis[i] = 0.0;
    }

    for (int i = 0; i < nRows; i++)
    {
        for (int j = 0; j < nCols; j++)
        {
            nSumPix[(int)srcImage.at<uchar>(i, j)]++;
        }
    }
    for (int i = 0; i < 256; i++)
    {
        nProDis[i] = (double)nSumPix[i] / (nRows * nCols);

    }
    AvePix[0] = 0;
    nSumProDis[0] = nProDis[0];
    for (int i = 1; i < 256; i++)
    {
        nSumProDis[i] = nSumProDis[i - 1] + nProDis[i];
        AvePix[i] = AvePix[i - 1] + i * nProDis[i];
    }
    double mean = AvePix[255];
    for (int k = 1; k < 256; k++)
    {
        double PA = nSumProDis[k];
        double PB = 1 - nSumProDis[k];
        double value = 0.0;
        if (fabs(PA) > 0.001 && fabs(PB) > 0.001)
        {
            double MA = AvePix[k];
            double MB = (mean - PA * MA) / PB;
            value = (double)(PA * PB * pow((MA - MB), 2));
            //或者这样value = (double)(PA * PB * pow((MA-MB),2));//类间方差  
            //pow(PA,1)* pow((MA - mean),2) + pow(PB,1)* pow((MB - mean),2)  
            if (value > max)
            {
                max = value;
                threshold = k;
            }
        }
    }
    return threshold;
}
int main()
{
    Mat srcImage = imread("D:/project/image_segment_with_cuda/image/raw/2.jpg");
    if (!srcImage.data)
    {
        printf("could not load image...\n");
        return -1;
    }

    Mat srcGray;
    cvtColor(srcImage, srcGray, CV_BGR2GRAY);
    imshow("srcGray", srcGray);

    double time1 = static_cast<double>(getTickCount());
    //调用二值化函数得到最佳阈值  
    int otsuThreshold = OTSU(srcGray);
    cout << otsuThreshold << endl;//输出最佳阈值  

    Mat otsuResultImage = Mat::zeros(srcGray.rows, srcGray.cols, CV_8UC1);
    //利用得到的阈值进行二值操作  
    for (int i = 0; i < srcGray.rows; i++)
    {
        for (int j = 0; j < srcGray.cols; j++)
        {
            if (srcGray.at<uchar>(i, j) > otsuThreshold)
            {
                otsuResultImage.at<uchar>(i, j) = 255;
            }
            else
            {
                otsuResultImage.at<uchar>(i, j) = 0;
            }
        }
    }
    time1 = ((double)getTickCount() - time1) / getTickFrequency();
    cout << "The Run Time of OSTU Algorithm with cpu is :" << time1 << "s" << endl;

    imshow("otsuResultImage", otsuResultImage);
    waitKey(0);
    return 0;
}