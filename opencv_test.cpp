#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main() {
    cv::Mat img = cv::imread("D:/project/image_segment_with_cuda/test.jpg", cv::IMREAD_REDUCED_COLOR_8);
    // imread( const String& filename, int flags = IMREAD_COLOR );
    if (img.empty()) return -1;
    cv::imshow("BGR", img);
    std::vector<cv::Mat> planes;
    cv::split(img, planes);
    cv::imshow("B", planes[0]);
    cv::imshow("G", planes[1]);
    cv::imshow("R", planes[2]);
    cv::imshow("BGR", img);

    cout << "test" << endl;

    img = cv::imread("D:/project/image_segment_with_cuda/test.jpg", cv::IMREAD_REDUCED_GRAYSCALE_8);
    cv::split(img, planes);  // 灰度图只能分离出一个通道
    cv::imshow("Gray", planes[0]);
    cv::waitKey(0);
    return 0;
}

//int main()
//{
//    //opencv版本号
//    cout << "opencv_version: " << cv_version << endl;
//
//    //读取图片
//    mat img = imread("d:/project/image_segment_with_cuda/test.jpg");
//
//    imshow("picture", img);
//    waitkey(0);
//    return 0;
//}
