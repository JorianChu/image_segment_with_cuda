#include <stdio.h>
#include<cuda_runtime.h>
#include <device_launch_parameters.h>


//__global__声明的函数，告诉编译器这段代码交由CPU调用，由GPU执行
__global__ void add(const int* dev_a, const int* dev_b, int* dev_c)
{
    int i = threadIdx.x;
    dev_c[i] = dev_a[i] + dev_b[i];
}

int main(void)
{
    //申请主机内存，并进行初始化
    int host_a[512], host_b[512], host_c[512];
    for (int i = 0; i < 512; i++)
    {
        host_a[i] = i;
        host_b[i] = i << 1;
    }

    //定义cudaError，默认为cudaSuccess(0)
    cudaError_t err = cudaSuccess;

    //申请GPU存储空间
    int* dev_a, * dev_b, * dev_c;
    err = cudaMalloc((void**)&dev_a, sizeof(int) * 512);
    err = cudaMalloc((void**)&dev_b, sizeof(int) * 512);
    err = cudaMalloc((void**)&dev_c, sizeof(int) * 512);
    if (err != cudaSuccess)
    {
        printf("the cudaMalloc on GPU is failed");
        return 1;
    }
    printf("SUCCESS");
    //将要计算的数据使用cudaMemcpy传送到GPU
    cudaMemcpy(dev_a, host_a, sizeof(host_a), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, sizeof(host_b), cudaMemcpyHostToDevice);

    //调用核函数在GPU上执行。数据较少，之使用一个Block，含有512个线程
    add << <1, 512 >> > (dev_a, dev_b, dev_c);
    cudaMemcpy(&host_c, dev_c, sizeof(host_c), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 512; i++)
        printf("host_a[%d] + host_b[%d] = %d + %d = %d\n", i, i, host_a[i], host_b[i], host_c[i]);
    cudaFree(dev_a);//释放GPU内存
    cudaFree(dev_b);//释放GPU内存
    cudaFree(dev_c);//释放GPU内存
    return 0;
}