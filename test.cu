#include <stdio.h>
#include<cuda_runtime.h>
#include <device_launch_parameters.h>


//__global__�����ĺ��������߱�������δ��뽻��CPU���ã���GPUִ��
__global__ void add(const int* dev_a, const int* dev_b, int* dev_c)
{
    int i = threadIdx.x;
    dev_c[i] = dev_a[i] + dev_b[i];
}

int main(void)
{
    //���������ڴ棬�����г�ʼ��
    int host_a[512], host_b[512], host_c[512];
    for (int i = 0; i < 512; i++)
    {
        host_a[i] = i;
        host_b[i] = i << 1;
    }

    //����cudaError��Ĭ��ΪcudaSuccess(0)
    cudaError_t err = cudaSuccess;

    //����GPU�洢�ռ�
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
    //��Ҫ���������ʹ��cudaMemcpy���͵�GPU
    cudaMemcpy(dev_a, host_a, sizeof(host_a), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, sizeof(host_b), cudaMemcpyHostToDevice);

    //���ú˺�����GPU��ִ�С����ݽ��٣�֮ʹ��һ��Block������512���߳�
    add << <1, 512 >> > (dev_a, dev_b, dev_c);
    cudaMemcpy(&host_c, dev_c, sizeof(host_c), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 512; i++)
        printf("host_a[%d] + host_b[%d] = %d + %d = %d\n", i, i, host_a[i], host_b[i], host_c[i]);
    cudaFree(dev_a);//�ͷ�GPU�ڴ�
    cudaFree(dev_b);//�ͷ�GPU�ڴ�
    cudaFree(dev_c);//�ͷ�GPU�ڴ�
    return 0;
}