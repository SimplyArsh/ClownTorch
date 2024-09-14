// #include "kernels.h"
// #include <cuda_runtime.h>

// Matrix multiplication kernel
__global__ void matrixMulKernel(const float* A, const float* B, float* C, int A_size, int AB_size, int B_size) {
    int YIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int XIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (YIdx < A_size && XIdx < B_size) {
        float value = 0.0f;
        for (int k = 0; k < AB_size; ++k)
            value += A[YIdx * AB_size + k] * B[k * B_size + XIdx];

        C[YIdx * B_size + XIdx] = value;
    }
}

__global__ void addKernel(float* A, float* B, int A_size, int AB_size) {
    int YIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int XIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (YIdx < A_size && XIdx < AB_size)
        A[YIdx * AB_size + XIdx] += B[XIdx];

}

__global__ void transposeKernel(float* input, float* output, int N, int D) {
    int YIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int XIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (YIdx < N && XIdx < D)
        output[XIdx * N + YIdx] = input[YIdx * D + XIdx];

}

// pre_axis is the product of axis shapes that occur before the axis to be summed
// sum_axis is the size of the axis to be summed
// post_axis is the product of axis shapes that occur after the axis to be summed
__global__ void sumAxisKernel(float* input, float* output, int pre_axis, int sum_axis, int post_axis) {
    int YIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int XIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (YIdx < pre_axis && XIdx < post_axis) {
        float value = 0.0f;
        for (int k = 0; k < sum_axis; ++k)
            value += input[YIdx * sum_axis * post_axis + XIdx + k * post_axis];

        output[YIdx * post_axis + XIdx] = value;
    }
}

__global__ void reluKernel(float* arr, int N, int D, int ones) {
    int YIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int XIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (YIdx < N && XIdx < D)
        arr[YIdx * D + XIdx] = arr[YIdx * D + XIdx] <= 0 ? 0 : (ones ? 1 : arr[YIdx * D + XIdx]);

}