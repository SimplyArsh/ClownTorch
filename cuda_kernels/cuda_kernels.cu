
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

__global__ void primitiveKernel(float *A, float *B, float *C, int op_code, 
int *A_shape, int *B_shape, int *C_shape, int A_ndim, int B_ndim, int C_ndim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = 1;
    
    for (int i = 0; i < C_ndim; ++i) {
        total_elements *= C_shape[i];
    }
    
    if (idx >= total_elements) return;

    int A_idx = 0, B_idx = 0;
    int A_stride = 1, B_stride = 1;
    
    for (int i = C_ndim - 1, stride = 1; i >= 0; --i) {
        int C_dim_idx = (idx / stride) % C_shape[i];
        stride *= C_shape[i];
        
        if (i < A_ndim) {
            if (A_shape[i] != 1) {
                A_idx += C_dim_idx * A_stride;
            }
            A_stride *= A_shape[i];
        }
        
        if (i < B_ndim) {
            if (B_shape[i] != 1) {
                B_idx += C_dim_idx * B_stride;
            }
            B_stride *= B_shape[i];
        }
    }

    // Perform elementwise addition
    if (op_code == 0) C[idx] = A[A_idx] + B[B_idx];
    else if (op_code == 1) C[idx] = A[A_idx] - B[B_idx];
    else if (op_code == 2) C[idx] = A[A_idx] * B[B_idx];
    else if (op_code == 3) C[idx] = A[A_idx] / B[B_idx];
    
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

__global__ void maxAxisKernel(float* input, float* output, int pre_axis, int sum_axis, int post_axis) {
    int YIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int XIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (YIdx < pre_axis && XIdx < post_axis) {
        float value = 0.0f;
        for (int k = 0; k < sum_axis; ++k)
            value = fmax(value, input[YIdx * sum_axis * post_axis + XIdx + k * post_axis]);

        output[YIdx * post_axis + XIdx] = value;
    }
}

__global__ void elementWiseKernel(float* A, float* B_scalar, int op_code, int *A_shape, int A_ndim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total_elements = 1;
    
    for (int i = 0; i < A_ndim; ++i) {
        total_elements *= A_shape[i];
    }
    
    if (idx >= total_elements) return;

    if (op_code == 0) A[idx] = expf(A[idx]);
    else if (op_code == 1) A[idx] = logf(A[idx]);
    else if (op_code == 2) A[idx] = A[idx] + B_scalar[0];
    else if (op_code == 3) A[idx] = A[idx] - B_scalar[0];
    else if (op_code == 4) A[idx] = A[idx] * B_scalar[0];
    else if (op_code == 5) A[idx] = A[idx] / B_scalar[0];

}


extern "C"
__global__ void convolutionKernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,        
    const int* A_shape, const int* B_shape, const int* C_shape, const int A_shape_size, const int B_shape_size,    
    const int C_shape_size, const int stride, const int padding) 
{
    int batch_idx = blockIdx.z;  // Batch dimension
    int out_channel = blockIdx.y;  // Output channel
    int out_row = blockIdx.x * blockDim.x + threadIdx.x;  // Output row
    int out_col = blockDim.y * threadIdx.y + threadIdx.y;  // Output column

    if (out_row >= C_shape[2] || out_col >= C_shape[3]) {
        return;
    }

    float output_val = 0.0f;

    for (int in_channel = 0; in_channel < A_shape[1]; ++in_channel) {
        for (int k_row = 0; k_row < B_shape[2]; ++k_row) {
            for (int k_col = 0; k_col < B_shape[3]; ++k_col) {
                int in_row = out_row * stride + k_row - padding;
                int in_col = out_col * stride + k_col - padding;

                if (in_row >= 0 && in_row < A_shape[2] && in_col >= 0 && in_col < A_shape[3]) {
                    int a_idx = ((batch_idx * A_shape[1] + in_channel) * A_shape[2] + in_row) * A_shape[3] + in_col;
                    int b_idx = ((out_channel * A_shape[1] + in_channel) * B_shape[2] + k_row) * B_shape[3] + k_col;

                    output_val += A[a_idx] * B[b_idx];
                }
            }
        }
    }

    int c_idx = ((batch_idx * C_shape[1] + out_channel) * C_shape[2] + out_row) * C_shape[3] + out_col;
    C[c_idx] = output_val;
}

extern "C"
__global__ void poolingKernel(const float* __restrict__ A, float* __restrict__ C, const int* A_shape,     
const int* C_shape, const int A_shape_size, const int C_shape_size, const int pool_size,   
const int stride, const int padding, const int pool_type)   
{
    int batch_idx = blockIdx.z;  // Batch dimension
    int out_channel = blockIdx.y;  // Output channel
    int out_row = blockIdx.x * blockDim.x + threadIdx.x;  // Output row
    int out_col = threadIdx.y;  // Output column

    if (out_row >= C_shape[2] || out_col >= C_shape[3]) {
        return;
    }

    float pool_val = (pool_type == 0) ? -INFINITY : 0.0f;
    int pool_count = 0;

    for (int p_row = 0; p_row < pool_size; ++p_row) {
        for (int p_col = 0; p_col < pool_size; ++p_col) {
            int in_row = out_row * stride + p_row - padding;
            int in_col = out_col * stride + p_col - padding;

            if (in_row >= 0 && in_row < A_shape[2] && in_col >= 0 && in_col < A_shape[3]) {
                int a_idx = ((batch_idx * A_shape[1] + out_channel) * A_shape[2] + in_row) * A_shape[3] + in_col;

                if (pool_type == 0) {
                    pool_val = fmaxf(pool_val, A[a_idx]);
                }
                else if (pool_type == 1) {
                    pool_val += A[a_idx];
                    pool_count++;
                }
            }
        }
    }

    if (pool_type == 1 && pool_count > 0) {
        pool_val /= pool_count;
    }

    int c_idx = ((batch_idx * C_shape[1] + out_channel) * C_shape[2] + out_row) * C_shape[3] + out_col;
    C[c_idx] = pool_val;
}

extern "C"
__global__ void dropoutKernel(const float* __restrict__ A,
float* __restrict__ C, const int* A_shape, const int A_shape_size,
const float dropout_prob, const int seed) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int total_elements = 1;

    for (int i = 0; i < A_shape_size; ++i) {
        total_elements *= A_shape[i];
    }

    if (idx >= total_elements) {
        return;
    }

    curandState state;
    curand_init(seed, idx, 0, &state);

    float random_val = curand_uniform(&state);

    if (random_val < dropout_prob) {
        C[idx] = 0.0f;  // Drop the element
    } else {
        C[idx] = A[idx] / (1.0f - dropout_prob);  // Scale the element
    }
}

