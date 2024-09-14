from NNet import NNet
import numpy as np
import pycuda.driver as cuda
from python_layers import *
from D_data import D_data


nn = NNet(batch_size=1000)
# nn.add_affine_layer(200, 212)
# nn.add_affine_layer(212, 300)
nn.add_relu(200)

h_in = np.random.randn(1000,200).astype(np.float32)

h_out = nn.forward(h_in)

h_loss = np.random.randn(1000,200).astype(np.float32)
h_dx = nn.backward(h_loss)
print(h_dx)

# h_W = nn.layers[0].params['d_W'].to_host()
# h_b = nn.layers[0].params['d_b'].to_host()
# np_out1, cache1 = affine_forward(h_in, h_W, h_b)
# h_W = nn.layers[1].params['d_W'].to_host()
# h_b = nn.layers[1].params['d_b'].to_host()
# np_out2, cache2 = affine_forward(np_out1, h_W, h_b)

# np_loss = np.random.randn(*np_out2.shape).astype(np.float32)
# h_dx = nn.backward(np_loss)

# np_dx1, np_dw1, np_db1 = affine_backward(np_loss, cache2)
# np_dx2, np_dw2, np_db2 = affine_backward(np_dx1, cache1)

# print(f'Forward: CUDA and NumPy {"DO" if np.allclose(h_out, np_out2) else "DO NOT"} match.')
# print(f'Backward: CUDA and NumPy {"DO" if np.allclose(h_dx, np_dx2) else "DO NOT"} match.')

# print(np.sum(np.abs(nn.layers[1].cache['d_W'].to_host() - np_dw1))/np.prod(np_dw1.shape))
# print(np.sum(np.abs(nn.layers[0].cache['d_W'].to_host() - np_dw2))/np.prod(np_dw2.shape))

# print(np.sum(np.abs(nn.layers[1].cache['d_b'].to_host() - np_db1))/np.prod(np_db1.shape))
# print(np.sum(np.abs(nn.layers[0].cache['d_b'].to_host() - np_db2))/np.prod(np_db2.shape))


# -------------------------------------------------------

# import pycuda.driver as cuda
# import pycuda.autoinit
# import numpy as np
# from pycuda.compiler import SourceModule
# import time

# # Define your CUDA kernels as a string
# cuda_code = """
# __global__ void matrixMulKernel(const float* A, const float* B, float* C, int A_size, int AB_size, int B_size) {
#     int row = blockIdx.y * blockDim.y + threadIdx.y;
#     int col = blockIdx.x * blockDim.x + threadIdx.x;
    
#     if (row < A_size && col < B_size) {
#         float value = 0.0f;
#         for (int k = 0; k < AB_size; ++k) {
#             value += A[row * AB_size + k] * B[k * B_size + col];
#         }
#         C[row * B_size + col] = value;
#     }
# }

# __global__ void addKernel(float* A, float* B, int A_size, int AB_size) {
#     int row = blockIdx.y * blockDim.y + threadIdx.y;
#     int col = blockIdx.x * blockDim.x + threadIdx.x;

#     if (row < A_size && col < AB_size) {
#         A[row * AB_size + col] += B[col];
#     }
# }
# """

# # Compile the CUDA code
# mod = SourceModule(cuda_code)

# # Get the kernel functions
# matrixMulKernel = mod.get_function("matrixMulKernel")
# addKernel = mod.get_function("addKernel")

# # Example matrix sizes
# A_size = 1000   # number of rows in A
# AB_size = 1000  # number of columns in A and number of rows in B
# B_size = 1000   # number of columns in B

# # Initialize matrices A, B, and C
# A = np.random.rand(A_size, AB_size).astype(np.float32)
# B = np.random.rand(AB_size, B_size).astype(np.float32)
# C = np.zeros((A_size, B_size), dtype=np.float32)

# # Allocate device memory for A, B, and C
# cuda_time = time.time()
# A_gpu = cuda.mem_alloc(A.nbytes)
# B_gpu = cuda.mem_alloc(B.nbytes)
# C_gpu = cuda.mem_alloc(C.nbytes)

# # Copy host memory to device memory
# cuda.memcpy_htod(A_gpu, A)
# cuda.memcpy_htod(B_gpu, B)

# # Set up the block and grid dimensions
# block_size = (16, 16, 1)
# grid_size = (int(np.ceil(B_size / block_size[0])), int(np.ceil(A_size / block_size[1])))

# # Launch matrix multiplication kernel
# matrixMulKernel(A_gpu, B_gpu, C_gpu, np.int32(A_size), np.int32(AB_size), np.int32(B_size),
#                 block=block_size, grid=grid_size)

# # Copy result from device to host
# cuda.memcpy_dtoh(C, C_gpu)
# print("CUDA time: ", time.time() - cuda_time)

# np_time = time.time()
# C_np = np.matmul(A, B)
# print("NumPy time: ", time.time() - np_time)

# # Print result
# print(f'CUDA and NumPy {"DO" if np.allclose(C, C_np) else "DO NOT"} match.')


# # Test the addKernel (element-wise addition of B to each row of A)
# # Reuse A and B for simplicity
# cuda.memcpy_htod(A_gpu, A)
# cuda.memcpy_htod(B_gpu, B[0])  # Assuming B is a vector for the addKernel

# # Launch add kernel
# addKernel(A_gpu, B_gpu, np.int32(A_size), np.int32(AB_size), block=block_size, grid=grid_size)

# # Copy result from device to host
# cuda.memcpy_dtoh(A, A_gpu)


# ------- transpose ---------

# import pycuda.driver as cuda
# import pycuda.autoinit
# from pycuda.compiler import SourceModule

# import numpy as np
# a = np.arange(200).reshape(2, 20, 5)

# a = a.astype(np.float32)
# a_gpu = cuda.mem_alloc(a.nbytes)
# cuda.memcpy_htod(a_gpu, a)

# b = np.empty(shape=(2, 5)).astype(np.float32)
# b_gpu = cuda.mem_alloc(b.nbytes)

# mod = SourceModule("""
# __global__ void sumAxisKernel(float* input, float* output, int pre_axis, int sum_axis, int post_axis) {
#     int row = blockIdx.y * blockDim.y + threadIdx.y;
#     int col = blockIdx.x * blockDim.x + threadIdx.x;

#     if (row < pre_axis && col < post_axis) {
#         float value = 0.0f;
#         for (int k = 0; k < sum_axis; ++k)
#             value += input[row * sum_axis * post_axis + col + k * post_axis];

#         output[row * post_axis + col] = value;
#     }
# }

#   """)

# func = mod.get_function("sumAxisKernel")
# func(a_gpu, b_gpu, np.int32(2), np.int32(20), np.int32(5), block=(5,2,1))
# cuda.memcpy_dtoh(b, b_gpu)
# print(a)
# print("out: ", b)