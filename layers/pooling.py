import numpy as np
from utils import inst_module, D_data
from cuda_kernels import POOL

class PoolLayer():
    def __init__(self, pool_size=2, stride=2, padding=0, pool_type='max'):
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        self.pool_type = pool_type  # 'max' or 'avg'

        self.pool = POOL(pool_size, stride, padding, pool_type)

    def forward(self, A):

        N, C, H, W = A.shape
        out_height = (H - self.pool_size + 2 * self.padding) // self.stride + 1
        out_width = (W - self.pool_size + 2 * self.padding) // self.stride + 1

        C_out = D_data(np.zeros((N, C, out_height, out_width), dtype=np.float32))

        self.pool.run(A, C_out)

        return C_out

    def __repr__(self):
        return f"Pooling Layer(pool_size={self.pool_size}, stride={self.stride}, padding={self.padding}, pool_type={self.pool_type})"
