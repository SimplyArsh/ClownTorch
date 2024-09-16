import numpy as np
from utils import inst_module, D_data

# element wise kernel
class EWK():
    def __init__(self):
        self.kernel = inst_module.module.get_function("elementWiseKernel")
        self.cache_avail = False

        # to be filled
        self.A_shape_size = None
        self.d_A_shape = None
        self.block_size = None
        self.grid_size = None
        self.operator = None
    
    def compute_meta(self, A, operator):
        self.cache_avail = True
        self.A_shape_size = np.int32(len(A.shape))
        self.d_A_shape = D_data(np.array(A.shape).astype(np.int32))
        self.block_size = (32, 1, 1)
        self.grid_size = (int(np.ceil(np.prod(A.shape) / 32)), 1, 1)
        self.operator = np.int32({'exp': 0, 'log': 1, '+': 2, '-': 3, '*': 4, '/': 5}[operator])

    def run(self, A, B=np.float32(0), operator=None):
        if not self.cache_avail:
            self.compute_meta(A, operator)
        self.kernel(A.data, D_data(h_data=np.array([B]).astype(np.float32)).data, self.operator, self.d_A_shape.data, self.A_shape_size,\
                     block=self.block_size, grid=self.grid_size)