import numpy as np
from utils import inst_module, D_data

# Pooling layer
class POOL():
    def __init__(self, pool_size, stride=1, padding=0, pool_type='max'):
        self.kernel = inst_module.module.get_function("poolingKernel")
        self.cache_avail = False

        self.pool_size = np.int32(pool_size)
        self.stride = np.int32(stride)
        self.padding = np.int32(padding)
        self.pool_type = np.int32(0 if pool_type == 'max' else 1)

        # to be filled later
        self.A_shape_size = None
        self.C_shape_size = None
        self.d_A_shape = None
        self.d_C_shape = None
        self.block_size = None
        self.grid_size = None
    
    def compute_meta(self, A, C):
        self.cache_avail = True
        self.A_shape_size = np.int32(len(A.shape))
        self.C_shape_size = np.int32(len(C.shape))
        self.d_A_shape = D_data(np.array(A.shape).astype(np.int32))
        self.d_C_shape = D_data(np.array(C.shape).astype(np.int32))
        
        self.block_size = (16, 16, 1)
        self.grid_size = (int(np.ceil(C.shape[2] / 16)), int(np.ceil(C.shape[3] / 16)), C.shape[0] * C.shape[1])

    def run(self, A, C, pool_size=None, stride=None, padding=None, pool_type=None):
        if pool_size is not None:
            self.pool_size = np.int32(pool_size)
        if stride is not None:
            self.stride = np.int32(stride)
        if padding is not None:
            self.padding = np.int32(padding)
        if pool_type is not None:
            self.pool_type = np.int32(0 if pool_type == 'max' else 1)

        if not self.cache_avail:
            self.compute_meta(A, C)

        self.kernel(A.data, C.data, self.d_A_shape.data, self.d_C_shape.data,
                    self.A_shape_size, self.C_shape_size, self.pool_size, self.stride, 
                    self.padding, self.pool_type, block=self.block_size, grid=self.grid_size)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"A_shape_size={self.A_shape_size}, "
                f"C_shape_size={self.C_shape_size}, "
                f"d_A_shape={self.d_A_shape.to_host()}, "
                f"d_C_shape={self.d_C_shape.to_host()}, "
                f"block_size={self.block_size}, "
                f"grid_size={self.grid_size}, "
                f"pool_size={self.pool_size}, "
                f"stride={self.stride}, "
                f"padding={self.padding}, "
                f"pool_type={'max' if self.pool_type == 0 else 'avg'})")
