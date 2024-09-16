import numpy as np
from utils import inst_module, D_data

# addition of vectors
class PK():
    def __init__(self):
        self.kernel = inst_module.module.get_function("primitiveKernel")
        self.cache_avail = False

        # to be filled
        self.A_shape_size = None
        self.B_shape_size = None
        self.C_shape_size = None
        self.d_A_shape = None
        self.d_B_shape = None
        self.d_C_shape = None
        self.block_size = None
        self.grid_size = None
        self.operator = None
    
    def compute_meta(self, A, B, C, operator):
        self.cache_avail = True
        self.A_shape_size = np.int32(len(A.shape))
        self.B_shape_size = np.int32(len(B.shape))
        self.C_shape_size = np.int32(len(C.shape))
        self.d_A_shape = D_data(np.array(A.shape).astype(np.int32))
        self.d_B_shape = D_data(np.array(B.shape).astype(np.int32))
        self.d_C_shape = D_data(np.array(C.shape).astype(np.int32))
        self.block_size = (512, 1, 1)
        self.grid_size = (int(np.ceil(np.prod(C.shape) / 32)), 1, 1)
        self.operator = np.int32({'+': 0, '-': 1, '*': 2, '/': 3}[operator])

    def run(self, A, B, C, operator=None):
        if not self.cache_avail:
            self.compute_meta(A, B, C, operator)
        self.kernel(A.data, B.data, C.data, self.operator, self.d_A_shape.data, self.d_B_shape.data,\
                     self.d_C_shape.data, self.A_shape_size, self.B_shape_size, self.C_shape_size, \
                        block=self.block_size, grid=self.grid_size)
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"A_shape_size={self.A_shape_size}, "
                f"B_shape_size={self.B_shape_size}, "
                f"C_shape_size={self.C_shape_size}, "
                f"d_A_shape={self.d_A_shape.to_host()}, "
                f"d_B_shape={self.d_B_shape.to_host()}, "
                f"d_C_shape={self.d_C_shape.to_host()}, "
                f"block_size={self.block_size}, "
                f"grid_size={self.grid_size}, "
                f"operator={self.operator})")
 