import numpy as np
from utils import inst_module, D_data

# Convolution operation
class CNV():
    def __init__(self, stride=1, padding=0):
        self.kernel = inst_module.module.get_function("convolutionKernel")
        self.cache_avail = False

        # to be filled
        self.A_shape_size = None  # Input shape
        self.B_shape_size = None  # Kernel shape
        self.C_shape_size = None  # Output shape
        self.d_A_shape = None
        self.d_B_shape = None
        self.d_C_shape = None
        self.block_size = None
        self.grid_size = None
        self.stride = np.int32(stride)
        self.padding = np.int32(padding)
    
    def compute_meta(self, A, B, C):
        self.cache_avail = True
        self.A_shape_size = np.int32(len(A.shape))
        self.B_shape_size = np.int32(len(B.shape))
        self.C_shape_size = np.int32(len(C.shape))
        self.d_A_shape = D_data(np.array(A.shape).astype(np.int32))
        self.d_B_shape = D_data(np.array(B.shape).astype(np.int32))
        self.d_C_shape = D_data(np.array(C.shape).astype(np.int32))
        
        self.block_size = (16, 16, 1)  # Blocks of 16x16 threads
        self.grid_size = (int(np.ceil(C.shape[2] / 16)), int(np.ceil(C.shape[3] / 16)), C.shape[0] * C.shape[1])  # Grid size based on output shape
    
    def run(self, A, B, C, stride=None, padding=None):
        if stride is not None:
            self.stride = np.int32(stride)
        if padding is not None:
            self.padding = np.int32(padding)

        if not self.cache_avail:
            self.compute_meta(A, B, C)

        self.kernel(A.data, B.data, C.data, self.d_A_shape.data, self.d_B_shape.data, self.d_C_shape.data,
                    self.A_shape_size, self.B_shape_size, self.C_shape_size, 
                    self.stride, self.padding, block=self.block_size, grid=self.grid_size)

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
                f"stride={self.stride}, "
                f"padding={self.padding})")
