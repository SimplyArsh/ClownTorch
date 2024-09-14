from pycuda.compiler import SourceModule
import numpy as np
import os

def instantiate_cuda_module():
    # read the source code from a file
    global module
    source_code = ""
    current_directory = os.path.dirname(os.path.abspath(__file__))

    with open(current_directory + '/kernels.cu', 'r') as f:
        source_code = f.read()
    
    module = SourceModule(source_code)

# multiplication
class MMK():
    def __init__(self):
        self.kernel = module.get_function("matrixMulKernel")
        self.cache_avail = False

        # to be filled
        self.A_size = None
        self.AB_size = None
        self.B_size = None
        self.block_size = None
        self.grid_size = None
    
    def compute_meta(self, A, B):
        self.cache_avail = True
        self.A_size = np.int32(A.flat_shape[0])
        self.AB_size = np.int32(A.flat_shape[1])
        self.B_size = np.int32(B.flat_shape[1])
        self.block_size = (32, 32, 1)
        self.grid_size = (int(np.ceil(self.B_size / 32)), int(np.ceil(self.A_size / 32)), 1)

    def run(self, A, B, C):
        if not self.cache_avail:
            self.compute_meta(A, B)
        self.kernel(A.data, B.data, C.data, self.A_size, self.AB_size,\
                     self.B_size, block=self.block_size, grid=self.grid_size)
    
# addition of vectors
class ADDK():
    def __init__(self):
        self.kernel = module.get_function("addKernel")
        self.cache_avail = False

        # to be filled
        self.A_size = None
        self.AB_size = None
        self.B_size = None
        self.block_size = None
        self.grid_size = None
    
    def compute_meta(self, A):
        self.cache_avail = True
        self.A_size = np.int32(A.flat_shape[0])
        self.AB_size = np.int32(A.flat_shape[1])
        self.block_size = (32, 32, 1)
        self.grid_size = (int(np.ceil(self.AB_size / 32)), int(np.ceil(self.A_size / 32)), 1)

    def run(self, A, B):
        if not self.cache_avail:
            self.compute_meta(A)
        self.kernel(A.data, B.data, self.A_size, self.AB_size,\
                     block=self.block_size, grid=self.grid_size)

# transpose 
class TK():
    def __init__(self):
        self.kernel = module.get_function("transposeKernel")
        self.cache_avail = False

        # to be filled
        self.N_size = None
        self.D_size = None
        self.block_size = None
        self.grid_size = None
    
    def compute_meta(self, A):
        self.cache_avail = True
        self.N_size = np.int32(A.flat_shape[0])
        self.D_size = np.int32(A.flat_shape[1])
        self.block_size = (32, 32, 1)
        self.grid_size = (int(np.ceil(self.D_size / 32)), int(np.ceil(self.N_size / 32)), 1)

    def run(self, A, B):
        if not self.cache_avail:
            self.compute_meta(A)
        self.kernel(A.data, B.data, self.N_size, self.D_size,\
                     block=self.block_size, grid=self.grid_size)

# summation at axis 
class SAK():
    def __init__(self):
        self.kernel = module.get_function("sumAxisKernel")
        self.cache_avail = False

        # to be filled
        self.pre_axis = None
        self.post_axis = None
        self.sum_axis = None
        self.block_size = None
        self.grid_size = None
    
    def compute_meta(self, A, axis):
        self.cache_avail = True
        self.pre_axis = np.int32(np.prod(A.shape[:axis])) if axis else np.int32(1)
        self.sum_axis = np.int32(A.shape[axis])
        self.post_axis = np.int32(1) if axis == len(A.shape) - 1 else np.int32(np.prod(A.shape[axis+1:]))
        self.block_size = (32, 32, 1)
        self.grid_size = (int(np.ceil(self.post_axis / 32)), int(np.ceil(self.pre_axis / 32)), 1)

    def run(self, A, B, axis=0):
        if not self.cache_avail:
            self.compute_meta(A, axis)
        self.kernel(A.data, B.data, self.pre_axis, self.sum_axis,\
                     self.post_axis, block=self.block_size, grid=self.grid_size)
    
        
# RELU kernel 
class RELUK():
    def __init__(self):
        self.kernel = module.get_function("reluKernel")
        self.cache_avail = False

        # to be filled
        self.N_size = None
        self.D_size = None
        self.ones = None
        self.block_size = None
        self.grid_size = None
    
    def compute_meta(self, A, backward_pass):
        self.cache_avail = True
        self.N_size = np.int32(A.flat_shape[0])
        self.D_size = np.int32(A.flat_shape[1])
        self.ones = np.int32(1) if backward_pass else np.int32(0)
        self.block_size = (32, 32, 1)
        self.grid_size = (int(np.ceil(self.D_size / 32)), int(np.ceil(self.N_size / 32)), 1)

    def run(self, A, backward_pass=False):
        if not self.cache_avail:
            self.compute_meta(A, backward_pass)
        self.kernel(A.data, self.N_size, self.D_size,\
                     self.ones, block=self.block_size, grid=self.grid_size)