
import numpy as np
from utils import inst_module

# summation at axis 
class SAK():
    def __init__(self):
        self.kernel = inst_module.module.get_function("sumAxisKernel")
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