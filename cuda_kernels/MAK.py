
import numpy as np
from utils import inst_module

# max at axis
class MAK():
    def __init__(self):
        self.kernel = inst_module.module.get_function("maxAxisKernel")
        self.cache_avail = False

        # to be filled
        self.pre_axis = None
        self.post_axis = None
        self.max_axis = None
        self.block_size = None
        self.grid_size = None
    
    def compute_meta(self, A, axis):
        self.cache_avail = True
        self.pre_axis = np.int32(np.prod(A.shape[:axis])) if axis else np.int32(1)
        self.max_axis = np.int32(A.shape[axis])
        self.post_axis = np.int32(1) if axis == len(A.shape) - 1 else np.int32(np.prod(A.shape[axis+1:]))
        self.block_size = (32, 32, 1)
        self.grid_size = (int(np.ceil(self.post_axis / 32)), int(np.ceil(self.pre_axis / 32)), 1)

    def run(self, A, B, axis=0):
        if not self.cache_avail:
            self.compute_meta(A, axis)
        self.kernel(A.data, B.data, self.pre_axis, self.max_axis,\
                     self.post_axis, block=self.block_size, grid=self.grid_size)