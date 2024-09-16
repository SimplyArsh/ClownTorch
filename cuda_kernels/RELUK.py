import numpy as np
from utils import inst_module

# RELU kernel 
class RELUK():
    def __init__(self):
        self.kernel = inst_module.module.get_function("reluKernel")
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