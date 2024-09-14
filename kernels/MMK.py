import numpy as np
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
        