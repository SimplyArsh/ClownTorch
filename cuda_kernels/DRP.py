import numpy as np
from utils import inst_module, D_data

# Dropout layer
class Dropout():
    def __init__(self, dropout_prob=0.5, seed=None):
        self.kernel = inst_module.module.get_function("dropoutKernel")
        self.cache_avail = False

        self.dropout_prob = np.float32(dropout_prob)
        self.seed = np.int32(seed) if seed is not None else np.int32(np.random.randint(0, 10000))

        # to be filled later
        self.A_shape_size = None
        self.d_A_shape = None
        self.block_size = None
        self.grid_size = None
    
    def compute_meta(self, A):
        self.cache_avail = True
        self.A_shape_size = np.int32(len(A.shape))
        self.d_A_shape = D_data(np.array(A.shape).astype(np.int32))
        
        # Define block and grid size for dropout
        self.block_size = (512, 1, 1)  # 512 threads per block
        self.grid_size = (int(np.ceil(np.prod(A.shape) / self.block_size[0])), 1, 1)

    def run(self, A, C, dropout_prob=None, seed=None):
        if dropout_prob is not None:
            self.dropout_prob = np.float32(dropout_prob)
        if seed is not None:
            self.seed = np.int32(seed)

        if not self.cache_avail:
            self.compute_meta(A)

        # Call the CUDA kernel for dropout
        self.kernel(A.data, C.data, self.d_A_shape.data, self.A_shape_size, 
                    self.dropout_prob, self.seed, block=self.block_size, grid=self.grid_size)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"A_shape_size={self.A_shape_size}, "
                f"d_A_shape={self.d_A_shape.to_host()}, "
                f"block_size={self.block_size}, "
                f"grid_size={self.grid_size}, "
                f"dropout_prob={self.dropout_prob}, "
                f"seed={self.seed})")
