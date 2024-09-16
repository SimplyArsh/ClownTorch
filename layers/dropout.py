import numpy as np
from utils import inst_module, D_data
from cuda_kernels import DRP

class DropoutLayer():
    def __init__(self, dropout_prob=0.5, seed=None):
        self.dropout_prob = dropout_prob
        self.seed = seed

        self.dropout = DRP(dropout_prob, seed)

    def forward(self, A, training=True):
        # If not training, simply return the input (no dropout)
        if not training:
            return A

        # Create output tensor
        C_out = D_data(np.zeros_like(A.to_host(), dtype=np.float32))

        # Perform dropout
        self.dropout.run(A, C_out)

        return C_out

    def __repr__(self):
        return f"Dropout Layer(dropout_prob={self.dropout_prob}, seed={self.seed})"
