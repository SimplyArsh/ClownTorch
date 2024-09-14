import numpy as np

# this is the base class for all layers in NN
class layer:
    def __init__(self, batch_size, in_features, out_features, cuda_module=None):

        self.params = {} # contains both the weights and the params both on the host and the device
        self.cache = {} # cache is stored on the device

        self.in_shape = (batch_size, *(in_features if isinstance(in_features, tuple) else (in_features,)))
        self.out_shape = (batch_size, *(out_features if isinstance(out_features, tuple) else (out_features,)))