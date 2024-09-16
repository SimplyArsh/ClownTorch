from layers.layers import layer
import numpy as np
from utils.D_data import D_data
from cuda_kernels import RELUK, MMK

class relu(layer):
    def __init__(self, in_features, out_features, batch_size):

        # instantiate parent class
        layer.__init__(self, batch_size, in_features, out_features)
        self.cache['d_in'] = D_data(shape=self.in_shape)

        # kernels
        self.kernels["f_reluk"] = RELUK()
        self.kernels["b_reluk"] = RELUK()
        self.kernels["b_mmk"] = MMK()

    def forward(self, d_in, d_out, **kwargs):

        self.cache['d_in'].copy_from(d_in)
        self.kernels["f_reluk"].run(d_in, backward_pass=False)
        d_out.copy_from(d_in)

    def backward(self, d_dout, d_din, **kwargs):
        # d_dout is the upstream gradient; d_din (dx) is what we want to fill

        self.kernels["f_reluk"].run(self.cache['d_in'], backward_pass=True)
        self.kernels["b_mmk"].run(d_dout, self.cache['d_in'], d_din)
        
