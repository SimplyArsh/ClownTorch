from layers import layer
import numpy as np
from D_data import D_data
from kernels import *

class affine(layer):
    def __init__(self, in_features, out_features, batch_size, weight_scale=1e-3):

        # instantiate parent class
        layer.__init__(self, batch_size, in_features, out_features)

        # W
        W = np.random.normal(scale=weight_scale, size=(in_features, out_features)).astype(np.float32)
        self.params['d_W'] = D_data(W)

        # b
        b = np.random.normal(scale=weight_scale, size=(out_features)).astype(np.float32)
        self.params['d_b'] = D_data(b)
        
        # cache
        self.cache['d_W'] = self.params['d_W'].empty_like()
        self.cache['d_b'] = self.params['d_b'].empty_like()
        self.cache['d_in'] = D_data(shape=self.in_shape)

        # kernels
        self.kernels = {}
        self.kernels["f_mmk"] = MMK()
        self.kernels["f_addk"] = ADDK()
        self.kernels["b_tk1"] = TK()
        self.kernels["b_mmk1"] = MMK()
        self.kernels["b_tk2"] = TK()
        self.kernels["b_mmk2"] = MMK()
        self.kernels["b_sak"] = SAK()

    def forward(self, d_in, d_out, **kwargs):

        # W*x + b
        self.kernels["f_mmk"].run(d_in, self.params['d_W'], d_out)
        self.kernels["f_addk"].run(d_out, self.params['d_b'])
        
        # update cache
        self.cache['d_in'].copy_from(d_in)
        self.cache['d_W'].copy_from(self.params['d_W'])
        self.cache['d_b'].copy_from(self.params['d_b'])

    def backward(self, d_dout, d_din, **kwargs):
        # d_dout is the upstream gradient; d_din (dx) is what we want to fill

        # dx
        N, D = self.params['d_W'].flat_shape
        d_WT = D_data(shape=(D, N))
        self.kernels["b_tk1"].run(self.cache['d_W'], d_WT)
        self.kernels["b_mmk1"].run(d_dout, d_WT, d_din)
        d_WT.free()

        # dw
        N, D = self.cache['d_in'].flat_shape
        d_inT = D_data(shape=(D, N))
        self.kernels["b_tk2"].run(self.cache['d_in'], d_inT)
        self.kernels["b_mmk2"].run(d_inT, d_dout, self.cache['d_W'])
        d_inT.free()

        #db
        self.kernels["b_sak"].run(d_dout, self.cache['d_b'], axis=0)