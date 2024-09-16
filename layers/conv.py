import numpy as np
from layers.layers import layer
from utils.D_data import D_data
from utils.inst_module import *
from cuda_kernels import CNV, PK, TK, SAK  # Assuming CNV is the convolution kernel

class conv(layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, input_shape, batch_size, weight_scale=1e-3):

        # Initialize parent class
        layer.__init__(self, batch_size, input_shape[0], out_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_shape = input_shape
        self.batch_size = batch_size

        W_shape = (out_channels, in_channels, kernel_size, kernel_size)
        W = np.random.normal(scale=weight_scale, size=W_shape).astype(np.float32)
        self.params['d_W'] = D_data(W)

        # Initialize biases
        b = np.random.normal(scale=weight_scale, size=(out_channels)).astype(np.float32)
        self.params['d_b'] = D_data(b)
        
        # Cache setup
        self.cache['d_W'] = self.params['d_W'].empty_like()
        self.cache['d_b'] = self.params['d_b'].empty_like()
        self.cache['d_in'] = D_data(shape=self.in_shape)

        # Kernels for forward and backward pass
        self.kernels["f_cnv"] = CNV(stride, padding)  # Assuming CNV is the convolution kernel
        self.kernels["f_addk"] = PK()
        self.kernels["b_tk1"] = TK()
        self.kernels["b_cnv1"] = CNV(stride, padding)
        self.kernels["b_tk2"] = TK()
        self.kernels["b_cnv2"] = CNV(stride, padding)
        self.kernels["b_sak"] = SAK()

    def forward(self, d_in, d_out, **kwargs):

        d_conv_out = d_out.empty_like()

        self.kernels["f_cnv"].run(d_in, self.params['d_W'], d_conv_out)
        self.kernels["f_addk"].run(d_conv_out, self.params['d_b'], d_out, '+')
        d_conv_out.free()

        self.cache['d_in'].copy_from(d_in)
        self.cache['d_W'].copy_from(self.params['d_W'])
        self.cache['d_b'].copy_from(self.params['d_b'])

    def backward(self, d_dout, d_din, **kwargs):

        N, C, H, W = self.cache['d_in'].shape  # Input dimensions
        d_WT = D_data(shape=self.params['d_W'].shape)  # Shape of W transpose for backward

        self.kernels["b_tk1"].run(self.cache['d_W'], d_WT)
        self.kernels["b_cnv1"].run(d_dout, d_WT, d_din)
        d_WT.free()

        d_inT = D_data(shape=self.cache['d_in'].shape)  # Transposed input
        self.kernels["b_tk2"].run(self.cache['d_in'], d_inT)
        self.kernels["b_cnv2"].run(d_inT, d_dout, self.cache['d_W'])
        d_inT.free()

        self.kernels["b_sak"].run(d_dout, self.cache['d_b'], axis=0)
