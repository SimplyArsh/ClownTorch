# instantiate a neural network
from D_data import D_data
from affine import affine
from relu import relu
from kernels import instantiate_cuda_module

class NNet():
    def __init__(self, batch_size=100):
        self.layers = []
        instantiate_cuda_module()
        self.batch_size = batch_size

    def add_layers(self, **kwargs):
        pass

    def add_affine_layer(self, in_size, out_size, weight_scale=1e-2):
        # adds an affine layer
        affine_layer = affine(in_size, out_size, self.batch_size, weight_scale=weight_scale)
        self.layers.append(affine_layer)
    
    def add_relu(self, features):
        relu_layer = relu(features, features, self.batch_size)
        self.layers.append(relu_layer)
    
    def forward(self, h_in):

        d_in = D_data(h_data=h_in)

        for layer in self.layers:
            d_out = D_data(shape=layer.out_shape)
            layer.forward(d_in, d_out)
            d_in.free()
            d_in = d_out
        
        h_out = d_in.to_host()
        d_in.free()
        return h_out
    
    def backward(self, h_loss): # the host sends us the loss

        d_dout = D_data(h_data=h_loss)

        for layer in self.layers[::-1]:
            d_din = D_data(shape=layer.in_shape)
            layer.backward(d_dout, d_din)
            d_dout.free()
            d_dout = d_din

        h_din = d_din.to_host()
        d_din.free()
        
        return h_din

    def loss(x, y):

        