# instantiate a neural network
from utils import D_data, softmax_loss
from layers import affine, relu, pooling, dropout, conv
from utils.inst_module import instantiate_cuda_module
import pycuda.autoinit

class NNet():
    def __init__(self, batch_size=100):
        self.layers = []
        instantiate_cuda_module()
        self.batch_size = batch_size
        self.loss_func = softmax_loss()

    def add_affine_layer(self, in_size, out_size, weight_scale=1e-2):
        # adds an affine layer
        affine_layer = affine(in_size, out_size, self.batch_size, weight_scale=weight_scale)
        self.layers.append(affine_layer)
    
    def add_relu(self):
        features = self.layers[-1].out_shape[1:]
        relu_layer = relu(features, features, self.batch_size)
        self.layers.append(relu_layer)
    
    def add_pooling_layer(self, pool_size=2, stride=2, pad=0, pool_type='max'):

        in_shape = self.layers[-1].out_shape
        pooling_layer = pooling(pool_size=pool_size, stride=stride, pad=pad, pool_type=pool_type)
        
        out_height = (in_shape[2] - pool_size + 2 * pad) // stride + 1
        out_width = (in_shape[3] - pool_size + 2 * pad) // stride + 1
        out_shape = (in_shape[0], in_shape[1], out_height, out_width)
        
        pooling_layer.out_shape = out_shape
        self.layers.append(pooling_layer)

    def add_dropout_layer(self, drp_prob=0.5):
        # Adds a dropout layer
        features = self.layers[-1].out_shape
        drp = dropout(drp_prob=drp_prob)

        drp.out_shape = features
        self.layers.append(drp)
    
    def add_conv_layer(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, weight_scale=1e-2):
        in_shape = self.layers[-1].out_shape
        conv_layer = conv(in_channels, out_channels, kernel_size, stride, pad,\
                           in_shape, self.batch_size, weight_scale=weight_scale)

        out_height = (in_shape[2] - kernel_size + 2 * pad) // stride + 1
        out_width = (in_shape[3] - kernel_size + 2 * pad) // stride + 1
        out_shape = (self.batch_size, out_channels, out_height, out_width)

        conv_layer.out_shape = out_shape
        self.layers.append(conv_layer)

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
    
    def backward(self, d_dloss): # the host sends us the loss

        if isinstance(d_dloss, D_data):
            d_dout = d_dloss
        else:
            d_dout = D_data(h_data=d_dloss)


        for layer in self.layers[::-1]:
            d_din = D_data(shape=layer.in_shape)
            layer.backward(d_dout, d_din)
            d_dout.free()
            d_dout = d_din
        h_din = d_din.to_host()
        d_din.free()
        return h_din
    
    def loss(self, h_x, h_y=None):

        h_y_pred = self.forward(h_x)

        d_y_pred, d_dloss = D_data(h_data=h_y_pred), D_data(h_data=h_y_pred)

        if h_y is None:
            return self.loss_func.loss(d_y_pred, d_dloss)
        
        h_loss = self.loss_func.loss(d_y_pred, d_dloss, h_y)

        self.backward(d_dloss)

        return h_loss

    def update_params(self, lr):
        for layer in self.layers:
            layer.update_params(lr)
        




