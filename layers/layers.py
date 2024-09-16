from utils.D_data import D_data
from cuda_kernels import EWK, PK

# this is the base class for all layers in NN
class layer:
    def __init__(self, batch_size, in_features, out_features, cuda_module=None):

        self.params = {} # contains both the weights and the params both on the host and the device
        self.cache = {} # cache is stored on the device

        self.kernels = {}
        self.kernels["weight_update_mult"] = EWK()
        self.kernels["weight_update_sub"] = PK()

        self.in_shape = (batch_size, *(in_features if isinstance(in_features, tuple) else (in_features,)))
        self.out_shape = (batch_size, *(out_features if isinstance(out_features, tuple) else (out_features,)))
    
    def update_params(self, lr):

        for key in self.params.keys():
            _update = self.cache[key]
            self.kernels["weight_update_mult"].run(_update, lr, '*')
            
            old_weight = D_data(shape=self.params[key].shape)
            old_weight.copy_from(self.params[key])
            self.kernels["weight_update_sub"].run(old_weight, _update, self.params[key], '-')
            

