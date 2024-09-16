
import numpy as np
from layers.layers import layer
from utils.D_data import D_data
from utils.inst_module import *
from cuda_kernels import MAK, PK, EWK, TK, SAK, MMK

class softmax_loss():

    def __init__(self):
        # kernels
        self.kernels = {}
        self.kernels["max"] = MAK()
        self.kernels["subtract1"] = PK()
        self.kernels["exp"] = EWK()
        self.kernels["sum1"] = SAK()
        self.kernels["divide1"] = PK()
        self.kernels["log1"] = EWK()
        self.kernels["sum2"] = SAK()
        self.kernels["divide2"] = EWK()
        self.kernels["subtract2"] = PK()
        self.kernels["multiply"] = PK()
        self.kernels["sum3"] = SAK()
        self.kernels["log2"] = PK()
        self.kernels["sum4"] = SAK()
        self.kernels["b_tk2"] = TK()
        self.kernels["b_mmk2"] = MMK()
        self.kernels["b_sak"] = SAK()


    def loss(self, d_x, d_dx, h_y=None):

        # h_probs = np.exp(x - np.max(x, axis=1, keepdims=True))
        shape = d_x.shape; shape = (shape[0], 1)
        _d_x_max = D_data(shape=shape)
        self.kernels["max"].run(d_x, _d_x_max, axis=1)
        _d_x_max_minus_n = D_data(shape=d_x.shape)
        self.kernels["subtract1"].run(d_x, _d_x_max, _d_x_max_minus_n, '-')
        self.kernels["exp"].run(_d_x_max_minus_n, operator='exp')
        probs = _d_x_max_minus_n
        _d_x_max.free()

        # h_probs /= np.sum(h_probs, axis=1, keepdims=True)
        shape = probs.shape; shape = (shape[0], 1)
        _probs_sum = D_data(shape=shape)
        self.kernels["sum1"].run(probs, _probs_sum, axis=1)
        _normalized_prob = probs.empty_like()
        self.kernels["divide1"].run(probs, _probs_sum, _normalized_prob, '/')

        # in the case we are just checking accuracy
        if h_y is None:
            return _normalized_prob.to_host()

        # loss = -np.sum(np.log(probs[np.arange(N), y])) / N
        N = d_x.shape[0]
        _h_mask = np.zeros(shape=probs.shape).astype(np.float32)
        _h_mask[np.arange(N), h_y] = 1
        _d_mask = D_data(h_data=_h_mask)
        _d_mask_out = D_data(shape=_h_mask.shape)
        self.kernels["multiply"].run(_normalized_prob, _d_mask, _d_mask_out, '*')
        _d_select_out = D_data(shape=(probs.shape[0],))
        self.kernels["sum3"].run(_d_mask_out, _d_select_out, axis=1)
        self.kernels["log1"].run(_d_select_out, operator='log')
        _scalar_almost_loss = D_data(shape=(1, 1))
        self.kernels["sum4"].run(_d_select_out, _scalar_almost_loss, axis=0)
        h_loss = - _scalar_almost_loss.to_host().item() / N

        # dx[np.arange(N), h_y] -= 1; dx /= N
        self.kernels["subtract2"].run(_normalized_prob, _d_mask, _d_mask_out, '-')
        self.kernels["divide2"].run(_d_mask_out, N, '/')
        d_dx.copy_from(_d_mask_out)

        return h_loss