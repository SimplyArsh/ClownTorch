from dataclasses import dataclass
import numpy as np
import pycuda.driver as cuda

@dataclass
class D_data():
    def __init__(self, h_data=None, dup=None, shape=None, type=np.float32):

        if h_data is not None:
            assert isinstance(h_data, np.ndarray), "Data is not of type np.ndarray"
            self.shape = h_data.shape
            self.flat_shape = (h_data.shape[0], np.prod(h_data.shape[1:]))
            self.type = h_data.dtype
            self.nbytes = h_data.nbytes
            d_data = cuda.mem_alloc(self.nbytes) # d_W holds the reference to W on the GPU
            cuda.memcpy_htod(d_data, h_data)
            self.data = d_data
        
        elif dup is not None:
            assert isinstance(dup, D_data), "Can not create duplicate of an object that is not D_data"
            self.shape = dup.shape
            self.flat_shape = dup.flat_shape
            self.nbytes = dup.nbytes
            self.type = dup.type
            d_data = cuda.mem_alloc(self.nbytes) # d_W holds the reference to W on the GPU
            self.data = d_data
        
        elif shape is not None:
            assert isinstance(shape, tuple), "Shape needs to be a tuple"
            self.shape = shape
            self.flat_shape = (shape[0], np.prod(shape[1:]))
            self.nbytes = int(np.dtype(type).itemsize*self.flat_shape[0]*self.flat_shape[1])
            self.type = np.dtype(type)
            d_data = cuda.mem_alloc(self.nbytes) # d_W holds the reference to W on the GPU
            self.data = d_data    

    def empty_like(self):
        return D_data(dup=self)
    
    def copy_from(self, other):
        assert isinstance(other, type(self)), "Only a D_data object can be copied"
        cuda.memcpy_dtod(self.data, other.data, self.nbytes)
        self.shape = other.shape # sometimes shape may be differnet
    
    def free(self):
        self.data.free()
        
    def to_host(self):
        h_out = np.zeros(shape=self.shape).astype(self.type)
        cuda.memcpy_dtoh(h_out, self.data)
        return h_out
    
