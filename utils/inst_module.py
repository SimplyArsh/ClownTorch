from pycuda.compiler import SourceModule
import os

def instantiate_cuda_module():
    # read the source code from a file
    global module
    source_code = ""
    current_directory = os.path.dirname(os.path.abspath(__file__))
    with open('/u/home/a/arshd/ClownTorch/NN/cuda_kernels/cuda_kernels.cu', 'r') as f:
        source_code = f.read()
    module = SourceModule(source_code)