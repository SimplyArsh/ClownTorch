# 🤡-Torch
A DL library written as an extension to C247 projects. Instead of using NumPy for operations, it uses in-house CUDA kernels.

Features:
 - support for affine, convolutional, dropout (buggy at the moment), relu and pooling layers
 - trained a model with over 58% accuracy on CIFAR-10
 - CUDA kernels to speed up processing w/ minimal overhead b/c meta-data computation occurs @ intialization & data stays on GPU.
 - NumPy comptability. Written modularly, where kernels can be replaced w/ NumPy layers

Use at your own risk.

# Updates to come
My server got corrupted because I was out of disk quota. I lost the most recent updates and there might be some minor bugs, which I will fix promptly. I will work on adding support for other optimizers like ADAM. I have code for them in NumPy, just need to write my own CUDA kernels.

module load conda; module load cuda; cd ~/ClownTorch/NN/; conda activate ~/ClownTorch/env/
