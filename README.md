# Gradient Comparison Between Neural Network MModels Implemented using Flux and Pytorch

This repository helps to compare the gradients of NN models developed in python and Julia. Pytorch is used to develop NN  models in python and Flux.jl is used to develop NN models in Julia.

NN models are configured with 2 inputs, one output and one hidden layer of size 5 with ReLU activation function.

To compare the gradients, first run codes inside `py_torch.py` file in python. Then run command sin `jl_flux.jl` file in julia. 