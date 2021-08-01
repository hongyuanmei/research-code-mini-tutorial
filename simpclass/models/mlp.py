import torch
from torch import nn

class MLP(nn.Module):
    # a simple 2-layer MLP model
    
    def __init__(self, input_dim = 2, output_dim=2, n_neurons = 16, activation=nn.Sigmoid):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_neurons = n_neurons
        self.h1=nn.Linear(input_dim, n_neurons)
        self.h2=nn.Linear(n_neurons, n_neurons)
        self.output=nn.Linear(n_neurons, output_dim)
        self.activation=activation()
    
    def forward(self, x):
        x=self.activation(self.h1(x.float()))
        x=self.activation(self.h2(x))
        x=self.output(x)
        return x