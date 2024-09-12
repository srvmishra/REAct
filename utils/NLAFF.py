import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np

gen = np.random.RandomState(seed=0)
def sample_params(num_params, device):
    params_list = []
    for _ in range(num_params):
        num = gen.normal(size=1)
        param = Parameter(torch.from_numpy(num).to(device).float(), requires_grad=True)
        params_list.append(param)
    return params_list

def affine_transform(a, b, x):
    return a * x + b

class STanAll(nn.Module):
    def __init__(self, device, num_outs):
        super(STanAll, self).__init__()
        num = gen.normal(size=num_outs)
        self.beta = Parameter(torch.from_numpy(num).to(self.device), requires_grad=True)        

    def forward(self, x):
        return (1.0 + self.beta * x) * torch.tanh(x)