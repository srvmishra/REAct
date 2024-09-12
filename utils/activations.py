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

class Tanh(nn.Module):
    def __init__(self, device):
        super(Tanh, self).__init__()
        self.device = device

    def forward(self, x):
        return torch.tanh(x)
    
class Sin(nn.Module):
    def __init__(self, device):
        super(Sin, self).__init__()
        self.device = device

    def forward(self, x):
        return torch.sin(x)
    
class Sech(nn.Module):
    def __init__(self, device):
        super(Sech, self).__init__()
        self.device = device

    def forward(self, x):
        z1 = torch.exp(x)
        z2 = torch.exp(-x)
        return 2.0/(z1 + z2)

class STan(nn.Module):
    def __init__(self, device):
        super(STan, self).__init__()
        self.device = device
        self.beta = sample_params(1, device)[0]

    def forward(self, x):
        z = affine_transform(self.beta, 1.0, x)
        return z * torch.tanh(x)
    
    def slope_recovery(self):
        return self.beta ** 2
    
class AffineTanh(nn.Module):
    def __init__(self, device):
        super(AffineTanh, self).__init__()
        self.device = device
        self.a, self.b, self.c, self.d = sample_params(4, device)

    def forward(self, x):
        z1 = affine_transform(self.a, self.b, x)
        z2 = affine_transform(self.c, self.d, x)
        return z1 * torch.tanh(z2)
    
    def slope_recovery(self):
        return self.a ** 2 + self.b ** 2 + self.c ** 2 + self.d ** 2
    
class SinTanh(nn.Module):
    def __init__(self, device):
        super(SinTanh, self).__init__()
        self.device = device
        self.a, self.b, self.c, self.d = sample_params(4, device)

    def forward(self, x):
        z1 = affine_transform(self.a, self.b, x)
        z2 = affine_transform(self.c, self.d, x)
        return torch.sin(z1) * torch.tanh(z2)
    
    def slope_recovery(self):
        return self.a ** 2 + self.b ** 2 + self.c ** 2 + self.d ** 2
    
class NewExp(nn.Module):
    def __init__(self, device):
        super(NewExp, self).__init__()
        self.device = device
        self.a, self.b, self.c, self.d = sample_params(4, device)

    def forward(self, x):
        z1 = affine_transform(self.a, self.b, x)
        z2 = affine_transform(self.c, self.d, x)
        return (1.0 - torch.exp(z1)) / (1.0 + torch.exp(z2))
    
    def slope_recovery(self):
        return self.a ** 2 + self.b ** 2 + self.c ** 2 + self.d ** 2
    
def create_activation_dict(device):
    activation_dict = {'tanh(x)': Tanh(device),
                       'sin(x)': Sin(device),
                       'sech(x)': Sech(device), 
                       '(1 + beta*x)tanh(x)': STan(device), 
                       '(ax + b)tanh(cx + d)': AffineTanh(device),
                       'sin(ax + b)tanh(cx + d)': SinTanh(device),
                       '(1 - exp(ax + b))/(1 + exp(ax + b))': NewExp(device)}
    return activation_dict

def create_baseline_activation_dict(device):
    activation_dict = {'ReLU': nn.ReLU(),
                       'Sigmoid': nn.Sigmoid(),
                       'Softplus': nn.Softplus(), 
                       'GELU': nn.GELU()}
    return activation_dict

# two neural nets - one for one input and one for two inputs - ode and pde nets
# without slope recovery part is done
# it will be done again in the models part and separate experiments will be written for it
class ODENeuralNet(nn.Module):
    def __init__(self, layer_sizes, custom_act):
        super(ODENeuralNet, self).__init__()
        layers = [nn.Linear(1, layer_sizes[1]), custom_act]
        for il, ol in zip(layer_sizes[1:-2], layer_sizes[2:-1]):
            # custom_act = choose_act(custom_act_name, device, ol)
            layers.append(nn.Linear(il, ol))
            layers.append(custom_act)
        self.fe = nn.Sequential(*layers)#.to(custom_act.device)
        self.fl = nn.Linear(layer_sizes[-2], layer_sizes[-1])#.to(custom_act.device)

    def forward(self, x):
        return self.fl(self.fe(x))
    
    def slope_recovery(self):
        l = 0
        for p in self.fe:
            if isinstance(p, STan) or isinstance(p, NewExp) or isinstance(p, SinTanh) or isinstance(p, AffineTanh):
                l = l + p.slope_recovery()
        return l
    
class PDENeuralNet(nn.Module):
    def __init__(self, layer_sizes, custom_act):
        super(PDENeuralNet, self).__init__()
        layers = [nn.Linear(2, layer_sizes[1]), custom_act]
        for il, ol in zip(layer_sizes[1:-2], layer_sizes[2:-1]):
            # custom_act = choose_act(custom_act_name, device, ol)
            layers.append(nn.Linear(il, ol))
            layers.append(custom_act)
        self.fe = nn.Sequential(*layers)#.to(custom_act.device)
        self.fl = nn.Linear(layer_sizes[-2], layer_sizes[-1])#.to(custom_act.device)

    def forward(self, x, t):
        X = torch.hstack([x, t])
        return self.fl(self.fe(X))
    
    def slope_recovery(self):
        l = 0
        for p in self.fe:
            if isinstance(p, STan) or isinstance(p, NewExp) or isinstance(p, SinTanh) or isinstance(p, AffineTanh):
                l = l + p.slope_recovery()
        return l

if __name__=="__main__":
    pass
