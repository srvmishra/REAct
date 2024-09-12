import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from utils.activations import *
from torchsummary import summary

gen = np.random.RandomState(seed=0)
create_param = lambda size, device: Parameter(torch.from_numpy(gen.normal(size=size)).to(device).float(), requires_grad=True)

class STan_(nn.Module):
    def __init__(self, device, size=1, typ='LAAF'):
        super(STan_, self).__init__()
        self.device = device
        if typ == 'LAAF':
            self.beta = create_param(1, device)
        if typ == 'NLAAF':
            self.beta = create_param(size, device)

    def forward(self, x):
        z = affine_transform(self.beta, 1.0, x)
        return z * torch.tanh(x)
    
    def slope_recovery(self):
        return torch.norm(self.beta) ** 2
    
class NewExp_(nn.Module):
    def __init__(self, device, size=1, typ='LAAF'):
        super(NewExp_, self).__init__()
        self.device = device
        if typ == 'LAAF':
            self.a = create_param(1, device)
            self.b = create_param(1, device)
            self.c = create_param(1, device)
            self.d = create_param(1, device)
        if typ == 'NLAAF':
            self.a = create_param(size, device)
            self.b = create_param(size, device)
            self.c = create_param(size, device)
            self.d = create_param(size, device)

    def forward(self, x):
        z1 = affine_transform(self.a, self.b, x)
        z2 = affine_transform(self.c, self.d, x)
        return (1.0 - torch.exp(z1)) / (1.0 + torch.exp(z2))
    
    def slope_recovery(self):
        return torch.norm(self.a) ** 2 + torch.norm(self.b) ** 2 + torch.norm(self.c) ** 2 + torch.norm(self.d) ** 2


class CustomAAFLayer(nn.Module):
    def __init__(self, in_size, out_size, custom_act, device, layer_typ='LAAF'):
        super(CustomAAFLayer, self).__init__()
        self.layer = nn.Linear(in_size, out_size).to(device)
        self.act = custom_act(device, size=out_size, typ=layer_typ)
    
    def forward(self, x):
        return self.act(self.layer(x))
    
    def slope_recovery(self):
        return self.act.slope_recovery()
    

class NormalNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, custom_act):
        super(NormalNeuralNetwork, self).__init__()
        layers = []
        # layers = [nn.Linear(1, layer_sizes[1]), custom_act]
        for il, ol in zip(layer_sizes[:-2], layer_sizes[1:-1]):
            # custom_act = choose_act(custom_act_name, device, ol)
            layers.append(nn.Linear(il, ol))
            layers.append(custom_act)
        self.fe = nn.Sequential(*layers)
        self.fl = nn.Linear(layer_sizes[-2], layer_sizes[-1]).to(device)

    def forward(self, x, t=None):
        if t is not None:
            X = torch.hstack([x, t])
        else:
            X = x
        return self.fl(self.fe(X))

class CustomNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, custom_act, device, layer_typ='LAAF'):
        super(CustomNeuralNetwork, self).__init__()
        in_sizes = layer_sizes[:-2]
        out_sizes = layer_sizes[1:-1]
        layers_ = []
        for in_, out_ in zip(in_sizes, out_sizes):
            layers_.append(CustomAAFLayer(in_, out_, custom_act, device, layer_typ=layer_typ))

        self.fe = nn.Sequential(*layers_)
        self.fl = nn.Linear(layer_sizes[-2], layer_sizes[-1]).to(device)

    def forward(self, x, t=None):
        if t is not None:
            X = torch.hstack([x, t])
        else:
            X = x
        return self.fl(self.fe(X))
    
    def slope_recovery(self):
        loss = 0
        for l in self.fe:
            loss = loss + l.slope_recovery()
        return loss

    
class ABULayer(nn.Module):
    def __init__(self, in_features, out_features, device):
        super(ABULayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features).to(device)
        
        # Initialize the weights for the activations
        # self.params = create_param(5, device)
        self.weights = create_param(5, device)

    def forward(self, x):
        # Apply the linear transformation
        x = self.linear(x)
        
        # Apply different activations
        relu_out = F.relu(x)
        sigmoid_out = torch.sigmoid(x)
        tanh_out = torch.tanh(x)
        softplus_out = F.softplus(x)
        sin_out = torch.sin(x)
        
        # Stack the outputs along a new dimension
        stacked_outputs = torch.stack([relu_out, sigmoid_out, tanh_out, softplus_out, sin_out], dim=0)
        
        # Apply weights and compute the weighted sum
        wts_ = F.softmax(self.weights, dim=0)
        weighted_sum = torch.sum(wts_.view(-1, 1, 1) * stacked_outputs, dim=0)
        
        return weighted_sum
    
class ABUPINN(nn.Module):
    def __init__(self, layer_sizes, device):
        super(ABUPINN, self).__init__()
        # self.params = create_param(5, device)
        in_sizes = layer_sizes[:-2]
        out_sizes = layer_sizes[1:-1]
        layers_ = []
        for in_, out_ in zip(in_sizes, out_sizes):
            # print(in_, out_)
            layers_.append(ABULayer(in_, out_, device))
        
        self.fe = nn.Sequential(*layers_)
        self.fl = nn.Linear(layer_sizes[-2], layer_sizes[-1]).to(device)

    def forward(self, x, t=None):
        if t is not None:
            X = torch.hstack([x, t])
        else:
            X = x
        return self.fl(self.fe(X))
    

def create_neural_networks(act_name, layer_sizes, device, problem_type='PDE'):
    if problem_type == 'PDE':
        net = PDENeuralNet
    else:
        net = ODENeuralNet
    if act_name == 'ReLU':
        act = nn.ReLU()
        model = net(layer_sizes, act).to(device)
    if act_name == 'Sigmoid':
        act = nn.Sigmoid()
        model = net(layer_sizes, act).to(device)
    if act_name == 'Softplus':
        act = nn.Softplus()
        model = net(layer_sizes, act).to(device)
    if act_name == 'GELU':
        act = nn.GELU()
        model = net(layer_sizes, act).to(device)
    if act_name == 'tanh(x)':
        act = Tanh(device)
        model = net(layer_sizes, act).to(device)
    if act_name == 'sin(x)':
        act = Sin(device)
        model = net(layer_sizes, act).to(device)
    if act_name == 'sech(x)':
        act = Sech(device)
        model = net(layer_sizes, act).to(device)
    if act_name == 'STan-LAAF':
        act = STan(device)
        model = net(layer_sizes, act).to(device)
        # model = CustomNeuralNetwork(layer_sizes, STan_, device, layer_typ='LAAF')
    if act_name == '(ax + b)tanh(cx + d)':
        act = AffineTanh(device)
        model = net(layer_sizes, act).to(device)
        # model = CustomNeuralNetwork(layer_sizes, AffineTanh, device, layer_typ='LAAF')
    if act_name == 'sin(ax + b)tanh(cx + d)':
        act = SinTanh(device)
        model = net(layer_sizes, act).to(device)
        # model = CustomNeuralNetwork(layer_sizes, SinTanh, device, layer_typ='LAAF')
    if act_name == 'STan-NLAAF':
        model = CustomNeuralNetwork(layer_sizes, STan_, device, layer_typ='NLAAF')
    if act_name == 'REAct-LAAF':
        act = NewExp(device)
        model = net(layer_sizes, act).to(device)
        # model = CustomNeuralNetwork(layer_sizes, NewExp_, device, layer_typ='LAAF')
    if act_name == 'REAct-NLAAF':
        model = CustomNeuralNetwork(layer_sizes, NewExp_, device, layer_typ='NLAAF')
    if act_name == 'ABU':
        model = ABUPINN(layer_sizes, device)
    return model

# changed order of activations
# if results are not good, then we will go with ode and pde net above for our activations
# modify hparams and experiments - done
# copy models, activations, hyperparameters, experiments to GPU5 and run all files again, 
# hopefully, this time results will be good
activation_names = ['ReLU', 'Sigmoid', 'Softplus', 'GELU', 'tanh(x)', 'sin(x)', 'sech(x)', 
                    '(ax + b)tanh(cx + d)', 'sin(ax + b)tanh(cx + d)', 'STan-LAAF', 'STan-NLAAF',
                    'REAct-LAAF', 'REAct-NLAAF', 'ABU']
#'REAct-NLAAF']

    
## LAAF with slope recovery - REAct - Done
## NLAAF with slope recovery - REAct - Done
## NLAAF without slope recovery - REAct - Done
## ABU-PINN with these activations - sin, tanh, Relu, softplus, sigmoid - Done

## NTK eigenspectrum - done - but taking long time to train - maybe use jax - best case
## slope recovery - done - but not including now, will only state the theorem - later
## further work - experiment with network size, learning rate, optimizer, LAAF and NLAAF comparison, 
## with and without slope recovery, ntk for best case out of all above to show convergence

if __name__=="__main__":
    layer_sizes = [2, 32, 32, 32, 1]
    custom_act = NewExp_
    device = torch.device('cuda')
    layer_typ = 'LAAF'
    # nn = NeuralNetwork(layer_sizes, custom_act, device, layer_typ=layer_typ)
    # summary(nn.cuda(), (32, 2))
    pass

