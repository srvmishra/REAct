import torch
from torch.nn.parameter import Parameter
import numpy as np
from copy import copy, deepcopy
from utils.Grids import OneDSpaceTimeGrid
from utils.Grids import array_to_meshgrid, meshgrid_to_array, numpy_to_tensor, tensor_to_numpy, _to, _like
from utils.Losses import compute_BC_loss, compute_IC_loss, compute_mixed_derivative, compute_single_derivative, MSE

gen = np.random.RandomState(seed=0)

def create_param(size):
    p = gen.uniform(low=0.2, high=2.5, size=size)
    return Parameter(torch.from_numpy(p).float(), requires_grad=True)

# create_param = lambda size: Parameter(torch.from_numpy(np.absolute(gen.normal(size=size))).float(), requires_grad=True)


def heat_loss(f, x, t, param, forcing_term=None):
  ut = compute_single_derivative(f, t, 1)
  uxx = compute_single_derivative(f, x, 2)
  pde_lhs = ut - param * uxx

  if forcing_term is not None:
    # external_force = forcing_fn(time, spatial_vars)
    pde_rhs = forcing_term.requires_grad_(False)
  else:
    pde_rhs = torch.zeros_like(pde_lhs).requires_grad_(False)

  loss = MSE(pde_lhs, pde_rhs)
  return loss

def wave_1d(f, x, t, param, forcing_term=None):
  utt = compute_single_derivative(f, t, 2)
  uxx = compute_single_derivative(f, x, 2)
  pde_lhs = utt - (param ** 2) * uxx

  if forcing_term is not None:
    # external_force = forcing_fn(time, spatial_vars)
    pde_rhs = forcing_term.requires_grad_(False)
  else:
    pde_rhs = torch.zeros_like(pde_lhs).requires_grad_(False)

  loss = MSE(pde_lhs, pde_rhs)
  return loss

class OneDSpaceTimeInverseEquations(OneDSpaceTimeGrid):
    def __init__(self, name, space_interval, time_interval, num_space, num_time, noise_std=0.1):
        super().__init__(space_interval, time_interval, num_space, num_time)

        self.init_loss_factor = 1.0
        self.boundary_loss_factor = 1.0 # 0.05
        self.ode_loss_factor = 1.0 # 10

        self.noise_std = noise_std

        self.name = name
        self.legends = ['Total Loss', 'Data Loss', 'BC Loss', 'IC Loss', 'PDE Loss']

        super().create_grid_1()
        self.initial_param_val = create_param(1)
        self.param = deepcopy(self.initial_param_val)

        # randomly sample points from training grid for analytical solution - data loss part
        indices = gen.choice(len(self.X_train), 5000, replace=False)
        self.data_x = self.X_train[indices, :].view(-1, 1)
        self.data_t = self.T_train[indices, :].view(-1, 1)

    def init_param(self):
        self.param = deepcopy(self.initial_param_val)

    def initial_condition_loss(self, net, init_fn, device):    
        xinit, tinit = _to([self.X_init, self.T_init], device, rg=False)
        yinit = init_fn(xinit) #.requires_grad_(False)
        xinit, tinit = xinit.requires_grad_(True), tinit.requires_grad_(True)
        out = net(xinit, t=tinit)
        loss = self.init_loss_factor * compute_IC_loss(out, tinit, yinit)
        return loss

    def boundary_condition_loss(self, net, vals, typs, mixing_fns, device):  
        '''
        vals, typs, mixing_fns in the format - left, right, top, bottom
        '''
        xleft, xright, tbdr = _to([self.X_left_bdr, self.X_right_bdr, self.T_bdr], device)
        out_left = net(xleft, t=tbdr)
        out_right = net(xright, t=tbdr)
        Vals = [_like(out_left, typ=vals[0]), _like(out_right, typ=vals[1])]
        loss = self.boundary_loss_factor * compute_BC_loss([out_left, out_right], [xleft, xright], Vals, typs, mixing_fns)
        return loss
    
    def set_noise_std(self, std):
       self.noise_std = std
    
    def data_loss(self, ana_fn, net, device):
       # impose data loss on the randomly sampled points in the domain
       x, t = _to([self.data_x, self.data_t], device, rg=False)
       rhs = ana_fn(x, t) + self.add_noise(x.shape, device)
       pred = net(x, t=t)
       return MSE(pred, rhs)
    
    def add_noise(self, size, device):
       # add noise to the analytical solution at the points sampled earlier
       noise = gen.normal(loc=0, scale=self.noise_std, size=size)
       return numpy_to_tensor(noise).to(device)
    
    def pde_loss(self, net, pde_fn, device, forcing_fn=None):
        if forcing_fn is not None:
            X, T = _to([self.X_train, self.T_train], device, rg=False)
            force = forcing_fn(X, T)
        else:
            force = None
        X, T = _to([self.X_train, self.T_train], device, rg=True)
        f = net(X, t=T)
        return pde_fn(f, X, T, self.param.to(device), forcing_term=force)
    
    def compute_analytical_solution(self, ana_fn, device):
        X, T = _to([self.X_test, self.T_test], device, rg=False)
        ana_sol = ana_fn(X, T)
        return tensor_to_numpy(ana_sol)
  
    def compute_PINN_solution(self, net, device):
        xeval, teval = _to([self.X_test, self.T_test], device, rg=False)  
        pinn_sol = net(xeval, t=teval)
        # sol = tensor_to_numpy(pinn_sol)
        return tensor_to_numpy(pinn_sol)

## Inverse heat and wave problem classes
class InverseHeatEquation(OneDSpaceTimeInverseEquations):
    def __init__(self):
        self.name = 'InverseHeatEquation'
        space_interval = [0.0, 1.0]
        time_interval = [0.0, 1.0]
        num_space = 100
        num_time = 100
        super().__init__(self.name, space_interval, time_interval, num_space, num_time)

        self.vals = [0.0, 0.0]
        self.typs = ['dirichlet', 'dirichlet']
        self.mixing_fns = [None, None]
        self.pde_fn = heat_loss

        self.param_ = 0.4

    def analytical_solution(self, x, t):
        if isinstance(x, torch.Tensor) and isinstance(t, torch.Tensor):
            return torch.exp(-self.param_ * (np.pi ** 2) * t) * torch.sin(np.pi * x)
        if isinstance(x, np.ndarray) and isinstance(t, np.ndarray):
            return np.exp(-self.param_ * (np.pi ** 2) * t) * np.sin(np.pi * x)

    def initial_condition(self, x):
        u0 = torch.sin(np.pi * x)
        # du0 = _like(x, typ=0)
        return [u0]
    
    def data_loss(self, net, device):
        return super().data_loss(self.analytical_solution, net, device)

    def initial_condition_loss(self, net, device):
        return super().initial_condition_loss(net, self.initial_condition, device)
    
    def boundary_condition_loss(self, net, device):
        return super().boundary_condition_loss(net, self.vals, self.typs, self.mixing_fns, device)
    
    def pde_loss(self, net, device):
        return super().pde_loss(net, self.pde_fn, device)
    
    def return_losses(self, net, device):
        data_loss = self.data_loss(net, device)
        bc_loss = self.boundary_condition_loss(net, device)
        ic_loss = self.initial_condition_loss(net, device)
        pde_loss = self.pde_loss(net, device)
        total_loss = data_loss + bc_loss + pde_loss + ic_loss
        return [total_loss, data_loss, bc_loss, ic_loss, pde_loss]
    
    def compute_analytical_solution(self, device):
        return super().compute_analytical_solution(self.analytical_solution, device)
    
    def compute_PINN_solution(self, net, device):
        return super().compute_PINN_solution(net, device)

class InverseWaveEquation(OneDSpaceTimeInverseEquations):
    def __init__(self):
        self.name = 'InverseWaveEquation'
        space_interval = [0.0, 2.0]
        time_interval = [0.0, 1.0]
        num_space = 100
        num_time = 100
        super().__init__(self.name, space_interval, time_interval, num_space, num_time)

        self.vals = [0.0, 0.0]
        self.typs = ['dirichlet', 'dirichlet']
        self.mixing_fns = [None, None]
        self.pde_fn = wave_1d

        self.param_ = 2.0

    def analytical_solution(self, x, t):
        if isinstance(x, torch.Tensor) and isinstance(t, torch.Tensor):
            return torch.sin(np.pi * x / 2.0) * torch.cos(np.pi * t)
        if isinstance(x, np.ndarray) and isinstance(t, np.ndarray):
            return np.sin(np.pi * x / 2.0) * np.cos(np.pi * t)
        
    def initial_condition(self, x):
        u0 = torch.sin(np.pi * x / 2.0)
        du0 = _like(x, typ=0)
        return [u0, du0]
    
    def data_loss(self, net, device):
        return super().data_loss(self.analytical_solution, net, device)
    
    def initial_condition_loss(self, net, device):
        return super().initial_condition_loss(net, self.initial_condition, device)
    
    def boundary_condition_loss(self, net, device):
        return super().boundary_condition_loss(net, self.vals, self.typs, self.mixing_fns, device)
    
    def pde_loss(self, net, device):
        return super().pde_loss(net, self.pde_fn, device)
    
    def return_losses(self, net, device):
        data_loss = self.data_loss(net, device)
        bc_loss = self.boundary_condition_loss(net, device)
        ic_loss = self.initial_condition_loss(net, device)
        pde_loss = self.pde_loss(net, device)
        total_loss = data_loss + bc_loss + pde_loss + ic_loss
        return [total_loss, data_loss, bc_loss, ic_loss, pde_loss]
    
    def compute_analytical_solution(self, device):
        return super().compute_analytical_solution(self.analytical_solution, device)
    
    def compute_PINN_solution(self, net, device):
        return super().compute_PINN_solution(net, device)