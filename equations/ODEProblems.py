from utils.Grids import OneDSpaceGrid, OneDTimeGrid
from utils.Grids import array_to_meshgrid, meshgrid_to_array, numpy_to_tensor, tensor_to_numpy, _to, _like
from utils.Losses import *

class OneDSpaceEquations(OneDSpaceGrid):
    def __init__(self, name, interval, num_points):
        super().__init__(interval, num_points)

        self.boundary_loss_factor = 1.0 # 0.05
        self.ode_loss_factor = 1.0 # 10

        self.name = name
        self.legends = ['Total Loss', 'BC Loss', 'ODE Loss']
        self.train_proto = None

    def boundary_condition_loss(self, net, vals, typs, mixing_fns, device):  
        '''
        vals, typs, mixing_fns in the format - left, right, top, bottom
        Euler Beam needs more
        Remember to make this general change to all equations in further implementations of more general equations
        '''
        xleft, xright = _to([self.left_bdr, self.right_bdr], device)
        out_left = net(xleft)
        out_right = net(xright)
        left_vals, right_vals = vals
        left_Vals = [_like(out_left, typ=v) for v in left_vals]
        right_Vals = [_like(out_right, typ=v) for v in right_vals]
        left_typs, right_typs = typs
        left_fns, right_fns = mixing_fns
        left_loss = compute_BC_loss([out_left]*len(left_Vals), [xleft]*len(left_Vals), left_Vals, left_typs, left_fns)
        right_loss = compute_BC_loss([out_right]*len(right_Vals), [xright]*len(right_Vals), 
                                     right_Vals, right_typs, right_fns)
        loss = self.boundary_loss_factor * (left_loss + right_loss)
        return loss
    
    def ode_loss(self, net, ode_fn, device, forcing_fn=None):
        if forcing_fn is not None:
            X = _to([self.X_train], device, rg=False)[0]
            force = forcing_fn(X)
        else:
            force = None
        X = _to([self.X_train], device, rg=True)[0]
        f = net(X)
        return ode_fn(f, X, forcing_term=force)
    
    def compute_ntk_eigenvalue(self, net, ode_fn, device, bc_typs):
        '''
        assuming that net is already in eval mode
        we are considering net with one output neuron - basically scalar fields
        bc_typs = left_bc_typs, right_bc_typs
        '''
        ## boundary condition data, output, and grad - modify
        xleft, xright = _to([self.left_bdr, self.right_bdr], device, rg=True)
        bcs = [xleft, xright]
        grads_ = []
        for i, p in enumerate(bc_typs):
            bcx = bcs[i]
            out_bdr = net(bcx)
            for p_ in p:
                if p_ == 'dirichlet':
                    out = out_bdr
                if p_ == 'neumann':
                    out = compute_single_derivative(out_bdr, bcx, 1)
                if p_ == 'neumann_2':
                    out = compute_single_derivative(out_bdr, bcx, 2)
                if p_ == 'neumann_3':
                    out = compute_single_derivative(out_bdr, bcx, 3)
                grad_params = torch.autograd.grad(out[:, -1].sum(), net.parameters(), retain_graph=True)
                grads_.extend([grad.view(-1) for grad in grad_params])
        grads_bdr = torch.cat(grads_)

        ## ode residual data, output, and grad - modify
        Xode = _to([self.X_train], device, rg=True)[0]
        out_ode = net(Xode)
        ode_res = ode_fn(out_ode, Xode, forcing_term=None, res=True)
        grad_params = torch.autograd.grad(ode_res[:, -1].sum(), net.parameters(), retain_graph=True)
        grads_ode = torch.cat([grad.view(-1) for grad in grad_params])

        ## construct the ntk - modify
        Kuu = torch.outer(grads_bdr.t(), grads_bdr)
        Kur = torch.outer(grads_bdr.t(), grads_ode)

        Kru = torch.outer(grads_ode.t(), grads_bdr)
        Krr = torch.outer(grads_ode.t(), grads_ode)

        ntk = torch.vstack([
            torch.hstack([Kuu, Kur]),
            torch.hstack([Kru, Krr])
        ])

        eigenvalues = torch.linalg.eigvalsh(ntk).cpu().numpy()
        smallest_eigenvalue = np.min(np.abs(eigenvalues))
        return smallest_eigenvalue

    def compute_analytical_solution(self, ana_fn, device):
        X = _to([self.X_test], device, rg=False)[0]
        ana_sol = ana_fn(X)
        return tensor_to_numpy(ana_sol)

    def compute_PINN_solution(self, net, device):
        xeval = _to([self.X_test], device, rg=False)[0]  
        pinn_sol = net(xeval)
        return tensor_to_numpy(pinn_sol)
    
class OneDTimeEquations(OneDTimeGrid):
    def __init__(self, name, interval, num_points, train_proto=1):
        super().__init__(interval, num_points)

        self.init_loss_factor = 1.0 # 0.05
        self.ode_loss_factor = 1.0 # 10

        self.name = name
        self.legends = ['Total Loss', 'IC Loss', 'ODE Loss']
        self.train_proto = train_proto

        if self.train_proto == 1:
            super().create_grid_1()
        if self.train_proto == 2:
            super().create_grid_2()

    def initial_condition_loss(self, net, init_fn, device):    
        tinit = _to([self.T_init], device, rg=False)[0]
        yinit = init_fn(tinit) #.requires_grad_(False), list of values [y0, dy0, d2y0, ...]
        tinit = tinit.requires_grad_(True)
        out = net(tinit)
        loss = self.init_loss_factor * compute_IC_loss(out, tinit, yinit)
        return loss
    
    def ode_loss(self, net, ode_fn, device, forcing_fn=None):
        if forcing_fn is not None:
            T = _to([self.T_train], device, rg=False)[0]
            force = forcing_fn(T)
        else:
            force = None
        T = _to([self.T_train], device, rg=True)[0]
        f = net(T)
        return ode_fn(f, T, forcing_term=force)
    
    '''
    the NTK part of the code
    '''
    def compute_ntk_eigenvalue(self, net, ode_fn, device, num_inits=1):
        ## initial condition part
        tinit = _to([self.T_init], device, rg=True)[0]
        out_init = net(tinit)
        grads_ = []
        for i in range(num_inits):
            f = compute_single_derivative(out_init, tinit, i)
            grad_params = torch.autograd.grad(f[:, -1].sum(), net.parameters(), retain_graph=True)
            grads_.extend([grad.view(-1) for grad in grad_params])
        grads_init = torch.cat(grads_)

        ## ode residual data, output, and grad - modify
        Tode = _to([self.T_train], device, rg=True)[0]
        out_ode = net(Tode)
        ode_res = ode_fn(out_ode, Tode, forcing_term=None, res=True)
        grad_params = torch.autograd.grad(ode_res[:, -1].sum(), net.parameters(), retain_graph=True)
        grads_ode = torch.cat([grad.view(-1) for grad in grad_params])

        ## construct the ntk - modify
        Kii = torch.outer(grads_init.t(), grads_init)
        Kir = torch.outer(grads_init.t(), grads_ode)

        Kri = torch.outer(grads_ode.t(), grads_init)
        Krr = torch.outer(grads_ode.t(), grads_ode)

        ntk = torch.vstack([
            torch.hstack([Kii, Kir]),
            torch.hstack([Kri, Krr])
        ])

        eigenvalues = torch.linalg.eigvalsh(ntk).cpu().numpy()
        smallest_eigenvalue = np.min(np.abs(eigenvalues))
        return smallest_eigenvalue

    def compute_analytical_solution(self, ana_fn, device):
        T = _to([self.T_test], device, rg=False)[0]
        ana_sol = ana_fn(T)
        return tensor_to_numpy(ana_sol)

    def compute_PINN_solution(self, net, device):
        teval = _to([self.T_test], device, rg=False)[0]  
        pinn_sol = net(teval)
        return tensor_to_numpy(pinn_sol)


class EulerBeamEquation(OneDSpaceEquations):
    def __init__(self):
        self.name = 'EulerBeam'
        interval = [0.0, 1.0]
        num_points = 1000
        super().__init__(self.name, interval, num_points)    
        
        self.vals = [[0.0, 0.0], [0.0, 0.0]]
        self.typs = [['dirichlet', 'neumann'], ['neumann_2', 'neumann_3']]
        self.mixing_fns = [[None, None], [None, None]]
        self.ode_fn = euler_beam_loss

    def analytical_solution(self, x):
        return - x**4/24 + x**3/6 - x**2/4
    
    def boundary_condition_loss(self, net, device):
        return super().boundary_condition_loss(net, self.vals, self.typs, self.mixing_fns, device)
    
    def ode_loss(self, net, device):
        return super().ode_loss(net, self.ode_fn, device)
    
    def return_losses(self, net, device):
        bc_loss = self.boundary_condition_loss(net, device)
        ode_loss = self.ode_loss(net, device)
        total_loss = bc_loss + ode_loss
        return [total_loss, bc_loss, ode_loss]
    
    def compute_ntk_eigenvalue(self, net, device):
        return super().compute_ntk_eigenvalue(net, self.ode_fn, device, self.typs)

    def compute_analytical_solution(self, device):
        return super().compute_analytical_solution(self.analytical_solution, device)
    
    def compute_PINN_solution(self, net, device):
        return super().compute_PINN_solution(net, device)


'''
We are not doing the below equations
'''
class Poisson1DEquation(OneDSpaceEquations):
    def __init__(self):
        self.name = 'Poisson1D'
        interval = [-1.0, 0.0]
        num_points = 1000
        super().__init__(self.name, interval, num_points)

        self.vals = [[0.0], [0.0]]
        self.typs = [['dirichlet'], ['dirichlet']]
        self.mixing_fns = [[None], [None]]
        self.ode_fn = poisson1d

    def analytical_solution(self, x):
        if isinstance(x, torch.Tensor):
            return torch.sin(np.pi * x)
        if isinstance(x, np.ndarray):
            return np.sin(np.pi * x)
        
    def forcing_fn(self, x):
        return (np.pi ** 2) * self.analytical_solution(x)
        
    def boundary_condition_loss(self, net, device):
        return super().boundary_condition_loss(net, self.vals, self.typs, self.mixing_fns, device)
    
    def ode_loss(self, net, device):
        return super().ode_loss(net, self.ode_fn, device, forcing_fn=self.forcing_fn)
    
    def return_losses(self, net, device):
        bc_loss = self.boundary_condition_loss(net, device)
        ode_loss = self.ode_loss(net, device)
        total_loss = bc_loss + ode_loss
        return [total_loss, bc_loss, ode_loss]
    
    def compute_ntk_eigenvalue(self, net, device):
        return super().compute_ntk_eigenvalue(net, self.ode_fn, device, self.typs)
        
    def compute_analytical_solution(self, device):
        return super().compute_analytical_solution(self.analytical_solution, device)
    
    def compute_PINN_solution(self, net, device):
        return super().compute_PINN_solution(net, device)

class UnderDampedVibrations(OneDTimeEquations):
    def __init__(self, train_proto=1):
        self.name = 'UnderDampedVibrations1D'
        interval = [0.0, 1.0]
        num_points = 1000
        self.omega_n = 3.0
        self.zeta = 0.5
        self.omega_d = self.omega_n * np.sqrt(1.0 - self.zeta ** 2)
        super().__init__(self.name, interval, num_points, train_proto)

        self.ode_fn = underdamped_vibration

    def analytical_solution(self, t):
        z = self.zeta * self.omega_n
        if isinstance(t, torch.Tensor):
            factor = torch.exp(-z * t)
            return factor * (torch.cos(self.omega_d * t) + (z/self.omega_d) * torch.sin(self.omega_d * t))
        if isinstance(t, np.ndarray):
            factor = np.exp(-z * t)
            return factor * (np.cos(self.omega_d * t) + (z/self.omega_d) * np.sin(self.omega_d * t))
        
    def initial_condition(self, t):
        y0 = _like(t, typ=1)
        dy0 = _like(t, typ=0)
        return [y0, dy0]
    
    def initial_condition_loss(self, net, device):
        return super().initial_condition_loss(net, self.initial_condition, device)
    
    def ode_loss(self, net, device):
        return super().ode_loss(net, self.ode_fn, device)
    
    def return_losses(self, net, device):
        ic_loss = self.initial_condition_loss(net, device)
        ode_loss = self.ode_loss(net, device)
        total_loss = ic_loss + ode_loss
        return [total_loss, ic_loss, ode_loss]
    
    def compute_ntk_eigenvalue(self, net, device):
        return super().compute_ntk_eigenvalue(net, self.ode_fn, device, num_inits=2)
    
    def compute_analytical_solution(self, device):
        return super().compute_analytical_solution(self.analytical_solution, device)
    
    def compute_PINN_solution(self, net, device):
        return super().compute_PINN_solution(net, device)
