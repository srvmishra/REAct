from utils.Grids import OneDSpaceTimeGrid
from utils.Grids import array_to_meshgrid, meshgrid_to_array, numpy_to_tensor, tensor_to_numpy, _to, _like
from utils.Losses import *
from scipy.io import loadmat
from scipy.interpolate import griddata

def load_burgers_data():
    data = np.load("./datasets/Burgers.npz")
    t, x, f = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    xx, tt = xx.flatten()[:, None], tt.flatten()[:, None]
    return xx, tt, f

def load_allen_cahn_data():
    data = loadmat("./datasets/Allen_Cahn.mat")
    t = data["t"]
    x = data["x"]
    f = data["u"]
    xx, tt = np.meshgrid(x, t)
    xx, tt = xx.flatten()[:, None], tt.flatten()[:, None]
    return xx, tt, f

class OneDSpaceTimeEquations(OneDSpaceTimeGrid):
    def __init__(self, name, space_interval, time_interval, num_space, num_time, train_proto=1):
        super().__init__(space_interval, time_interval, num_space, num_time)

        self.init_loss_factor = 1.0
        self.boundary_loss_factor = 1.0 # 0.05
        self.ode_loss_factor = 1.0 # 10

        self.name = name
        self.legends = ['Total Loss', 'BC Loss', 'IC Loss', 'PDE Loss']
        self.train_proto = train_proto

        if train_proto == 1:
            super().create_grid_1()
        if train_proto == 2:
            super().create_grid_2()

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
    
    def pde_loss(self, net, pde_fn, device, forcing_fn=None):
        if forcing_fn is not None:
            X, T = _to([self.X_train, self.T_train], device, rg=False)
            force = forcing_fn(X, T)
        else:
            force = None
        X, T = _to([self.X_train, self.T_train], device, rg=True)
        f = net(X, t=T)
        return pde_fn(f, X, T, forcing_term=force)
    
    def compute_ntk_eigenvalue(self, net, pde_fn, device, bc_typs, num_inits=1):
        '''
        assuming that net is already in eval mode
        we are considering net with one output neuron - basically scalar fields
        we are excluding wave equation for now because we are assuming only one initial condition
        for multiple initial conditions, the init grad needs to be modified
        include multiple initial conditions to make it good. take one extra parameter, anyways only
        network output derivatives matter, not the actual initial or boundary conditions.
        '''
        ## initial condition data, output, and grad
        xinit, tinit = _to([self.X_init, self.T_init], device, rg=True)
        out_init = net(xinit, t=tinit)
        grads_ = []
        for i in range(num_inits):
            f = compute_single_derivative(out_init, tinit, i)
            grad_params = torch.autograd.grad(f[:, -1].sum(), net.parameters(), retain_graph=True)
            grads_.extend([grad.view(-1) for grad in grad_params])
        grads_init = torch.cat(grads_)

        ## boundary condition data, output, and grad
        xleft, xright, tbdr = _to([self.X_left_bdr, self.X_right_bdr, self.T_bdr], device, rg=True)
        bcs = [(xleft, tbdr), (xright, tbdr)]
        grads_ = []
        for i, p in enumerate(bc_typs):
            bcx, bct = bcs[i]
            out_bdr = net(bcx, t=bct)
            '''
            make it more generalized for higher order space equations with more BCs like the OneDSpaceEquations
            '''
            if p == 'dirichlet':
                out = out_bdr
            if p == 'neumann':
                out = compute_single_derivative(out_bdr, bct, 1)
            if p == 'neumann_2':
                out = compute_single_derivative(out_bdr, bct, 2)
            if p == 'neumann_3':
                out = compute_single_derivative(out_bdr, bct, 3)
            grad_params = torch.autograd.grad(out[:, -1].sum(), net.parameters(), retain_graph=True)
            grads_.extend([grad.view(-1) for grad in grad_params])
        grads_bdr = torch.cat(grads_)

        ## pde residual data, output, and grad
        Xpde, Tpde = _to([self.X_train, self.T_train], device, rg=True)
        out_pde = net(Xpde, t=Tpde)
        pde_res = pde_fn(out_pde, Xpde, Tpde, forcing_term=None, res=True)
        grad_params = torch.autograd.grad(pde_res[:, -1].sum(), net.parameters(), retain_graph=True)
        grads_pde = torch.cat([grad.view(-1) for grad in grad_params])

        ## construct the ntk
        Kuu = torch.outer(grads_bdr.t(), grads_bdr)
        Kui = torch.outer(grads_bdr.t(), grads_init)
        Kur = torch.outer(grads_bdr.t(), grads_pde)

        Kiu = torch.outer(grads_init.t(), grads_bdr)
        Kii = torch.outer(grads_init.t(), grads_init)
        Kir = torch.outer(grads_init.t(), grads_pde)

        Kru = torch.outer(grads_pde.t(), grads_bdr)
        Kri = torch.outer(grads_pde.t(), grads_init)
        Krr = torch.outer(grads_pde.t(), grads_pde)

        ntk = torch.vstack([
            torch.hstack([Kuu, Kui, Kur]),
            torch.hstack([Kiu, Kii, Kir]),
            torch.hstack([Kru, Kri, Krr])
        ])

        eigenvalues = torch.linalg.eigvalsh(ntk).cpu().numpy()
        smallest_eigenvalue = np.min(np.abs(eigenvalues))
        return smallest_eigenvalue
    
    def compute_analytical_solution(self, ana_fn, device):
        X, T = _to([self.X_test, self.T_test], device, rg=False)
        ana_sol = ana_fn(X, T)
        return tensor_to_numpy(ana_sol)
  
    def compute_PINN_solution(self, net, device):
        xeval, teval = _to([self.X_test, self.T_test], device, rg=False)  
        pinn_sol = net(xeval, t=teval)
        # sol = tensor_to_numpy(pinn_sol)
        return tensor_to_numpy(pinn_sol)
    
    def interpolate_grid_data(self, x, t, f, xx, tt, method='cubic'):
        # xx, tt = array_to_meshgrid(xx, self.test_shape), array_to_meshgrid(tt, self.test_shape)
        xt = np.hstack((x, t))
        X = np.hstack((xx, tt))
        # print(f.shape, xt.shape, X.shape)
        sol = griddata(xt, f.flatten()[:, None], X, method=method)
        return sol
        # return array_to_meshgrid(sol, self.test_shape)
    

class HeatEquation(OneDSpaceTimeEquations):
    def __init__(self, train_proto=1):
        self.name = 'HeatEquation'
        space_interval = [0.0, 1.0]
        time_interval = [0.0, 1.0]
        num_space = 100
        num_time = 100
        super().__init__(self.name, space_interval, time_interval, num_space, num_time, train_proto)

        self.vals = [0.0, 0.0]
        self.typs = ['dirichlet', 'dirichlet']
        self.mixing_fns = [None, None]
        self.pde_fn = heat_loss

    def analytical_solution(self, x, t):
        if isinstance(x, torch.Tensor) and isinstance(t, torch.Tensor):
            return torch.exp(-0.4 * (np.pi ** 2) * t) * torch.sin(np.pi * x)
        if isinstance(x, np.ndarray) and isinstance(t, np.ndarray):
            return np.exp(-0.4 * (np.pi ** 2) * t) * np.sin(np.pi * x)

    def initial_condition(self, x):
        u0 = torch.sin(np.pi * x)
        # du0 = _like(x, typ=0)
        return [u0]

    def initial_condition_loss(self, net, device):
        return super().initial_condition_loss(net, self.initial_condition, device)
    
    def boundary_condition_loss(self, net, device):
        return super().boundary_condition_loss(net, self.vals, self.typs, self.mixing_fns, device)
    
    def pde_loss(self, net, device):
        return super().pde_loss(net, self.pde_fn, device)
    
    def return_losses(self, net, device):
        bc_loss = self.boundary_condition_loss(net, device)
        ic_loss = self.initial_condition_loss(net, device)
        pde_loss = self.pde_loss(net, device)
        total_loss = bc_loss + pde_loss + ic_loss
        return [total_loss, bc_loss, ic_loss, pde_loss]
    
    def compute_ntk_eigenvalue(self, net, device):
        return super().compute_ntk_eigenvalue(net, self.pde_fn, device, self.typs, num_inits=1)
    
    def compute_analytical_solution(self, device):
        return super().compute_analytical_solution(self.analytical_solution, device)
    
    def compute_PINN_solution(self, net, device):
        return super().compute_PINN_solution(net, device)

class WaveEquation(OneDSpaceTimeEquations):
    def __init__(self, train_proto=1):
        self.name = 'WaveEquation'
        space_interval = [0.0, 2.0]
        time_interval = [0.0, 1.0]
        num_space = 100
        num_time = 100
        super().__init__(self.name, space_interval, time_interval, num_space, num_time, train_proto)

        self.vals = [0.0, 0.0]
        self.typs = ['dirichlet', 'dirichlet']
        self.mixing_fns = [None, None]
        self.pde_fn = wave_1d

    def analytical_solution(self, x, t):
        if isinstance(x, torch.Tensor) and isinstance(t, torch.Tensor):
            return torch.sin(np.pi * x / 2.0) * torch.cos(np.pi * t)
        if isinstance(x, np.ndarray) and isinstance(t, np.ndarray):
            return np.sin(np.pi * x / 2.0) * np.cos(np.pi * t)
        
    def initial_condition(self, x):
        u0 = torch.sin(np.pi * x / 2.0)
        du0 = _like(x, typ=0)
        return [u0, du0]
    
    def initial_condition_loss(self, net, device):
        return super().initial_condition_loss(net, self.initial_condition, device)
    
    def boundary_condition_loss(self, net, device):
        return super().boundary_condition_loss(net, self.vals, self.typs, self.mixing_fns, device)
    
    def pde_loss(self, net, device):
        return super().pde_loss(net, self.pde_fn, device)
    
    def return_losses(self, net, device):
        bc_loss = self.boundary_condition_loss(net, device)
        ic_loss = self.initial_condition_loss(net, device)
        pde_loss = self.pde_loss(net, device)
        total_loss = bc_loss + pde_loss + ic_loss
        return [total_loss, bc_loss, ic_loss, pde_loss]
    
    def compute_ntk_eigenvalue(self, net, device):
        return super().compute_ntk_eigenvalue(net, self.pde_fn, device, self.typs, num_inits=2)
    
    def compute_analytical_solution(self, device):
        return super().compute_analytical_solution(self.analytical_solution, device)
    
    def compute_PINN_solution(self, net, device):
        return super().compute_PINN_solution(net, device)

class DiffusionReaction(OneDSpaceTimeEquations):
    def __init__(self, train_proto=1):
        self.name = 'DiffusionReaction'
        space_interval = [-np.pi, np.pi]
        time_interval = [0.0, 1.0]
        num_space = 100
        num_time = 100
        super().__init__(self.name, space_interval, time_interval, num_space, num_time, train_proto)

        self.vals = [0.0, 0.0]
        self.typs = ['dirichlet', 'dirichlet']
        self.mixing_fns = [None, None]
        self.pde_fn = diffusion_rxn_loss

    def analytical_solution(self, x, t):
        if isinstance(x, torch.Tensor) and isinstance(t, torch.Tensor):
            sin = torch.sin
            exp = torch.exp
        if isinstance(x, np.ndarray) and isinstance(t, np.ndarray):
            sin = np.sin
            exp = np.exp
        return exp(-t) * (sin(x) + sin(2.0*x)/2.0 + sin(3.0*x)/3.0 + sin(4.0*x)/4.0 + sin(8.0*x)/8.0)
    
    def forcing_fn(self, x, t):
        sin = torch.sin
        exp = torch.exp
        return exp(-t) * (3.0*sin(2.0*x)/2.0 + 8.0*sin(3.0*x)/3.0 + 15*sin(4.0*x)/4.0 + 63*sin(8.0*x)/8.0)
    
    def initial_condition(self, x):
        sin = torch.sin
        u0 = sin(x) + sin(2.0*x)/2.0 + sin(3.0*x)/3.0 + sin(4.0*x)/4.0 + sin(8.0*x)/8.0
        return [u0]
    
    def initial_condition_loss(self, net, device):
        return super().initial_condition_loss(net, self.initial_condition, device)
    
    def boundary_condition_loss(self, net, device):
        return super().boundary_condition_loss(net, self.vals, self.typs, self.mixing_fns, device)
    
    def pde_loss(self, net, device):
        return super().pde_loss(net, self.pde_fn, device, forcing_fn=self.forcing_fn)
    
    def return_losses(self, net, device):
        bc_loss = self.boundary_condition_loss(net, device)
        ic_loss = self.initial_condition_loss(net, device)
        pde_loss = self.pde_loss(net, device)
        total_loss = bc_loss + pde_loss + ic_loss
        return [total_loss, bc_loss, ic_loss, pde_loss]
    
    def compute_ntk_eigenvalue(self, net, device):
        return super().compute_ntk_eigenvalue(net, self.pde_fn, device, self.typs, num_inits=1)
    
    def compute_analytical_solution(self, device):
        return super().compute_analytical_solution(self.analytical_solution, device)
    
    def compute_PINN_solution(self, net, device):
        return super().compute_PINN_solution(net, device)
    
class Diffusion(OneDSpaceTimeEquations):
    def __init__(self, train_proto=1):
        self.name = 'Diffusion'
        space_interval = [-1.0, 1.0]
        time_interval = [0.0, 1.0]
        num_space = 100
        num_time = 100
        super().__init__(self.name, space_interval, time_interval, num_space, num_time, train_proto)

        self.vals = [0.0, 0.0]
        self.typs = ['dirichlet', 'dirichlet']
        self.mixing_fns = [None, None]
        self.pde_fn = diffusion_rxn_loss

    def analytical_solution(self, x, t):
        if isinstance(x, torch.Tensor) and isinstance(t, torch.Tensor):
            sin = torch.sin
            exp = torch.exp
        if isinstance(x, np.ndarray) and isinstance(t, np.ndarray):
            sin = np.sin
            exp = np.exp
        return exp(-t) * sin(np.pi * x)
    
    def forcing_fn(self, x, t):
        sin = torch.sin
        exp = torch.exp
        return -exp(-t) * (sin(np.pi * x) - (np.pi ** 2) * sin(np.pi * x))
    
    def initial_condition(self, x):
        sin = torch.sin
        u0 = sin(np.pi * x) 
        return [u0]
    
    def initial_condition_loss(self, net, device):
        return super().initial_condition_loss(net, self.initial_condition, device)
    
    def boundary_condition_loss(self, net, device):
        return super().boundary_condition_loss(net, self.vals, self.typs, self.mixing_fns, device)
    
    def pde_loss(self, net, device):
        return super().pde_loss(net, self.pde_fn, device, forcing_fn=self.forcing_fn)
    
    def return_losses(self, net, device):
        bc_loss = self.boundary_condition_loss(net, device)
        ic_loss = self.initial_condition_loss(net, device)
        pde_loss = self.pde_loss(net, device)
        total_loss = bc_loss + pde_loss + ic_loss
        return [total_loss, bc_loss, ic_loss, pde_loss]
    
    def compute_ntk_eigenvalue(self, net, device):
        return super().compute_ntk_eigenvalue(net, self.pde_fn, device, self.typs, num_inits=1)
    
    def compute_analytical_solution(self, device):
        return super().compute_analytical_solution(self.analytical_solution, device)
    
    def compute_PINN_solution(self, net, device):
        return super().compute_PINN_solution(net, device)

class BurgersEquation(OneDSpaceTimeEquations):
    def __init__(self, train_proto=1):
        self.name = 'BurgersEquation'
        space_interval = [-1.0, 1.0]
        time_interval = [0.0, 1.0]
        num_space = 256
        num_time = 100
        super().__init__(self.name, space_interval, time_interval, num_space, num_time, train_proto)

        self.vals = [0.0, 0.0]
        self.typs = ['dirichlet', 'dirichlet']
        self.mixing_fns = [None, None]
        self.pde_fn = burgers_loss

    def analytical_solution(self, x, t):
        xx, tt, f = load_burgers_data()
        # print(f.shape)
        # xx, tt = np.meshgrid(xx, tt)
        x, t = tensor_to_numpy(x), tensor_to_numpy(t)
        sol = super().interpolate_grid_data(xx, tt, f, x, t, method='nearest')
        return sol

    def initial_condition(self, x):
        u0 = -torch.sin(np.pi * x)
        return [u0]
    
    def initial_condition_loss(self, net, device):
        return super().initial_condition_loss(net, self.initial_condition, device)
    
    def boundary_condition_loss(self, net, device):
        return super().boundary_condition_loss(net, self.vals, self.typs, self.mixing_fns, device)
    
    def pde_loss(self, net, device):
        return super().pde_loss(net, self.pde_fn, device)
    
    def return_losses(self, net, device):
        bc_loss = self.boundary_condition_loss(net, device)
        ic_loss = self.initial_condition_loss(net, device)
        pde_loss = self.pde_loss(net, device)
        total_loss = bc_loss + pde_loss + ic_loss
        return [total_loss, bc_loss, ic_loss, pde_loss]
    
    def compute_ntk_eigenvalue(self, net, device):
        return super().compute_ntk_eigenvalue(net, self.pde_fn, device, self.typs, num_inits=1)
    
    def compute_analytical_solution(self, device):
        x, t = self.X_test, self.T_test
        return self.analytical_solution(x, t)
    
    def compute_PINN_solution(self, net, device):
        return super().compute_PINN_solution(net, device)

class AllenCahnEquation(OneDSpaceTimeEquations):
    def __init__(self, train_proto=1):
        self.name = 'AllenCahnEquation'
        space_interval = [-1.0, 1.0]
        time_interval = [0.0, 1.0]
        num_space = 100
        num_time = 100
        super().__init__(self.name, space_interval, time_interval, num_space, num_time, train_proto)

        self.vals = [-1.0, -1.0]
        self.typs = ['dirichlet', 'dirichlet']
        self.mixing_fns = [None, None]
        self.pde_fn = allen_cahn_loss

    def analytical_solution(self, x, t):
        xx, tt, f = load_allen_cahn_data()
        x, t = tensor_to_numpy(x), tensor_to_numpy(t)
        sol = super().interpolate_grid_data(xx, tt, f, x, t)
        return sol

    def initial_condition(self, x):
        u0 = (x ** 2.0) * torch.cos(np.pi * x)
        return [u0]
    
    def initial_condition_loss(self, net, device):
        return super().initial_condition_loss(net, self.initial_condition, device)
    
    def boundary_condition_loss(self, net, device):
        return super().boundary_condition_loss(net, self.vals, self.typs, self.mixing_fns, device)
    
    def pde_loss(self, net, device):
        return super().pde_loss(net, self.pde_fn, device)
    
    def return_losses(self, net, device):
        bc_loss = self.boundary_condition_loss(net, device)
        ic_loss = self.initial_condition_loss(net, device)
        pde_loss = self.pde_loss(net, device)
        total_loss = bc_loss + pde_loss + ic_loss
        return [total_loss, bc_loss, ic_loss, pde_loss]
    
    def compute_ntk_eigenvalue(self, net, device):
        return super().compute_ntk_eigenvalue(net, self.pde_fn, device, self.typs, num_inits=1)
    
    def compute_analytical_solution(self, device):
        x, t = self.X_test, self.T_test
        return self.analytical_solution(x, t)
    
    def compute_PINN_solution(self, net, device):
        return super().compute_PINN_solution(net, device)
