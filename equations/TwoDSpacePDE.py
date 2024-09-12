from utils.Grids import TwoDSpaceGrid
from utils.Grids import array_to_meshgrid, meshgrid_to_array, numpy_to_tensor, tensor_to_numpy, _to, _like
from utils.Losses import *

class TwoDSpaceEquations(TwoDSpaceGrid):
    def __init__(self, name, space_interval, space_points):
        super().__init__(space_interval, space_points)

        self.boundary_loss_factor = 1.0 # 0.05
        self.pde_loss_factor = 1.0 # 10

        self.name = name
        self.legends = ['Total Loss', 'BC Loss', 'PDE Loss']
        self.train_proto = None

    def boundary_condition_loss(self, net, vals, typs, mixing_fns, device):  
        '''
        vals, typs, mixing_fns in the format - left, right, top, bottom
        '''
        xleft, xright, xtop, xbottom = _to([self.X_left_bdr, self.X_right_bdr,
                                            self.X_top_bdr, self.X_bottom_bdr], device)
        yleft, yright, ytop, ybottom = _to([self.Y_left_bdr, self.Y_right_bdr,
                                            self.Y_top_bdr, self.Y_bottom_bdr], device)
        out_left = net(xleft, yleft)
        out_right = net(xright, yright)
        out_top = net(xtop, ytop)
        out_bottom = net(xbottom, ybottom)
        Vals = [_like(out_left, typ=vals[0]), _like(out_right, typ=vals[1]),
                _like(out_top, typ=vals[2]), _like(out_bottom, typ=vals[3])]
        loss = self.boundary_loss_factor * compute_BC_loss([out_left, out_right, out_top, out_bottom], 
                                                           [xleft, xright, ytop, ybottom], 
                                                           Vals, typs, mixing_fns)
        return loss
    
    def pde_loss(self, net, pde_fn, device, forcing_fn=None):
        if forcing_fn is not None:
            X, Y = _to([self.X_train, self.Y_train], device, rg=False)
            force = forcing_fn(X, Y)
        else:
            force = None
        X, Y = _to([self.X_train, self.Y_train], device, rg=True)
        f = net(X, Y)
        return pde_fn(f, X, Y, forcing_term=force)
    
    def compute_analytical_solution(self, ana_fn, device):
        X, Y = _to([self.X_test, self.Y_test], device, rg=False)
        ana_sol = ana_fn(X, Y)
        return tensor_to_numpy(ana_sol)
  
    def compute_PINN_solution(self, net, device):
        xeval, yeval = _to([self.X_test, self.Y_test], device, rg=False)  
        pinn_sol = net(xeval, yeval)
        return tensor_to_numpy(pinn_sol)
    
class HelmholtzEquation(TwoDSpaceEquations):
    def __init__(self):
        self.name = 'Helmholtz'
        space_interval = [0.0, 1.0, 0.0, 1.0] # down, up, left, right
        space_points = 250
        super().__init__(self.name, space_interval, space_points)

    def analytical_solution(self, x, y):
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            return torch.sin(4.0 * np.pi * x) * torch.sin(4.0 * np.pi * y)
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            return np.sin(4.0 * np.pi * x) * np.sin(4.0 * np.pi * y)
        
    def forcing_fn(self, x, y):
        return (16.0 * np.pi ** 2) * self.analytical_solution(x, y)
    
    def boundary_condition_loss(self, net, device):
        vals = [0.0, 0.0, 0.0, 0.0]
        typs = ['dirichlet', 'dirichlet', 'dirichlet', 'dirichlet']
        mixing_fns = [None, None, None, None]
        return super().boundary_condition_loss(net, vals, typs, mixing_fns, device)
    
    def pde_loss(self, net, device):
        pde_fn = helmholtz_loss
        forcing_fn = self.forcing_fn
        return super().pde_loss(net, pde_fn, device, forcing_fn=forcing_fn)
    
    def return_losses(self, net, device):
        bc_loss = self.boundary_condition_loss(net, device)
        pde_loss = self.pde_loss(net, device)
        total_loss = bc_loss + pde_loss
        return [total_loss, bc_loss, pde_loss]
    
    def compute_analytical_solution(self, device):
        return super().compute_analytical_solution(self.analytical_solution, device)
    
    def compute_PINN_solution(self, net, device):
        return super().compute_PINN_solution(net, device)
    