from utils.Grids import OneDSpaceGrid
from utils.Grids import array_to_meshgrid, meshgrid_to_array, numpy_to_tensor, tensor_to_numpy, _to, _like
from utils.Losses import *

# def piecewise_smooth(x):
#     if isinstance(x, torch.Tensor):
#         x = tensor_to_numpy(x)
#         y = np.empty(x.shape)
#     for i in range(len(x)):
#         if -np.pi <= x[i] and x[i] < -np.pi/2.0:
#             y[i] = x[i] + np.pi
#         if -np.pi/2.0 <= x[i] and x[i] < np.pi/2.0:
#             y[i] = -4.0 * (x[i] ** 2)/(np.pi ** 2)
#         if np.pi/2.0 <= x[i] and x[i] <= np.pi:
#             y[i] = np.pi - x[i]
#     # if isinstance(x, torch.Tensor):
#     #     y = numpy_to_tensor(y).to(x.device)
#     return y

def piecewise_smooth_1(x):
    if isinstance(x, torch.Tensor):
        y = _like(x, typ=0, rg=False)
    if isinstance(x, np.ndarray):
        y = np.zeros_like(x)
    n = len(y)
    y[:n//2] = x[:n//2] + np.pi
    y[n//2:] = x[n//2:] - np.pi
    return y

def piecewise_smooth_2(x):
    if isinstance(x, torch.Tensor):
        y = _like(x, typ=0, rg=False)
    if isinstance(x, np.ndarray):
        y = np.zeros_like(x)
    n = len(y)//3
    y[:n] = x[:n] + np.pi
    y[n:2*n] = -4.0 * (x[n:2*n]/np.pi) ** 2 
    y[2*n:] = np.pi - x[2*n:]
    return y
        

class FunctionApproximationTasks(OneDSpaceGrid):
    def __init__(self, name, interval, num_points):
        super().__init__(interval, num_points)

        self.name = name
        self.legends = ['Total Loss']
        self.train_proto = None

    def compute_loss(self, net, ana_fn, device):
        X = _to([self.X_train], device, rg=False)[0]
        y = ana_fn(X)
        y_pred = net(X)
        loss = MSE(y_pred, y)
        return loss
    
    def compute_ntk_eigenvalue(self, net, device):
        X = _to([self.X_train], device, rg=True)[0]
        func_out = net(X)
        grad_params = torch.autograd.grad(func_out[:, -1].sum(), net.parameters(), retain_graph=True)
        grads_func = torch.cat([grad.view(-1) for grad in grad_params])
        ntk = torch.matmul(grads_func.t(), grads_func)
        eigenvalues = torch.linalg.eigvalsh(ntk).cpu().numpy()
        smallest_eigenvalue = np.min(np.abs(eigenvalues))
        return smallest_eigenvalue

    def compute_analytical_solution(self, ana_fn, device):
        X = _to([self.X_test], device, rg=False)[0]
        y = ana_fn(X)
        if isinstance(y, torch.Tensor):
            return tensor_to_numpy(y)
        else:
            return y

    def compute_PINN_solution(self, net, device):
        X = _to([self.X_test], device, rg=False)[0]
        y = net(X)
        return tensor_to_numpy(y)
    

class ExponentialSinusoid(FunctionApproximationTasks):
    def __init__(self):
        self.name = 'ExponentialSinusoid'
        interval = [0.0, 2.0 * np.pi]
        num_points = 1000
        super().__init__(self.name, interval, num_points)

    def analytical_solution(self, x):
        if isinstance(x, torch.Tensor):
            return torch.exp(-0.5 * x) * torch.sin(4.0*x + np.pi/6.0)
        if isinstance(x, np.ndarray):
            return np.exp(-0.5 * x) * np.sin(4.0*x + np.pi/6.0)
        
    def compute_loss(self, net, device):
        return super().compute_loss(net, self.analytical_solution, device)
    
    def return_losses(self, net, device):
        return [self.compute_loss(net, device)]

    def compute_ntk_eigenvalue(self, net, device):
        return super().compute_ntk_eigenvalue(net, device)
    
    def compute_analytical_solution(self, device):
        return super().compute_analytical_solution(self.analytical_solution, device)
    
    def compute_PINN_solution(self, net, device):
        return super().compute_PINN_solution(net, device)
    
class SinusoidalBeats(FunctionApproximationTasks):
    def __init__(self):
        self.name = 'ExponentialSinusoid'
        interval = [0.0, 2.0 * np.pi]
        num_points = 1000
        super().__init__(self.name, interval, num_points)

    def analytical_solution(self, x):
        if isinstance(x, torch.Tensor):
            return torch.sin(2.0*x + np.pi/3.0) * torch.sin(4.0*x + np.pi/6.0)
        if isinstance(x, np.ndarray):
            return np.sin(2.0*x + np.pi/3.0) * np.sin(4.0*x + np.pi/6.0)
        
    def compute_loss(self, net, device):
        return super().compute_loss(net, self.analytical_solution, device)
    
    def return_losses(self, net, device):
        return [self.compute_loss(net, device)]
    
    def compute_ntk_eigenvalue(self, net, device):
        return super().compute_ntk_eigenvalue(net, device)
    
    def compute_analytical_solution(self, device):
        return super().compute_analytical_solution(self.analytical_solution, device)
    
    def compute_PINN_solution(self, net, device):
        return super().compute_PINN_solution(net, device)
    
class AbsoluteExponentialSinusoid(FunctionApproximationTasks):
    def __init__(self):
        self.name = 'AbsoluteExponentialSinusoid'
        interval = [-np.pi, np.pi]
        num_points = 1000
        super().__init__(self.name, interval, num_points)

    def analytical_solution(self, x):
        if isinstance(x, torch.Tensor):
            return torch.exp(torch.absolute(x)) * torch.sin(6.0*x)
        if isinstance(x, np.ndarray):
            return np.exp(np.absolute(x)) * np.sin(6.0*x)
        
    def compute_loss(self, net, device):
        return super().compute_loss(net, self.analytical_solution, device)
    
    def return_losses(self, net, device):
        return [self.compute_loss(net, device)]
    
    def compute_ntk_eigenvalue(self, net, device):
        return super().compute_ntk_eigenvalue(net, device)
    
    def compute_analytical_solution(self, device):
        return super().compute_analytical_solution(self.analytical_solution, device)
    
    def compute_PINN_solution(self, net, device):
        return super().compute_PINN_solution(net, device)
    
class ParabolicSinusoid(FunctionApproximationTasks):
    def __init__(self):
        self.name = 'ParabolicSinusoid'
        interval = [-np.pi, np.pi]
        num_points = 1000
        super().__init__(self.name, interval, num_points)

    def analytical_solution(self, x):
        if isinstance(x, torch.Tensor):
            return torch.sin(2.0*x) * (x ** 2)
        if isinstance(x, np.ndarray):
            return np.sin(2.0*x) * (x ** 2)
        
    def compute_loss(self, net, device):
        return super().compute_loss(net, self.analytical_solution, device)
    
    def return_losses(self, net, device):
        return [self.compute_loss(net, device)]
    
    def compute_ntk_eigenvalue(self, net, device):
        return super().compute_ntk_eigenvalue(net, device)
    
    def compute_analytical_solution(self, device):
        return super().compute_analytical_solution(self.analytical_solution, device)
    
    def compute_PINN_solution(self, net, device):
        return super().compute_PINN_solution(net, device)

class PolynomialSinusoid(FunctionApproximationTasks):
    def __init__(self):
        self.name = 'PolynomialSinusoid'
        interval = [-np.pi, np.pi]
        num_points = 1000
        super().__init__(self.name, interval, num_points)

    def analytical_solution(self, x):
        if isinstance(x, torch.Tensor):
            return (x ** 3 - x) * torch.sin(7.0*x)/7.0 + torch.sin(12.0*x)
        if isinstance(x, np.ndarray):
            return (x ** 3 - x) * np.sin(7.0*x)/7.0 + np.sin(12.0*x)
        
    def compute_loss(self, net, device):
        return super().compute_loss(net, self.analytical_solution, device)
    
    def return_losses(self, net, device):
        return [self.compute_loss(net, device)]
    
    def compute_ntk_eigenvalue(self, net, device):
        return super().compute_ntk_eigenvalue(net, device)
    
    def compute_analytical_solution(self, device):
        return super().compute_analytical_solution(self.analytical_solution, device)
    
    def compute_PINN_solution(self, net, device):
        return super().compute_PINN_solution(net, device)
    
class PiecewiseSmooth1(FunctionApproximationTasks):
    def __init__(self):
        self.name = 'PiecewiseSmooth_1'
        interval = [-np.pi, np.pi]
        num_points = 1000
        super().__init__(self.name, interval, num_points)

    def analytical_solution(self, x):
        return piecewise_smooth_1(x)
        
    def compute_loss(self, net, device):
        return super().compute_loss(net, self.analytical_solution, device)
    
    def return_losses(self, net, device):
        return [self.compute_loss(net, device)]
    
    def compute_ntk_eigenvalue(self, net, device):
        return super().compute_ntk_eigenvalue(net, device)
    
    def compute_analytical_solution(self, device):
        return super().compute_analytical_solution(self.analytical_solution, device)
    
    def compute_PINN_solution(self, net, device):
        return super().compute_PINN_solution(net, device)
    
class PiecewiseSmooth2(FunctionApproximationTasks):
    def __init__(self):
        self.name = 'PiecewiseSmooth_2'
        interval = [-np.pi, np.pi]
        num_points = 1200
        super().__init__(self.name, interval, num_points)

    def analytical_solution(self, x):
        return piecewise_smooth_2(x)
        
    def compute_loss(self, net, device):
        return super().compute_loss(net, self.analytical_solution, device)
    
    def return_losses(self, net, device):
        return [self.compute_loss(net, device)]
    
    def compute_ntk_eigenvalue(self, net, device):
        return super().compute_ntk_eigenvalue(net, device)
    
    def compute_analytical_solution(self, device):
        return super().compute_analytical_solution(self.analytical_solution, device)
    
    def compute_PINN_solution(self, net, device):
        return super().compute_PINN_solution(net, device)
    
class TanhSin(FunctionApproximationTasks):
    def __init__(self):
        self.name = 'TanhSin'
        interval = [-np.pi, np.pi]
        num_points = 1000
        super().__init__(self.name, interval, num_points)

    def analytical_solution(self, x):
        if isinstance(x, torch.Tensor):
            return torch.tanh(2.0 * x) * torch.sin(5.0 * x)
        if isinstance(x, np.ndarray):
            return np.tanh(2.0 * x) * np.sin(5.0 * x)
        
    def compute_loss(self, net, device):
        return super().compute_loss(net, self.analytical_solution, device)
    
    def return_losses(self, net, device):
        return [self.compute_loss(net, device)]
    
    def compute_ntk_eigenvalue(self, net, device):
        return super().compute_ntk_eigenvalue(net, device)
    
    def compute_analytical_solution(self, device):
        return super().compute_analytical_solution(self.analytical_solution, device)
    
    def compute_PINN_solution(self, net, device):
        return super().compute_PINN_solution(net, device)
