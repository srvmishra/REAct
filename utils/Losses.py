import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.autograd import Variable, grad
import numpy as np

MSE = torch.nn.MSELoss(reduction='mean')

def compute_single_derivative(f, var, order):
  df = f
  for n in range(order):
    df = grad(df, var, grad_outputs=torch.ones_like(df), create_graph=True)[0]
  return df

def compute_mixed_derivative(f, vars, orders):
  df = f
  for var, n in zip(vars, orders):
    df = compute_single_derivative(df, var, n)
  return df

def compute_BC_loss(fs, vars, values, typs, mixing_fns):
  '''
  vars - specifies the boundary locations
  values - the rhs of boundary conditions at corresponding locations above
  fs - NN predictions at corresponding boundaries
  mixing_fn is there only for robin boundary conditions, so if type=='robin', mixing_fn must be present
  IMPLEMENT mixing_fn - supports only upto first order derivative
  '''
  loss = 0
  for f, var, value, typ, mixing_fn in zip(fs, vars, values, typs, mixing_fns):
    if typ == 'dirichlet':
      lhs = f
    if typ == 'neumann':
      lhs = compute_single_derivative(f, var, 1)
    if typ == 'neumann_2':
      lhs = compute_single_derivative(f, var, 2)
    if typ == 'neumann_3':
      lhs = compute_single_derivative(f, var, 3)
    if typ == 'robin' and mixing_fn is not None:
      df = compute_single_derivative(f, var, 1)
      lhs = mixing_fn(f, df)
    loss = loss + MSE(lhs, value)
  return loss

def compute_IC_loss(f, var, values):
  '''
  values specified as [u0, du0] as we have only upto two orders
  the code is generalizable for any order of derivatives
  '''
  loss = 0
  for order, val in enumerate(values):
    lhs = compute_single_derivative(f, var, order)
    loss = loss + MSE(lhs, val)
  return loss


### record the parameters and the forcing functions and put them in the equations ###
def euler_beam_loss(f, x, forcing_term=None, res=False):
  ux4 = compute_single_derivative(f, x, 4)
  ode_lhs = ux4 + 1.0

  if forcing_term is not None:
    # external_force = forcing_fn(time, spatial_vars)
    ode_rhs = forcing_term.requires_grad_(False)
  else:
    ode_rhs = torch.zeros_like(ode_lhs).requires_grad_(False)

  loss = MSE(ode_lhs, ode_rhs)
  if res:
    return ode_lhs
  else:
    return loss

def allen_cahn_loss(f, x, t, forcing_term=None, res=False):
  ut = compute_single_derivative(f, t, 1)
  uxx = compute_single_derivative(f, x, 2)
  func = 5.0*(f - f**3)
  pde_lhs = ut - 0.001 * uxx - func

  if forcing_term is not None:
    # external_force = forcing_fn(time, spatial_vars)
    pde_rhs = forcing_term.requires_grad_(False)
  else:
    pde_rhs = torch.zeros_like(pde_lhs).requires_grad_(False)

  loss = MSE(pde_lhs, pde_rhs)
  if res:
    return pde_lhs
  else:
    return loss

def burgers_loss(f, x, t, forcing_term=None, res=False):
  ut = compute_single_derivative(f, t, 1)
  ux = compute_single_derivative(f, x, 1)
  uxx = compute_single_derivative(f, x, 2)
  pde_lhs = ut + f * ux - (0.01/np.pi) * uxx

  if forcing_term is not None:
    # external_force = forcing_fn(time, spatial_vars)
    pde_rhs = forcing_term.requires_grad_(False)
  else:
    pde_rhs = torch.zeros_like(pde_lhs).requires_grad_(False)

  loss = MSE(pde_lhs, pde_rhs)
  if res:
    return pde_lhs
  else:
    return loss

def heat_loss(f, x, t, forcing_term=None, res=False):
  ut = compute_single_derivative(f, t, 1)
  uxx = compute_single_derivative(f, x, 2)
  pde_lhs = ut - 0.4 * uxx

  if forcing_term is not None:
    # external_force = forcing_fn(time, spatial_vars)
    pde_rhs = forcing_term.requires_grad_(False)
  else:
    pde_rhs = torch.zeros_like(pde_lhs).requires_grad_(False)

  loss = MSE(pde_lhs, pde_rhs)
  if res:
    return pde_lhs
  else:
    return loss

def diffusion_rxn_loss(f, x, t, forcing_term=None, res=False):
  ut = compute_single_derivative(f, t, 1)
  uxx = compute_single_derivative(f, x, 2)
  pde_lhs = ut - 1.0 * uxx

  if forcing_term is not None:
    # external_force = forcing_fn(time, spatial_vars)
    pde_rhs = forcing_term.requires_grad_(False)
  else:
    pde_rhs = torch.zeros_like(pde_lhs).requires_grad_(False)

  loss = MSE(pde_lhs, pde_rhs)
  if res:
    return pde_lhs
  else:
    return loss

def helmholtz_loss(f, x, y, forcing_term=None, res=False):
  ## take negative in the forcing term in the equation part
  uxx = compute_single_derivative(f, x, 2)
  uyy = compute_single_derivative(f, y, 2)
  ksq = (4.0 * np.pi) ** 2
  pde_lhs = uxx + uyy + ksq * f

  if forcing_term is not None:
    # external_force = forcing_fn(time, spatial_vars)
    pde_rhs = forcing_term.requires_grad_(False)
  else:
    pde_rhs = torch.zeros_like(pde_lhs).requires_grad_(False)

  loss = MSE(pde_lhs, pde_rhs)
  if res:
    return pde_lhs
  else:
    return loss

def underdamped_vibration(f, t, forcing_term=None, res=False):
  ## define c and k and a proper solution to the equation in the equation part
  ut = compute_single_derivative(f, t, 1)
  utt = compute_single_derivative(f, t, 2)
  omega_n = 3.0
  zeta = 0.5
  c = 2.0 * zeta * omega_n
  k = omega_n ** 2
  ode_lhs = utt + c * ut + k * f

  if forcing_term is not None:
    # external_force = forcing_fn(time, spatial_vars)
    ode_rhs = forcing_term.requires_grad_(False)
  else:
    ode_rhs = torch.zeros_like(ode_lhs).requires_grad_(False)

  loss = MSE(ode_lhs, ode_rhs)
  if res:
    return ode_lhs
  else:
    return loss

def wave_1d(f, x, t, forcing_term=None, res=False):
  utt = compute_single_derivative(f, t, 2)
  uxx = compute_single_derivative(f, x, 2)
  pde_lhs = utt - 4.0 * uxx

  if forcing_term is not None:
    # external_force = forcing_fn(time, spatial_vars)
    pde_rhs = forcing_term.requires_grad_(False)
  else:
    pde_rhs = torch.zeros_like(pde_lhs).requires_grad_(False)

  loss = MSE(pde_lhs, pde_rhs)
  if res:
    return pde_lhs
  else:
    return loss

def poisson1d(f, x, forcing_term=None, res=False):
  uxx = compute_single_derivative(f, x, 2)
  ode_lhs = uxx

  if forcing_term is not None:
    # external_force = forcing_fn(time, spatial_vars)
    ode_rhs = forcing_term.requires_grad_(False)
  else:
    ode_rhs = torch.zeros_like(ode_lhs).requires_grad_(False)

  loss = MSE(ode_lhs, ode_rhs)
  if res:
    return ode_lhs
  else:
    return loss