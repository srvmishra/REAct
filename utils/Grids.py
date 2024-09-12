import numpy as np
import torch
from utils.metrics import *
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def error(ground_truth, estimate, typ='rel'):
    if typ == 'rel':
        err = l2_relative_error(ground_truth, estimate)
    if typ == 'abs':
        err = mean_absolute_error(ground_truth, estimate)
    return err

def pc_error(ground_truth, estimate):
    return 100.0 * np.absolute(ground_truth - estimate)/np.absolute(ground_truth)

# Space, Time, SpaceTime
# There is only one trianing and testing protocol for space
# But two for time
fmt = lambda x: "{:.2f}".format(x)

def _to(tensors, device, rg=True):
    return [t.to(device).requires_grad_(rg) for t in tensors]

def _like(tensor_a, typ=0, rg=False):
    if typ == 1:
        return torch.ones_like(tensor_a).float().requires_grad_(rg)
    if typ == 0:
        return torch.zeros_like(tensor_a).float().requires_grad_(rg)
    if typ not in [0, 1]:
        return typ * torch.ones_like(tensor_a).float().requires_grad_(rg)

numpy_to_tensor = lambda x: torch.from_numpy(x.flatten()).float().view(-1, 1)
tensor_to_numpy = lambda x: x.detach().cpu().numpy()

# meshgrid -> array
def array_to_meshgrid(arr, shape):
    return arr.reshape(shape)

# array -> meshgrid
def meshgrid_to_array(arr):
    return arr.flatten()

## function to compute ntk of a neural network
# def compute_ntk(model, inputs):
#     # Initialize NTK
#     ntk = None

#     # Get the output of the network
#     outputs = model(inputs)
#     num_outputs = outputs.shape[-1]

#     # Compute the gradients of the output with respect to each parameter
#     for i in range(num_outputs):
#         grad_params = torch.autograd.grad(outputs[:, i].sum(), model.parameters(), retain_graph=True)
#         grads = torch.cat([grad.view(-1) for grad in grad_params])

#         # Calculate the NTK as the outer product of the gradients
#         grads = grads.unsqueeze(0)
#         if ntk is None:
#             ntk = torch.matmul(grads.t(), grads)
#         else:
#             ntk += torch.matmul(grads.t(), grads)

#     return ntk


## We assume that space is of the form [a, b]^n where n is the dimensions and time is of the form [0, T]

class OneDSpaceGrid(object):
    def __init__(self, interval, num_points):
        self.left = interval[0]
        self.right = interval[1]
        self.train_pts = num_points
        self.test_pts = self.train_pts * 10
        self.create_grid()

    def create_grid(self):
        X_train = np.linspace(self.left, self.right, self.train_pts)
        self.X_train = numpy_to_tensor(X_train)
        X_test = np.linspace(self.left, self.right, self.test_pts)
        self.X_test = numpy_to_tensor(X_test)
        self.left_bdr = torch.tensor(self.left).float().view((1, 1))
        self.right_bdr = torch.tensor(self.right).float().view((1, 1))

    def make_sim_and_error_plots(self, name, ana_sol, results_, save_path):
        X_test = np.linspace(self.left, self.right, self.test_pts)
        # print(t.shape)
        # print(sol.shape)
        # print(sol)
        errors = np.absolute(ana_sol - results_)
        # print(results_.shape, errors.shape)
        plt.tight_layout()

        x_unit = 0.25 * (self.right - self.left)
        xticks = [fmt(self.left + v * x_unit) for v in range(5)]

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].plot(X_test, errors, lw=2, label='Absolute Errors')
        axs[0].set_xticks([self.left + v * x_unit for v in range(5)])
        axs[0].set_xticklabels(labels=xticks, fontsize=12)
        # axs[0].set_yticklabels(fontsize=12)
        axs[0].xaxis.set_tick_params(labelsize=12)
        axs[0].yaxis.set_tick_params(labelsize=12)
        axs[0].set_ylabel('errors', fontsize=15)
        axs[0].set_xlabel('x', fontsize=15)
        axs[0].legend(fontsize=15)

        axs[1].plot(X_test, ana_sol, 'g-', lw=2, label='Analytical Solution')
        axs[1].plot(X_test, results_, 'b--', lw=2, label='PINN Solution')
        axs[1].set_xticks([self.left + v * x_unit for v in range(5)])
        axs[1].set_xticklabels(labels=xticks, fontsize=12)
        # axs[1].set_yticklabels(fontsize=12)
        axs[1].xaxis.set_tick_params(labelsize=12)
        axs[1].yaxis.set_tick_params(labelsize=12)
        axs[1].set_ylabel('solutions', fontsize=15)
        axs[1].set_xlabel('x', fontsize=15)
        axs[1].legend(fontsize=15)

        #   figsavepath = save_path + problem.name + '.png'
        plt.savefig(save_path + name + '_sim_and_error.png')
        plt.show()

        rel_err = error(ana_sol, results_, typ='rel')
        abs_err = error(ana_sol, results_, typ='abs')
        print(name + " Relative Error: {:.4f}".format(rel_err))
        print(name + " Absolute Error: {:.4f}".format(abs_err))
       
class OneDTimeGrid(object):
    def __init__(self, interval, num_points):
        self.start = interval[0]
        self.stop = interval[1]
        self.train_pts = num_points
        
    def create_grid_1(self):
        self.test_pts = self.train_pts * 10
        T_train = np.linspace(self.start, self.stop, self.train_pts)
        self.T_train = numpy_to_tensor(T_train)
        T_test = np.linspace(self.start, self.stop, self.test_pts)
        self.T_test = numpy_to_tensor(T_test)
        self.T_init = torch.tensor(self.start).float().view((1, 1))

    def create_grid_2(self):
        num_pts = int(1.25 * self.train_pts) + 1
        timestamps = np.linspace(self.start, self.stop, num_pts)
        T_train = timestamps[:self.train_pts]
        self.T_train = numpy_to_tensor(T_train)
        T_test = timestamps[self.train_pts:]
        self.T_test = numpy_to_tensor(T_test)
        self.T_init = torch.tensor(self.start).float().view((1, 1))

    def make_sim_and_error_plots(self, name, ana_sol, results_, save_path, train_proto):       
        if train_proto == 1:
            T_test = np.linspace(self.start, self.stop, self.train_pts * 10)
            t_unit = 0.25 * self.stop
            y_ticks_ = [v * t_unit for v in range(5)]
            yticks = [fmt(v * t_unit) for v in range(5)]
        if train_proto == 2:
            num_pts = int(1.25 * self.train_pts) + 1
            timestamps = np.linspace(self.start, self.stop, num_pts)
            T_test = timestamps[self.train_pts:]
            t_unit = 0.25 * (T_test[-1] - T_test[0])
            y_ticks_ = [T_test[0] + v * t_unit for v in range(5)]
            yticks = [fmt(T_test[0] + v * t_unit) for v in range(5)]

        errors = np.absolute(ana_sol - results_)
        plt.tight_layout()

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].plot(T_test, errors, lw=2, label='Absolute Errors')
        axs[0].set_xticks(y_ticks_)
        axs[0].set_xticklabels(labels=yticks, fontsize=12)
        # axs[0].set_yticklabels(fontsize=12)
        axs[0].xaxis.set_tick_params(labelsize=12)
        axs[0].yaxis.set_tick_params(labelsize=12)
        axs[0].set_ylabel('errors', fontsize=15)
        axs[0].set_xlabel('t', fontsize=15)
        axs[0].legend(fontsize=15)

        axs[1].plot(T_test, ana_sol, 'g-', lw=2, label='Analytical Solution')
        axs[1].plot(T_test, results_, 'b--', lw=2, label='PINN Solution')
        axs[1].set_xticks(y_ticks_)
        axs[1].set_xticklabels(labels=yticks, fontsize=12)
        # axs[1].set_yticklabels(fontsize=12)
        axs[1].xaxis.set_tick_params(labelsize=12)
        axs[1].yaxis.set_tick_params(labelsize=12)
        axs[1].set_ylabel('solutions', fontsize=15)
        axs[1].set_xlabel('t', fontsize=15)
        axs[1].legend(fontsize=15)

        #   figsavepath = save_path + problem.name + '.png'
        plt.savefig(save_path + name + '_train_proto_' + str(train_proto) + '_sim_and_error.png')
        plt.show()

        rel_err = error(ana_sol, results_, typ='rel')
        abs_err = error(ana_sol, results_, typ='abs')
        print(name + " train_proto_{} ".format(train_proto) + "Relative Error: {:.4f}".format(rel_err))
        print(name + " train_proto_{} ".format(train_proto) + "Absolute Error: {:.4f}".format(abs_err))

class OneDSpaceTimeGrid(object):
    def __init__(self, space_interval, time_interval, num_space, num_time):
        self.left = space_interval[0]
        self.right = space_interval[1]
        self.start = time_interval[0]
        self.stop = time_interval[1]
        self.space_pts = num_space
        self.time_pts = num_time

    def create_grid_1(self):
        # train
        space_points = np.linspace(self.left, self.right, self.space_pts)
        time_points = np.linspace(self.start, self.stop, self.time_pts)
        Xgrid, Tgrid = np.meshgrid(space_points, time_points)
        self.X_train, self.T_train = numpy_to_tensor(Xgrid), numpy_to_tensor(Tgrid)

        # initial
        self.X_init = numpy_to_tensor(space_points)
        self.T_init = _like(self.X_init)

        # boundary
        self.T_bdr = numpy_to_tensor(time_points)
        self.X_left_bdr = self.left * _like(self.T_bdr, typ=1)
        self.X_right_bdr = self.right * _like(self.T_bdr, typ=1)

        # test
        space_points = np.linspace(self.left, self.right, self.space_pts * 10)
        time_points = np.linspace(self.start, self.stop, self.time_pts * 10)
        Xgrid, Tgrid = np.meshgrid(space_points, time_points)
        self.test_shape = Xgrid.shape
        self.X_test, self.T_test = numpy_to_tensor(Xgrid), numpy_to_tensor(Tgrid)

    def create_grid_2(self):
        # train
        space_points = np.linspace(self.left, self.right, self.space_pts)
        time_points = np.linspace(self.start, 0.8 * self.stop, int(0.8 * self.time_pts))
        Xgrid, Tgrid = np.meshgrid(space_points, time_points)
        self.X_train, self.T_train = numpy_to_tensor(Xgrid), numpy_to_tensor(Tgrid)

        # initial
        self.X_init = numpy_to_tensor(space_points)
        self.T_init = _like(self.X_init)

        # boundary
        self.T_bdr = numpy_to_tensor(time_points)
        self.X_left_bdr = self.left * _like(self.T_bdr, typ=1)
        self.X_right_bdr = self.right * _like(self.T_bdr, typ=1)

        # test
        space_points = np.linspace(self.left, self.right, self.space_pts)
        time_points = np.linspace(0.8 * self.stop, self.stop, int(0.2 * self.time_pts))
        Xgrid, Tgrid = np.meshgrid(space_points, time_points)
        self.test_shape = Xgrid.shape
        self.X_test, self.T_test = numpy_to_tensor(Xgrid), numpy_to_tensor(Tgrid)

    def make_sim_and_error_plots(self, name, ana_sol, results_, save_path, train_proto):
        if train_proto == 1:
            time_points = np.linspace(self.start, self.stop, self.time_pts * 10)
        if train_proto == 2:
            time_points = np.linspace(0.8 * self.stop, self.stop, int(0.2 * self.time_pts))

        x_unit = 0.25 * (self.right - self.left)    
        t_unit = 0.25 * (time_points[-1] - time_points[0])
        xticks_ = [self.left + v * x_unit for v in range(5)]
        xticks = [fmt(self.left + v * x_unit) for v in range(5)]
        tticks_ = [time_points[0] + v * t_unit for v in range(5)]
        tticks = [fmt(time_points[0] + v * t_unit) for v in range(5)]

        ana_sol = array_to_meshgrid(ana_sol, self.test_shape)
        results_ = array_to_meshgrid(results_, self.test_shape)
        errors = np.absolute(ana_sol - results_)
        plt.tight_layout()

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        im = ax.imshow(errors.T, cmap='gray')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=1.0)
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=15, rotation=90) 
        cbar.set_label('Absolute Error', fontsize=15)
        ax.set_yticks(xticks_)
        ax.set_xticks(tticks_)
        ax.set_yticklabels(labels=xticks, fontsize=15)
        ax.set_xticklabels(labels=tticks, fontsize=15, rotation=90)
        ax.set_ylabel('x', fontsize=15)
        ax.set_xlabel('t', fontsize=15)

        save_path_err = save_path + name + '_train_proto_' + str(train_proto) + '_errors.png'
        plt.savefig(save_path_err)

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        im = ax.imshow(ana_sol.T, cmap='spring')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=1.0)
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=15, rotation=90) 
        cbar.set_label('Exact Solution', fontsize=15)
        ax.set_yticks(xticks_)
        ax.set_xticks(tticks_)
        ax.set_yticklabels(labels=xticks, fontsize=15)
        ax.set_xticklabels(labels=tticks, fontsize=15, rotation=90)
        ax.set_ylabel('x', fontsize=15)
        ax.set_xlabel('t', fontsize=15)

        save_path_err = save_path + name + '_train_proto_' + str(train_proto) + '_anasol.png'
        plt.savefig(save_path_err)

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        im = ax.imshow(results_.T, cmap='spring')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=1.0)
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=15, rotation=90) 
        cbar.set_label('PINN Solution', fontsize=15)
        ax.set_yticks(xticks_)
        ax.set_xticks(tticks_)
        ax.set_yticklabels(labels=xticks, fontsize=15)
        ax.set_xticklabels(labels=tticks, fontsize=15, rotation=90)
        ax.set_ylabel('x', fontsize=15)
        ax.set_xlabel('t', fontsize=15)

        save_path_err = save_path + name + '_train_proto_' + str(train_proto) + '_pinnsol.png'
        plt.savefig(save_path_err)
        plt.show()

        rel_err = error(ana_sol, results_, typ='rel')
        abs_err = error(ana_sol, results_, typ='abs')
        print(name + " train_proto_{} ".format(train_proto) + "Relative Error: {:.4f}".format(rel_err))
        print(name + " train_proto_{} ".format(train_proto) + "Absolute Error: {:.4f}".format(abs_err))

    def make_three_time_instant_plots(self, name, ana_sol, results_, save_path, train_proto):
        if train_proto == 1:
            time_points = np.linspace(self.start, self.stop, self.time_pts * 10)
            time_indices_ = [int(0.6*len(time_points)), int(0.75*len(time_points)), int(0.9*len(time_points))]
            time_stamps = [time_points[i] for i in time_indices_]
        if train_proto == 2:
            time_points = np.linspace(0.8 * self.stop, self.stop, int(0.2 * self.time_pts))
            time_indices_ = [int(0.1*len(time_points)), int(0.5*len(time_points)), int(0.9*len(time_points))]
            time_stamps = [time_points[i] for i in time_indices_]

        ana_sol = array_to_meshgrid(ana_sol, self.test_shape)
        results_ = array_to_meshgrid(results_, self.test_shape)

        x_unit = 0.25 * (self.right - self.left) 
        xticks_ = [self.left + v * x_unit for v in range(5)]
        xticks = [fmt(self.left + v * x_unit) for v in range(5)]

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        plt.tight_layout()
        for i in range(3):
            ax = axs[i]
            act_sol = ana_sol[:, i]
            pinn_sol = results_[:, i]

            ax.plot(act_sol, 'g--', lw=2, label='analytical')
            ax.plot(pinn_sol, 'b--', lw=2, label='PINN')
            ax.set_xticks(xticks_)
            ax.set_xticklabels(labels=xticks, fontsize=12, rotation=90)
            ax.set_ylabel('x', fontsize=12)
            ax.set_xlabel('t', fontsize=12)
            ax.legend(fontsize=12)
            ax.set_title('t = {:.4f}s'.format(time_stamps[i]), fontsize=15)

        save_path = save_path + name  + '_train_proto_' + str(train_proto) + '_three_instants.pngs'
        plt.savefig(save_path)
        plt.show()
        return

class TwoDSpaceGrid(object):
    def __init__(self, space_interval, space_points):
        self.down = space_interval[0]
        self.up = space_interval[1]
        self.left = space_interval[2]
        self.right = space_interval[3]
        self.pts = space_points
        self.create_grid()

    def create_grid(self):
        # train
        horz_points = np.linspace(self.left, self.right, self.pts)
        vert_points = np.linspace(self.down, self.up, self.pts)
        Xgrid, Ygrid = np.meshgrid(horz_points, vert_points)
        self.X_train, self.Y_train = numpy_to_tensor(Xgrid), numpy_to_tensor(Ygrid)

        # top and bottom boundary
        self.X_top_bdr = numpy_to_tensor(horz_points)
        self.Y_top_bdr = self.up * _like(self.X_top_bdr, typ=1)
        self.X_bottom_bdr = numpy_to_tensor(horz_points)
        self.Y_bottom_bdr = self.down * _like(self.X_bottom_bdr, typ=1)

        # left and right boundary
        self.Y_left_bdr = numpy_to_tensor(vert_points)
        self.X_left_bdr = self.left * _like(self.Y_left_bdr, typ=1)
        self.Y_right_bdr = numpy_to_tensor(vert_points)
        self.X_right_bdr = self.right * _like(self.Y_right_bdr, typ=1)

        # test
        horz_points = np.linspace(self.left, self.right, self.pts * 10)
        vert_points = np.linspace(self.down, self.up, self.pts * 10)
        Xgrid, Ygrid = np.meshgrid(horz_points, vert_points)
        self.X_test, self.Y_test = numpy_to_tensor(Xgrid), numpy_to_tensor(Ygrid)

    def make_sim_and_error_plots(self, name, ana_sol, results_, save_path):
        x_unit = 0.25 * (self.right - self.left)
        y_unit = 0.25 * (self.up - self.down)

        xticks = [fmt(self.left + v * x_unit) for v in range(5)]
        yticks = [fmt(self.down + v * y_unit) for v in range(5)]

        ana_sol = array_to_meshgrid(ana_sol, (self.pts, self.pts))
        results_ = array_to_meshgrid(results_, (self.pts, self.pts))
        errors = np.absolute(ana_sol - results_)
        plt.tight_layout()

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        im = ax.imshow(errors.T, cmap='gray')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=1.0)
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=15, rotation=90) 
        cbar.set_label('Absolute Error', fontsize=15)
        ax.set_yticks([self.left + v * x_unit for v in range(5)])
        ax.set_xticks([self.down + v * y_unit for v in range(5)])
        ax.set_yticklabels(labels=xticks, fontsize=15)
        ax.set_xticklabels(labels=yticks, fontsize=15, rotation=90)
        ax.set_ylabel('x', fontsize=15)
        ax.set_xlabel('y', fontsize=15)

        save_path_err = save_path + name + '_errors.png'
        plt.savefig(save_path_err)

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        im = ax.imshow(ana_sol.T, cmap='spring')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=1.0)
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=15, rotation=90) 
        cbar.set_label('Exact Solution', fontsize=15)
        ax.set_yticks([self.left + v * x_unit for v in range(5)])
        ax.set_xticks([self.down + v * y_unit for v in range(5)])
        ax.set_yticklabels(labels=xticks, fontsize=15)
        ax.set_xticklabels(labels=yticks, fontsize=15, rotation=90)
        ax.set_ylabel('x', fontsize=15)
        ax.set_xlabel('y', fontsize=15)

        save_path_err = save_path + name + '_anasol.png'
        plt.savefig(save_path_err)

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        im = ax.imshow(results_.T, cmap='spring')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=1.0)
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=15, rotation=90) 
        cbar.set_label('PINN Solution', fontsize=15)
        ax.set_yticks([self.left + v * x_unit for v in range(5)])
        ax.set_xticks([self.down + v * y_unit for v in range(5)])
        ax.set_yticklabels(labels=xticks, fontsize=15)
        ax.set_xticklabels(labels=yticks, fontsize=15, rotation=90)
        ax.set_ylabel('x', fontsize=15)
        ax.set_xlabel('y', fontsize=15)

        save_path_err = save_path + name + '_pinnsol.png'
        plt.savefig(save_path_err)
        plt.show()

        rel_err = error(ana_sol, results_, typ='rel')
        abs_err = error(ana_sol, results_, typ='abs')
        print(name + " Relative Error: {:.4f}".format(rel_err))
        print(name + " Absolute Error: {:.4f}".format(abs_err))