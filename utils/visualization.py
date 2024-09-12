import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.metrics import l2_relative_error, mean_absolute_error
from utils.Grids import fmt
from equations.OneDSpaceTimePDE import *
from equations.ODEProblems import *
from equations.TwoDSpacePDE import *
from equations.FuncApprox import *
import pickle

## save file
def save_pkl_file(dict_to_save, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(dict_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    return

## load file
def load_pkl_file(load_path):
    with open(load_path, 'rb') as f:
        res = pickle.load(f)
    f.close()
    return res

## Need to make changes to this file: the problem parts after coding the individual equations
def error(ground_truth, estimate, typ='rel'):
    if typ == 'rel':
        err = l2_relative_error(ground_truth, estimate)
    if typ == 'abs':
        err = mean_absolute_error(ground_truth, estimate)
    return err

'''
All save_paths below are directories only
'''

## compare loss functions for one equation across activations in same plot:
'''
Read loss dict file and plot total losses with legends
Make legends as an optional argument so that only relevant activations can be used
'''

'''
Put all files combined - read all of the files and combine the loss dicts - no need to combine now
Then plot them. Use semilogx
'''
def compare_losses(problem_name, best_act_name, loss_file_path, legends, save_path):
    # move this section to the corresponding function itself.
    losses = load_pkl_file(loss_file_path)
    best_act_losses = losses[best_act_name]
    make_loss_plots(problem_name, best_act_name, best_act_losses, losses['legends'], save_path)

    # read all files and combine dicts

    # plot everything together
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.tight_layout()
    for l in legends:
        losses_ = losses[l]
        epochs = len(losses_)
        total_losses = [l[0] for l in losses_]
        ax.plot(range(epochs), total_losses, label=l)
    unit = epochs//10
    xticks = [i*unit for i in range(11)]
    labels = [fmt(x) for x in xticks]
    ax.legend(fontsize=12)
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=12)
    ax.set_xlabel('Epochs', fontsize=15)
    ax.set_ylabel('Losses', fontsize=15)
    plt.savefig(save_path + problem_name + '_loss_compare.png')
    plt.show()
    return

## loss plots for one equation and one activation
def make_loss_plots(name, act_name, losses, legends, save_path):
    '''
    total loss
    boundary condition loss
    initial condition loss
    PDE loss
    in that sequence
    '''
    # legends = ['Total Loss', 'BC Loss', 'IC Loss', 'PDE Loss']
    plt.figure(figsize=(6, 6))
    plt.tight_layout()
    for loss, lgd in zip(losses, legends):
        plt.plot(range(len(loss)), loss, lw=2, label=lgd)
    plt.xticks(ticks=range(0, len(loss)+1, len(loss)//10), labels=[v*len(loss)//10 for v in range(11)], fontsize=12, rotation=90)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=15)
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Losses', fontsize=15)
    plt.savefig(save_path + name + '_' + act_name + '_losses.png')
    plt.show()
    return

def make_ntk_spectra_plot(problem_name, ntk_spectra_file_path, legends, save_path):
    ntk_spectra = load_pkl_file(ntk_spectra_file_path)

    # plot everything together
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.tight_layout()
    for l in legends:
        eigen_vals = ntk_spectra[l]
        epochs = len(eigen_vals)
        ax.semilogx(range(epochs), eigen_vals, label=l)
    # unit = epochs//10
    # xticks = [i*unit for i in range(11)]
    # labels = [fmt(x) for x in xticks]
    ax.legend(fontsize=12)
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(labels, rotation=90, fontsize=12)
    ax.set_xlabel('Epochs', fontsize=15)
    ax.set_ylabel('Eigenvalue', fontsize=15)
    plt.savefig(save_path + problem_name + '_ntk_spectra_compare.png')
    plt.show()
    return

## visualization of solutions for one equation and one activation
'''
Inputs:
    problem_object -> problem.name, problem.train_proto
    best_activation_name -> add activation name to problem.name so that it reflects in the filename
    plot save path: save_path
    saved results file path -> read the file in the function and extract: ana_sol
                            -> also extract the results of the best activation: results_
                            -> call problem.plot_whatever based on the class of the problem one by one
'''
def visualize_problem(problem, best_activation_name, results_path, plot_save_path):
    res = load_pkl_file(results_path)
    ana_sol = res['analytical']
    results_ = res[best_activation_name]
    if isinstance(problem, OneDSpaceTimeGrid) or isinstance(problem, OneDTimeGrid):
        problem.make_sim_and_error_plots(problem.name, ana_sol, results_, plot_save_path, problem.train_proto)
        problem.make_three_time_instant_plots(problem.name, ana_sol, results_, plot_save_path, problem.train_proto)
    if isinstance(problem, TwoDSpaceGrid) or isinstance(problem, OneDSpaceGrid):
        problem.make_sim_and_error_plots(problem.name, ana_sol, results_, plot_save_path)
    return
# inputs - problem_object -> name, saved results -> ana_sol, results_, save_path, train_proto



# def make_ode_loss_plots(losses, save_path):
#     '''
#     ODE loss only
#     in that sequence
#     '''
#     legends = ['Total Loss', typ + ' Loss', 'ODE Loss']
#     assert len(legends) == len(losses)
#     plt.figure(figsize=(6, 6))
#     plt.tight_layout()
#     for loss, lgd in zip(losses, legends):
#         plt.plot(range(len(loss)), loss, lw=2, label=lgd)
#     plt.xticks(ticks=range(0, len(loss)+1, len(loss)//10), labels=[v*len(loss)//10 for v in range(11)], fontsize=12, rotation=90)
#     plt.yticks(fontsize=12)
#     plt.legend(fontsize=15)
#     plt.xlabel('Epochs', fontsize=15)
#     plt.ylabel('Losses', fontsize=15)
#     plt.savefig(save_path)
#     plt.show()
#     return

# def make_pde_sim_and_error_plots(problem, results_, save_path):
#     '''
#     error field in x, t
#     solution field in x, t
#     '''
#     sol, _ = problem.compute_analytical_solutions_and_initial_profiles()
#     errors = np.absolute(sol - results_)
#     plt.tight_layout()

#     x_unit = 0.25 * problem.L
#     t_unit = 0.25 * problem.T
#     fmt = lambda x: "{:.2f}".format(x)

#     xticks = [fmt(v * x_unit) for v in range(5)]
#     yticks = [fmt(v * t_unit) for v in range(5)]

#     fig, ax = plt.subplots(1, 1, figsize=(6, 6))
#     im = ax.imshow(errors.T, cmap='gray')
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("bottom", size="5%", pad=1.0)
#     cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
#     cbar.ax.tick_params(labelsize=15, rotation=90) 
#     cbar.set_label('Absolute Error', fontsize=15)
#     ax.set_yticks(range(0, problem.x_points, problem.x_points//4))
#     ax.set_xticks(range(0, problem.t_points, problem.t_points//4))
#     ax.set_yticklabels(labels=xticks, fontsize=15)
#     ax.set_xticklabels(labels=yticks, fontsize=15, rotation=90)
#     ax.set_ylabel('x', fontsize=15)
#     ax.set_xlabel('t', fontsize=15)

#     save_path_err = save_path + '_errors.png'
#     plt.savefig(save_path_err)

#     fig, ax = plt.subplots(1, 1, figsize=(6, 6))
#     im = ax.imshow(sol.T, cmap='spring')
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("bottom", size="5%", pad=1.0)
#     cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
#     cbar.ax.tick_params(labelsize=15, rotation=90) 
#     cbar.set_label('Exact Solution', fontsize=15)
#     ax.set_yticks(range(0, problem.x_points, problem.x_points//4))
#     ax.set_xticks(range(0, problem.t_points, problem.t_points//4))
#     ax.set_yticklabels(labels=xticks, fontsize=15)
#     ax.set_xticklabels(labels=yticks, fontsize=15, rotation=90)
#     ax.set_ylabel('x', fontsize=15)
#     ax.set_xlabel('t', fontsize=15)

#     save_path_err = save_path + '_anasol.png'
#     plt.savefig(save_path_err)

#     fig, ax = plt.subplots(1, 1, figsize=(6, 6))
#     im = ax.imshow(results_.T, cmap='spring')
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("bottom", size="5%", pad=1.0)
#     cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
#     cbar.ax.tick_params(labelsize=15, rotation=90) 
#     cbar.set_label('PINN Solution', fontsize=15)
#     ax.set_yticks(range(0, problem.x_points, problem.x_points//4))
#     ax.set_xticks(range(0, problem.t_points, problem.t_points//4))
#     ax.set_yticklabels(labels=xticks, fontsize=15)
#     ax.set_xticklabels(labels=yticks, fontsize=15, rotation=90)
#     ax.set_ylabel('x', fontsize=15)
#     ax.set_xlabel('t', fontsize=15)

#     save_path_err = save_path + '_pinnsol.png'
#     plt.savefig(save_path_err)
#     plt.show()

#     rel_err = error(sol, results_, typ='rel')
#     abs_err = error(sol, results_, typ='abs')
#     print(problem.name + " Relative Error: {:.4f}".format(rel_err))
#     print(problem.name + " Absolute Error: {:.4f}".format(abs_err))
#     return

# def make_ode_sim_and_error_plots(problem, results_, save_path):
#     '''
#     error field in t
#     solution field in t
#     '''
#     t = problem.Tpinn_eval.view(-1).numpy()
#     # print(t.shape)
#     sol = problem.compute_analytical_solution()
#     # print(sol.shape)
#     # print(sol)
#     errors = np.absolute(sol - results_)
#     # print(results_.shape, errors.shape)
#     plt.tight_layout()

#     t_unit = 0.25 * problem.T
#     fmt = lambda x: "{:.2f}".format(x)
#     yticks = [fmt(v * t_unit) for v in range(5)]

#     fig, axs = plt.subplots(1, 2, figsize=(12, 6))
#     axs[0].plot(t, errors, lw=2, label='Absolute Errors')
#     # axs[0].set_xticks(range(0, problem.t_points, problem.t_points//4))
#     # axs[0].set_xticklabels(labels=yticks, fontsize=12)
#     # axs[0].set_yticklabels(fontsize=12)
#     axs[0].xaxis.set_tick_params(labelsize=12)
#     axs[0].yaxis.set_tick_params(labelsize=12)
#     axs[0].set_ylabel('errors', fontsize=15)
#     axs[0].set_xlabel('t', fontsize=15)
#     axs[0].legend(fontsize=15)

#     axs[1].plot(t[::250], sol[::250], 'g^-', lw=2, label='Analytical Solution')
#     axs[1].plot(t[::200], results_[::200], 'bo--', lw=2, label='PINN Solution')
#     # axs[1].set_xticks(range(0, problem.t_points, problem.t_points//4))
#     # axs[1].set_xticklabels(labels=yticks, fontsize=12)
#     # axs[1].set_yticklabels(fontsize=12)
#     axs[1].xaxis.set_tick_params(labelsize=12)
#     axs[1].yaxis.set_tick_params(labelsize=12)
#     axs[1].set_ylabel('solutions', fontsize=15)
#     axs[1].set_xlabel('t', fontsize=15)
#     axs[1].legend(fontsize=15)

#     #   figsavepath = save_path + problem.name + '.png'
#     plt.savefig(save_path)
#     plt.show()

#     rel_err = error(sol, results_, typ='rel')
#     abs_err = error(sol, results_, typ='abs')
#     print(problem.name + " Relative Error: {:.4f}".format(rel_err))
#     print(problem.name + " Absolute Error: {:.4f}".format(abs_err))
#     return

# def make_three_time_instant_plots(problem, results_, save_path):
#     '''
#     fix three time instants from test time interval for PDEs only
#     plot the analytical solution from the problem at these instants
#     plot the PINN solution at these instants
#     choose integer indices lying between 0.8*T to T for these plots
#     '''
#     return