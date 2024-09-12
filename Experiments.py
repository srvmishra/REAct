from utils.activations import *
from utils.Grids import *
from utils.metrics import *
from utils.hyperparameters import *
from utils.Losses import *
from utils.visualization import *
from equations.TwoDSpacePDE import *
from equations.FuncApprox import *
from equations.ODEProblems import *
from equations.OneDSpaceTimePDE import *
from equations.InverseProblems import *
from tqdm import tqdm
import pickle
import pandas as pd
import markdown


'''
save_paths are full paths including filename but excluding other stuff and extensions
- include optimizer name in filename too
like ./results/HeatEquation_Adam_train_1
additional_str contains the optimizer and train protocol like _Adam_train_1
'''

## experiments for convergence - loss curves, with and without slope recovery, metrics values, NTK eigenspectrum
## forward problems - create loss plots and stuff
class ForwardExperiement(object):
    def __init__(self, problem, hyperparameters, optimizer, additional_str, ntk=True):
        self.problem = problem
        self.hparams = hyperparameters
        self.optimizer_name = optimizer
        self.ntk = ntk

        self.loss_save_path = './losses/' + self.problem.name + additional_str + '_losses.pkl'
        self.results_save_path = './results/' + self.problem.name + additional_str + '_results.pkl'
        self.ntk_spectra_save_path = './results/' + self.problem.name + additional_str + '_ntk_eval.pkl'
        
        self.results_dict = {k: None for k in activation_names}
        self.losses_dict = {k: None for k in activation_names}
        self.ntk_eigen_dict = {k: None for k in activation_names}
        self.losses_dict['legends'] = self.problem.legends

        # if isinstance(self.problem, OneDSpaceTimeEquations) or isinstance(self.problem, TwoDSpaceEquations):
        #     self.hparams.set_problem_type('PDE')
        if isinstance(self.problem, OneDSpaceGrid) or isinstance(self.problem, OneDTimeGrid):
            self.hparams.set_problem_type('ODE')
        else:
            self.hparams.set_problem_type('PDE')

    def train(self):
        # return losses list
        # format: [[total, bc, ic, ode/pde] ... epochs]
        losses, evals = [], []
        for i in tqdm(range(self.hparams.epochs)):
            self.net.train()
            losses_ = self.problem.return_losses(self.net, self.hparams.device)
            loss = losses_[0]
            # if isinstance(self.net, CustomNeuralNetwork):
            #     slope = self.net.slope_recovery()
            #     loss = loss + slope
            #     losses_[0] = losses_[0] + slope
            losses.append([tensor_to_numpy(l).item() for l in losses_])
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.ntk:
                self.net.eval()
                ev = self.problem.compute_ntk_eigenvalue(self.net, self.hparams.device)
                evals.append(ev)
        
        if self.ntk:
            return losses, evals
        else:
            return losses

    def test(self):
        return self.problem.compute_PINN_solution(self.net, self.hparams.device)

    def analytical_solution(self):
        ana_sol = self.problem.compute_analytical_solution(self.hparams.device)
        self.results_dict['analytical'] = ana_sol
        # print('Nan values in analytical sol: {}'.format(np.isnan(ana_sol)))
    
    def train_and_test_on_all_activations(self):
        print('Computing and Saving Analytical Solution')
        self.analytical_solution()
        for act_name in activation_names:
            print("Training for activation fn: {}".format(act_name))
            self.hparams.set_activation(act_name)
            self.net = self.hparams.create_model()
            self.optimizer = self.hparams.set_optimizer(self.net, self.optimizer_name)

            if self.ntk:
                losses, evals = self.train()
            else:
                losses = self.train()

            self.losses_dict[act_name] = losses
            self.results_dict[act_name] = self.test()
            if self.ntk:
                self.ntk_eigen_dict[act_name] = evals
            
            torch.cuda.empty_cache()
        
        save_pkl_file(self.results_dict, self.results_save_path)
        save_pkl_file(self.losses_dict, self.loss_save_path)
        if self.ntk:
            save_pkl_file(self.ntk_eigen_dict, self.ntk_spectra_save_path)
        
        self.evaluate()

    def evaluate(self):
        results_table = {k: {'l2_relative_error': None,
                             'mean_squared_error': None,
                             'mean_absolute_error': None,
                             'explained_variance_score': None} 
                         for k in activation_names}
        y_true = self.results_dict['analytical']
        for act_name in activation_names:
            y_pred = self.results_dict[act_name]
            results_table[act_name]['l2_relative_error'] = l2_relative_error(y_true, y_pred)
            results_table[act_name]['mean_squared_error'] = mean_squared_error(y_true, y_pred)
            results_table[act_name]['mean_absolute_error'] = mean_absolute_error(y_true, y_pred)
            results_table[act_name]['explained_variance_score'] = explained_variance_score(y_true, y_pred)
        results_df = pd.DataFrame.from_dict(results_table)
        print(results_df.T.to_markdown())

    def plot_results(self, best_act_name, acts_to_plot):
        save_path = './plots'
        print('Comparing Losses ...')
        compare_losses(self.problem.name, best_act_name, self.loss_save_path, acts_to_plot, save_path)
        print('Comparing NTK Spectra ...')
        make_ntk_spectra_plot(self.problem.name, self.ntk_spectra_save_path, acts_to_plot, save_path)
        print('Visualize Solutions ...')
        visualize_problem(self.problem, best_act_name, self.results_save_path, save_path)
        return


## experiments for inverse problems - all activations - compare percentage error vs noise level (1%, 5%, 10%)
## keeping other configurations same
## no ntk, but there will be loss plots
## Make 3 experiments - one for each noise level - compare real and obtained values of parameters and solutions
class InverseExperiement(object):
    def __init__(self, problem, hyperparameters, optimizer, additional_str, noise_std=0.1):
        self.problem = problem
        self.hparams = hyperparameters
        self.optimizer_name = optimizer
        self.noise_std = noise_std

        self.loss_save_path = './losses/' + self.problem.name + additional_str + '_losses.pkl'
        self.results_save_path = './results/' + self.problem.name + additional_str + '_results.pkl'
        
        self.results_dict = {k: None for k in activation_names}
        self.losses_dict = {k: None for k in activation_names}
        # self.ntk_eigen_dict = {k: None for k in activation_names}
        self.losses_dict['legends'] = self.problem.legends

        if isinstance(self.problem, OneDSpaceGrid) or isinstance(self.problem, OneDTimeGrid):
            self.hparams.set_problem_type('ODE')
        else:
            self.hparams.set_problem_type('PDE')

    def set_noise_std(self, std):
        self.problem.set_noise_std(std)

    def train(self):
        # return losses list
        # format: [[total, bc, ic, ode/pde] ... epochs]
        losses = []
        for i in tqdm(range(self.hparams.epochs)):
            losses_ = self.problem.return_losses(self.net, self.hparams.device)
            loss = losses_[0]
            # if isinstance(self.net, CustomNeuralNetwork):
            #     slope = self.net.slope_recovery()
            #     loss = loss + slope
            #     losses_[0] = losses_[0] + slope
            losses.append([tensor_to_numpy(l).item() for l in losses_])
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        print('final')
        print(self.problem.param)
        return losses

    def test(self):
        return self.problem.compute_PINN_solution(self.net, self.hparams.device)

    def analytical_solution(self):
        ana_sol = self.problem.compute_analytical_solution(self.hparams.device)
        self.results_dict['analytical'] = ana_sol
        # print('Nan values in analytical sol: {}'.format(np.isnan(ana_sol)))
    
    def train_and_test_on_all_activations(self):
        print('Computing and Saving Analytical Solution')
        self.analytical_solution()
        for act_name in activation_names:
            print("Training for activation fn: {}".format(act_name))
            self.problem.init_param()
            print('init')
            print(self.problem.param)
            self.hparams.set_activation(act_name)
            self.net = self.hparams.create_model()
            self.optimizer = self.hparams.set_optimizer(self.net, self.optimizer_name, 
                                                        additional_params=[self.problem.param])
            losses = self.train()

            self.losses_dict[act_name] = losses
            self.results_dict[act_name] = self.test()
            
            torch.cuda.empty_cache()
        
        save_pkl_file(self.results_dict, self.results_save_path)
        save_pkl_file(self.losses_dict, self.loss_save_path)
        
        self.evaluate()

    def evaluate(self):
        results_table = {k: {'l2_relative_error': None,
                             'mean_squared_error': None,
                             'mean_absolute_error': None,
                             'explained_variance_score': None,
                             'est. param value': None,
                             'pc. error in param value': None} 
                         for k in activation_names}
        y_true = self.results_dict['analytical']
        for act_name in activation_names:
            # this parameter value is the value from the last experiment
            # so this is not correct for all activation cases
            # use the printed value from the above train function and use it manually to calculate pc
            param = tensor_to_numpy(self.problem.param).item()
            y_pred = self.results_dict[act_name]
            results_table[act_name]['l2_relative_error'] = l2_relative_error(y_true, y_pred)
            results_table[act_name]['mean_squared_error'] = mean_squared_error(y_true, y_pred)
            results_table[act_name]['mean_absolute_error'] = mean_absolute_error(y_true, y_pred)
            results_table[act_name]['explained_variance_score'] = explained_variance_score(y_true, y_pred)
            results_table[act_name]['est. param value'] = param
            results_table[act_name]['pc. error in param value'] = pc_error(self.problem.param_, param)
        results_df = pd.DataFrame.from_dict(results_table)
        print(results_df.T.to_markdown())

    def train_across_noise_stds(self, act_name):
        self.noise_stds = [0.1, 0.5, 1.0, 3.0, 5.0]
        results_table = {k: {'l2_relative_error': None,
                             'mean_squared_error': None,
                             'mean_absolute_error': None,
                             'explained_variance_score': None,
                             'est. param value': None,
                             'pc. error in param value': None} 
                         for k in self.noise_stds}
        self.analytical_solution()
        y_true = self.results_dict['analytical']
        
        for st_ in self.noise_stds:
            self.problem.set_noise_std(st_)
            self.problem.init_param()
            print('init')
            print(self.problem.param)
            self.hparams.set_activation(act_name)
            self.net = self.hparams.create_model()
            self.optimizer = self.hparams.set_optimizer(self.net, self.optimizer_name, 
                                                        additional_params=[self.problem.param])
            _ = self.train()
            y_pred = self.test()

            param = tensor_to_numpy(self.problem.param).item()
            print('final')
            print(param)
            results_table[st_]['l2_relative_error'] = l2_relative_error(y_true, y_pred)
            results_table[st_]['mean_squared_error'] = mean_squared_error(y_true, y_pred)
            results_table[st_]['mean_absolute_error'] = mean_absolute_error(y_true, y_pred)
            results_table[st_]['explained_variance_score'] = explained_variance_score(y_true, y_pred)
            results_table[st_]['est. param value'] = param
            results_table[st_]['pc. error in param value'] = pc_error(self.problem.param_, param)

            torch.cuda.empty_cache()

        results_df = pd.DataFrame.from_dict(results_table)
        print(results_df.T.to_markdown())

    def plot_results(self, best_act_name, acts_to_plot):
        save_path = './plots'
        print('Comparing Losses ...')
        compare_losses(self.problem.name, best_act_name, self.loss_save_path, acts_to_plot, save_path)
        print('Visualize Solutions ...')
        visualize_problem(self.problem, best_act_name, self.results_save_path, save_path)
        return


## generalization experiments - compare with Neural Oscillators

## slope recovery experiments - copy the above classes and modify them to include slope recovery
class ForwardExperiementwithSlopeRecovery(object):
    def __init__(self, problem, hyperparameters, optimizer, additional_str, ntk=True):
        self.problem = problem
        self.hparams = hyperparameters
        self.optimizer_name = optimizer
        self.ntk = ntk

        self.loss_save_path = './slope_losses/' + self.problem.name + additional_str + '_losses.pkl'
        self.results_save_path = './slope_results/' + self.problem.name + additional_str + '_results.pkl'
        self.ntk_spectra_save_path = './slope_results/' + self.problem.name + additional_str + '_ntk_eval.pkl'

        self.act_names = ['STan-LAAF', 'STan-NLAAF', 'REAct-LAAF', 'REAct-NLAAF']
        
        self.results_dict = {k: None for k in self.act_names}
        self.losses_dict = {k: None for k in self.act_names}
        self.ntk_eigen_dict = {k: None for k in self.act_names}
        self.losses_dict['legends'] = self.problem.legends

        # if isinstance(self.problem, OneDSpaceTimeEquations) or isinstance(self.problem, TwoDSpaceEquations):
        #     self.hparams.set_problem_type('PDE')
        if isinstance(self.problem, OneDSpaceGrid) or isinstance(self.problem, OneDTimeGrid):
            self.hparams.set_problem_type('ODE')
        else:
            self.hparams.set_problem_type('PDE')

    def train(self):
        # return losses list
        # format: [[total, bc, ic, ode/pde] ... epochs]
        losses, evals = [], []
        for i in tqdm(range(self.hparams.epochs)):
            self.net.train()
            losses_ = self.problem.return_losses(self.net, self.hparams.device)
            loss = losses_[0] + self.net.slope_recovery()
            # if isinstance(self.net, CustomNeuralNetwork):
            #     slope = self.net.slope_recovery()
            #     loss = loss + slope
            #     losses_[0] = losses_[0] + slope
            losses.append([tensor_to_numpy(l).item() for l in losses_])
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.ntk:
                self.net.eval()
                ev = self.problem.compute_ntk_eigenvalue(self.net, self.hparams.device)
                evals.append(ev)
        
        if self.ntk:
            return losses, evals
        else:
            return losses

    def test(self):
        return self.problem.compute_PINN_solution(self.net, self.hparams.device)

    def analytical_solution(self):
        ana_sol = self.problem.compute_analytical_solution(self.hparams.device)
        self.results_dict['analytical'] = ana_sol
        # print('Nan values in analytical sol: {}'.format(np.isnan(ana_sol)))
    
    def train_and_test_on_all_activations(self):
        print('Computing and Saving Analytical Solution')
        self.analytical_solution()
        for act_name in self.act_names:
            print("Training for activation fn: {}".format(act_name))
            self.hparams.set_activation(act_name)
            self.net = self.hparams.create_model()
            self.optimizer = self.hparams.set_optimizer(self.net, self.optimizer_name)

            if self.ntk:
                losses, evals = self.train()
            else:
                losses = self.train()

            self.losses_dict[act_name] = losses
            self.results_dict[act_name] = self.test()
            if self.ntk:
                self.ntk_eigen_dict[act_name] = evals
            
            torch.cuda.empty_cache()
        
        save_pkl_file(self.results_dict, self.results_save_path)
        save_pkl_file(self.losses_dict, self.loss_save_path)
        if self.ntk:
            save_pkl_file(self.ntk_eigen_dict, self.ntk_spectra_save_path)
        
        self.evaluate()

    def evaluate(self):
        results_table = {k: {'l2_relative_error': None,
                             'mean_squared_error': None,
                             'mean_absolute_error': None,
                             'explained_variance_score': None} 
                         for k in self.act_names}
        y_true = self.results_dict['analytical']
        for act_name in self.act_names:
            y_pred = self.results_dict[act_name]
            results_table[act_name]['l2_relative_error'] = l2_relative_error(y_true, y_pred)
            results_table[act_name]['mean_squared_error'] = mean_squared_error(y_true, y_pred)
            results_table[act_name]['mean_absolute_error'] = mean_absolute_error(y_true, y_pred)
            results_table[act_name]['explained_variance_score'] = explained_variance_score(y_true, y_pred)
        results_df = pd.DataFrame.from_dict(results_table)
        print(results_df.T.to_markdown())

    def plot_results(self, best_act_name, acts_to_plot):
        save_path = './plots'
        print('Comparing Losses ...')
        compare_losses(self.problem.name, best_act_name, self.loss_save_path, acts_to_plot, save_path)
        print('Comparing NTK Spectra ...')
        make_ntk_spectra_plot(self.problem.name, self.ntk_spectra_save_path, acts_to_plot, save_path)
        print('Visualize Solutions ...')
        visualize_problem(self.problem, best_act_name, self.results_save_path, save_path)
        return

class InverseExperiementwithSlopeRecovery(object):
    def __init__(self, problem, hyperparameters, optimizer, additional_str, noise_std=0.1):
        self.problem = problem
        self.hparams = hyperparameters
        self.optimizer_name = optimizer
        self.noise_std = noise_std

        self.loss_save_path = './slope_losses/' + self.problem.name + additional_str + '_losses.pkl'
        self.results_save_path = './slope_results/' + self.problem.name + additional_str + '_results.pkl'

        self.act_names = ['STan-LAAF', 'STan-NLAAF', 'REAct-LAAF', 'REAct-NLAAF']
        
        self.results_dict = {k: None for k in self.act_names}
        self.losses_dict = {k: None for k in self.act_names}
        # self.ntk_eigen_dict = {k: None for k in self.act_names}
        self.losses_dict['legends'] = self.problem.legends

        if isinstance(self.problem, OneDSpaceGrid) or isinstance(self.problem, OneDTimeGrid):
            self.hparams.set_problem_type('ODE')
        else:
            self.hparams.set_problem_type('PDE')

    def set_noise_std(self, std):
        self.problem.set_noise_std(std)

    def train(self):
        # return losses list
        # format: [[total, bc, ic, ode/pde] ... epochs]
        losses = []
        for i in tqdm(range(self.hparams.epochs)):
            losses_ = self.problem.return_losses(self.net, self.hparams.device)
            loss = losses_[0] + self.net.slope_recovery()
            # if isinstance(self.net, CustomNeuralNetwork):
            #     slope = self.net.slope_recovery()
            #     loss = loss + slope
            #     losses_[0] = losses_[0] + slope
            losses.append([tensor_to_numpy(l).item() for l in losses_])
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        print('final')
        print(self.problem.param)
        return losses

    def test(self):
        return self.problem.compute_PINN_solution(self.net, self.hparams.device)

    def analytical_solution(self):
        ana_sol = self.problem.compute_analytical_solution(self.hparams.device)
        self.results_dict['analytical'] = ana_sol
        # print('Nan values in analytical sol: {}'.format(np.isnan(ana_sol)))
    
    def train_and_test_on_all_activations(self):
        print('Computing and Saving Analytical Solution')
        self.analytical_solution()
        for act_name in self.act_names:
            print("Training for activation fn: {}".format(act_name))
            self.problem.init_param()
            print('init')
            print(self.problem.param)
            self.hparams.set_activation(act_name)
            self.net = self.hparams.create_model()
            self.optimizer = self.hparams.set_optimizer(self.net, self.optimizer_name, 
                                                        additional_params=[self.problem.param])
            losses = self.train()

            self.losses_dict[act_name] = losses
            self.results_dict[act_name] = self.test()
            
            torch.cuda.empty_cache()
        
        save_pkl_file(self.results_dict, self.results_save_path)
        save_pkl_file(self.losses_dict, self.loss_save_path)
        
        self.evaluate()

    def evaluate(self):
        results_table = {k: {'l2_relative_error': None,
                             'mean_squared_error': None,
                             'mean_absolute_error': None,
                             'explained_variance_score': None,
                             'est. param value': None,
                             'pc. error in param value': None} 
                         for k in self.act_names}
        y_true = self.results_dict['analytical']
        for act_name in self.act_names:
            # this parameter value is the value from the last experiment
            # so this is not correct for all activation cases
            # use the printed value from the above train function and use it manually to calculate pc
            param = tensor_to_numpy(self.problem.param).item()
            y_pred = self.results_dict[act_name]
            results_table[act_name]['l2_relative_error'] = l2_relative_error(y_true, y_pred)
            results_table[act_name]['mean_squared_error'] = mean_squared_error(y_true, y_pred)
            results_table[act_name]['mean_absolute_error'] = mean_absolute_error(y_true, y_pred)
            results_table[act_name]['explained_variance_score'] = explained_variance_score(y_true, y_pred)
            results_table[act_name]['est. param value'] = param
            results_table[act_name]['pc. error in param value'] = pc_error(self.problem.param_, param)
        results_df = pd.DataFrame.from_dict(results_table)
        print(results_df.T.to_markdown())

    def train_across_noise_stds(self, act_name):
        self.noise_stds = [0.1, 0.5, 1.0, 3.0, 5.0]
        results_table = {k: {'l2_relative_error': None,
                             'mean_squared_error': None,
                             'mean_absolute_error': None,
                             'explained_variance_score': None,
                             'est. param value': None,
                             'pc. error in param value': None} 
                         for k in self.noise_stds}
        self.analytical_solution()
        y_true = self.results_dict['analytical']
        
        for st_ in self.noise_stds:
            self.problem.set_noise_std(st_)
            self.problem.init_param()
            print('init')
            print(self.problem.param)
            self.hparams.set_activation(act_name)
            self.net = self.hparams.create_model()
            self.optimizer = self.hparams.set_optimizer(self.net, self.optimizer_name, 
                                                        additional_params=[self.problem.param])
            _ = self.train()
            y_pred = self.test()

            param = tensor_to_numpy(self.problem.param).item()
            print('final')
            print(param)
            results_table[st_]['l2_relative_error'] = l2_relative_error(y_true, y_pred)
            results_table[st_]['mean_squared_error'] = mean_squared_error(y_true, y_pred)
            results_table[st_]['mean_absolute_error'] = mean_absolute_error(y_true, y_pred)
            results_table[st_]['explained_variance_score'] = explained_variance_score(y_true, y_pred)
            results_table[st_]['est. param value'] = param
            results_table[st_]['pc. error in param value'] = pc_error(self.problem.param_, param)

            torch.cuda.empty_cache()

        results_df = pd.DataFrame.from_dict(results_table)
        print(results_df.T.to_markdown())

    def plot_results(self, best_act_name, acts_to_plot):
        save_path = './plots'
        print('Comparing Losses ...')
        compare_losses(self.problem.name, best_act_name, self.loss_save_path, acts_to_plot, save_path)
        print('Visualize Solutions ...')
        visualize_problem(self.problem, best_act_name, self.results_save_path, save_path)
        return
