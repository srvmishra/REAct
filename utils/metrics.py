import numpy as np
from sklearn import metrics
# from utils.activations import *
# from utils.visualization import *
# from utils.Grids import *

var_ = lambda x: np.mean((x - np.mean(x)) ** 2)

def l2_relative_error(y_true, y_pred):
    return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)

def mean_squared_error(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2) / len(y_true)
    # return metrics.mean_squared_error(y_true, y_pred)

def mean_absolute_error(y_true, y_pred):
    return np.sum(np.absolute(y_true - y_pred)) / len(y_true)
    # return metrics.mean_absolute_error(y_true, y_pred)

def explained_variance_score(y_true, y_pred):
    return 1.0 - var_(y_true - y_pred)/var_(y_true)
    # return metrics.explained_variance_score(y_true, y_pred)

# class Results(object):
#     def __init__(self, results_dir, plot_save_dir):
#         self.results_dir = results_dir
#         self.plot_save_dir = plot_save_dir

#     def save_losses(self):
#         pass

#     def save_analytical_sol(self):
#         pass

#     def save_PINN_sol(self):
#         pass

#     def save(self):
#         pass

#     def load(self):
#         pass
