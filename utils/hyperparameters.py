import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.activations import *
from utils.Models import *

class Hyperparameters(object):
    def __init__(self):
        self.lr = 1e-3
        self.epochs = 10000

    def set_model_size(self, layers):
        self.layers = layers

    def set_activation(self, act_name):
        self.act_name = act_name

    def set_problem_type(self, problem_type):
        self.problem_type = problem_type

    def create_model(self):
        return create_neural_networks(self.act_name, self.layers, self.device, problem_type=self.problem_type)

    def set_device(self, device):
        self.device = device

    def set_lr(self, lr):
        self.lr = lr

    def set_epochs(self, epochs):
        self.epochs = epochs

    def set_optimizer(self, model, optimizer, additional_params=None):
        if additional_params is not None:
            parameters = list(model.parameters()) + additional_params
        else:
            parameters = model.parameters()
        return optimizer(parameters, lr=self.lr)


