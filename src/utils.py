from torch import nn
import torch
import gym
import time
import numpy as np
import os
from collections import deque, namedtuple


def t(x): return torch.from_numpy(x).float()

def n(x): return x.detach().float()

def create_dir(save_path):
    path = ""
    for directory in os.path.split(save_path):
        path = os.path.join(path, directory)
        if not os.path.exists(path):
            os.mkdir(path)

def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter 
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


def preprocess_transition(data, force_to_float = False):
    t = torch.tensor(data)
    dim = len(list(t.shape))
    if (dim == 0):
        t = t.unsqueeze(0).unsqueeze(0)
    elif (dim == 1):
        t = t.unsqueeze(0)
    t = t.reshape(1,-1)   
    if force_to_float:
        t = t.float()   
    return t

    # num -> 1, 1 
    # list -> 1 list   len(lst), 1
    # array -> 1, array shape[0], 1

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = 0

    def update(self, xt):
        x = xt.numpy()
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

        res =  ((xt - self.mean) / self.var).float()

        if np.isnan(res).any():
            return (xt - self.mean)
        else:
            return ((xt - self.mean) / self.var).float()

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

class RunningMeanStdOne:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    def __init__(self):
        self.mean = 0.
        self.var = 0.
        self.capacity = 1000

        self.memory = deque(maxlen = self.capacity)
        self.memory.append(0.)

    def update(self, xt):
        x = xt.numpy()
        self.memory.append(x)
        self.calculate()

        #xm = self.mean + (x - self.mean)/self.count
        #xv = self.var + ( (x - self.mean)*(x - xm) - self.var)/self.count

        res =  1*((xt - self.mean) / self.var).float() - 1
        return res

    def calculate(self):
        self.mean = np.mean(self.memory, axis = 0)
        self.var = np.var(self.memory, axis = 0)

    def reset(self):
        self.memory = [0.]

class RunningFixed:
    def __init__(self):
        self.bias_arb = 8
        self.range_arb = 8 

    def update(self,x):
        return (x - self.bias_arb)/self.range_arb

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 1.0)
        m.bias.data.fill_(0)