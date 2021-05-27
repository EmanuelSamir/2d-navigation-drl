import sys
sys.path.insert(0, '../')

from algorithms.a2cilstmrndpn import A2CiLSTMRNDPNAgent
import gym_robot2d
#from robot2d import Robot2D
from torch import nn
import torch
import gym
import time
import numpy as np
from matplotlib import pyplot as plt

env = gym.make('robot2d-v0')

# Environments parameters
s = env.reset()



state_dim = s.shape[0]
actions =       [
                    np.array([3., 0., 0.]),
                    np.array([0., 0., -5.]),
                    np.array([0., 0., 5.]),
                ]
action_dim = 3

#lr_lst = [1e-3, 1e-4, 1e-5]
lr_lst = [1e-4]
gamma_lst = [0.999]

# Mejor caso fue para 0.99, deberia ser 0.999

trained = True

sets = [True, False]

if trained:
    for lr in lr_lst:
        for gamma in gamma_lst:
            #for set in sets:
            fn = '_0526_01-37-31/best_e=876_ri=310.7063903808594_re=198.0_steps=402.pth'
            agent = A2CiLSTMRNDPNAgent(env, state_dim, action_dim, actions, 50, lr = lr, gamma=gamma, intrinsic_set=True, load_model_fn = fn)
            agent.train('For tuning.')

else:
    # path = '../checkpoints/A2CiLSTMRNDPN/model/'
    # fn = '_0513_17-51-41/tmp_model.pth'
    # fn = path + 'tmp_model.pth'#'e400_rtensor([43.9384]).pth'# 'best_model_e266_r102.66022491455078.pth'#  'best_model_e579_r200.0.pth
    # '
    for lr in lr_lst:
        for gamma in gamma_lst:
            #fn = '_0514_01-22-57/best_e=1899_ri=0_re=201.0_steps=402.pth'
            #fn = '_0519_01-05-52/e=1150_ri=25.208454132080078_re=177.0_steps=402.pth'
            fn = '_0526_01-37-31/best_e=876_ri=310.7063903808594_re=198.0_steps=402.pth'
            agent = A2CiLSTMRNDPNAgent(env, state_dim, action_dim, actions, 1000, lr = lr, gamma=gamma, load_model_fn = fn, trial = True)
            agent.test()