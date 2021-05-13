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
lr_lst = [5e-5]
gamma_lst = [0.999]

# Mejor caso fue para 0.99, deberia ser 0.999

trained = True

if trained:
    for lr in lr_lst:
        for gamma in gamma_lst:
            fn = '_0512_07-57-12/e=1950_ri=0_re=-22.0_steps=202.pth'
            agent = A2CiLSTMRNDPNAgent(env, state_dim, action_dim, actions, 50, lr = lr, gamma=gamma, intrinsic_set=False, load_model_fn=fn, trial = True)
            agent.train('Trial of _0512_07-57-12/e=1950_ri=0_re=-22.0_steps=202.pth for collecting special data. RND')

else:
    path = '../checkpoints/A2CiLSTMRNDPN/model/'
    fn = path + 'tmp_model.pth'#'e400_rtensor([43.9384]).pth'# 'best_model_e266_r102.66022491455078.pth'#  'best_model_e579_r200.0.pth'

    agent = A2CiLSTMRNDPNAgent(env, state_dim, action_dim, actions, 1000, lr = 1e-4, load_model_path = fn)
    agent.test()