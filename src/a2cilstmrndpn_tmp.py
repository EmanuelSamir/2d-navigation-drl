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

if trained:
    for lr in lr_lst:
        for gamma in gamma_lst:
            #fn = '_0513_01-26-37/best_e=2978_ri=0_re=101.0_steps=202.pth'
            agent = A2CiLSTMRNDPNAgent(env, state_dim, action_dim, actions, 1000, lr = lr, gamma=gamma, intrinsic_set=False)#, load_model_fn = fn)
            agent.train('Changed again. Changed env steps from 200 to 400. Changed, death to -100. Changed be alive to 0.5. Changed time no motion to 8.')

else:
    path = '../checkpoints/A2CiLSTMRNDPN/model/'
    fn = path + 'tmp_model.pth'#'e400_rtensor([43.9384]).pth'# 'best_model_e266_r102.66022491455078.pth'#  'best_model_e579_r200.0.pth'

    agent = A2CiLSTMRNDPNAgent(env, state_dim, action_dim, actions, 1000, lr = 1e-4, load_model_path = fn)
    agent.test()