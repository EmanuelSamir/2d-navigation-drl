import sys
sys.path.insert(0, '../')

from algorithms.a2cilstmrnd import A2CiLSTMRNDAgent
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

#lr_lst = [5e-3, 1e-3, 5e-4, 1e-4, 5e-5]
lr = 1e-4

#or lr in lr_lst:
agent = A2CiLSTMRNDAgent(env, state_dim, action_dim, actions, 1000, lr = lr, load_model_path='../checkpoints/A2CiLSTMRND/model/best_model_e962_r10.0.pth')

#agent.train()  

agent.test()

#agent.actor_critic_logger.plot_reward(show = True, save = True, fn = 'rewards_lr_exp_{}.png'.format(lr))
#agent.actor_critic_logger.plot_loss(show = True, save = True, fn = 'losses_lr_exp_{}.png'.format(lr))
#agent.rnd_logger.plot_reward(show = True, save = True, fn = 'losses_lr_exp_{}.png'.format(lr))