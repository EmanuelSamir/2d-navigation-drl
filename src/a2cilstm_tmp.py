import sys
sys.path.insert(0, '../')

from algorithms.a2cilstm import A2CiLSTMAgent
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

# lr_lst = [5e-3, 1e-3, 5e-4, 1e-4, 5e-5]

# for lr in lr_lst:
agent = A2CiLSTMAgent(env, state_dim, action_dim, actions, 800, load_model_path='../checkpoints/A2CiLSTM/model/best_model_e464_r9.6604391910946.pth')

agent.train()  

agent.test()

#agent.actor_critic_logger.plot_reward(show = False, save = True, fn = 'rewards_lr_{}.png'.format(lr))
#agent.actor_critic_logger.plot_loss(show = False, save = True, fn = 'losses_lr_{}.png'.format(lr))
