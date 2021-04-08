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

lr_lst = [5e-3, 1e-3, 5e-4, 1e-4, 5e-5]

trained = True
show = False
if trained:
    for lr in lr_lst:
        agent = A2CiLSTMRNDPNAgent(env, state_dim, action_dim, actions, 1500, lr = lr)

        agent.train()  

        #agent.test()

        agent.actor_critic_logger.plot_reward(show = show, save = True, fn = 'rewards_lr_exp_{}.png'.format(lr))
        agent.actor_critic_logger.plot_loss(show = show, save = True, fn = 'losses_lr_exp_{}.png'.format(lr))
        agent.rnd_logger.plot_reward(show = show, save = True, fn = 'rewards_lr_exp_{}.png'.format(lr))

else:
    for lr in lr_lst:
        path = '../checkpoints/A2CiLSTMRNDPN/model/'
        fn = path + 'tmp_model.pth'

        agent = A2CiLSTMRNDPNAgent(env, state_dim, action_dim, actions, 1000, lr = lr, load_model_path = fn)
        agent.test()