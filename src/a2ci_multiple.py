import sys
sys.path.insert(0, '../')

from algorithms.a2ci import A2CiAgent
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
                    np.array([0., 2.]),
                    np.array([0., -2.]),
                    np.array([2., 0.]),
                    np.array([-2., 0.]),
                ]
action_dim = 4

lr_lst = [5e-3, 1e-3, 5e-4, 1e-4, 5e-5]

for lr in lr_lst:
    print('Case for lr = {}.'.format(lr))
    print( 100*'-' )
    agent = A2CiAgent(env, state_dim, action_dim, actions, 1000)
    agent.train()  

    #agent.test()

    agent.actor_critic_logger.plot_reward(show = False, save = True, fn = 'rewards_lr_{}.png'.format(lr))
    agent.actor_critic_logger.plot_loss(show = False, save = True, fn = 'losses_lr_{}.png'.format(lr))
