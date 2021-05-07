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

lr_lst = [1e-3, 1e-4, 1e-5]

trained = True
show = False
save = True
if trained:
    for lr in lr_lst:
        #for (lr, rnd_i) in [(i, j) for i in lr_lst for j in rnd_type]:
        #print(lr, rnd_i)
        #path = '../checkpoints/A2CiLSTMRNDPN/model/'
        #fn = path + 'tmp_model.pth'

        agent = A2CiLSTMRNDPNAgent(env, state_dim, action_dim, actions, 1500, lr = lr)
        agent.train()  

        #agent.test()
        #agent.actor_critic_logger.plot_reward(show = show, save = save, fn = 'rewards_lr_no_ri_{}.png'.format(lr))
        #agent.actor_critic_logger.plot_loss(show = show, save = save, fn = 'losses_lr_no_ri_{}.png'.format(lr))
        #agent.rnd_logger.plot_reward(show = show, save = save, fn = 'rewards_lr_exp_12_int{}.png'.format(lr))

else:
    path = '../checkpoints/A2CiLSTMRNDPN/model/'
    fn = path + 'tmp_model.pth'#'e400_rtensor([43.9384]).pth'# 'best_model_e266_r102.66022491455078.pth'#  'best_model_e579_r200.0.pth'

    agent = A2CiLSTMRNDPNAgent(env, state_dim, action_dim, actions, 1000, lr = 1e-4, load_model_path = fn)
    agent.test()