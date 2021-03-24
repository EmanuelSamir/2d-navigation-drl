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

#action_dim = env.action_space.n
#actions = None

agent = A2CiAgent(env, state_dim, action_dim, actions, 500, load_model_path = '../checkpoints/A2Ci/model/best_model_e753_r9.696371180001229.pth')#best_model_e122_r9.221048075495599.pth#best_model_e485_r5.052320332792627.pth')

# agent.train()  

agent.test()

# agent.actor_critic_logger.plot_reward(show = True, save = True)
# agent.actor_critic_logger.plot_loss(show = True, save = True)