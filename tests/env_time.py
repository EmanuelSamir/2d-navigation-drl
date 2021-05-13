import sys
sys.path.insert(0, '../')
import gym_robot2d

from torch import nn
import torch
import gym
import numpy as np
from matplotlib import pyplot as plt
import time

env = gym.make('robot2d-v0')

# Environments parameters
s = env.reset()
print(s[0:3])

actions =       [
                    np.array([3., 0., 0.]),
                    np.array([0., 0., -5.]),
                    np.array([0., 0., 5.]),
                ]
action_ix = 0
start = time.time()
for i in range(10):
    next_state, ext_reward, is_done, info = env.step(actions[action_ix])
    env.render()
    #time.sleep(0.1)
    print('state: {}, r_e: {}, done: {} '.format(next_state[0:3], ext_reward, is_done))

print('time step: {} '.format(( time.time() - start)/10))

print( 0.009 * 1000 )