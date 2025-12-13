# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 18:58:34 2021

@author: Nikola
"""

import gym
env = gym.make("FrozenLake-v1")
state = env.reset()
env.render()
num_timesteps = 20
for t in range(num_timesteps):
    random_action = env.action_space.sample()
    new_state, reward, done, info = env.step(random_action)
    print ('Time Step {} :'.format(t+1))
    env.render()
    if done:
         break