# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 19:00:49 2021

@author: Nikola
"""

import gym

env = gym.make("FrozenLake-v1")
num_episodes = 10
num_timesteps = 20
for i in range(num_episodes):
    state = env.reset()
    print('Time Step 0 :')
    env.render()
    for t in range(num_timesteps):
        random_action = env.action_space.sample()
        new_state, reward, done, info = env.step(random_action)
        print('Time Step {} :'.format(t + 1))
        env.render()
        if done:
            break
