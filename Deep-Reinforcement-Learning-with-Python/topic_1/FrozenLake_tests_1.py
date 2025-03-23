# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 20:37:35 2021

@author: Nikola
"""


## Tests
import gym 
env = gym.make("FrozenLake-v1")
state = env.reset()
env.render()

print(env.observation_space)
print(env.action_space)
print(env.P[14][2]) 

state = env.reset()

env.render()

env.step(2)

env.step(2)

env.step(1)

env.step(0)

env.render()

(next_state, reward, done, info) = env.step(1)


random_action = env.action_space.sample()
next_state, reward, done, info = env.step(random_action)
env.render()




