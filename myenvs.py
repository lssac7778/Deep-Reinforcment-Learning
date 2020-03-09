# -*- coding: utf-8 -*-
import logging as lg
lg.basicConfig(format='[%(levelname)s//%(filename)s//%(funcName)s//%(lineno)s] > %(message)s',
               level = lg.INFO)

import gym
import time
import numpy as np
from abc import *

class myenv(metaclass=ABCMeta):
    @abstractmethod
    def step(self, action):
        new_state, reward, done = None, None, None
        return new_state, reward, done
    
    @abstractmethod
    def reset(self):
        new_state = None
        return new_state

    @abstractmethod
    def play_random(self):
        pass
    
    @property
    @abstractmethod
    def state_shape(self):
        pass

    @property
    @abstractmethod
    def action_size(self):
        pass


class cartpole(myenv):
    action_size = 2
    state_shape = (4,)
    def __init__(self):
        env = gym.make('CartPole-v1')
        env._max_episode_steps = 10001
        self.env = env
    
    def step(self, action):
        new_state, reward, done, _ = self.env.step(action)
        
        if done:
            reward = -5
        
        return new_state, reward, done
    
    def reset(self):
        return self.env.reset()
    
    def play_random(self):
        state = self.env.reset()
        done = False
        while not done:
            action = self.env.action_space.sample()
            state, reward, done, _ = self.env.step(action)
            self.env.render()
        self.env.close()

from skimage.transform import resize
from skimage.color import rgb2gray
import random

class pong(myenv):
    action_size = 3
    action_space = [0, 2, 3]
    state_shape = (84, 84, 1)
    
    def __init__(self):
        env = gym.make('Pong-v0')
        self.env = env
    
    def step(self, action):
        observation, reward, done, _ = self.env.step(self.action_space[action])
        new_state = self.preprocess(observation)
        
        if reward<0:
            reward = -10
            done = True
        else:
            reward = 1
        
        return new_state, reward, done
    
    def reset(self):
        self.env.reset()
        for _ in range(20):
            state, _, _, _ = self.env.step(self.env.action_space.sample())
        return self.preprocess(state)
    
    def play_random(self):
        state = self.reset()
        done = False
        step = 1
        while not done:
            action = random.randint(0,2)
            state, reward, done = self.step(action)
            print(reward, done, step)
            self.env.render()
            step += 1
        self.env.close()
        
    def play_user(self):
        state = self.reset()
        done = False
        while not done:
            action = int(input())
            state, reward, done = self.step(action)
            print(reward, done)
            self.env.render()
        self.env.close()
        
    def preprocess(self, state):
        temp = resize(rgb2gray(state), (84, 84), mode='reflect')
        temp = np.expand_dims(temp, axis=2)
        return temp
    

if __name__=="__main__":
    env = pong()
    env.play_random()


