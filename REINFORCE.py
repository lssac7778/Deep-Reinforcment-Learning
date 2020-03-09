# -*- coding: utf-8 -*-
import logging as lg
lg.basicConfig(format='[%(levelname)s//%(filename)s//%(funcName)s//%(lineno)s] > %(message)s',
               level = lg.INFO)

import numpy as np
import tensorflow as tf

class PGagent():
    def __init__(self, state_shape,
                       action_size,
                       network,
                       gamma = 0.95,
                       learning_rate = 0.001,
                       entropy_rate = 0.01
                       ):
    
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.entropy_rate = entropy_rate

        self.network = network
        self.states, self.actions, self.rewards = [], [], []
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate = learning_rate)
        
        self.loss_metric = tf.keras.metrics.Mean(name='loss')
        self.entropy_metric = tf.keras.metrics.Mean(name='entropy')

    
    def act(self, state):
        policy = self.network.predict(np.expand_dims(state, axis=0))[0]
        return np.random.choice(self.action_size, 1, p=policy)[0]
    
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def memorize(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
    
    def train(self):
        states, actions = np.array(self.states), np.array(self.actions)
        discounted_rewards = self.discount_rewards(self.rewards)
        
        with tf.GradientTape() as tape:
            action_prob = tf.reduce_sum(tf.multiply(self.network(states), actions), axis = 1)
            
            cross_entropy = tf.multiply(tf.math.log(action_prob + 1e-10), discounted_rewards)
            cross_entropy = -tf.reduce_sum(cross_entropy)
            
            #entropy for exploration
            entropy = -tf.reduce_sum(tf.multiply(tf.math.log(action_prob + 1e-10), action_prob))
            
            loss = cross_entropy + self.entropy_rate * (-entropy)
            
        gradients = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
      
        self.loss_metric(loss)
        self.entropy_metric(entropy)
        
        history = {"loss":self.loss_metric.result().numpy(),
                   "entropy":self.entropy_metric.result().numpy()}
        return history


if __name__=="__main__":
    from tensorflow.keras import models
    from tensorflow.keras.layers import Dense
    
    import gym
    import plotter
    
    def network_builder(input_size, ouput_size):
        model = models.Sequential()
        model.add(Dense(16, "relu", input_shape=(input_size,)))
        model.add(Dense(16, "relu"))
        model.add(Dense(16, "relu"))
        model.add(Dense(ouput_size, "softmax"))
        
        return model
    
    state_size = (4,)
    action_size = 2
    
    network = network_builder(state_size[0], action_size)
    
    agent = PGagent(state_size,
                    action_size,
                    network,
                    learning_rate = 0.00002,
                    entropy_rate = 0.001)
    
    env = gym.make('CartPole-v1')
    env._max_episode_steps = 10001
    
    
    history = {"episode":[0],
               "reward" : [],
               "loss" : [],
               "entropy" : []}
    
    
    
    EPISODES = 100
    
    for epi in range(1, EPISODES+1):
        total_reward = 0
        state = env.reset()
        done = False
        
        while not done:
    
    
            action = agent.act(state)
            new_state, reward, done, _ = env.step(action)
            if done:
                reward = -10
        
            agent.memorize(state, action, reward)
    
            total_reward += reward
            state = new_state
        
        if total_reward > 2000:
            break
        
        hist = agent.train()
        
        history["episode"][0] += 1
        history["reward"].append(total_reward)
        history["loss"].append(hist["loss"])
        history["entropy"].append(hist["entropy"])
        
        for met in history.keys():
            print(met,":", history[met][-1], end="  ")
        print()
            
        if epi%50==0:
            for met in history.keys():
                plotter.plotWithSmooth(history[met], met, dark=True)
    
