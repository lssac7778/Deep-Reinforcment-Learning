# -*- coding: utf-8 -*-
import logging as lg
lg.basicConfig(format='[%(levelname)s//%(filename)s//%(funcName)s//%(lineno)s] > %(message)s',
               level = lg.INFO)

import tensorflow as tf
import numpy as np

class ACagent():
    def __init__(self, state_shape,
                       action_size,
                       actor_network,
                       critic_network,
                       gamma = 0.95,
                       actor_learning_rate = 0.001,
                       critic_learning_rate = 0.001,
                       entropy_rate = 0.01
                       ):
    
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = gamma
        
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        
        self.entropy_rate = entropy_rate

        self.actor = actor_network
        self.critic = critic_network
        
        self.actor_optimizer = tf.keras.optimizers.RMSprop(learning_rate = actor_learning_rate)
        self.critic_optimizer = tf.keras.optimizers.RMSprop(learning_rate = critic_learning_rate)

        
        self.actor_loss_metric = tf.keras.metrics.Mean(name='critic_loss')
        self.critic_loss_metric = tf.keras.metrics.Mean(name='actor_loss')
        self.entropy_metric = tf.keras.metrics.Mean(name='entropy')

    def act(self, state):
        policy = self.actor.predict(np.expand_dims(state, axis=0))[0]
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def train_actor(self, states, actions, advantages):
        
        with tf.GradientTape() as tape:
            action_prob = tf.reduce_sum(tf.multiply(self.actor(states), actions), axis = 1)
            
            cross_entropy = tf.multiply(tf.math.log(action_prob + 1e-10), advantages)
            cross_entropy = -tf.reduce_sum(cross_entropy)
            
            entropy = -tf.reduce_sum(tf.multiply(tf.math.log(action_prob + 1e-10), action_prob))
            
            loss = cross_entropy + self.entropy_rate * (-entropy)
            
        gradients = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))
        
        self.actor_loss_metric(loss)
        self.entropy_metric(entropy)
        history = {"actor_loss":self.actor_loss_metric.result().numpy(),
                   "entropy":self.entropy_metric.result().numpy()}
        return history

    def train_critic(self, states, targets):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(self.critic(states) - targets))
            
        gradients = tape.gradient(loss, self.critic.trainable_variables)
        
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))
        
        self.critic_loss_metric(loss)
        history = {"critic_loss":self.critic_loss_metric.result().numpy()}
        return history

    def train_step(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        
        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        act = np.zeros([1, self.action_size])
        act[0][action] = 1

        if done:
            advantage = reward - value
            target = [reward]
        else:
            advantage = (reward + self.gamma * next_value) - value
            target = reward + self.gamma * next_value

        actor_history = self.train_actor(state, act, advantage)
        critic_history = self.train_critic(state, target)
        
        history = {}
        history.update(actor_history)
        history.update(critic_history)
        return history

if __name__=="__main__":
    from tensorflow.keras import models
    from tensorflow.keras.layers import Dense
    
    import gym
    import plotter
    
    


    
    state_size = (4,)
    action_size = 2
    
    actor = models.Sequential()
    actor.add(Dense(16, "relu", input_shape=(state_size[0],)))
    actor.add(Dense(16, "relu"))
    actor.add(Dense(16, "relu"))
    actor.add(Dense(action_size, "softmax"))

    critic = models.Sequential()
    critic.add(Dense(16, "relu", input_shape=(state_size[0],)))
    critic.add(Dense(16, "relu"))
    critic.add(Dense(1, "linear"))
        
        
    agent = ACagent(state_size,
                    action_size,
                    actor,
                    critic,
                    actor_learning_rate = 0.001,
                    critic_learning_rate = 0.01,
                    entropy_rate = 0.01)
    
    env = gym.make('CartPole-v1')
    env._max_episode_steps = 10001
    
    
    history = {"episode":[0],
               "reward" : [],
               "actor_loss" : [],
               "critic_loss" : [],
               "entropy" : []}
    
    
    
    EPISODES = 2000
    
    for epi in range(1, EPISODES+1):
        total_reward = 0
        state = env.reset()
        done = False
        
        while not done:
    
    
            action = agent.act(state)
            new_state, reward, done, _ = env.step(action)
            if done:
                reward = -10
        
            hist = agent.train_step(state, action, reward, new_state, done)
            
            history["actor_loss"].append(hist["actor_loss"])
            history["critic_loss"].append(hist["critic_loss"])
            history["entropy"].append(hist["entropy"])
    
            total_reward += reward
            state = new_state
        
        if total_reward > 2000:
            break
        
        history["episode"][0] += 1
        history["reward"].append(total_reward)
        
        for met in history.keys():
            print(met,":", history[met][-1], end="  ")
        print()
            
        if epi%50==0:
            for met in history.keys():
                plotter.plotWithSmooth(history[met], met, dark=True)

