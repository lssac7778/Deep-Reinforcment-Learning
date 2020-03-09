# -*- coding: utf-8 -*-
import logging as lg
lg.basicConfig(format='[%(levelname)s//%(filename)s//%(funcName)s//%(lineno)s] > %(message)s',
               level = lg.INFO)

import tensorflow as tf
import numpy as np

class PPOagent():
    def __init__(self, state_shape,
                       action_size,
                       actor_network,
                       critic_network,
                       gamma = 0.95,
                       learning_rate = 0.001,
                       entropy_rate = 0.01,
                       loss_clip = 0.1,
                       epoch = 10,
                       horizon = 64,
                       gae_lambda = 0.95
                       ):
        
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.learning_rate = learning_rate
        self.entropy_rate = entropy_rate
        self.epoch = epoch
        self.horizon = horizon
        self.loss_clip = loss_clip

        self.actor = actor_network
        self.critic = critic_network
        
        self.memory = []
        
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate = learning_rate)
        
        self.actor_loss_metric = tf.keras.metrics.Mean(name='critic_loss')
        self.critic_loss_metric = tf.keras.metrics.Mean(name='actor_loss')
        self.entropy_metric = tf.keras.metrics.Mean(name='entropy')
        

    def act(self, state):
        policy = self.actor.predict(np.expand_dims(state, axis=0))[0]
        action = np.random.choice(self.action_size, 1, p=policy)[0]
        action_prob = policy[action]
        return action, action_prob
    
    def memorize(self, state, action, action_prob, reward, next_state, done):
        act = np.zeros(self.action_size)
        act[action] = 1
        self.memory.append((state, act, action_prob, reward, next_state, done))
    
    def get_batch(self):
        result = []
        data_len = len(self.memory[0])
        for i in range(data_len):
            temp = []
            for data in self.memory:
                temp.append(data[i])
            result.append(np.array(temp))
        
        self.memory = []
        return result
    
    def train(self):
        '''
        this function should be called when done==True
        '''
        states, actions, action_probs, rewards, next_states, dones = self.get_batch()
        
        for _ in range(self.epoch):
            targets = rewards + dones * self.gamma * self.critic.predict(next_states)[0]
            td_errors = targets - self.critic.predict(states)[0]
            td_errors = td_errors.tolist()
            
            #compute Generalized Advantage Estimation
            advantages = []
            advantage = 0
            for td_error in reversed(td_errors):
                advantage = self.gamma * self.gae_lambda * advantage + td_error
                advantages.append(advantage)
            advantages.reverse()
            advantages = np.array(advantages)

            
            with tf.GradientTape() as actor_tape:
                new_action_probs = tf.reduce_sum(tf.multiply(self.actor(states), actions), axis = 1)
                
                ratio = tf.exp(tf.math.log(new_action_probs + 1e-10) - tf.math.log(action_probs + 1e-10))
                
                surrogate1 = ratio * advantages
                surrogate2 = tf.clip_by_value(ratio, 1 - self.loss_clip, 1 + self.loss_clip) * advantages
                
                entropy = -tf.reduce_sum(tf.multiply(tf.math.log(new_action_probs + 1e-10), new_action_probs))
                
                actor_loss = -tf.minimum(surrogate1, surrogate2) + self.entropy_rate * (-entropy)
            
            actor_gradients = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
            self.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
            
            self.actor_loss_metric(actor_loss)
            self.entropy_metric(entropy)
            
            
            with tf.GradientTape() as critic_tape:
                critirc_loss = tf.reduce_mean(tf.square(targets - self.critic(states)))
            
            critic_gradients = critic_tape.gradient(critirc_loss, self.critic.trainable_variables)
            self.optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
            
            self.critic_loss_metric(critirc_loss)
            
        
        history = {"actor_loss":self.actor_loss_metric.result().numpy(),
                   "critic_loss":self.critic_loss_metric.result().numpy(),
                   "entropy":self.entropy_metric.result().numpy()}
        return history



if __name__=="__main__":
    from tensorflow.keras import models
    from tensorflow.keras.layers import Dense
    
    import gym
    import plotter
    
    def play_env(agent):
        state = env.reset()
        done = False
        while not done:
            action, action_prob = agent.act(state)
            state, reward, done, _ = env.step(action)
            env.render()
    
    state_size = (4,)
    action_size = 2
    
    actor = models.Sequential()
    actor.add(Dense(32, "relu", input_shape=(state_size[0],)))
    actor.add(Dense(32, "relu"))
    actor.add(Dense(32, "relu"))
    actor.add(Dense(action_size, "softmax"))

    critic = models.Sequential()
    critic.add(Dense(32, "relu", input_shape=(state_size[0],)))
    critic.add(Dense(32, "relu"))
    critic.add(Dense(1, "linear"))
    
    
    agent = PPOagent(state_size,
                     action_size,
                     actor,
                     critic,
                     gamma = 0.98,
                     learning_rate = 0.001,
                     entropy_rate = 0.00001,
                     loss_clip = 0.1,
                     epoch = 3,
                     horizon = 20,
                     gae_lambda = 0.95)
    
    env = gym.make('CartPole-v1')
    env._max_episode_steps = 10001
    
    
    history = {"episode":0,
               "step":0,
               "reward" : [],
               "actor_loss" : [],
               "critic_loss" : [],
               "entropy" : []}
    
    prints = ["reward",
              "actor_loss",
              "critic_loss",
              "entropy"]
    
    
    EPISODES = 1000
    total_reward = 0
    state = env.reset()
    done = False
    
    while history["episode"] <= EPISODES:

        action, action_prob = agent.act(state)
        new_state, reward, done, _ = env.step(action)
        if done: reward = -10
        agent.memorize(state, action, action_prob, reward, new_state, done)
        
        total_reward += reward
        state = new_state
        history["step"] += 1
        
        if history["step"] % agent.horizon == 0 or done:
            hist = agent.train()
            
            history["actor_loss"].append(hist["actor_loss"])
            history["critic_loss"].append(hist["critic_loss"])
            history["entropy"].append(hist["entropy"])


        if done:
            history["episode"] += 1
            history["reward"].append(total_reward)
            
            total_reward = 0
            state = env.reset()
            done = False
            
            
            for met in history.keys():
                if met in prints:
                    
                    if len(history[met])==0: printval = "-"
                    else: printval = history[met][-1]
                    
                else:
                    printval = history[met]
                
                print(met,":", printval, end="  ")
            print()
                
            if history["episode"]%50==0:
                for met in prints:
                    plotter.plotWithSmooth(history[met], met, dark=True)
        
            if total_reward > 10000:
                break
    
    

