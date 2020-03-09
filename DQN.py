# -*- coding: utf-8 -*-
import logging as lg
lg.basicConfig(format='[%(levelname)s//%(filename)s//%(funcName)s//%(lineno)s] > %(message)s',
               level = lg.INFO)
import random
import numpy as np
from collections import deque
import tensorflow as tf

class DQNagent():
    def __init__(self, state_shape,
                       action_size,
                       main_network,
                       target_network,
                       batch_size = 50,
                       gamma = 0.95,
                       memory_len = 100000,
                       epsilon_init = 1,
                       epsilon_min = 0.1,
                       epsilon_decay = 0.99,
                       learning_rate = 0.01,
                       target_update_step = 50,
                       Double_Q = True
                       ):

        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=memory_len)
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon_init  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.Double_Q = Double_Q
        self.target_update_step = target_update_step

        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate = learning_rate)
        self.loss_function = tf.keras.losses.Huber()
        
        self.loss_metric = tf.keras.metrics.Mean(name='loss')
        self.avg_q_metric = tf.keras.metrics.Mean(name='avg_q')


        self.total_update_step = 0
        self.batch_size = batch_size

        self.main_network = main_network
        self.target_network = target_network
        self.update_target()
    
    def update_target(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        act = np.zeros(self.action_size)
        act[action] = 1
        
        self.memory.append((state, act, reward, next_state, done))

    def act(self, state, use_epsilon=True):
        if np.random.rand() <= self.epsilon and use_epsilon:
            return random.randrange(self.action_size)
        act_values = self.main_network.predict(np.expand_dims(state, axis=0))
        return np.argmax(act_values[0])  # returns action
    
    def get_batch(self):
        
        batch_size = self.batch_size

        if batch_size >= len(self.memory):
            minibatch = self.memory
        else:
            minibatch = random.sample(self.memory, batch_size)
            
        result = []
        for i in range(len(minibatch[0])):
            temp = []
            for experience in minibatch:
                temp.append(experience[i])
            result.append(np.array(temp))
        
        return result

    def train(self):
        states, actions, rewards, next_states, dones = self.get_batch()
        
        if self.Double_Q:
            indices = np.argmax(self.main_network.predict(next_states), axis=1)
            indices = np.expand_dims(indices, axis = 1)
            qvalues = self.target_network.predict(next_states)
            qvalues = np.take_along_axis(qvalues, indices, axis=1)
            qvalues = np.squeeze(qvalues, axis=1)
        else:
            qvalues = self.target_network.predict(next_states)
            qvalues = np.max(qvalues, axis=1)
            
        targets = rewards + dones * (self.gamma * qvalues)
            
            
        with tf.GradientTape() as tape:
            action_values = self.main_network(states)
            average_action_value = tf.reduce_mean(action_values)
            
            action_value = tf.reduce_sum(tf.multiply(action_values, actions), axis = 1)
            
            loss = self.loss_function(targets, action_value)
            
            
        gradients = tape.gradient(loss, self.main_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.main_network.trainable_variables))

        self.loss_metric(loss)
        self.avg_q_metric(average_action_value)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.total_update_step += 1
        if self.total_update_step%self.target_update_step==0:
            self.update_target()

        history = {"loss":self.loss_metric.result().numpy(),
                   "avg_q":self.avg_q_metric.result().numpy()}
        return history

from PER import Memory

class PDQNagent(Model.nn_model):
    def __init__(self, state_shape,
                       action_size,
                       main_network,
                       target_network,
                       batch_size = 50,
                       gamma = 0.95,
                       memory_len = 100000,
                       epsilon_init = 1,
                       epsilon_min = 0.1,
                       epsilon_decay = 0.99,
                       learning_rate = 0.01,
                       target_update_step = 500,
                       Double_Q = True,
                       PER_alpha = 0.6,
                       PER_beta = 0.4,
                       PER_beta_increment = 0.001,
                       ):

        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = Memory(memory_len,
                             alpha = PER_alpha,
                             beta = PER_beta,
                             beta_increment = PER_beta_increment)
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon_init  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.Double_Q = Double_Q
        self.target_update_step = target_update_step

        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate = learning_rate)
        self.loss_function = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
        
        self.loss_metric = tf.keras.metrics.Mean(name='loss')
        self.avg_q_metric = tf.keras.metrics.Mean(name='avg_q')


        self.total_update_step = 0
        self.batch_size = batch_size

        self.main_network = main_network
        self.target_network = target_network
        self.update_target()
    
    def update_target(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        act = np.zeros(self.action_size)
        act[action] = 1
        
        self.memory.store((state, act, reward, next_state, done))

    def act(self, state, use_epsilon=True):
        if np.random.rand() <= self.epsilon and use_epsilon:
            return random.randrange(self.action_size)
        act_values = self.main_network.predict(np.expand_dims(state, axis=0))
        return np.argmax(act_values[0])  # returns action
    
    def get_batch(self):
        
        batch_size = self.batch_size
        if batch_size > self.memory.data_num:
            batch_size = self.memory.data_num

        tree_indexs, datas, ISweights = self.memory.sample(batch_size)

        result = []
        for i in range(len(datas[0])):
            temp = []
            for experience in datas:
                temp.append(experience[i])
            result.append(np.array(temp))
        
        return result, tree_indexs, np.array(ISweights)

    def train(self):
        experiences, tree_indexs, ISweights = self.get_batch()
        
        states, actions, rewards, next_states, dones = experiences
        
        if self.Double_Q:
            indices = np.argmax(self.main_network.predict(next_states), axis=1)
            indices = np.expand_dims(indices, axis = 1)
            qvalues = self.target_network.predict(next_states)
            qvalues = np.take_along_axis(qvalues, indices, axis=1)
            qvalues = np.squeeze(qvalues, axis=1)
        else:
            qvalues = self.target_network.predict(next_states)
            qvalues = np.max(qvalues, axis=1)
            
        targets = rewards + dones * (self.gamma * qvalues)
            
            
        with tf.GradientTape() as tape:
            action_values = self.main_network(states)
            average_action_value = tf.reduce_mean(action_values)
            
            action_value = tf.reduce_sum(tf.multiply(action_values, actions), axis = 1)
            
            td_errors = self.loss_function(targets, action_value)
            loss = tf.reduce_mean(td_errors * ISweights)
            
        gradients = tape.gradient(loss, self.main_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.main_network.trainable_variables))
        
        #update priority
        priority = td_errors.numpy()
        self.memory.batch_update(tree_indexs, priority)

        self.loss_metric(loss)
        self.avg_q_metric(average_action_value)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.total_update_step += 1
        if self.total_update_step%self.target_update_step==0:
            self.update_target()

        history = {"loss":self.loss_metric.result().numpy(),
                   "avg_q":self.avg_q_metric.result().numpy()}
        return history



if __name__=="__main__":
    import gym
    import plotter
    from tensorflow.keras import models
    from tensorflow.keras.layers import Dense


    def network_builder(input_size, ouput_size):
        model = models.Sequential()
        model.add(Dense(16, "relu", input_shape=(input_size,)))
        model.add(Dense(32, "relu"))
        model.add(Dense(16, "relu"))
        model.add(Dense(ouput_size, "linear"))
        
        return model
    
    state_size = (4,)
    action_size = 2
    
    main = network_builder(state_size[0], action_size)
    target = network_builder(state_size[0], action_size)
    
    agent = PDQNagent(state_size,
                    action_size,
                    main,
                    target,
                    learning_rate = 0.001,
                    target_update_step=5000,
                    epsilon_decay = 0.99,
                    epsilon_min = 0.1,
                    batch_size=128,
                    memory_len=50000)
    
    env = gym.make('CartPole-v1')
    env._max_episode_steps = 10001

    history = {"step" : 0,
               "reward" : [],
               "loss" : [0],
               "avg_q" : [0]}
    
    prints = ["reward", "loss", "avg_q"]
    
    

    steps = 30000
    plot_rate = 1000
    train_start = 10000
    
    done = False
    total_reward = 0
    state = env.reset()
    
    for step in range(1, steps+1):
    
    
        action = agent.act(state)
        new_state, reward, done, _ = env.step(action)
        if done:
            reward = -10
    
        agent.memorize(state, action, reward, new_state, done)
    
        total_reward += reward
        state = new_state
        
        history["step"] += 1
    
        if history["step"] > train_start:
            hist = agent.train()
            loss = round(hist["loss"], 3)
            avg_q = round(hist["avg_q"], 3)
    
            history["loss"].append(loss)
            history["avg_q"].append(avg_q)
        
            
            if history["step"]%plot_rate==0:
                for met in prints:
                    plotter.plotWithSmooth(history[met], met, dark=True)
                    
        if done:
            
            print("step :", history["step"],
                  "  current reward :", total_reward,
                  "  loss :", history["loss"][-1],
                  "  avg_q :", history["avg_q"][-1],
                  "  epsilon", round(agent.epsilon, 3))
            
            state = env.reset()
            done = False
            history["reward"].append(total_reward)
            total_reward = 0                    
                    
