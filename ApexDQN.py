# -*- coding: utf-8 -*-
import logging as lg
lg.basicConfig(format='[%(levelname)s//%(filename)s//%(funcName)s//%(lineno)s] > %(message)s',
               level = lg.INFO)
import random
import numpy as np
import tensorflow as tf
import threading
import time

import plotter
from PER import Memory


class ApexDQN():
    def __init__(self, state_shape,
                       action_size,
                       network_builder,
                       env_class,
                       actor_num = 8,
                       batch_size = 50,
                       gamma = 0.95,
                       memory_len = 100000,
                       epsilon_max = 0.4,
                       epsilon_alpha = 7,
                       learning_rate = 0.001,
                       target_update_step = 50,
                       Double_Q = True,
                       train_start_step = 10000,
                       local_network_update_step = 50,
                       n_step = 3,
                       ):

        self.state_shape = state_shape
        self.action_size = action_size
        self.env_class = env_class
        self.memory = Memory(memory_len)
        self.gamma = gamma    # discount rate
        self.epsilon_max = epsilon_max  # exploration rate
        self.epsilon_alpha = epsilon_alpha
        self.learning_rate = learning_rate
        self.Double_Q = Double_Q
        self.target_update_step = target_update_step
        self.network_builder = network_builder
        self.actor_num = actor_num
        self.train_start_step = train_start_step
        self.local_network_update_step = local_network_update_step
        self.n_step = n_step
        
        self.history = {"reward":[], "loss":[], "avg_q":[], "step":0}
        self.end_training = [False]

        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate = learning_rate)
        self.loss_function = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
        
        self.loss_metric = tf.keras.metrics.Mean(name='loss')
        self.avg_q_metric = tf.keras.metrics.Mean(name='avg_q')


        self.total_update_step = 0
        self.batch_size = batch_size

        self.main_network = network_builder()
        self.target_network = network_builder()
        self.update_target()
        
        self.generate_local_networks()
        
        self.device = '/gpu:0'
        
    def generate_local_networks(self):
        local_networks = []
        for _ in range(self.actor_num):
            local_networks.append((self.network_builder(), self.network_builder()))
        
        self.local_networks = local_networks
    
    def update_target(self):
        self.target_network.set_weights(self.main_network.get_weights())
    
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
        with tf.device(self.device):
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
    
            self.total_update_step += 1
            if self.total_update_step%self.target_update_step==0:
                self.update_target()
    
            history = {"loss":self.loss_metric.result().numpy(),
                       "avg_q":self.avg_q_metric.result().numpy()}
        return history
    
    def main(self, train_until, plot_every):
        self.end_training[0] = False
        
        history = self.history
        plot_metrics = ["reward", "loss", "avg_q"]
        
        print("generating actors")
        
        actors = []
        for i in range(self.actor_num):
            
            epsilon = self.epsilon_max ** (1 + i/(self.actor_num - 1)*self.epsilon_alpha)
            
            actors.append(Actor(
                            self.state_shape,
                            self.action_size,
                            self.local_networks[i][0],
                            self.local_networks[i][1],
                            self.main_network,
                            self.target_network,
                            self.memory,
                            epsilon,
                            self.Double_Q,
                            self.loss_function,
                            self.end_training,
                            self.gamma,
                            self.env_class(),
                            history,
                            self.local_network_update_step,
                            self.n_step
                            ))
        
        print("start acting for {} steps".format(self.train_start_step))
        for actor in actors:
            actor.start()
        
        
        
        while self.history["step"] < self.train_start_step:
            time.sleep(1)
            print("step :",self.history["step"])
        
        
        
        print("start training")
        
        metrics = ["loss", "avg_q"]
        
        episode = 0
        train_num = 0
        
        while train_num <= train_until:
            episode = len(history["reward"])
            
            
            hist = self.train()
            train_num += 1
            
            for met in metrics:
                history[met].append(hist[met])
            
            if train_num%plot_every==0:
                for met in plot_metrics:
                    plotter.plotWithSmooth(history[met], met)
            
            print("episodes :", episode,
                  "  train_num :",train_num,
                  "  memory len :",self.memory.data_num)
            
        
        self.end_training[0] = True
        for actor in actors:
            actor.join()
    
class Actor(threading.Thread):
    
    def __init__(self, state_shape,
                       action_size,
                       local_main_network,
                       local_target_network,
                       global_main_network,
                       global_target_network,
                       global_memory,
                       epsilon,
                       Double_Q,
                       loss_function,
                       end_training,
                       gamma,
                       env,
                       history,
                       network_update_step,
                       n_step
                       ):
    
        threading.Thread.__init__(self)
        
        self.state_shape = state_shape
        self.action_size = action_size
        
        self.local_main_network = local_main_network
        self.local_target_network = local_target_network
        # self.local_main_network = global_main_network
        # self.local_target_network = global_target_network
        
        self.global_main_network = global_main_network
        self.global_target_network = global_target_network
        self.global_memory = global_memory
        
        self.Double_Q = Double_Q
        self.epsilon = epsilon
        
        self.loss_function = loss_function
        self.end_training = end_training
        self.gamma = gamma
        self.env = env
        self.history = history
        
        self.network_update_step = network_update_step
        self.n_step = n_step
        
        self.update_local_network()
        
        self.device = '/cpu:0'
    
    def act(self, state, use_epsilon=True):
        if np.random.rand() <= self.epsilon and use_epsilon:
            return random.randrange(self.action_size)
        with tf.device(self.device):
            act_values = self.local_main_network.predict(np.expand_dims(state, axis=0))
        return np.argmax(act_values[0])  # returns action
    
    def update_local_network(self):
        self.local_main_network.set_weights(self.global_main_network.get_weights())
        self.local_target_network.set_weights(self.global_target_network.get_weights())
    
    def discount_rewards(self, rewards):
        discounted = []
        before = 0
        for r in reversed(rewards):
            before = r + self.gamma * before
            discounted.append(before)
        return sum(discounted)
    
    def update_global_memory(self, state, action_num, rewards, next_state, done):
        
        with tf.device(self.device):
            #compute priority
            action = np.zeros(self.action_size)
            action[action_num] = 1
            
            one_next_state = np.expand_dims(next_state, axis=0)
            one_state = np.expand_dims(state, axis=0)
            reward = self.discount_rewards(rewards)
            
            if self.Double_Q:
                
                index = self.local_main_network.predict(one_next_state)[0]
                index = np.argmax(index)
                
                qvalues = self.local_target_network.predict(one_next_state)[0]
                qvalue = qvalues[index]
                
            else:
                qvalues = self.local_target_network.predict(one_next_state)[0]
                qvalue = np.max(qvalues)
                
            target = reward + done * (self.gamma * qvalue)
            
            action_values = self.local_main_network.predict(one_state)[0]
            action_value = np.sum(action_values * action)
            
            priority = self.loss_function(target, action_value).numpy()
            
            #store at memory
            data = (state, action, reward, next_state, done)
            self.global_memory.store(data, priority = priority)
    
    def run(self):
        
        rewards = []
        score = 0
        total_step = 0
        state = self.env.reset()
        
        while not self.end_training[0]:
            action = self.act(state)
            next_state, reward, done = self.env.step(action)
            
            score += reward
            rewards.append(reward)
            total_step += 1
            self.history["step"] += 1
            
            if total_step%self.network_update_step==0:
                self.update_local_network()
            
            if total_step%self.n_step==0:
                self.update_global_memory(state, action, rewards, next_state, done)
                rewards = []
            
            if done:
                self.update_global_memory(state, action, rewards, next_state, done)
                self.history["reward"].append(score)
                
                rewards = []
                score = 0
                
                state = self.env.reset()
            else:
                state = next_state
        print("end thread")



    
if __name__=="__main__":

    from tensorflow.keras import models
    from tensorflow.keras.layers import Dense
    from tensorflow.keras import layers
    from myenvs import pong

    env_class = pong
    
    def network_builder():
        action_size = env_class.action_size
        state_shape = env_class.state_shape
        
        model = models.Sequential()
        model.add(layers.Conv2D(16, (3, 3), input_shape = state_shape, activation = "relu"))
        model.add(layers.Conv2D(32, (3, 3), activation = "relu"))
        model.add(layers.Conv2D(64, (3, 3), activation = "relu"))
        model.add(layers.Flatten())
        model.add(Dense(64, "relu"))
        model.add(Dense(32, "relu"))
        model.add(Dense(action_size, "linear"))
        return model
    
    
    agent = ApexDQN(env_class.state_shape,
                   env_class.action_size,
                   network_builder,
                   env_class,
                   actor_num = 8,
                   batch_size = 32,
                   gamma = 0.99,
                   memory_len = 1000,
                   epsilon_max = 0.4,
                   epsilon_alpha = 7,
                   learning_rate = 0.001,
                   target_update_step = 1000,
                   Double_Q = True,
                   train_start_step = 1000,
                   local_network_update_step = 10,
                   n_step = 5
                   )
    
    
    train_until = 3000
    plot_every = 25
    
    agent.main(train_until, plot_every)
    
