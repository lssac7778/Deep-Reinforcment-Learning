# -*- coding: utf-8 -*-
import logging as lg
lg.basicConfig(format='[%(levelname)s//%(filename)s//%(funcName)s//%(lineno)s] > %(message)s',
               level = lg.INFO)

import numpy as np

class SumTree:

    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)

        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):

        tree_index = self.data_pointer + self.capacity - 1

        self.data[self.data_pointer] = data

        self.update(tree_index, priority)

        self.data_pointer += 1

        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
    
    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        while tree_index != 0:
            tree_index = (tree_index - 1)//2
            self.tree[tree_index] += change
    
    def get_leaf(self, value):

        parent_index = 0
        leaf_index = -1

        while True:

            left_index = parent_index * 2 + 1
            right_index = left_index + 1

            if left_index >= len(self.tree):
                leaf_index = parent_index
                break

            if value <= self.tree[left_index]:
                parent_index = left_index
            
            else:
                value -= self.tree[left_index]
                parent_index = right_index
        
        data_index = leaf_index - self.capacity + 1
        
        if type(self.data[data_index])==int:
            lg.info("leaf_index : " + str(leaf_index))
            lg.info("self.tree[leaf_index] : " + str(self.tree[leaf_index]))
            lg.info("self.data[data_index] : " + str(self.data[data_index]))
            
            lg.info("value : " + str(value))
            lg.info("self.tree[0] : " + str(self.tree[0]))

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]

class Memory:

    def __init__(self, 
                capacity,
                alpha = 0.6,
                beta = 0.4,
                beta_increment = 0.001,
                error_upper_bound = 1.,
                epsilon = 0.01):

        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.error_upper_bound = error_upper_bound
        self.epsilon = epsilon
        self.data_num = 0

        self.tree = SumTree(capacity)
    
    def store(self, experience, priority=None):
        
        if priority==None:
            max_priority = np.max(self.tree.tree[-self.tree.capacity:])
            if max_priority == 0:
                max_priority = self.error_upper_bound
            self.tree.add(max_priority, experience)
        else:
            priority += self.epsilon
            self.tree.add(priority, experience)
        
        
        if self.data_num < self.capacity:
            self.data_num += 1
    
    def sample(self, batch_size):
        sample_range_distance = self.tree.total_priority / batch_size

        self.beta = np.min([1., self.beta + self.beta_increment])

        avail_start = -self.tree.capacity
        avail_end = avail_start + self.data_num
        
        if avail_end < 0:
            avail_values = self.tree.tree[avail_start:avail_end]
        else:
            avail_values = self.tree.tree[avail_start:]
        
        min_priority = np.min(avail_values) / self.tree.total_priority
        max_ISweight = (min_priority * batch_size) ** (-self.beta)
        
        lg.debug("total_priority : " + str(self.tree.total_priority))
        lg.debug("min_priority : " + str(min_priority))

        
        index_array = []
        data_array = []
        ISweight_array = []

        for i in range(batch_size):
            a, b = sample_range_distance * i, sample_range_distance * (i + 1)
            value = np.random.uniform(a, b)

            index, priority, data = self.tree.get_leaf(value)

            samling_prob = priority / self.tree.total_priority
            lg.debug("batch_size : " + str(batch_size))
            lg.debug("samling_prob : " + str(samling_prob))
            lg.debug("max_ISweight : " + str(max_ISweight))
            ISweight = ( (batch_size * samling_prob) ** (-self.beta) )/ max_ISweight
            
            index_array.append(index)
            data_array.append(data)
            ISweight_array.append(ISweight)
        
        return index_array, data_array, ISweight_array
    
    def batch_update(self, tree_indexs, abs_errors):
        abs_errors += self.epsilon
        #clipped_errors = np.minimum(abs_errors, self.error_upper_bound)
        powered_errors = np.power(abs_errors, self.alpha)
        
        for idx, err in zip(tree_indexs, powered_errors):
            self.tree.update(idx, err)

if __name__=="__main__":
    import matplotlib.pyplot as plt
    
    memory = Memory(100000)
    for i in range(1, 10):
        for _ in range(100):
            memory.tree.add(i, i)
    
    _, datas, _ = memory.sample(100)
    
    plt.hist(datas)
    plt.show()
