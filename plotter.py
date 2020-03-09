# -*- coding: utf-8 -*-
import logging as lg
lg.basicConfig(format='[%(levelname)s//%(filename)s//%(funcName)s//%(lineno)s] > %(message)s',
               level = lg.INFO)


import matplotlib.pyplot as plt
import copy


def plotWithSmooth(array_, title, dark=False):
    if dark:
        plt.style.use(['dark_background'])

    array = copy.deepcopy(array_)
    if len(array) < 5000:
        smooth_space = 20
    else:
        smooth_space = 50
    x = [i for i in range(0, len(array))]
    y = array
    # smooth curve
    x_ = []
    y_ = []
    for i in range(0, len(x)):
        x_.append(x[i])
        if i+smooth_space >= len(x):
            y_.append(sum(y[i-smooth_space:i]) / float(smooth_space))
        else:
            y_.append(sum(y[i:i+smooth_space]) / float(smooth_space))
    
    plt.title(title)
    plt.plot(x, y, ('#ffc9ba'))
    if len(array) > 100:
        plt.plot(x_, y_, color='r', linewidth=1)
    plt.show()
