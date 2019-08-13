from __future__ import print_function,division
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length


def genData():
    x = np.array(np.random.choice(2,total_series_length, p =[0.5,0.5])
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0
    x = x.reshape((batch_size, -1))
    y = y.reshape((batch_size, -1))
    return (x,y)

