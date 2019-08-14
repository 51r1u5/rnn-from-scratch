from __future__ import print_function, division
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

#Initializing params
epochs = 100
series_length = 100000
bptt = 15
state_size = 6
batch_size = 10
echo_step = 4
num_classes = 2
num_batches = series_length//batch_size//bptt

#Data Generation
def genData():
    x = np.random.choice(2,series_length,p=[0.5,0.5])
    y = np.roll(x,4)
    y[0:echo_step] = 0
    x = x.reshape((batch_size, -1))
    y = y.reshape((batch_size, -1))
    return (x,y)
#ploting loss
def plot(loss_list):
    plt.subplot(2,3,1)
    plt.cla()
    plt.plot(loss_list)

    plt.draw()
    plt.pause(0.0001)

#Initializing placeholders
init_state = tf.placeholder(tf.float32, [batch_size,state_size])
Xbatch_placeholder = tf.placeholder(tf.float32, [batch_size, bptt])
Ybatch_placeholder = tf.placeholder(tf.int32, [batch_size, bptt])

#init variables
W = tf.Variable(np.random.rand(state_size+1, state_size), dtype = tf.float32)
b = tf.Variable(np.zeros((1, state_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(state_size, num_classes), dtype=tf.float32)
b2 = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float32)

#train step
input_series = tf.split(axis = 1, num_or_size_splits = bptt, value= Xbatch_placeholder)
labels_series = tf.unstack(Ybatch_placeholder, axis=1)

#forward prop
cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
states_series, current_state = tf.contrib.rnn.static_rnn(cell, input_series, init_state)

logits_series = [tf.matmul(state, W2) + b2 for state in states_series]
prediction_series = [tf.nn.softmax(logits) for logits in logits_series]

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits =logits, labels=labels) for logits, labels in zip(logits_series, labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(epochs):
        x, y = genData()
        _current_state = np.zeros((batch_size, state_size))

        print("New data and epoch = ", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx*bptt
            end_idx = start_idx+bptt

            batchX = x[:,start_idx:end_idx]
            batchY = y[:, start_idx:end_idx]

            _total_loss, _train_step, _current_state, _predictions_series = \
                sess.run([total_loss, train_step, current_state, prediction_series],\
                    feed_dict ={\
                        Xbatch_placeholder:batchX, Ybatch_placeholder:batchY, init_state:_current_state})

            loss_list.append(_total_loss)

            if batch_idx%100 == 0:
                print("Step", batch_idx, "Loss", _total_loss)
                plot(loss_list)
            
plt.ioff()
plt.show()
        