import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/")

# Begin constructing the DNN!

# First, specify inputs
n_inputs = 28*28
n_hidden = 400
n_outputs = 10

# Then, specify placeholders
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')

# Create the DNN!
with tf.name_scope('DNN_V2'):
    hidden1 = fully_connected(X, n_hidden, scope='hidden1')
    hidden2 = fully_connected(hidden1, n_hidden, scope='hidden2')
    hidden3 = fully_connected(hidden2, n_hidden, scope='hidden3')
    hidden4 = fully_connected(hidden3, n_hidden, scope='hidden4')
    hidden5 = fully_connected(hidden4, n_hidden, scope='hidden5')
    logits = fully_connected(hidden5, n_outputs, scope='outputs', activation_fn=None)

# Create the loss function...
with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')

learning_rate = 0.01
# Optimizer
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Run the thangs!!!
n_epochs = 400
batch_size = 64

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images,
                                            y: mnist.test.labels})
        
        print(epoch,  'Train accuracy:', acc_train, 'Test accuracy:', acc_test)
        
    save_path = saver.save(sess, "./DNN_MNIST.ckpt")

