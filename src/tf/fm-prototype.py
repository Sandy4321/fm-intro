'''
Attempt to make a Factorization Machine in TensorFlow
Blog: http://nowave.it/factorization-machines-with-tensorflow.html
'''


import argparse
import sys
import tempfile
import os

import tensorflow as tf
import numpy as np
#import pandas as pd

import load_data

graph_location = tempfile.mkdtemp()


# Load data and get dense matrix form
data = load_data.sparse_to_dense(
    path="../../data/ml-100k", 
    filename="u.data", 
    header=['user_id','item_id','rating','timestamp'],
    cols=['user_id','item_id','rating'])

print(data.filepath)
print(data.data.head())
print(data.data_dense.head())
print(data.data_dense.shape)


# Data All
x_data, y_data = data.data_dense_mat


#=== Input Data ===================================================================================

# Shape
n, p = x_data.shape
print(n, p)


# Latent Factors
k = 5

# Feature matrix and target
X = tf.placeholder(tf.float32, shape=[n, p], name='X_train')
y = tf.placeholder(tf.float32, shape=[n, 1], name='y_train')

# Weights
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.zeros([p]))

# Interaction Variables
V = tf.Variable(tf.random_normal([k, p], stddev=0.01))

# Prediction
y_hat = tf.Variable(tf.zeros([n, 1]))


#=== Model ========================================================================================

# Linear Terms
linear_terms = tf.add(b, tf.reduce_sum(tf.multiply(W, X), 1, keep_dims=True))

# Factorization Interactions
interactions = (tf.multiply(0.5,
	tf.reduce_sum(tf.subtract(
		tf.pow(tf.matmul(X, tf.transpose(V)), 2),
		tf.matmul(tf.pow(X, 2), tf.transpose(tf.pow(V, 2)))),
	1, keep_dims=True)))

# Factorization Machine
y_hat = tf.add(linear_terms, interactions)



#=== Regularization ===============================================================================

# L1 L2
lambda_w = tf.constant(0.001, name='lambda_w')
lambda_v = tf.constant(0.001, name='lambda_v')

# L2
l2_norm = (tf.reduce_sum(
            tf.add(
                tf.multiply(lambda_w, tf.pow(W, 2)),
                tf.multiply(lambda_v, tf.pow(V, 2)))))

# Loss
error = tf.reduce_mean(tf.square(tf.subtract(y, y_hat)))
loss = tf.add(error, l2_norm)


#=== Optimizer ====================================================================================
eta = tf.constant(0.1)
optimizer = tf.train.AdagradOptimizer(eta).minimize(loss)


#=== Train ========================================================================================
# that's a lot of iterations
N_EPOCHS = 100
# Launch the graph.
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print("======== START TRAINING ======")
    for epoch in range(N_EPOCHS):
        indices = np.arange(n)
        np.random.shuffle(indices)
        x_data, y_data = x_data[indices], y_data[indices]
        sess.run(optimizer, feed_dict={X: x_data, y: y_data})

        # Logging
        if epoch % 10 == 0:
        	print('Epoch: ', epoch)
        	print('\tMSE: ', sess.run(error, feed_dict={X: x_data, y: y_data}))
        	print('\tLoss (regularized error):', sess.run(loss, feed_dict={X: x_data, y: y_data}))


    print("================ FINAL ==============")
    print('MSE: ', sess.run(error, feed_dict={X: x_data, y: y_data}))
    print('Loss (regularized error):', sess.run(loss, feed_dict={X: x_data, y: y_data}))
    print('Predictions:', sess.run(y_hat, feed_dict={X: x_data, y: y_data}))
    print('Learnt weights:', sess.run(W, feed_dict={X: x_data, y: y_data}))
    print('Learnt factors:', sess.run(V, feed_dict={X: x_data, y: y_data}))
