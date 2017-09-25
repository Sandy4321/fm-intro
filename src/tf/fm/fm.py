
'''
Define a vanilla Factorization machine
'''


import numpy as np
import tensorflow as tf
import math

LATENT_FACTORS = 5


# Factorization Machine
def model(input_layer, latent_factors, num_feature):

	# Get shape of data
	p = num_feature

	# 1st order weights
	with tf.name_scope('single_order'):
		weights = tf.Variable(tf.zeros([p]), name='weights')
		biases  = tf.Variable(tf.zeros([1]), name='biases')

	# Factorizations
	with tf.name_scope('factorization'):
		factors = tf.Variable(
			tf.random_normal([latent_factors, p], stddev=1.0 / math.sqrt(float(p))),
			name='factors')

	# Linear Regression terms
	linear_terms = tf.add(biases, tf.reduce_sum(tf.multiply(weights, input_layer), 1, keep_dims=True))

	print(linear_terms)

	# Factorization interactions
	interaction_terms = (tf.multiply(
		0.5,
		# Variance
		tf.reduce_sum(
			tf.subtract(
				tf.pow(tf.matmul(input_layer, tf.transpose(factors)), 2),
				tf.matmul(tf.pow(input_layer, 2), tf.transpose(tf.pow(factors, 2)))),
			axis=1, keep_dims=True)))

	# Complete inference
	y_hat = tf.add(linear_terms, interaction_terms)
	return y_hat


# Loss
def loss(logits, labels):

	# # Get variables from model
	# weights = [v for v in tf.global_variables() if v.name == "single_order/weights:0"]
	# factors = [v for v in tf.global_variables() if v.name == "factorization/factors:0"]

	# # # Regularization: L1 L2
	# lambda_w = tf.constant(0.001, name='lambda_w')
	# lambda_v = tf.constant(0.001, name='lambda_v')
	# l2_norm = (tf.reduce_sum(
 #            tf.add(
 #                tf.multiply(lambda_w, tf.pow(weights, 2)),
 #                tf.multiply(lambda_v, tf.pow(factors, 2)))))

	# labels = tf.to_int64(labels)
	# cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
	# 	labels=labels, logits=logits, name='xentropy')

	# error = tf.reduce_mean(cross_entropy, name='xentropy_mean')

	# loss = tf.add(error, l2_norm)
	# return loss

	labels = tf.to_int64(labels)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels=labels, logits=logits, name='xentropy')
	loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
	return loss


def training(loss, learning_rate):
	"""Setup Training

	Return:
		The training op
	"""
	# Set the summary snapshot loss
	tf.summary.scalar('loss', loss)
	# Create gradient descent
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	# Create a variable to track steps
	global_step = tf.Variable(0, name='global_step', trainable=False)
	# Use Optimizer to apply the gradients that minimize loss
	train_op = optimizer.minimize(loss, global_step=global_step)
	return train_op


def evaluation(logits, labels):
	"""Evaluate given labels for snapshot and diagnostics

	Args:
		logits: Logits tensors, float - [batch_size, NUM_CLASSES]

	Return:
		A scalar int32 tensor with batch_size
	"""


	# logits = np.squeeze(logits)
	# labels = np.squeeze(labels)
	print('================ logits')
	print(logits)

	print('================ labels')
	print(type(labels))
	print(labels)

	correct = tf.nn.in_top_k(logits, labels, 1)
	# Returnthe number of true entries
	return tf.reduce_sum(tf.cast(correct, tf.int32))

