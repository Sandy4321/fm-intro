

'''
Here we define the model to be used by "train.py"
'''

import tensorflow as tf
import math

# Target labels
NUM_CLASSES = 10

# Feature size
NUM_FEATURE = 28 * 28



def inference(data, hidden_1_units, hidden_2_units):
	"""Define the model
	
	Args:
		data: Placeholder from inputs()
		hidden_1_units: layer 1
		hidden_2_units: layer 2

	Return:
		softmax_linear: Output logits
	"""

	# Layer 1
	with tf.name_scope('hidden_1'):
		weights = tf.Variable(
			tf.truncated_normal([NUM_FEATURE, hidden_1_units],
				stddev=1.0 / math.sqrt(float(NUM_FEATURE))),
			name='weights')
		biases = tf.Variable(
			tf.zeros([hidden_1_units]),
			name='biases')
		hidden_1 = tf.nn.relu(tf.matmul(data, weights) + biases)
		
	# Layer 2
	with tf.name_scope('hidden_2'):
		weights = tf.Variable(
			tf.truncated_normal([hidden_1_units, hidden_2_units],
				stddev=1.0 / math.sqrt(float(hidden_1_units))),
			name='weights')
		biases = tf.Variable(
			tf.zeros([hidden_2_units]),
			name='biases')
		hidden_2 = tf.nn.relu(tf.matmul(hidden_1, weights) + biases)

	# Dense
	with tf.name_scope('softmax_layer'):
		weights = tf.Variable(
			tf.truncated_normal([hidden_2_units, NUM_CLASSES],
				stddev=1.0 / math.sqrt(float(hidden_2_units))),
			name='weights')
		biases = tf.Variable(
			tf.zeros([NUM_CLASSES]),
			name='biases')
		logits = tf.matmul(hidden_2, weights) + biases
	return logits


def loss(logits, labels):
	"""Get the loss from the logits

	Args:
		logits: get return from inference()
		labels: actual labels

	Return:
		loss: Loss tensor of type float

	"""
	labels = tf.to_int64(labels)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels=labels, logits=logits, name='xentropy')
	return tf.reduce_mean(cross_entropy, name='xentropy_mean')


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
	correct = tf.nn.in_top_k(logits, labels, 1)
	# Returnthe number of true entries
	return tf.reduce_sum(tf.cast(correct, tf.int32))

















