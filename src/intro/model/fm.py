'''
Vanilla Factorization Machines

'''

import math
import os
import argparse
from time import time

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, accuracy_score
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


class FM(BaseEstimator, TransformerMixin):

	def __init__(self, features_p, factor_k, col_m,
				 lamda_bilinear, dropout_keep_rate, 
				 epoch, batch_size, learning_rate, optimizer_type, batch_norm, save_file,
				 self_terminate,
				 pretrain_flag, verbose, seed=1337, **kwargs):
		self.name = "Factorization Machine"
		self.features_p = features_p
		self.factor_k = factor_k
		self.col_m = col_m

		self.lamda_bilinear = lamda_bilinear
		self.dropout_keep_rate = dropout_keep_rate

		self.epoch = epoch
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.optimizer_type = optimizer_type
		self.batch_norm = batch_norm
		self.save_file = save_file
		self.self_terminate = self_terminate

		self.pretrain_flag = pretrain_flag
		self.verbose = verbose
		self.seed = seed

		# performance of each epoch
		self.train_rmse, self.valid_rmse, self.test_rmse = [], [], []

		# initialize graph
		self._init_graph()


	#==============================================================================================
	# Model
	#==============================================================================================
	def _init_graph(self):
		'''
		Initialize graph and define the model
		'''
		self.graph = tf.Graph()
		with self.graph.as_default():

			# Set Seed
			tf.set_random_seed(self.seed)

			# Input Data
			self.train_features = tf.placeholder(tf.int32, shape=[None, None], name='train_features_fm')  # None * features_p
			self.train_labels   = tf.placeholder(tf.float32, shape=[None, 1], name='train_labels_fm')  # None * 1
			self.dropout_keep   = tf.placeholder(tf.float32, name='dropout_keep_fm')
			self.train_phase    = tf.placeholder(tf.bool, name='train_phase_fm')

			# Variables
			self.weights = self._init_weights()

			#=== Model
			# get the summed up embeddings of the features
			# we do a lookup because the features are categorical/long format
			self.nonzero_embeddings = tf.nn.embedding_lookup(
				params=self.weights['feature_embeddings'],
				ids=self.train_features,
				name='nonzero_embeddings')
			self.summed_features_embeddings = tf.reduce_sum(
				self.nonzero_embeddings, 1, keep_dims=True)  # None * 1 * k

			# get the element-multiplication factors
			self.summed_feature_embeddings_square = tf.square(self.summed_features_embeddings)

			# Sum Squared
			self.squared_feature_embeddings = tf.square(self.nonzero_embeddings)
			self.squared_sum_feature_embeddings = tf.reduce_sum(self.squared_feature_embeddings, 1, keep_dims=True)  # None * 1 * k

			# Factorization Machine (FM)
			self.FM = 0.5 * tf.subtract(self.summed_feature_embeddings_square, self.squared_sum_feature_embeddings, name='fm')

			# The data will be in long format with m columns. 
			# We normalize to make sure the sum of weights is 1
			# self.FM = self.FM / self.col_m
			self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase, scope_bn='bn_fm')
			self.FM_OUT = tf.reduce_sum(self.FM, 1, name='fm_out')

			# Dropout Regularizer
			self.FM_OUT = tf.nn.dropout(self.FM_OUT, self.dropout_keep)

			# Output Layer
			Bilinear = tf.reduce_sum(self.FM_OUT, 1, keep_dims=True)  # None * 1
			self.Feature_bias = tf.reduce_sum(tf.nn.embedding_lookup(
				params=self.weights['feature_bias'],
				ids=self.train_features), 1)
			Bias = self.weights['bias'] * tf.ones_like(self.train_labels)  # None * 1
			self.out = tf.add_n([Bilinear, self.Feature_bias, Bias], name='out')  # None * 1

			#=== Loss
			# L2 Regularizaer
			if self.lamda_bilinear > 0:
				self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out)) + tf.contrib.layers.l2_regularizer(self.lamda_bilinear)(self.weights['feature_embeddings'])  # regulizer
			else:
				self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out))
			
			# Set the summary snapshot loss
			self.summary_train = tf.summary.scalar('loss_train', self.loss)
			self.summary_valid = tf.summary.scalar('loss_valid', self.loss)


			#=== Optimizer
			# Select Optimizer
			if self.optimizer_type == 'AdamOptimizer':
				optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
			elif self.optimizer_type == 'AdagradOptimizer':
				optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8)
			elif self.optimizer_type == 'GradientDescentOptimizer':
				optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
			elif self.optimizer_type == 'MomentumOptimizer':
				optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95)

			# Create a variable to track steps
			global_step = tf.Variable(0, name='global_step', trainable=False)

			# Minimize
			self.optimizer = optimizer.minimize(self.loss, global_step=global_step)

			#=== Initialize
			self.sess = self._init_session()
			self.saver = tf.train.Saver()
			init = tf.global_variables_initializer()
			self.sess.run(init)

			#=== Summary
			# Get Summary Tensor
			# self.summary = tf.summary.merge_all()

			# Instantiate a SummaryWriter to output summaries and the Graph.
			self.summary_writer = tf.summary.FileWriter(self.save_file, self.sess.graph)

			#=== Print
			if self.verbose > 0:
				# Number of parameters
				total_parameters = 0
				for variable in self.weights.values():
					shape = variable.get_shape() # shape is an array of tf.Dimension
					variable_parameters = 1
					for dim in shape:
						variable_parameters *= dim.value
					total_parameters += variable_parameters
				print("#params: %d" %total_parameters)


	def _init_session(self):
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		return tf.Session(config = config)


	def _init_weights(self):
		# Store all weights
		all_weights = dict()

		all_weights['feature_embeddings'] = tf.Variable(
			tf.random_normal([self.features_p, self.factor_k], stddev=0.01),
			name='feature_embeddings')  # features_p * factor_k
		all_weights['feature_bias'] = tf.Variable(
			tf.random_uniform([self.features_p, 1], 0.0, 0.0),
			name='feature_bias')  # features_p * 1
		all_weights['bias'] = tf.Variable(
			tf.constant(0.0),
			name='bias')  # 1 * 1
		return all_weights


	def batch_norm_layer(self, x, train_phase, scope_bn):
		bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
			is_training=True, reuse=None, trainable=True, scope=scope_bn)
		bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
			is_training=False, reuse=True, trainable=True, scope=scope_bn)
		z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
		return z



	def variable_summaries(var):
		"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean', mean)
			with tf.name_scope('stddev'):
				stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			tf.summary.scalar('stddev', stddev)
			tf.summary.scalar('max', tf.reduce_max(var))
			tf.summary.scalar('min', tf.reduce_min(var))
			tf.summary.histogram('histogram', var)

	#==============================================================================================
	# Evaluation
	#==============================================================================================
	def train(self, data_train, data_valid, data_test):
		'''
		Fit a dataset
		'''


		# # Report
		# if self.verbose:
		# 	t2 = time()
		# 	init_train = self.evaluate(data_train, epoch=None, data_type='train')
		# 	init_valid = self.evaluate(data_valid, epoch=None, data_type='valid')
		# 	print("Init: \t train=%.4f, validation=%.4f [%.1f s]" %(init_train, init_valid, time()-t2))

		# Training Epochs
		for epoch in range(self.epoch):

			# Form Batch
			t1 = time()

			# Shuffle data
			self.shuffle_data(data_train['X'], data_train['Y'])
			total_batch = int(len(data_train['Y']) / self.batch_size)

			# Batches
			for i in range(total_batch):
				batch_xs = self.get_random_block_from_data(data_train, self.batch_size)
				# Fit training
				self.partial_fit(batch_xs)

			t2 = time()

			# output validation
			train_result = self.evaluate(data_train, epoch, data_type='train')
			valid_result = self.evaluate(data_valid, epoch, data_type='valid')

			self.train_rmse.append(train_result)
			self.valid_rmse.append(valid_result)

			# Print
			if self.verbose > 0 and epoch%self.verbose == 0:
				print("Epoch %d [%.1f s]\ttrain=%.4f, validation=%.4f [%.1f s]"
					  %(epoch+1, t2-t1, train_result, valid_result, time()-t2))
			if self.self_terminate and self.evaluate_termination(self.valid_rmse):
				break


		if self.pretrain_flag < 0:
			print("Save model to file as pretrain.")
			self.saver.save(self.sess, self.save_file)

	def shuffle_data(self, X, y):
		'''
		Randomizes data
		'''
		current_state = np.random.get_state()
		np.random.shuffle(X)
		np.random.set_state(current_state)
		np.random.shuffle(y)


	def get_random_block_from_data(self, data, batch_size):  # generate a random block of training data
		start_index = np.random.randint(0, len(data['Y']) - batch_size)
		X , Y = [], []
		# forward get sample
		i = start_index
		while len(X) < batch_size and i < len(data['X']):
			if len(data['X'][i]) == len(data['X'][start_index]):
				Y.append([data['Y'][i]])
				X.append(data['X'][i])
				i = i + 1
			else:
				break
		# backward get sample
		i = start_index
		while len(X) < batch_size and i >= 0:
			if len(data['X'][i]) == len(data['X'][start_index]):
				Y.append([data['Y'][i]])
				X.append(data['X'][i])
				i = i - 1
			else:
				break
		return {'X': X, 'Y': Y}

		
	def partial_fit(self, data):  # fit a batch
		feed_dict = {
			self.train_features: data['X'], 
			self.train_labels: data['Y'], 
			self.dropout_keep: self.dropout_keep_rate, 
			self.train_phase: True}
		loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)

		return loss


	def training_op(self):
		"""Setup Training

		Return:
			The training op
		"""
		# Set the summary snapshot loss
		tf.summary.scalar('loss', self.loss)
		# # Create gradient descent
		# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		# Create a variable to track steps
		global_step = tf.Variable(0, name='global_step', trainable=False)
		# Use Optimizer to apply the gradients that minimize loss
		train_op = self.optimizer.minimize(self.loss, global_step=global_step)
		return train_op

	def evaluate(self, data, epoch=None, data_type='train'):  # evaluate the results for an input set
		num_example = len(data['Y'])
		feed_dict = {
			self.train_features: data['X'], 
			self.train_labels: [[y] for y in data['Y']], 
			self.dropout_keep: 1.0, 
			self.train_phase: False}
		predictions = self.sess.run((self.out), feed_dict=feed_dict)
		y_pred = np.reshape(predictions, (num_example,))
		y_true = np.reshape(data['Y'], (num_example,))

		# Loss
		loss = self.sess.run(self.loss, feed_dict=feed_dict)

		# Datatype
		if data_type == 'train':
			summary_type = self.summary_train
		elif data_type == 'valid':
			summary_type = self.summary_valid
		elif data_type == 'test':
			summary_type = self.summary_test

		# Update the events file.
		if epoch is not None:
			summary_str = self.sess.run(summary_type, feed_dict=feed_dict)
			self.summary_writer.add_summary(summary_str, epoch)
			self.summary_writer.flush()
		
		predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
		predictions_bounded = np.minimum(predictions_bounded, np.ones(num_example) * max(y_true))  # bound the higher values
		RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
		return RMSE


	def evaluate_termination(self, valid):
		if len(valid) > 5:
			if (valid[-1] > valid[-2] and 
				valid[-2] > valid[-3] and 
				valid[-3] > valid[-4] and 
				valid[-4] > valid[-5]):
				return True
		return False


if __name__ == '__main__':
	pass












