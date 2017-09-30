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


class FM(BaseEstimator, TransformerMixin):

	def __init__(self, features_p, factor_k, col_m,
				 lamda_bilinear, dropout_keep_rate, 
				 epoch, batch_size, learning_rate, optimizer_type, batch_norm, 
				 verbose, seed=1337):
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

		self.verbose = verbose
		self.seed = seed

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
			self.train_labels   = tf.placeholder(tf.int32, shape=[None, 1], name='train_label_fm')  # None * 1
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

			#=== Optimizer.
			if self.optimizer_type == 'AdamOptimizer':
				self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
			elif self.optimizer_type == 'AdagradOptimizer':
				self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)
			elif self.optimizer_type == 'GradientDescentOptimizer':
				self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
			elif self.optimizer_type == 'MomentumOptimizer':
				self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(self.loss)

			#=== Initialize
			self.sess = self._init_session()
			self.saver = tf.train.Saver()
			init = tf.global_variables_initializer()
			self.sess.run(init)

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
				print "#params: %d" %total_parameters 


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


	#==============================================================================================
	# Evaluation
	#==============================================================================================
	def train(self, data_train, data_valid, data_test):
		'''
		Fit a dataset
		'''

		# Report
		if self.verbose:
			t2 = time()
			init_train = self.evaluate(data_train)
			init_valid = self.evaluate(data_valid)
			print("Init: \t train=%.4f, validation=%.4f [%.1f s]" %(init_train, init_valid, time()-t2))

		# Training Epochs
		for epoch in xrange(self.epoch):
			t1 = time()

if __name__ == '__main__':
	pass















