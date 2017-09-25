'''
Data pre process for AFM and FM
@author: 
Lizi Liao (liaolizi.llz@gmail.com)
Xiangnan He (xiangnanhe@gmail.com)
'''
import collections
import numpy as np
import os
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn import model_selection
from tensorflow.python.framework import dtypes

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


class DataSet(object):

	def __init__(self,
							 images,
							 labels,
							 fake_data=False,
							 one_hot=False,
							 dtype=dtypes.float32,
							 reshape=True):
		"""Construct a DataSet.
		one_hot arg is used only if fake_data is true.  `dtype` can be either
		`uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
		`[0, 1]`.
		"""
		dtype = dtypes.as_dtype(dtype).base_dtype
		if dtype not in (dtypes.uint8, dtypes.float32):
			raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
											dtype)
		if fake_data:
			self._num_examples = 10000
			self.one_hot = one_hot
		else:
			assert images.shape[0] == labels.shape[0], (
					'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
			self._num_examples = images.shape[0]

			# Convert shape from [num examples, rows, columns, depth]
			# to [num examples, rows*columns] (assuming depth == 1)
			if reshape:
				assert images.shape[3] == 1
				images = images.reshape(images.shape[0],
																images.shape[1] * images.shape[2])
			if dtype == dtypes.float32:
				# Convert from [0, 255] -> [0.0, 1.0].
				images = images.astype(np.float32)
				images = np.multiply(images, 1.0 / 255.0)
		self._images = images
		self._labels = labels
		self._epochs_completed = 0
		self._index_in_epoch = 0

	@property
	def images(self):
		return self._images

	@property
	def labels(self):
		return self._labels

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_completed(self):
		return self._epochs_completed

	def next_batch(self, batch_size, fake_data=False):
		"""Return the next `batch_size` examples from this data set."""
		if fake_data:
			fake_image = [1] * 784
			if self.one_hot:
				fake_label = [1] + [0] * 9
			else:
				fake_label = 0
			return [fake_image for _ in xrange(batch_size)], [
					fake_label for _ in xrange(batch_size)
			]
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_examples:
			# Finished epoch
			self._epochs_completed += 1
			# Shuffle the data
			perm = np.arange(self._num_examples)
			np.random.shuffle(perm)
			self._images = self._images[perm]
			self._labels = self._labels[perm]
			# Start next epoch
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples
		end = self._index_in_epoch
		return self._images[start:end], self._labels[start:end]



class sparse_to_dense(object):

	def __init__(self, path, filename, header=None, cols=[], data_split=[0.8, 0.1, 0.1], seed=1337):
		self.path       = path + "/"
		self.header     = header
		self.cols       = cols
		self.seed       = seed
		self.data_split = data_split
		self.filepath   = self.path + filename

		# For batch indexer
		self._index_in_epoch = 0
		self._epochs_completed = 0

		self.data           = self.read_data(self.filepath, self.header)
		self.data_dense     = self.long_to_wide(self.data, self.cols)
		self.data_dense_mat = self.dense_matrix(self.data)

		# Split and assign
		self.data_splitter(self.data_dense_mat, self.data_split, self.seed)

	# Read in delimited file using Pandas
	def read_data(self, filepath, header=None):
		print("Reading datafile [" + filepath + "]...")

		data = pd.read_table(filepath, sep="\t", header=None, names=header)
		return(data)

	# Convert from Long to Wide format
	def long_to_wide(self, data, cols=[], row_id=None, col_id=None):

		col_names = list(data)
		group_row = col_names[0] if row_id is None else row_id
		group_col = col_names[1] if col_id is None else col_id

		# Subset
		if(len(cols)):
			data = data[cols]

		# Dense Matrix
		print("Grouping by [" + group_row + "] and [" + group_col + "]...")
		data = data.groupby([group_row, group_col]).size().unstack(fill_value=0)

		return(data)

	# Convert to Dense Matrix Format
	def dense_matrix(self, data, label_id=None, row_id=None, col_id=None):

		col_names   = list(data)
		group_row   = col_names[0] if row_id is None else row_id
		group_col   = col_names[1] if col_id is None else col_id
		group_label = col_names[2] if label_id is None else label_id

		# Convert to List of Dictionaries
		X_raw = data.drop(group_label, axis=1)

		# Convert int to string so that sklearn indexes them
		X_raw.item_id = X_raw.item_id.astype(str)
		X_raw.user_id = X_raw.user_id.astype(str)

		# y = Labels
		y = data.as_matrix(columns=[group_label])

		# X - Features
		data_to_dict = X_raw.T.to_dict().values()
		data_to_dict = list(data_to_dict)

		v = DictVectorizer(sparse=True)
		X = v.fit_transform(data_to_dict)
		X_data = X.toarray()

		return X_data, y

	# Split into train, valid, test
	def data_splitter(self, data_dense_mat, data_split, seed):
		print('Splitting data...')
		features, labels = data_dense_mat

		# Squeeze the labels so they are (n, ) instead of (n, 1)
		labels = np.squeeze(labels)

		X_train, X_inter, y_train, y_inter = model_selection.train_test_split(
			features,
			labels,
			test_size=sum(data_split[1:3]),
			random_state=seed)
		X_valid, X_test, y_valid, y_test = model_selection.train_test_split(
			X_inter,
			y_inter,
			test_size=data_split[2] / sum(data_split[1:3]),
			random_state=seed)

		self.train = DataSet(X_train, y_train, reshape=False)
		self.valid = DataSet(X_valid, y_valid, reshape=False)
		self.test  = DataSet(X_test, y_test, reshape=False)

		# self.train = X_train, y_train
		# self.valid = X_valid, y_valid
		# self.test  = X_test, y_test
		self._num_examples, self.num_feature = features.shape


	# Indicate next batch
	def next_batch(self, batch_size, fake_data=False):
		"""Return the next `batch_size` examples from this data set."""
		if fake_data:
			fake_image = [1] * 784
			if self.one_hot:
				fake_label = [1] + [0] * 9
			else:
				fake_label = 0
				return [fake_image for _ in xrange(batch_size)], [
					fake_label for _ in xrange(batch_size)
					]
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_examples:
			# Finished epoch
			self._epochs_completed += 1
			# Shuffle the data
			perm = np.arange(self._num_examples)
			np.random.shuffle(perm)
			self._images = self._images[perm]
			self._labels = self._labels[perm]
			# Start next epoch
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples
		end = self._index_in_epoch
		return self._images[start:end], self._labels[start:end]


# Load data and get dense matrix form
# https://github.com/coreylynch/pyFM
# data = sparse_to_dense(
#     path="../../../data/ml-100k", 
#     filename="u.data", 
#     header=['user_id','item_id','rating','timestamp'],
#     cols=['user_id','item_id','rating'])


# X_train, y_train = data.train.next_batch(100)
# print(X_train)
# print(y_train)
# print(X_train.shape)
# print(y_train.shape)
# print(type(X_train))
# print(type(y_train))
# X_valid, y_valid = data.valid
# X_test, y_test   = data.test


# print(X_train.shape)
# print(X_valid.shape)
# print(X_test.shape)



