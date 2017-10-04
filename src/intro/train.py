'''
Get model and data and train accordingly.

We separate out the training from the models in order to grid search not just
hyper-parameters but also models.

Also, we can set 

'''

import math
import os
import numpy as np
import pandas as pd
import argparse
from time import time

import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

import load_data as DATA  # Data loading module
import model as model_classes


#===== Parse Arguments
def parse_args_train():
	parser = argparse.ArgumentParser(description="Run Collaborative Filter models.")

	parser.add_argument('--filename', nargs=1, default='u.data',
						help='Filename of the dataset.')
	parser.add_argument('--path', nargs=1, default='./data/ml-100k',
						help='Path to the dataset folder.')
	parser.add_argument('--path_output', nargs=1, default='./output/ml-100k',
						help='Path to the output folder.')
	parser.add_argument('--dataset', nargs=1, default='movielens-100k',
						help='Which dataset type is it so that it is parsed accordingly.')
	parser.add_argument('--seed', nargs=1, type=int, default=1337,
						help='Set seed for repoducibility.')
	parser.add_argument('--split', nargs=1, default=[0.8, 0.1, 0.1],
						help='Dataset split proportion [train, valid, test].')

	# Model
	parser.add_argument('--model_type', nargs='?', default='AFM',
						help='Model to run.')
	parser.add_argument('--process', nargs='?', default='train',
						help='Process type: train, evaluate.')
	parser.add_argument('--mla', type=int, default=0,
						help='Set the experiment mode to be Micro Level Analysis or not: 0-disable, 1-enable.')
	parser.add_argument('--epoch', type=int, default=20,
						help='Number of epochs.')
	parser.add_argument('--pretrain', type=int, default=-1,
						help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save to pretrain file')
	parser.add_argument('--batch_size', type=int, default=20480,
						help='Batch size.')
	parser.add_argument('--self_terminate', type=bool, default=False,
						help='Should the model stop on its own if no improvements?')
	parser.add_argument('--factor_k', type=int, default=16,
						help='Number of hidden factors.')
	parser.add_argument('--lamda_bilinear', type=float, default=0,
						help='Regularizer for bilinear part.')
	parser.add_argument('--dropout_keep_rate', type=float, default=0.7, 
						help='Keep probility (1-dropout) for the bilinear interaction layer. 1: no dropout')
	parser.add_argument('--learning_rate', type=float, default=0.01,
						help='Learning rate.')
	parser.add_argument('--optimizer_type', nargs='?', default='AdagradOptimizer',
						help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
	parser.add_argument('--verbose', type=int, default=True,
						help='Whether to show the performance of each epoch (0 or 1)')
	parser.add_argument('--batch_norm', type=int, default=1,
						help='Whether to perform batch normaization (0 or 1)')


	# Attentional Factorization
	parser.add_argument('--hidden_factor_1', nargs=1, type=int, default=16,
                        help='Number of hidden factors.')
	parser.add_argument('--hidden_factor_2', nargs=1, type=int, default=16,
                        help='Number of hidden factors.')
	parser.add_argument('--valid_dimen', type=int, default=3,
						help='Valid dimension of the dataset. The number of columns (e.g. frappe=10, ml-tag=3)')
	parser.add_argument('--attention', type=int, default=1,
						help='flag for attention. 1: use attention; 0: no attention')
	parser.add_argument('--lamda_attention', type=float, default=1e+2,
						help='Regularizer for attention part.')
	# parser.add_argument('--keep', nargs='+', type=float, default=[1.0, 0.5],
	# 					help='Keep probility (1-dropout) of each layer. 1: no dropout. The first index is for the attention-aware pairwise interaction layer.')
	parser.add_argument('--keep_1', nargs=1, type=float, default=1.0,
						help='Keep probility (1-dropout) of each layer. 1: no dropout. The first index is for the attention-aware pairwise interaction layer.')
	parser.add_argument('--keep_2', nargs=1, type=float, default=0.5,
						help='Keep probility (1-dropout) of each layer. 1: no dropout. The first index is for the attention-aware pairwise interaction layer.')
	parser.add_argument('--lr', type=float, default=0.1,
						help='Learning rate.')
	parser.add_argument('--freeze_fm', type=int, default=0,
						help='Freese all params of fm and learn attention params only.')
	parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
						help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
	parser.add_argument('--decay', type=float, default=0.999,
						help='Decay value for batch norm')
	parser.add_argument('--activation', nargs='?', default='relu',
						help='Which activation function to use for deep layers: relu, sigmoid, tanh, identity')


	return parser.parse_args()




def make_save_file(args):


	if args.model_type == 'FM':
		pretrain_path = './pretrain/' + args.model_type + '_%s_%d' %(args.dataset, args.factor_k)
	elif args.model_type == 'AFM':
		pretrain_path = './pretrain/' + args.model_type + '_%s_%d_%d' %(args.dataset, args.hidden_factor_1, args.hidden_factor_2)

	if args.mla:
		pretrain_path += '_mla'
	if not os.path.exists(pretrain_path):
		os.makedirs(pretrain_path)

	if args.model_type == 'FM':
		save_file = pretrain_path + '/%s_%d' %(args.dataset, args.factor_k)
	elif args.model_type == 'AFM':
		save_file = pretrain_path + '/%s_%d_%d' %(args.dataset, args.hidden_factor_1, args.hidden_factor_2)
	return save_file

def train(args):

	# Dictionary of arguments
	argv = vars(args)

	# Data loading
	data = DATA.LoadData(args.path, args.dataset, args.seed)

	# Get arguments from data
	argv['features_p'] = data.features_p
	argv['col_m'] = data.col_m

	if args.verbose > 0:
		print("FM: dataset=%s, factors=%d, #epoch=%d, batch=%d, lr=%.4f, lambda=%.1e, keep=%.2f, optimizer=%s, batch_norm=%d"
			  %(args.dataset, args.factor_k, args.epoch, args.batch_size, args.learning_rate, args.lamda_bilinear, args.dropout_keep_rate, args.optimizer_type, args.batch_norm))


	t1 = time()

	# Choose model
	if args.model_type == 'FM':
		model_class = model_classes.fm.FM
		model = model_class(
				features_p = data.features_p, 
				factor_k = args.factor_k, 
				col_m = data.col_m,
				lamda_bilinear = args.lamda_bilinear, 
				dropout_keep_rate = args.dropout_keep_rate, 
				epoch = args.epoch, 
				batch_size = args.batch_size, 
				learning_rate = args.learning_rate, 
				optimizer_type = args.optimizer_type, 
				batch_norm = args.batch_norm, 
				pretrain_flag = args.pretrain,
				save_file = make_save_file(args),
				self_terminate = args.self_terminate,
				verbose = args.verbose, 
				seed=1337)
	elif args.model_type == 'AFM':
		model_class = model_classes.afm.AFM
		model = model_class(
				features_p=data.features_p, 
				pretrain_flag= args.pretrain, 
				save_file=make_save_file(args), 
				attention=args.attention, 
				hidden_factor_1=args.hidden_factor_1, 
				hidden_factor_2=args.hidden_factor_2, 
				valid_dimension=data.col_m, 
				activation_function=args.activation, 
				freeze_fm=args.freeze_fm, 
				epoch = args.epoch, 
				batch_size = args.batch_size, 
				learning_rate = args.learning_rate, 
				lamda_attention=args.lamda_attention, 
				keep_1=args.keep_1, 
				keep_2=args.keep_2, 
				optimizer_type = args.optimizer_type, 
				batch_norm = args.batch_norm, 
				decay=args.decay, 
				verbose = args.verbose, 
				micro_level_analysis=args.mla, 
				random_seed=args.seed)
	else:
		print("=== Please select a model type.")
		return


	# Begin Training
	model.train(data.data_train, data.data_valid, data.data_test)
	
	
	# Find the best validation result across iterations
	best_valid_score = 0
	best_valid_score = min(model.valid_rmse)
	best_epoch = model.valid_rmse.index(best_valid_score)
	print ("Best Iter(validation)= %d\t train = %.4f, valid = %.4f [%.1f s]" 
		   %(best_epoch+1, model.train_rmse[best_epoch], model.valid_rmse[best_epoch], time()-t1))

def evaluate(args):
	# load test data
	data = DATA.LoadData(args.path, args.dataset, args.seed).data_test
	save_file = make_save_file(args)
	
	# load the graph
	weight_saver = tf.train.import_meta_graph(save_file + '.meta')
	pretrain_graph = tf.get_default_graph()

	# load tensors 
	feature_embeddings = pretrain_graph.get_tensor_by_name('feature_embeddings:0')
	nonzero_embeddings = pretrain_graph.get_tensor_by_name('nonzero_embeddings:0')
	feature_bias = pretrain_graph.get_tensor_by_name('feature_bias:0')
	bias = pretrain_graph.get_tensor_by_name('bias:0')
	fm = pretrain_graph.get_tensor_by_name('fm:0')
	fm_out = pretrain_graph.get_tensor_by_name('fm_out:0')
	out = pretrain_graph.get_tensor_by_name('out:0')
	train_features = pretrain_graph.get_tensor_by_name('train_features_fm:0')
	train_labels = pretrain_graph.get_tensor_by_name('train_labels_fm:0')
	dropout_keep = pretrain_graph.get_tensor_by_name('dropout_keep_fm:0')
	train_phase = pretrain_graph.get_tensor_by_name('train_phase_fm:0')


	# restore session
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	weight_saver.restore(sess, save_file)

	# start evaluation
	num_example = len(data['Y'])
	feed_dict = {train_features: data['X'], train_labels: [[y] for y in data['Y']], dropout_keep: 1.0, train_phase: False}
	ne, fe = sess.run((nonzero_embeddings, feature_embeddings), feed_dict=feed_dict)
	_fm, _fm_out, predictions = sess.run((fm, fm_out, out), feed_dict=feed_dict)

	# calculate rmse
	y_pred = np.reshape(predictions, (num_example,))
	y_true = np.reshape(data['Y'], (num_example,))
	
	predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
	predictions_bounded = np.minimum(predictions_bounded, np.ones(num_example) * max(y_true))  # bound the higher values
	RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))

	print("Test RMSE: %.4f"%(RMSE))

	# Unify into dataframe
	y_df = pd.DataFrame({'label':y_true, 'pred':y_pred})


	# Write
	if not os.path.exists(args.path_output):
		os.makedirs(args.path_output)
	fullpath_output = args.path_output + '/predictions_%s_%d.csv' %(args.dataset, args.factor_k)
	y_df.to_csv(fullpath_output, index=False)


if __name__ == '__main__':
	args = parse_args_train()

	# initialize the optimal parameters
	if args.mla:
		args.lr = 0.05
		args.keep = 0.7
		args.batch_norm = 0
	else:
		args.lr = 0.01
		args.keep = 0.7
		args.batch_norm = 1

	args.process = 'train'
	args.epoch = 2000
	args.factor_k = 4
	args.hidden_factor_1 = 12
	args.hidden_factor_2 = 12

	if args.process == 'train':
		train(args)
	elif args.process == 'evaluate':
		evaluate(args)