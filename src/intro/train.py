'''
Get model and data and train accordingly.

We separate out the training from the models in order to grid search not just
hyper-parameters but also models.

Also, we can set 

'''

import argparse

import tensorflow as tf


import load_data as DATA  # Data loading module
import model


#===== Parse Arguments
class parse_args():
	parser = argparse.ArgumentParser(description="Run Collaborative Filter models.")

	parser.add_argument('--filename', nargs=1, default='u.data',
                        help='Filename of the dataset.')
	parser.add_argument('--path', nargs=1, default='./data/ml-100k',
                        help='Path to the dataset folder.')
	parser.add_argument('--dataset', nargs=1, default='movielens-100k',
                        help='Which dataset type is it so that it is parsed accordingly.')
	parser.add_argument('--seed', nargs=1, default=1337,
                        help='Set seed for repoducibility.')
	parser.add_argument('--split', nargs=1, default=[0.8, 0.1, 0.1],
                        help='Dataset split proportion [train, valid, test].')



if __name__ == '__main__':
	pass