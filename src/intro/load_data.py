'''
Load Data 

	Generalized data loader that can read CSVs for models.
	The training script will import any kind of data and any kind of model
	that fits the model class:

			Eval 
		   /	\
	Model 1   	 load_data (this)
	Model 2
	  etc.


'''

import sys
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf


# Parse arguments for loading data
def parse_args():
    parser = argparse.ArgumentParser(description="Load Data.")
    parser.add_argument('--filename', nargs=1, default='u.data',
                        help='Filename of the dataset.')
    parser.add_argument('--path', nargs=1, default='./data/movielens-100k',
                        help='Path to the dataset folder.')
    parser.add_argument('--dataset', nargs=1, default='movielens-100k',
                        help='Which dataset type is it so that it is parsed accordingly.')
    return parser.parse_args()



# Load the data    
class LoadData(object):

	def __init__(self, path, filename, dataset='none'):
		self.path = path
		self.filename = filename
		self.full_path = self.path + '/' + self.filename
		self.dataset = dataset

		self.data = self.ParseRaw()


	# If the dataset is raw then parse through it
	def ParseRaw(self):

		# handle datasets accordingly
		if(self.dataset == 'movielens-100k'):
			return self.ML100K()
		elif(self.dataset == 'ml-latest-small'):
			return self.MLLatestSmall()
		else:
		elif(self.dataset == 'none'):
			print("No dataset selected")
		else:
			print("Dataset parsing method not available.")


	def ML100K(self):
		'''
		Movie-lens 100k dataset: http://files.grouplens.org/datasets/movielens/ml-100k/README
		Ratings: 100,000
		Users:       943
		Items:     1,682
		'''

		# User
		u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
		users = pd.read_csv(self.path + '/' + 'u.user', sep='|', names=u_cols, 
			encoding='latin-1')

		# Rating
		r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
		ratings = pd.read_csv(self.path + '/' + 'u.data', sep='\t', names=r_cols, 
			encoding='latin-1')

		# Movie
		m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
		m_cols_type = ['unknown', 'action', 'adventure', 'animation', 'childrens', 'comedy', 
			'crime', 'documentary', 'drama', 'fantasy', 'noir', 'horror', 'musical', 
			'mystery', 'romance', 'scifi', 'thriller', 'war', 'western']
		movies = pd.read_csv(self.path + '/' + 'u.item', sep='|', names=m_cols + m_cols_type,
			encoding='latin-1')

		# Merge
		movie_ratings = pd.merge(movies, ratings)
		lens = pd.merge(movie_ratings, users)

		# Drop columns not needed


		return lens
		

	def MLLatestSmall(self):
		
		# User
		u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
		users = pd.read_csv(self.path + '/' + 'u.user', sep='|', names=u_cols, 
			encoding='latin-1')
		




# If run alone then simply read
if __name__ == '__main__':

	# Parse arguments
	args = parse_args()
	argv = vars(args)

	# Pass argumants
	data = LoadData(**argv)

	print(data.data.head())