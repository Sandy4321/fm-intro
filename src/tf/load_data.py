'''
Data pre process for AFM and FM
@author: 
Lizi Liao (liaolizi.llz@gmail.com)
Xiangnan He (xiangnanhe@gmail.com)
'''
import numpy as np
import os
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer


class LoadData(object):
    '''given the path of data, return the data format for AFM and FM
    :param path
    return:
    Train_data: a dictionary, 'Y' refers to a list of y values; 'X' refers to a list of features_M dimension vectors with 0 or 1 entries
    Test_data: same as Train_data
    Validation_data: same as Train_data
    '''

    # Three files are needed in the path
    def __init__(self, path, name_train, name_valid, name_test, loss_type="square_loss"):
        self.path               = path + "/"
        self.trainfile          = self.path + name_train
        self.testfile           = self.path + name_test
        self.validationfile     = self.path + name_valid
        self.features_M         = self.map_features( )
        self.Train_data, self.Validation_data, self.Test_data = self.construct_data( loss_type )

    def map_features(self): # map the feature entries in all files, kept in self.features dictionary
        self.features = {}
        self.read_features(self.trainfile)
        self.read_features(self.testfile)
        self.read_features(self.validationfile)
        # print("features_M:", len(self.features))
        return  len(self.features)

    def read_features(self, file): # read a feature file
        f = open( file )
        line = f.readline()
        i = len(self.features)
        while line:
            items = line.strip().split(' ')
            for item in items[1:]:
                if item not in self.features:
                    self.features[ item ] = i
                    i = i + 1
            line = f.readline()
        f.close()

    def construct_data(self, loss_type):
        X_, Y_ , Y_for_logloss= self.read_data(self.trainfile)
        if loss_type == 'log_loss':
            Train_data = self.construct_dataset(X_, Y_for_logloss)
        else:
            Train_data = self.construct_dataset(X_, Y_)
        #print("Number of samples in Train:" , len(Y_))

        X_, Y_ , Y_for_logloss= self.read_data(self.validationfile)
        if loss_type == 'log_loss':
            Validation_data = self.construct_dataset(X_, Y_for_logloss)
        else:
            Validation_data = self.construct_dataset(X_, Y_)
        #print("Number of samples in Validation:", len(Y_))

        X_, Y_ , Y_for_logloss = self.read_data(self.testfile)
        if loss_type == 'log_loss':
            Test_data = self.construct_dataset(X_, Y_for_logloss)
        else:
            Test_data = self.construct_dataset(X_, Y_)
        #print("Number of samples in Test:", len(Y_))

        return Train_data,  Validation_data,  Test_data

    def read_data(self, file):
        # read a data file. For a row, the first column goes into Y_;
        # the other columns become a row in X_ and entries are maped to indexs in self.features
        f = open( file )
        X_ = []
        Y_ = []
        Y_for_logloss = []
        line = f.readline()
        while line:
            items = line.strip().split(' ')
            Y_.append( 1.0*float(items[0]) )

            if float(items[0]) > 0:# > 0 as 1; others as 0
                v = 1.0
            else:
                v = 0.0
            Y_for_logloss.append( v )

            X_.append( [ self.features[item] for item in items[1:]] )
            line = f.readline()
        f.close()
        return X_, Y_, Y_for_logloss

    def construct_dataset(self, X_, Y_):
        Data_Dic = {}
        X_lens = [ len(line) for line in X_]
        indexs = np.argsort(X_lens)
        Data_Dic['Y'] = [ Y_[i] for i in indexs]
        Data_Dic['X'] = [ X_[i] for i in indexs]
        return Data_Dic
    
    def truncate_features(self):
        """
        Make sure each feature vector is of the same length
        """
        num_variable = len(self.Train_data['X'][0])
        for i in xrange(len(self.Train_data['X'])):
            num_variable = min([num_variable, len(self.Train_data['X'][i])])
        # truncate train, validation and test
        for i in xrange(len(self.Train_data['X'])):
            self.Train_data['X'][i] = self.Train_data['X'][i][0:num_variable]
        for i in xrange(len(self.Validation_data['X'])):
            self.Validation_data['X'][i] = self.Validation_data['X'][i][0:num_variable]
        for i in xrange(len(self.Test_data['X'])):
            self.Test_data['X'][i] = self.Test_data['X'][i][0:num_variable]
        return num_variable




class sparse_to_dense(object):

    def __init__(self, path, filename, header=None, cols=[]):
        self.path       = path + "/"
        self.header     = header
        self.cols       = cols
        self.filepath   = self.path + filename
        self.data       = self.read_data(self.filepath, self.header)
        self.data_dense = self.long_to_wide(self.data, self.cols)
        self.data_dense_mat = self.dense_matrix(self.data)

    # Read in delimited file using Pandas
    def read_data(self, filepath, header=None):
        print("\tReading datafile [" + filepath + "]...")

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
        print("\tGrouping by [" + group_row + "] and [" + group_col + "]")
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



# # Load data and get dense matrix form
# # https://github.com/coreylynch/pyFM
# data = sparse_to_dense(
#     path="../../data/ml-100k", 
#     filename="u.data", 
#     header=['user_id','item_id','rating','timestamp'],
#     cols=['user_id','item_id','rating'])


# print(data.data_dense_mat)

# X, y = data.data_dense_mat


# print(X)
# print(y)
# print(X.shape)



