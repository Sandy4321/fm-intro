
'''
Learning the basics of TensorFlow through a basic linear model
https://www.tensorflow.org/tutorials/wide
'''

import tempfile  # Create Temp Files of public datasets
import urllib

import tensorflow as tf
import pandas as pd

# train_file = tempfile.NamedTemporaryFile()
# test_file = tempfile.NamedTemporaryFile()

#urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)
#urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)

data_path = '../../../data/lm'

# import data
CSV_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "gender",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income_bracket"]
df_train = pd.read_csv(data_path + '/' + 'adult.data', names=CSV_COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(data_path + '/' + 'adult.test', names=CSV_COLUMNS, skipinitialspace=True, skiprows=1)

print(df_train.shape)
print(df_test.shape)

# clean labels
train_labels = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
test_labels = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)


def input_fn(data_file, num_epochs, shuffle):
  """Input builder function."""
  df_data = pd.read_csv(
      tf.gfile.Open(data_file),
      names=CSV_COLUMNS,
      skipinitialspace=True,
      engine="python",
      skiprows=1)
  # remove NaN elements
  df_data = df_data.dropna(how="any", axis=0)
  labels = df_data["income_bracket"].apply(lambda x: ">50K" in x).astype(int)
  return tf.estimator.inputs.pandas_input_fn(
      x=df_data,
      y=labels,
      batch_size=100,
      num_epochs=num_epochs,
      shuffle=shuffle,
      num_threads=5)

